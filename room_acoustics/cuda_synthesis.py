"""
CUDA GPU synthesis kernel for the unified modal pipeline.

Parallelizes the recursive oscillator across modes using CuPy raw
CUDA kernels. Each GPU thread handles one mode, accumulating into
a thread-local buffer, then reduced to the final IR via atomicAdd.

Expected performance on RTX 2060 (1920 cores):
  37K modes × 30K samples → ~10-50ms (vs 2.9s CPU Numba = 60-300x)
  Calibration (35 evals): ~2s total (vs 100 min CPU)

Falls back to CPU Numba/numpy if CuPy is not available.

Usage:
    from room_acoustics.cuda_synthesis import synthesize_gpu, has_gpu

    if has_gpu():
        ir = synthesize_gpu(amp, gam, wd, nc, n_samples, dt)
    else:
        # falls back to CPU
        ir = synthesize_gpu(amp, gam, wd, nc, n_samples, dt)
"""

import numpy as np

# Check for CuPy
_HAVE_CUPY = False
try:
    import cupy as cp
    _HAVE_CUPY = True
except ImportError:
    cp = None


def has_gpu():
    """Check if CUDA GPU is available via CuPy."""
    return _HAVE_CUPY


# CUDA kernel: one thread per mode, recursive oscillator,
# atomicAdd into shared output array
_CUDA_KERNEL_SRC = r"""
extern "C" __global__
void modal_synthesis(
    const double* amp,      // (n_modes,) amplitude per mode
    const double* gam,      // (n_modes,) decay rate per mode
    const double* wd,       // (n_modes,) damped angular frequency
    const long long* nc,    // (n_modes,) early termination sample count
    double* ir,             // (n_samples,) output IR (atomicAdd)
    int n_modes,
    int n_samples,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_modes) return;

    double a = amp[i];
    double g = gam[i];
    double w = wd[i];
    int n = (int)nc[i];
    if (n <= 0 || n > n_samples) n = n_samples;
    if (abs(a) < 1e-30) return;

    // Precompute per-mode constants (only transcendentals here, once)
    double decay_step = exp(-g * dt);
    double cos_coeff = 2.0 * cos(w * dt);

    // Initial values
    double env = 1.0;
    double cos_prev = 1.0;
    double cos_curr = cos(w * dt);

    // k=0
    atomicAdd(&ir[0], a * env * cos_prev);

    // k=1
    if (n > 1) {
        env *= decay_step;
        atomicAdd(&ir[1], a * env * cos_curr);
    }

    // k=2..n-1: pure multiply+subtract, no transcendentals
    for (int k = 2; k < n; k++) {
        env *= decay_step;
        double cos_next = cos_coeff * cos_curr - cos_prev;
        atomicAdd(&ir[k], a * env * cos_next);
        cos_prev = cos_curr;
        cos_curr = cos_next;
    }
}
"""

# Compiled kernel (lazy init)
_kernel = None


def _get_kernel():
    """Compile CUDA kernel on first use."""
    global _kernel
    if _kernel is None and _HAVE_CUPY:
        _kernel = cp.RawKernel(_CUDA_KERNEL_SRC, 'modal_synthesis')
    return _kernel


def synthesize_gpu(amp, gam, wd, nc, n_samples, dt):
    """
    Synthesize IR on GPU using CUDA recursive oscillator kernel.

    Falls back to CPU Numba if CuPy not available.

    Parameters
    ----------
    amp : array (n_modes,), float64
        Mode amplitudes.
    gam : array (n_modes,), float64
        Decay rates.
    wd : array (n_modes,), float64
        Damped angular frequencies.
    nc : array (n_modes,), int64
        Per-mode early termination sample count.
    n_samples : int
        Output IR length.
    dt : float
        Time step (1/sr).

    Returns
    -------
    ir : array (n_samples,), float64
    """
    if not _HAVE_CUPY:
        return _synthesize_cpu_fallback(amp, gam, wd, nc, n_samples, dt)

    kernel = _get_kernel()
    n_modes = len(amp)

    # Transfer to GPU
    d_amp = cp.asarray(amp, dtype=cp.float64)
    d_gam = cp.asarray(gam, dtype=cp.float64)
    d_wd = cp.asarray(wd, dtype=cp.float64)
    d_nc = cp.asarray(nc, dtype=cp.int64)
    d_ir = cp.zeros(n_samples, dtype=cp.float64)

    # Launch: one thread per mode
    block_size = 256
    grid_size = (n_modes + block_size - 1) // block_size

    kernel((grid_size,), (block_size,),
           (d_amp, d_gam, d_wd, d_nc, d_ir,
            np.int32(n_modes), np.int32(n_samples), np.float64(dt)))

    # Transfer back
    ir = cp.asnumpy(d_ir)
    return ir


def _synthesize_cpu_fallback(amp, gam, wd, nc, n_samples, dt):
    """CPU fallback using Numba JIT or numpy."""
    try:
        from .analytical_modes import _synthesize_numba, _HAVE_NUMBA
        if _HAVE_NUMBA:
            ir = np.zeros(n_samples, dtype=np.float64)
            return _synthesize_numba(ir, amp, gam, wd, nc, dt)
    except ImportError:
        pass

    # Pure numpy fallback
    ir = np.zeros(n_samples, dtype=np.float64)
    t = np.arange(n_samples, dtype=np.float64) * dt
    for i in range(len(amp)):
        if abs(amp[i]) < 1e-30 or nc[i] < 10:
            continue
        n = min(int(nc[i]), n_samples)
        ir[:n] += amp[i] * np.exp(-gam[i] * t[:n]) * np.cos(wd[i] * t[:n])
    return ir


def synthesize_ir_gpu(modes, T=3.0, sr=44100, ism_ir=None):
    """
    Drop-in replacement for unified_modes.synthesize_ir using GPU.

    Parameters
    ----------
    modes : structured array (MODE_DTYPE)
    T : float
    sr : int
    ism_ir : array or None

    Returns
    -------
    ir : array (n_samples,), float64
    """
    n_samples = int(T * sr)
    dt = 1.0 / sr

    # Filter active modes
    mask = (np.abs(modes['amplitude']) > 1e-20) & (modes['omega_d'] > 0)
    active = modes[mask]

    if len(active) == 0:
        ir = np.zeros(n_samples, dtype=np.float64)
    else:
        amp = np.ascontiguousarray(active['amplitude'], dtype=np.float64)
        gam = np.ascontiguousarray(active['decay_rate'], dtype=np.float64)
        wd = np.ascontiguousarray(active['omega_d'], dtype=np.float64)

        # Per-mode early termination
        safe_gamma = np.maximum(gam, 0.01)
        t_80dB = 80.0 * np.log(10) / (20.0 * safe_gamma)
        nc = np.minimum(np.floor(t_80dB * sr).astype(np.int64) + 1, n_samples)

        # Amplitude threshold
        amp_thresh = max(np.max(np.abs(amp)) * 1e-3, 1e-20)
        sig = np.abs(amp) > amp_thresh
        amp, gam, wd, nc = amp[sig], gam[sig], wd[sig], nc[sig]

        ir = synthesize_gpu(amp, gam, wd, nc, n_samples, dt)

    # Add ISM
    if ism_ir is not None:
        n = min(len(ism_ir), n_samples)
        ir[:n] += ism_ir[:n]

    return ir
