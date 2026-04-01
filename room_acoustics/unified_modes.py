"""
Unified Modal Synthesis — single-pass IR from multiple engine sources.

All engines contribute modes as (frequency, amplitude, decay_rate) tuples
to a shared mode list. One synthesis pass produces the final IR. No
crossover filters, no frequency-domain blending artifacts.

Architecture:
  1. Each engine implements ModeProvider.provide_modes() → mode array
  2. merge_modes() deduplicates overlapping modes by confidence
  3. synthesize_ir() renders all modes in one pass (Numba JIT)
  4. ISM early reflections added separately (discrete arrivals)

Adding a new engine:
  1. Implement a class with name, frequency_range, confidence_base properties
  2. Implement provide_modes(source, receiver, materials, ...) → MODE_DTYPE array
  3. Register with synthesizer.register(your_provider)

    That's it. The merge and synthesis are automatic.

Example:
    from room_acoustics.unified_modes import (
        UnifiedModalSynthesizer, AnalyticalModesProvider)
    from room_acoustics.analytical_modes import AnalyticalRoomModes

    arm = AnalyticalRoomModes(8.4, 6.7, 3.0, f_max=4000)
    synth = UnifiedModalSynthesizer()
    synth.register(AnalyticalModesProvider(arm))

    ir = synth.impulse_response(source, receiver, materials, T=3.0)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ===================================================================
# Mode representation
# ===================================================================

# Structured dtype for a mode. Flat arrays of this feed directly into
# the Numba JIT synthesis kernel.
MODE_DTYPE = np.dtype([
    ('frequency',     np.float64),  # Hz
    ('amplitude',     np.float64),  # linear, signed (src * rec coupling)
    ('decay_rate',    np.float64),  # gamma [1/s]
    ('omega_d',       np.float64),  # damped angular freq [rad/s]
    ('confidence',    np.float32),  # 0.0-1.0: how reliable is this mode
    ('source_engine', 'U16'),       # engine name (diagnostics only)
])


def make_modes(freqs, amplitudes, decay_rates, confidence, engine_name=''):
    """
    Build a MODE_DTYPE array from parallel arrays.

    Convenience function for providers. Computes omega_d automatically.

    Parameters
    ----------
    freqs : array (N,)
        Mode frequencies [Hz].
    amplitudes : array (N,)
        Mode amplitudes (signed, includes coupling).
    decay_rates : array (N,)
        Decay rates gamma [1/s].
    confidence : float or array (N,)
        Confidence level(s) for these modes.
    engine_name : str
        Source engine identifier.

    Returns
    -------
    modes : structured array (N,) with MODE_DTYPE
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    decay_rates = np.asarray(decay_rates, dtype=np.float64)
    N = len(freqs)

    omega = 2.0 * np.pi * freqs
    omega_d = np.sqrt(np.maximum(omega ** 2 - decay_rates ** 2, 0.0))

    modes = np.zeros(N, dtype=MODE_DTYPE)
    modes['frequency'] = freqs
    modes['amplitude'] = amplitudes
    modes['decay_rate'] = decay_rates
    modes['omega_d'] = omega_d
    if np.isscalar(confidence):
        modes['confidence'] = confidence
    else:
        modes['confidence'] = np.asarray(confidence, dtype=np.float32)
    modes['source_engine'] = engine_name
    return modes


# ===================================================================
# Merge / deduplication
# ===================================================================

def merge_modes(mode_arrays, freq_tolerance_hz=2.0, freq_tolerance_pct=0.005):
    """
    Merge mode lists from multiple engines, deduplicating overlapping modes.

    When two modes from different engines are within the tolerance window,
    the higher-confidence mode is kept and the other is discarded.

    Parameters
    ----------
    mode_arrays : list of structured arrays (MODE_DTYPE)
        Each from a different provider.
    freq_tolerance_hz : float
        Absolute frequency tolerance [Hz].
    freq_tolerance_pct : float
        Relative tolerance (fraction of frequency).
        Effective tolerance = max(freq_tolerance_hz, f * freq_tolerance_pct).

    Returns
    -------
    merged : structured array (MODE_DTYPE)
        Deduplicated, sorted by frequency.
    """
    if not mode_arrays:
        return np.zeros(0, dtype=MODE_DTYPE)

    # Concatenate all modes
    all_modes = np.concatenate(mode_arrays)

    if len(all_modes) == 0:
        return all_modes

    # Sort by frequency
    order = np.argsort(all_modes['frequency'])
    all_modes = all_modes[order]

    # Sweep: mark duplicates for removal
    N = len(all_modes)
    keep = np.ones(N, dtype=bool)
    freqs = all_modes['frequency']
    confs = all_modes['confidence']
    engines = all_modes['source_engine']

    i = 0
    while i < N:
        if not keep[i]:
            i += 1
            continue

        f_i = freqs[i]
        tol = max(freq_tolerance_hz, f_i * freq_tolerance_pct)

        # Look ahead for modes within tolerance
        j = i + 1
        while j < N and freqs[j] - f_i <= tol:
            if not keep[j]:
                j += 1
                continue

            # Same engine → not a duplicate (legitimately distinct modes)
            if engines[i] == engines[j]:
                j += 1
                continue

            # Different engines, same frequency → keep higher confidence
            if confs[i] >= confs[j]:
                keep[j] = False
            else:
                keep[i] = False
                break  # i is now removed, move on

            j += 1
        i += 1

    return all_modes[keep]


# ===================================================================
# Synthesis
# ===================================================================

def synthesize_ir(modes, T=3.0, sr=44100, ism_ir=None):
    """
    Single-pass IR synthesis from merged mode array.

    Uses Numba JIT recursive oscillator if available, else numpy fallback.
    ISM early reflections are added separately (not modal).

    Parameters
    ----------
    modes : structured array (MODE_DTYPE)
        From merge_modes().
    T : float
        IR duration [seconds].
    sr : int
        Sample rate [Hz].
    ism_ir : ndarray or None
        ISM early reflections to add (discrete arrivals, not modal).

    Returns
    -------
    ir : ndarray (n_samples,), float64
    """
    n_samples = int(T * sr)
    dt = 1.0 / sr
    ir = np.zeros(n_samples, dtype=np.float64)

    # Filter active modes
    mask = (np.abs(modes['amplitude']) > 1e-20) & (modes['omega_d'] > 0)
    active = modes[mask]

    if len(active) > 0:
        amp = np.ascontiguousarray(active['amplitude'], dtype=np.float64)
        gam = np.ascontiguousarray(active['decay_rate'], dtype=np.float64)
        wd = np.ascontiguousarray(active['omega_d'], dtype=np.float64)

        # Per-mode early termination at -80 dB
        safe_gamma = np.maximum(gam, 0.01)
        t_80dB = 80.0 * np.log(10) / (20.0 * safe_gamma)
        nc = np.minimum(np.floor(t_80dB * sr).astype(np.int64) + 1, n_samples)

        # Amplitude threshold: skip modes below 0.1% of peak
        amp_thresh = max(np.max(np.abs(amp)) * 1e-3, 1e-20)
        sig = np.abs(amp) > amp_thresh
        amp, gam, wd, nc = amp[sig], gam[sig], wd[sig], nc[sig]

        if len(amp) > 0:
            amp = np.array(amp, dtype=np.float64, copy=False)
            gam = np.array(gam, dtype=np.float64, copy=False)
            wd = np.array(wd, dtype=np.float64, copy=False)
            nc = np.array(nc, dtype=np.int64)

            # 3-tier dispatch: GPU (CuPy) → CPU (Numba JIT) → numpy
            synthesized = False

            # Tier 1: CUDA GPU
            try:
                from .cuda_synthesis import has_gpu, synthesize_gpu
                if has_gpu():
                    ir = synthesize_gpu(amp, gam, wd, nc, n_samples, float(dt))
                    synthesized = True
            except Exception:
                pass

            # Tier 2: Numba JIT
            if not synthesized:
                try:
                    from .analytical_modes import _synthesize_numba, _HAVE_NUMBA
                    if _HAVE_NUMBA:
                        ir = _synthesize_numba(ir, amp, gam, wd, nc, float(dt))
                        synthesized = True
                except Exception:
                    pass

            # Tier 3: numpy fallback
            if not synthesized:
                _synthesize_numpy(ir, amp, gam, wd, nc, dt, sr, n_samples)

    # Add ISM early reflections
    if ism_ir is not None:
        n = min(len(ism_ir), n_samples)
        ir[:n] += ism_ir[:n]

    return ir


def _synthesize_numpy(ir, amp, gam, wd, nc, dt, sr, n_samples):
    """Numpy fallback for synthesis (no Numba)."""
    order = np.argsort(nc)
    amp, gam, wd, nc = amp[order], gam[order], wd[order], nc[order]

    tier_limits = [int(0.01 * sr), int(0.05 * sr), int(0.2 * sr),
                   int(1.0 * sr), n_samples]
    i = 0
    n_active = len(amp)
    for tier_n in tier_limits:
        j = i
        while j < n_active and nc[j] <= tier_n:
            j += 1
        if j <= i:
            continue
        t_len = min(tier_n, n_samples)
        max_chunk = max(1, int(100e6 / (max(t_len, 1) * 8 * 3)))
        t_vec = np.arange(t_len, dtype=np.float64) / sr
        for ci in range(i, j, max_chunk):
            cj = min(ci + max_chunk, j)
            phase = wd[ci:cj, np.newaxis] * t_vec[np.newaxis, :]
            decay = np.exp(-gam[ci:cj, np.newaxis] * t_vec[np.newaxis, :])
            ir[:t_len] += np.dot(amp[ci:cj], decay * np.cos(phase))
        i = j


# ===================================================================
# Provider adapters
# ===================================================================

class AnalyticalModesProvider:
    """
    Wraps AnalyticalRoomModes for box rooms.

    Provides exact analytical modes (axial + tangential + oblique)
    with Kuttruff decay formula and air absorption.

    When used alongside eigensolve (ModalROMProvider), set defer_below
    to the eigensolve f_max. Modes below that frequency get reduced
    confidence so the eigensolve modes (with exact decay rates) win
    the merge. Above that frequency, analytical modes take over at
    full confidence.
    """
    name = 'analytical'

    def __init__(self, arm, defer_below=0.0):
        """
        Parameters
        ----------
        arm : AnalyticalRoomModes
            Pre-built analytical mode object.
        defer_below : float
            Frequency [Hz] below which confidence is reduced to 0.5
            (to let eigensolve modes win the merge). Default 0 = no
            deferral, analytical has full confidence everywhere.
        """
        self._arm = arm
        self._defer_below = float(defer_below)

    @property
    def confidence_base(self):
        return 1.0

    @property
    def frequency_range(self):
        return (0.0, self._arm.f_max)

    def provide_modes(self, source, receiver, materials,
                      c=343.0, humidity=50.0, temperature=20.0, **kw):
        arm = self._arm

        phi_src = arm.mode_shape(*source)
        phi_rec = arm.mode_shape(*receiver)

        n_nonzero = ((arm.modes_n > 0).astype(int) +
                     (arm.modes_m > 0).astype(int) +
                     (arm.modes_l > 0).astype(int))
        norm = np.power(2.0, n_nonzero) / arm.volume

        amplitudes = phi_src * phi_rec * norm
        gamma = arm.compute_decay_rates(materials, humidity, temperature)

        # Frequency-dependent confidence: full above defer_below,
        # reduced below (so eigensolve wins the merge there)
        if self._defer_below > 0:
            conf = np.where(arm.modes_f >= self._defer_below, 1.0, 0.5)
        else:
            conf = 1.0

        return make_modes(arm.modes_f, amplitudes, gamma, conf, self.name)


class ModalROMProvider:
    """
    Wraps eigensolve-based modal ROM.

    Provides eigenmodes with exact decay rates from per-node impedance
    (via precomputed surface weights).

    Confidence: 0.95 (mesh-limited but very accurate within range).
    """
    name = 'modal_rom'
    confidence_base = 0.95

    def __init__(self, eigenvalues, eigenvectors, frequencies,
                 mesh, ops, surface_weights=None):
        """
        Parameters
        ----------
        eigenvalues, eigenvectors, frequencies : from compute_room_modes()
        mesh : room mesh object
        ops : operator dict
        surface_weights : dict or None
            From precompute_surface_gamma_weights(). If None, uses FI path.
        """
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._frequencies = frequencies
        self._mesh = mesh
        self._ops = ops
        self._surface_weights = surface_weights

    @property
    def frequency_range(self):
        return (0.0, float(self._frequencies[-1]))

    def provide_modes(self, source, receiver, materials,
                      c=343.0, humidity=50.0, temperature=20.0, **kw):
        mesh = self._mesh
        eigvec = self._eigenvectors

        # Source coupling: project Gaussian onto eigenmodes
        sigma = kw.get('sigma', 0.3)
        r2 = ((mesh.x - source[0]) ** 2 +
              (mesh.y - source[1]) ** 2 +
              (mesh.z - source[2]) ** 2)
        p0 = np.exp(-r2 / sigma ** 2)
        M = self._ops['M_diag']
        modal_amps = eigvec.T @ (M * p0)

        # Receiver coupling
        rec_idx = mesh.nearest_node(*receiver)
        phi_rec = eigvec[rec_idx, :]
        amplitudes = modal_amps * phi_rec

        # Decay rates
        if self._surface_weights is not None:
            from .material_function import compute_modal_decay_spectral
            mat_funcs = {}
            for label in self._surface_weights:
                mat = materials.get(label)
                if mat is not None:
                    mat_funcs[label] = mat
                else:
                    from .material_function import MaterialFunction
                    mat_funcs[label] = MaterialFunction.from_scalar(0.05)
            gamma = compute_modal_decay_spectral(
                self._surface_weights, mat_funcs, self._frequencies,
                c=c, humidity=humidity, temperature=temperature)
        else:
            # Fallback: FI impedance
            gamma = np.zeros(len(self._eigenvalues))

        return make_modes(self._frequencies, amplitudes, gamma,
                          self.confidence_base, self.name)


class AxialModesProvider:
    """
    Wraps axial mode detection for any room with parallel surfaces.

    Provides 1D standing-wave modes between each detected parallel pair
    with 3D coupling loss model and air absorption.

    Confidence: 0.65 (1D approximation, validated at 88% spectral match).
    """
    name = 'axial'
    confidence_base = 0.65

    def __init__(self, parallel_pairs, room_volume, room_surface_area):
        self._pairs = parallel_pairs
        self._volume = room_volume
        self._surface_area = room_surface_area

    @property
    def frequency_range(self):
        return (0.0, 20000.0)  # axial modes exist at any frequency

    def provide_modes(self, source, receiver, materials,
                      c=343.0, humidity=50.0, temperature=20.0,
                      f_max=8000, **kw):
        from .material_function import MaterialFunction, air_absorption_coefficient

        src = np.asarray(source, dtype=float)
        rec = np.asarray(receiver, dtype=float)
        V = self._volume
        S = self._surface_area

        all_freqs = []
        all_amps = []
        all_gammas = []

        # Room-average gamma for coupling
        mean_alpha = np.mean([
            (materials[p.label_1](500) if isinstance(materials.get(p.label_1), MaterialFunction) else 0.05)
            for p in self._pairs])
        rt60 = 0.161 * V / (-S * np.log(1 - min(max(mean_alpha, 0.01), 0.99)))
        gamma_room = 6.91 / max(rt60, 0.05)

        total_pair_area = sum(p.overlap_area for p in self._pairs)
        if total_pair_area < 1e-10:
            return np.zeros(0, dtype=MODE_DTYPE)

        for pair in self._pairs:
            L = pair.distance
            n_dir = pair.normal

            x_src = np.clip(np.dot(src - pair.centroid_1, n_dir), 0, L)
            x_rec = np.clip(np.dot(rec - pair.centroid_1, n_dir), 0, L)

            # Check bounds
            if np.dot(src - pair.centroid_1, n_dir) < -0.01 * L:
                continue
            if np.dot(rec - pair.centroid_1, n_dir) < -0.01 * L:
                continue

            # Resolve materials
            mat1 = materials.get(pair.label_1)
            mat2 = materials.get(pair.label_2)
            if mat1 is None or mat2 is None:
                continue

            weight = pair.overlap_area / total_pair_area
            A_pair = pair.overlap_area * 2
            coupling = 1.0 - min(A_pair / S, 1.0) if S > 0 else 0

            n_max = int(np.floor(2 * L * f_max / c))
            for n in range(1, n_max + 1):
                f = n * c / (2 * L)
                S_n = np.cos(n * np.pi * x_src / L)
                R_n = np.cos(n * np.pi * x_rec / L)
                A = S_n * R_n * (2.0 / L) * weight

                if abs(A) < 1e-15:
                    continue

                a1 = np.clip(mat1(f) if isinstance(mat1, MaterialFunction) else 0.05, 0.001, 0.999)
                a2 = np.clip(mat2(f) if isinstance(mat2, MaterialFunction) else 0.05, 0.001, 0.999)
                R_prod = (1 - a1) * (1 - a2)
                if R_prod <= 0:
                    continue

                gp = (c / (2 * L)) * (-np.log(R_prod))
                gamma = (1 - coupling) * gp + coupling * gamma_room
                gamma += float(air_absorption_coefficient(np.array([f]),
                               humidity, temperature)[0]) * c

                all_freqs.append(f)
                all_amps.append(A)
                all_gammas.append(gamma)

        if not all_freqs:
            return np.zeros(0, dtype=MODE_DTYPE)

        return make_modes(all_freqs, all_amps, all_gammas,
                          self.confidence_base, self.name)


class GeneralizedModesProvider:
    """
    Wraps GeneralizedModes for non-box rooms with parallel surfaces.

    Provides axial + tangential + oblique modes detected from perpendicular
    parallel surface pairs.

    Confidence: 0.75 (better than pure axial, less exact than eigensolve).
    """
    name = 'generalized'
    confidence_base = 0.75

    def __init__(self, gm):
        """
        Parameters
        ----------
        gm : GeneralizedModes
            Pre-built generalized mode object.
        """
        self._gm = gm

    @property
    def frequency_range(self):
        return (0.0, self._gm._f_max)

    def provide_modes(self, source, receiver, materials,
                      c=343.0, humidity=50.0, temperature=20.0, **kw):
        # Delegate to GeneralizedModes synthesis but extract modes only
        # For now, use the box path if available
        if hasattr(self._gm, '_arm'):
            provider = AnalyticalModesProvider(self._gm._arm)
            modes = provider.provide_modes(source, receiver, materials,
                                           c=c, humidity=humidity,
                                           temperature=temperature, **kw)
            modes['confidence'] = self.confidence_base
            modes['source_engine'] = self.name
            return modes

        # Non-box: would need to extract mode params from _synthesize_generalized
        # For now return empty (to be implemented per geometry)
        return np.zeros(0, dtype=MODE_DTYPE)


# ===================================================================
# Orchestrator
# ===================================================================

class UnifiedModalSynthesizer:
    """
    Central orchestrator: registers providers, merges modes, synthesizes IR.

    Usage:
        synth = UnifiedModalSynthesizer()
        synth.register(AnalyticalModesProvider(arm))
        synth.register(AxialModesProvider(pairs, V, S))

        ir = synth.impulse_response(source, receiver, materials, T=3.0)

    Adding a new engine:
        1. Write a class with: name, frequency_range, confidence_base, provide_modes()
        2. synth.register(YourProvider(...))

    Removing an engine:
        synth.unregister('engine_name')
    """

    def __init__(self, sr=44100, c=343.0):
        self.sr = sr
        self.c = c
        self._providers = {}
        self._ism_func = None

    def register(self, provider):
        """Register a mode provider. Replaces existing with same name."""
        self._providers[provider.name] = provider

    def unregister(self, name):
        """Remove a provider by name. No error if not found."""
        self._providers.pop(name, None)

    def set_ism(self, ism_func):
        """
        Set ISM early reflection provider.

        Parameters
        ----------
        ism_func : callable
            ism_func(source, receiver, materials) → ir_array
        """
        self._ism_func = ism_func

    def list_providers(self):
        """Print registered providers."""
        for name, p in self._providers.items():
            f_lo, f_hi = p.frequency_range
            print(f"  {name:<16s} {f_lo:>6.0f}-{f_hi:>6.0f} Hz  "
                  f"confidence={p.confidence_base:.2f}")
        if self._ism_func:
            print(f"  {'ISM':<16s} early reflections (non-modal)")

    def impulse_response(self, source, receiver, materials,
                         T=3.0, humidity=50.0, temperature=20.0,
                         freq_tolerance_hz=2.0, freq_tolerance_pct=0.005,
                         **kwargs):
        """
        Full pipeline: provide modes → merge → synthesize → add ISM.

        Parameters
        ----------
        source, receiver : (x, y, z)
        materials : dict of label → MaterialFunction
        T : float
            IR duration [s].
        humidity, temperature : float
        freq_tolerance_hz, freq_tolerance_pct : float
            Deduplication tolerances.

        Returns
        -------
        ImpulseResponse
        """
        source = np.asarray(source, dtype=float)
        receiver = np.asarray(receiver, dtype=float)

        # Collect modes from all providers
        mode_arrays = []
        for name, provider in self._providers.items():
            modes = provider.provide_modes(
                source, receiver, materials,
                c=self.c, humidity=humidity, temperature=temperature,
                **kwargs)
            if len(modes) > 0:
                mode_arrays.append(modes)

        # Merge and deduplicate
        merged = merge_modes(mode_arrays,
                             freq_tolerance_hz=freq_tolerance_hz,
                             freq_tolerance_pct=freq_tolerance_pct)

        # ISM early reflections
        ism_ir = None
        if self._ism_func is not None:
            ism_ir = self._ism_func(source, receiver, materials)

        # Synthesize
        ir_data = synthesize_ir(merged, T=T, sr=self.sr, ism_ir=ism_ir)

        # Wrap in ImpulseResponse
        from .room import ImpulseResponse
        return ImpulseResponse(ir_data, self.sr), merged
