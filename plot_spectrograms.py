"""Generate spectrograms for all replicated IRs (2D + 3D)."""

import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import spectrogram
from scipy.sparse.linalg import spsolve
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from room_acoustics.sem import RectMesh2D, assemble_2d_operators
from room_acoustics.sem import BoxMesh3D, assemble_3d_operators

C, RHO = 343.0, 1.2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'results', 'sampedro_spectrograms')
os.makedirs(OUT, exist_ok=True)

# ── Weeks helpers ────────────────────────────────────────────
def weeks_s_values(sigma, b, N):
    k = np.arange(N)
    theta = 2 * np.pi * k / N
    z = np.exp(1j * theta)
    z_safe = np.where(np.abs(1 - z) < 1e-10, 1 - 1e-10, z)
    return sigma + b * (1 + z_safe) / (1 - z_safe), z_safe

def weeks_coefficients(H, b, z_safe):
    return np.fft.fft(H * (2 * b / (1 - z_safe))) / len(H)

def laguerre_eval(n, x):
    L = np.zeros((n, len(x)))
    L[0] = 1
    if n > 1:
        L[1] = 1 - x
    for k in range(1, n - 1):
        L[k + 1] = ((2*k + 1 - x) * L[k] - k * L[k - 1]) / (k + 1)
    return L

def weeks_reconstruct(a, sigma, b, t):
    return np.exp((sigma - b) * t) * np.real(a @ laguerre_eval(len(a), 2*b*t))

def miki_admittance(f, sf, d):
    f = max(abs(f), 1.0)
    X = f / sf
    Zc = RHO * C * (1 + 0.0699*X**(-0.632) - 1j*0.107*X**(-0.632))
    kc = (2*np.pi*f/C) * (1 + 0.109*X**(-0.618) - 1j*0.160*X**(-0.618))
    return 1.0 / (-1j * Zc * np.cos(kc * d) / np.sin(kc * d))


# ── Plot function ────────────────────────────────────────────
def plot_ir_spectrogram(title, subtitle, ir, fs, fname):
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 2, 0.8], hspace=0.35)
    t_ms = np.arange(len(ir)) / fs * 1000

    # Waveform
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t_ms, ir, 'b-', lw=0.5)
    ax0.set_ylabel('Pressure [Pa]')
    ax0.set_title(title + ' — ' + subtitle, fontsize=13, fontweight='bold')
    ax0.set_xlim(0, t_ms[-1])
    ax0.grid(True, alpha=0.3)

    # Spectrogram
    ax1 = fig.add_subplot(gs[1])
    nperseg = min(256, len(ir) // 4)
    f_spec, t_spec, Sxx = spectrogram(
        ir, fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4, window='hann')
    Sxx_db = 10 * np.log10(Sxx + 1e-30)
    vmax = Sxx_db.max()
    valid = Sxx_db[Sxx_db > -300]
    vmin = max(vmax - 80, valid.min()) if len(valid) > 0 else vmax - 80
    im = ax1.pcolormesh(t_spec * 1000, f_spec, Sxx_db,
                        shading='gouraud', cmap='inferno',
                        vmin=vmin, vmax=vmax)
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlabel('Time [ms]')
    f_upper = min(int(fs / 2), 5000)
    ax1.set_ylim(0, f_upper)
    cb = plt.colorbar(im, ax=ax1, pad=0.02)
    cb.set_label('Power [dB]')

    # Energy decay curve (Schroeder integration)
    ax2 = fig.add_subplot(gs[2])
    ir2 = ir ** 2
    edc = np.cumsum(ir2[::-1])[::-1]
    edc_db = 10 * np.log10(edc / max(edc[0], 1e-30) + 1e-30)
    ax2.plot(t_ms, edc_db, 'r-', lw=1)
    ax2.set_ylabel('EDC [dB]')
    ax2.set_xlabel('Time [ms]')
    ax2.set_xlim(0, t_ms[-1])
    ax2.set_ylim(-60, 0)
    ax2.grid(True, alpha=0.3)

    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')


# ═════════════════════════════════════════════════════════════
# 2D CASES
# ═════════════════════════════════════════════════════════════
print("Building 2D mesh...")
mesh2d = RectMesh2D(2, 2, 20, 20, 4)
ops2d = assemble_2d_operators(mesh2d)
N2d = mesh2d.N_dof
r2 = (mesh2d.x - 1)**2 + (mesh2d.y - 1)**2
p0_2d = np.exp(-r2 / 0.3**2)
rec2d = mesh2d.nearest_node(0.2, 0.2)

c2S_2d = (C**2 * ops2d['S']).tocsc()
M_2d = ops2d['M_diag']
B_2d = np.array(ops2d['B_total'].diagonal())

def fom_2d(s, Zs):
    Br = C**2 * RHO * B_2d / Zs
    sig, omg = s.real, s.imag
    Kr = c2S_2d + sparse.diags((sig**2 - omg**2)*M_2d + sig*Br, format='csc')
    Kc = sparse.diags(2*sig*omg*M_2d + omg*Br, format='csc')
    A = sparse.bmat([[Kr, -Kc], [Kc, Kr]], format='csc')
    rhs = np.concatenate([sig*p0_2d*M_2d, omg*p0_2d*M_2d])
    x = spsolve(A, rhs)
    return (x[:N2d] + 1j * x[N2d:])[rec2d]

sigma_w, b_w = 20.0, 800.0
fs = 44100
t_eval = np.arange(0, 0.1, 1.0 / fs)

for Zs in [5000, 15000]:
    s_vals, z_safe = weeks_s_values(sigma_w, b_w, 1000)
    print(f"2D FOM Zs={Zs} (1000 solves)...", end='', flush=True)
    t0 = time.perf_counter()
    H = np.array([fom_2d(s, Zs) for s in s_vals])
    print(f" {time.perf_counter()-t0:.0f}s")
    a = weeks_coefficients(H, b_w, z_safe)
    ir = weeks_reconstruct(a, sigma_w, b_w, t_eval)
    plot_ir_spectrogram(
        '2D FI (2m x 2m, N=6561)',
        'Zs = %d Pa s/m3' % Zs,
        ir, fs,
        os.path.join(OUT, 'ir_2d_Z%d.png' % Zs))

# ═════════════════════════════════════════════════════════════
# 3D CASES (GPU)
# ═════════════════════════════════════════════════════════════
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import gmres as gpu_gmres
    from cupyx.scipy.sparse.linalg import LinearOperator as gpuLO
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("No GPU — skipping 3D")

if HAS_GPU:
    print("\nBuilding 3D mesh...")
    mesh3d = BoxMesh3D(1, 1, 1, 8, 8, 8, 4)
    ops3d = assemble_3d_operators(mesh3d)
    N3d = mesh3d.N_dof
    mesh3d._ensure_coords()
    r2_3d = (mesh3d.x - 0.5)**2 + (mesh3d.y - 0.5)**2 + (mesh3d.z - 0.5)**2
    p0_3d = np.exp(-r2_3d / 0.2**2)
    rec3d = mesh3d.nearest_node(0.25, 0.1, 0.8)

    c2S_3d_gpu = csp.csr_matrix((C**2 * ops3d['S']).tocsr())
    M_3d_gpu = cp.asarray(ops3d['M_diag'])
    B_3d_gpu = cp.asarray(np.array(ops3d['B_total'].diagonal()))
    rhs_base_3d = cp.asarray(p0_3d * ops3d['M_diag'])

    def fom_3d(s, d_mat):
        f = max(abs(s.imag) / (2 * np.pi), 1.0)
        Ys = complex(miki_admittance(f, 10000.0, d_mat))
        diag = s**2 * M_3d_gpu + s * C**2 * RHO * Ys * B_3d_gpu
        A = c2S_3d_gpu + csp.diags(diag)
        rhs = s * rhs_base_3d
        prec = gpuLO((N3d, N3d),
                      matvec=lambda x: x / A.diagonal(), dtype=complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, _ = gpu_gmres(A, rhs, M=prec,
                             tol=1e-7, maxiter=500, restart=100)
        cp.cuda.Device().synchronize()
        return complex(cp.asnumpy(x[rec3d]))

    s_vals_3d, z_safe_3d = weeks_s_values(sigma_w, b_w, 500)

    for d in [0.05, 0.15]:
        print("3D FOM d=%.3fm (500 solves)..." % d, end='', flush=True)
        t0 = time.perf_counter()
        H = np.array([fom_3d(s, d) for s in s_vals_3d])
        print(" %.0fs" % (time.perf_counter() - t0))
        a = weeks_coefficients(H, b_w, z_safe_3d)
        ir = weeks_reconstruct(a, sigma_w, b_w, t_eval)
        plot_ir_spectrogram(
            '3D Freq-Dep (1m cube, N=35937)',
            'd_mat = %.3f m' % d,
            ir, fs,
            os.path.join(OUT, 'ir_3d_d%dmm.png' % int(d * 1000)))

print("\nAll done.")
