"""
validate_fom.py
===============
Validate our Laplace-domain FOM against:
  1. Analytical eigenfrequencies (rigid rect room)
  2. Time-domain p-Phi RK4 solver (same SEM mesh)

Uses the same 2D setup as the Sampedro replication:
  Domain: 2m x 2m, Ne=20, P=4, N=6561
  Source: Gaussian pulse at (1.0, 1.0), sigma=0.3
  Receiver: (0.2, 0.2)

Test cases:
  A. Perfectly rigid walls (PR) — eigenfrequencies known exactly
  B. Frequency-independent walls (FI, Z=5000) — cross-check TD vs Laplace
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from room_acoustics.sem import RectMesh2D, assemble_2d_operators
from room_acoustics.solvers import (fom_pphi, analytical_rigid_rect,
                                     C_AIR, RHO_AIR)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'results', 'fom_validation')
os.makedirs(OUT, exist_ok=True)

# ── Config (matches Sampedro replication) ────────────────────
Lx, Ly = 2.0, 2.0
Ne, P = 20, 4
SRC = (1.0, 1.0)
SIGMA = 0.3
REC = (0.2, 0.2)

# ── Laplace+Weeks helpers ────────────────────────────────────
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
    L[0] = 1.0
    if n > 1:
        L[1] = 1.0 - x
    for k in range(1, n - 1):
        L[k + 1] = ((2*k + 1 - x) * L[k] - k * L[k - 1]) / (k + 1)
    return L

def weeks_reconstruct(a, sigma, b, t):
    return np.exp((sigma - b) * t) * np.real(a @ laguerre_eval(len(a), 2*b*t))

def laplace_fom_solve(c2S, M_diag, B_diag, p0, N, s, Zs):
    """Solve Laplace FOM at one complex frequency."""
    sig, omg = s.real, s.imag
    if Zs > 1e14:
        Br = np.zeros_like(B_diag)
    else:
        Br = C_AIR**2 * RHO_AIR * B_diag / Zs
    Kr = c2S + sparse.diags((sig**2 - omg**2)*M_diag + sig*Br, format='csc')
    Kc = sparse.diags(2*sig*omg*M_diag + omg*Br, format='csc')
    A = sparse.bmat([[Kr, -Kc], [Kc, Kr]], format='csc')
    rhs = np.concatenate([sig*p0*M_diag, omg*p0*M_diag])
    x = spsolve(A, rhs)
    return x[:N] + 1j * x[N:]


def main():
    print("=" * 64)
    print("  FOM VALIDATION")
    print("=" * 64)

    # ── Build mesh ───────────────────────────────────────────
    mesh = RectMesh2D(Lx, Ly, Ne, Ne, P)
    ops = assemble_2d_operators(mesh)
    N = mesh.N_dof
    rec_idx = mesh.nearest_node(*REC)
    print(f"Mesh: {Lx}x{Ly}m, Ne={Ne}, P={P}, N={N}")
    print(f"Receiver node {rec_idx}: "
          f"({mesh.x[rec_idx]:.4f}, {mesh.y[rec_idx]:.4f})")

    r2 = (mesh.x - SRC[0])**2 + (mesh.y - SRC[1])**2
    p0 = np.exp(-r2 / SIGMA**2)

    c2S = (C_AIR**2 * ops['S']).tocsc()
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())

    # ==========================================================
    # TEST A: Eigenfrequencies (rigid walls)
    # ==========================================================
    print(f"\n{'='*64}")
    print("  TEST A: Eigenfrequencies (rigid walls)")
    print(f"{'='*64}")

    # Analytical eigenfrequencies for 2D rigid rect
    f_analytical = []
    for m in range(30):
        for n in range(30):
            if m == 0 and n == 0:
                continue
            f_mn = (C_AIR / 2) * np.sqrt((m / Lx)**2 + (n / Ly)**2)
            if f_mn < 2000:
                f_analytical.append(f_mn)
    f_analytical = sorted(set(f_analytical))
    print(f"  Analytical modes below 2kHz: {len(f_analytical)}")

    # Laplace FOM IR with rigid walls
    sigma_w, b_w, Ns = 20.0, 800.0, 1000
    s_vals, z_safe = weeks_s_values(sigma_w, b_w, Ns)

    print(f"  Laplace FOM (rigid, {Ns} solves)...", end='', flush=True)
    t0 = time.perf_counter()
    H_rigid = np.array([
        laplace_fom_solve(c2S, M_diag, B_diag, p0, N, s, 1e15)[rec_idx]
        for s in s_vals])
    print(f" {time.perf_counter()-t0:.0f}s")

    fs_ir = 44100
    t_ir = np.arange(0, 0.1, 1.0/fs_ir)
    a_rigid = weeks_coefficients(H_rigid, b_w, z_safe)
    ir_laplace_rigid = weeks_reconstruct(a_rigid, sigma_w, b_w, t_ir)

    # Also get analytical IR
    print("  Analytical IR (modal expansion)...", end='', flush=True)
    t_ana, ir_ana = analytical_rigid_rect(
        mesh, SRC[0], SRC[1], SIGMA, REC[0], REC[1],
        dt=1.0/fs_ir, T=0.1, n_modes=120)
    print(" done")

    # Also get TD solver IR
    print("  TD p-Phi RK4 (rigid)...", end='', flush=True)
    t0 = time.perf_counter()
    # CFL condition: dt < h_min / (c * sqrt(2)) * safety
    h_min = min(mesh.hx, mesh.hy)
    dt_cfl = 0.5 * h_min / (C_AIR * np.sqrt(2))
    dt_td = min(dt_cfl, 1e-5)
    td_result = fom_pphi(mesh, ops, 'PR', {}, SRC[0], SRC[1], SIGMA,
                         dt=dt_td, T=0.1, rec_idx=rec_idx)
    t_td = np.arange(len(td_result['ir'])) * dt_td
    print(f" {time.perf_counter()-t0:.0f}s (dt={dt_td:.2e})")

    # FFT of all three
    def get_peaks(ir, fs, f_max=2000, n_peaks=50):
        """Find spectral peaks in IR."""
        from scipy.signal import find_peaks
        spec = np.abs(np.fft.rfft(ir))
        freqs = np.fft.rfftfreq(len(ir), 1.0/fs)
        mask = freqs < f_max
        spec_masked = spec[mask]
        freqs_masked = freqs[mask]
        # Normalize
        spec_masked = spec_masked / spec_masked.max()
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        peaks, props = find_peaks(spec_masked, height=0.01,
                                  distance=max(1, int(5 / df)))
        peak_freqs = freqs_masked[peaks]
        peak_heights = spec_masked[peaks]
        # Sort by height
        order = np.argsort(peak_heights)[::-1][:n_peaks]
        return peak_freqs[order], peak_heights[order]

    peaks_lap, h_lap = get_peaks(ir_laplace_rigid, fs_ir)
    peaks_td, h_td = get_peaks(td_result['ir'],
                                1.0/dt_td,
                                f_max=2000,
                                n_peaks=50)

    # Match peaks to analytical
    def match_peaks(peaks_found, f_analytical, tol_hz=5.0):
        matched = 0
        errors = []
        for fp in peaks_found:
            diffs = [abs(fp - fa) for fa in f_analytical]
            best = min(diffs)
            if best < tol_hz:
                matched += 1
                errors.append(best)
        return matched, errors

    m_lap, e_lap = match_peaks(peaks_lap, f_analytical)
    m_td, e_td = match_peaks(peaks_td, f_analytical)

    print(f"\n  Eigenfrequency matching (tolerance: 5 Hz):")
    print(f"    Laplace+Weeks: {m_lap}/{len(peaks_lap)} peaks match "
          f"analytical, mean error {np.mean(e_lap):.2f} Hz")
    print(f"    TD RK4:        {m_td}/{len(peaks_td)} peaks match "
          f"analytical, mean error {np.mean(e_td):.2f} Hz")

    # ── Plot A: Three-way comparison ─────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Waveform overlay
    ax = axes[0]
    ax.plot(t_ir*1000, ir_laplace_rigid, 'b-', lw=0.7, label='Laplace+Weeks')
    ax.plot(t_td*1000, td_result['ir'], 'r--', lw=0.7, label='TD RK4')
    ax.plot(t_ana*1000, ir_ana, 'g:', lw=0.7, label='Analytical')
    ax.set_ylabel('Pressure [Pa]')
    ax.set_xlim(0, 50)
    ax.legend()
    ax.set_title('TEST A: Rigid Walls — Three-Way Waveform Comparison')
    ax.grid(True, alpha=0.3)

    # Spectrum overlay
    ax = axes[1]
    spec_lap = np.abs(np.fft.rfft(ir_laplace_rigid))
    freqs_lap = np.fft.rfftfreq(len(ir_laplace_rigid), 1.0/fs_ir)
    spec_td = np.abs(np.fft.rfft(td_result['ir']))
    freqs_td = np.fft.rfftfreq(len(td_result['ir']), dt_td)
    spec_ana = np.abs(np.fft.rfft(ir_ana))
    freqs_ana = np.fft.rfftfreq(len(ir_ana), 1.0/fs_ir)

    ax.semilogy(freqs_lap, spec_lap/spec_lap.max(), 'b-', lw=0.5,
                label='Laplace+Weeks')
    ax.semilogy(freqs_td, spec_td/spec_td.max(), 'r-', lw=0.5, alpha=0.7,
                label='TD RK4')
    ax.semilogy(freqs_ana, spec_ana/spec_ana.max(), 'g-', lw=0.5, alpha=0.5,
                label='Analytical')
    # Mark analytical eigenfrequencies
    for i, fa in enumerate(f_analytical[:30]):
        ax.axvline(fa, color='k', alpha=0.15, lw=0.5)
    ax.set_xlim(0, 2000)
    ax.set_ylim(1e-4, 2)
    ax.set_ylabel('Normalized |FFT|')
    ax.legend()
    ax.set_title('Spectrum — peaks should align with analytical eigenfrequencies (grey lines)')
    ax.grid(True, alpha=0.3)

    # Error: Laplace vs TD
    ax = axes[2]
    # Interpolate TD to same time grid as Laplace
    ir_td_interp = np.interp(t_ir, t_td, td_result['ir'])
    err_lt = np.abs(ir_laplace_rigid - ir_td_interp)
    ir_ana_interp = np.interp(t_ir, t_ana, ir_ana)
    err_la = np.abs(ir_laplace_rigid - ir_ana_interp)

    ax.semilogy(t_ir*1000, err_lt + 1e-30, 'r-', lw=0.5,
                label='|Laplace - TD|')
    ax.semilogy(t_ir*1000, err_la + 1e-30, 'g-', lw=0.5,
                label='|Laplace - Analytical|')
    p_max = max(np.max(np.abs(ir_laplace_rigid)), 1e-30)
    ax.set_ylabel('Absolute error [Pa]')
    ax.set_xlabel('Time [ms]')
    rel_lt = np.max(err_lt) / p_max
    rel_la = np.max(err_la) / p_max
    ax.set_title('Laplace vs TD: rel=%.2e  |  Laplace vs Analytical: rel=%.2e'
                 % (rel_lt, rel_la))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUT, 'test_A_rigid.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Plot: {fname}")

    # ==========================================================
    # TEST B: FI walls (Z=5000) — TD vs Laplace
    # ==========================================================
    print(f"\n{'='*64}")
    print("  TEST B: FI walls (Z=5000) — TD vs Laplace")
    print(f"{'='*64}")

    Z_test = 5000.0

    # Laplace
    print(f"  Laplace FOM (Z={Z_test}, {Ns} solves)...", end='', flush=True)
    t0 = time.perf_counter()
    H_fi = np.array([
        laplace_fom_solve(c2S, M_diag, B_diag, p0, N, s, Z_test)[rec_idx]
        for s in s_vals])
    print(f" {time.perf_counter()-t0:.0f}s")

    a_fi = weeks_coefficients(H_fi, b_w, z_safe)
    ir_laplace_fi = weeks_reconstruct(a_fi, sigma_w, b_w, t_ir)

    # TD
    print(f"  TD p-Phi RK4 (Z={Z_test})...", end='', flush=True)
    t0 = time.perf_counter()
    td_fi = fom_pphi(mesh, ops, 'FI', {'Z': Z_test},
                     SRC[0], SRC[1], SIGMA,
                     dt=dt_td, T=0.1, rec_idx=rec_idx)
    print(f" {time.perf_counter()-t0:.0f}s")

    t_td_fi = np.arange(len(td_fi['ir'])) * dt_td
    ir_td_fi_interp = np.interp(t_ir, t_td_fi, td_fi['ir'])

    err_fi = np.abs(ir_laplace_fi - ir_td_fi_interp)
    p_max_fi = max(np.max(np.abs(ir_laplace_fi)), 1e-30)
    rel_fi = np.max(err_fi) / p_max_fi

    print(f"\n  Results:")
    print(f"    |p|_max          = {p_max_fi:.4e}")
    print(f"    |Laplace - TD|   = {np.max(err_fi):.4e}")
    print(f"    Relative error   = {rel_fi:.4e}")

    # ── Plot B ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    ax.plot(t_ir*1000, ir_laplace_fi, 'b-', lw=0.7, label='Laplace+Weeks')
    ax.plot(t_td_fi*1000, td_fi['ir'], 'r--', lw=0.7, label='TD RK4')
    ax.set_ylabel('Pressure [Pa]')
    ax.legend()
    ax.set_title('TEST B: FI walls (Z=5000) — Laplace vs TD')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(t_ir*1000, err_fi + 1e-30, 'k-', lw=0.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('|Laplace - TD| [Pa]')
    ax.set_title('Relative error: %.2e' % rel_fi)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUT, 'test_B_fi.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Plot: {fname}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*64}")
    print(f"  A. Eigenfrequency match (rigid):")
    print(f"     Laplace: {m_lap}/{len(peaks_lap)} peaks, "
          f"mean err {np.mean(e_lap):.2f} Hz")
    print(f"     TD:      {m_td}/{len(peaks_td)} peaks, "
          f"mean err {np.mean(e_td):.2f} Hz")
    print(f"  B. FI walls (Z=5000):")
    print(f"     Laplace vs TD relative error: {rel_fi:.2e}")
    print(f"  A. Rigid walls:")
    print(f"     Laplace vs TD:         {rel_lt:.2e}")
    print(f"     Laplace vs Analytical:  {rel_la:.2e}")

    PASS_A = m_lap >= len(peaks_lap) * 0.8 and np.mean(e_lap) < 5.0
    PASS_B = rel_fi < 0.05
    print(f"\n  TEST A: {'PASS' if PASS_A else 'FAIL'}")
    print(f"  TEST B: {'PASS' if PASS_B else 'FAIL'}")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
