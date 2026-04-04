"""
validate_bras.py — BRAS Scene 9 validation for romacoustics
=============================================================
Compares romacoustics simulation against measured RIRs from the
BRAS benchmark (Brinkmann et al., JASA 2019).

Room: 8.4 x 6.7 x 3.0 m seminar room at RWTH Aachen
Materials: plaster walls, suspended ceiling, linoleum floor, windows, concrete
Source: LS1 = (2.0, 3.35, 1.5) — dodecahedron
Receiver: MP1 = (6.0, 1.5, 1.2)

Outputs: comparison plots, octave-band metric tables, WAV files
"""

import os, sys, time
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'romacoustics'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from romacoustics.sem import BoxMesh3D, assemble_3d
from romacoustics.solver import (
    C_AIR, RHO_AIR,
    weeks_s_values, laplace_to_ir,
)
from romacoustics.ir import ImpulseResponse
from romacoustics.metrics import octave_band_metrics, octave_band_filter, band_metrics
from romacoustics.materials import absorption_to_impedance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_bras')
os.makedirs(OUT, exist_ok=True)

# ── BRAS paths ───────────────────────────────────────────────
BRAS_DIR = r"C:\Users\bsaka\Downloads\BRAS"
BRAS_MAT_DIR = os.path.join(BRAS_DIR, "surface_data", "3 Surface descriptions",
                             "_csv", "fitted_estimates")
BRAS_RIR_DIR = os.path.join(BRAS_DIR, "scene9_data", "1 Scene descriptions",
                             "09 small room (seminar room)", "RIRs", "wav")

# ── Room geometry ────────────────────────────────────────────
Lx, Ly, Lz = 8.4, 6.7, 3.0

# Source/receiver positions (from SketchUp / calibrate_absorption.py)
SOURCES = {
    'LS1': (2.0, 3.35, 1.5),
    'LS2': (4.5, 4.0, 1.5),
}
RECEIVERS = {
    'MP1': (6.0, 1.5, 1.2),
    'MP2': (6.0, 3.0, 1.2),
    'MP3': (6.0, 4.5, 1.2),
    'MP4': (3.0, 1.5, 1.2),
    'MP5': (3.0, 5.0, 1.2),
}

# Surface material mapping (box faces -> BRAS materials)
# 3D box: x_min/x_max, y_min/y_max, z_min/z_max
SURFACE_MAP = {
    'z_min': 'floor',       # bottom = floor
    'z_max': 'ceiling',     # top = ceiling
    'y_min': 'plaster',     # front wall
    'y_max': 'plaster',     # back wall
    'x_min': 'windows',     # window wall
    'x_max': 'concrete',    # concrete wall
}

OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000]


def load_bras_material(name):
    """Load BRAS absorption data. Returns (freqs, alpha_fitted)."""
    path = os.path.join(BRAS_MAT_DIR, f"mat_scene09_{name}.csv")
    lines = open(path).read().strip().split('\n')
    freqs = np.array([float(x) for x in lines[0].split(',')])
    alpha_fitted = np.array([float(x) for x in lines[2].split(',')])
    return freqs, alpha_fitted


def load_measured_rir(source, mic):
    """Load measured RIR from BRAS."""
    path = os.path.join(BRAS_RIR_DIR,
                        f"scene9_RIR_{source}_{mic}_Dodecahedron.wav")
    sr, data = wavfile.read(path)
    return data.astype(np.float64), sr


def bras_Z_per_surface(f_target):
    """Get impedance per surface at target frequency."""
    Z = {}
    for face, mat_name in SURFACE_MAP.items():
        freqs, alpha = load_bras_material(mat_name)
        alpha_at_f = np.interp(f_target, freqs, alpha)
        Z[face] = absorption_to_impedance(alpha_at_f)
    return Z


def solve_laplace_persurface(mesh, ops, src_pos, rec_idx,
                              Z_per_surface, Ns=500,
                              sigma_w=20.0, b_w=800.0, t_max=1.0, fs=44100):
    """Laplace FOM with per-surface impedance."""
    from scipy.sparse.linalg import spsolve
    from scipy import sparse

    N = mesh.N_dof
    mesh._ensure_coords()
    r2 = ((mesh.x - src_pos[0])**2 + (mesh.y - src_pos[1])**2
          + (mesh.z - src_pos[2])**2)
    p0 = np.exp(-r2 / 0.2**2)

    c2S = (C_AIR**2 * ops['S']).tocsc()
    M = ops['M_diag']
    B_labels = ops['B_labels']

    # Build combined Br diagonal from per-surface impedances
    Br_diag = np.zeros(N)
    for face, Z in Z_per_surface.items():
        if face in B_labels:
            Br_diag += C_AIR**2 * RHO_AIR * B_labels[face] / Z

    s_vals, z_safe = weeks_s_values(sigma_w, b_w, Ns)
    t_eval = np.arange(0, t_max, 1.0/fs)

    H = np.zeros(Ns, dtype=complex)
    t0 = time.perf_counter()
    for i, s in enumerate(s_vals):
        sig, omg = s.real, s.imag
        Kr = c2S + sparse.diags(
            (sig**2 - omg**2)*M + sig*Br_diag, format='csc')
        Kc = sparse.diags(
            2*sig*omg*M + omg*Br_diag, format='csc')
        A = sparse.bmat([[Kr, -Kc], [Kc, Kr]], format='csc')
        rhs = np.concatenate([sig*p0*M, omg*p0*M])
        x = spsolve(A, rhs)
        H[i] = x[rec_idx] + 1j*x[N + rec_idx]
        if (i+1) % max(1, Ns//10) == 0:
            el = time.perf_counter() - t0
            print(f'  {i+1}/{Ns} ({el:.0f}s)', end='', flush=True)
    print(f' done ({time.perf_counter()-t0:.0f}s)')

    ir = laplace_to_ir(H, sigma_w, b_w, t_eval)
    return ir, t_eval


def main():
    print("=" * 64)
    print("  BRAS Scene 9 Validation")
    print("  8.4 x 6.7 x 3.0 m seminar room")
    print("=" * 64)

    # ── Build mesh ───────────────────────────────────────────
    # Use small Ne for speed — we're validating metrics, not waveform details
    Ne = 4  # N = (4*4+1)^3 = 4913
    P = 4
    print(f"\nBuilding 3D mesh: Ne={Ne}, P={P}...")
    mesh = BoxMesh3D(Lx, Ly, Lz, Ne, Ne, Ne, P)
    ops = assemble_3d(mesh)
    N = mesh.N_dof
    print(f"  N = {N}")
    print(f"  B_labels: {list(ops['B_labels'].keys())}")

    # ── Load materials ───────────────────────────────────────
    print("\nMaterials:")
    for face, mat_name in SURFACE_MAP.items():
        freqs, alpha = load_bras_material(mat_name)
        # Show octave-band values
        alpha_oct = [np.interp(f, freqs, alpha) for f in OCTAVE_BANDS]
        print(f"  {face:6s} -> {mat_name:10s}: "
              + " ".join(f"{a:.3f}" for a in alpha_oct))

    # Use average impedance across octave bands
    Z_avg = {}
    for face, mat_name in SURFACE_MAP.items():
        freqs, alpha = load_bras_material(mat_name)
        alpha_oct = np.array([np.interp(f, freqs, alpha) for f in OCTAVE_BANDS])
        Z_avg[face] = absorption_to_impedance(np.mean(alpha_oct))
    print("\nAverage Z per surface:")
    for face, Z in Z_avg.items():
        print(f"  {face:6s}: Z = {Z:.0f}")

    # ── Source-receiver pair ─────────────────────────────────
    src_name, rec_name = 'LS1', 'MP1'
    src_pos = SOURCES[src_name]
    rec_pos = RECEIVERS[rec_name]
    rec_idx = mesh.nearest_node(*rec_pos)
    print(f"\nSource: {src_name} = {src_pos}")
    print(f"Receiver: {rec_name} = {rec_pos} -> node {rec_idx}")

    # ── Load measured RIR ────────────────────────────────────
    print(f"\nLoading measured RIR ({src_name}_{rec_name})...")
    ir_meas, fs_meas = load_measured_rir(src_name, rec_name)
    print(f"  {len(ir_meas)} samples, {fs_meas} Hz, {len(ir_meas)/fs_meas:.2f}s")

    # ── Compute metrics on measured ──────────────────────────
    print("\nMeasured metrics:")
    meas_metrics = octave_band_metrics(ir_meas, fs_meas, OCTAVE_BANDS)
    print(f"  T30: {meas_metrics['T30']}")
    print(f"  EDT: {meas_metrics['EDT']}")
    print(f"  C80: {meas_metrics['C80']}")

    # ── Run simulation ───────────────────────────────────────
    # Weeks parameters: b must satisfy 2*b*t_max < ~200 for Laguerre stability
    # t_max=1.0s -> b=80. Ns=500 is enough for this b.
    t_max_sim = 1.0
    b_w = 80.0
    sigma_w = 5.0
    Ns_sim = 500
    print(f"\nSimulating (Ns={Ns_sim}, t_max={t_max_sim}s, sigma={sigma_w}, b={b_w})...")
    ir_sim, t_sim = solve_laplace_persurface(
        mesh, ops, src_pos, rec_idx, Z_avg,
        Ns=Ns_sim, sigma_w=sigma_w, b_w=b_w, t_max=t_max_sim, fs=fs_meas)

    sim_metrics = octave_band_metrics(ir_sim, fs_meas, OCTAVE_BANDS)
    print("\nSimulated metrics:")
    print(f"  T30: {sim_metrics['T30']}")
    print(f"  EDT: {sim_metrics['EDT']}")
    print(f"  C80: {sim_metrics['C80']}")

    # ── Comparison table ─────────────────────────────────────
    print("\n" + "=" * 64)
    print("  OCTAVE-BAND COMPARISON")
    print("=" * 64)
    print(f"{'Band':>6s}  {'T30_meas':>8s} {'T30_sim':>8s} {'err%':>6s}  "
          f"{'EDT_meas':>8s} {'EDT_sim':>8s}  {'C80_meas':>8s} {'C80_sim':>8s}")
    print("-" * 80)

    t30_errors = []
    for i, fc in enumerate(OCTAVE_BANDS):
        t30_m = meas_metrics['T30'][i]
        t30_s = sim_metrics['T30'][i]
        edt_m = meas_metrics['EDT'][i]
        edt_s = sim_metrics['EDT'][i]
        c80_m = meas_metrics['C80'][i]
        c80_s = sim_metrics['C80'][i]

        t30_m_str = f"{t30_m:.3f}" if t30_m is not None else "N/A"
        t30_s_str = f"{t30_s:.3f}" if t30_s is not None else "N/A"
        if t30_m is not None and t30_s is not None and t30_m > 0:
            err = abs(t30_s - t30_m) / t30_m * 100
            t30_errors.append(err)
            err_str = f"{err:.1f}%"
        else:
            err_str = "N/A"

        edt_m_str = f"{edt_m:.3f}" if edt_m is not None else "N/A"
        edt_s_str = f"{edt_s:.3f}" if edt_s is not None else "N/A"
        c80_m_str = f"{c80_m:.1f}" if c80_m is not None else "N/A"
        c80_s_str = f"{c80_s:.1f}" if c80_s is not None else "N/A"

        print(f"{fc:>5d}Hz  {t30_m_str:>8s} {t30_s_str:>8s} {err_str:>6s}  "
              f"{edt_m_str:>8s} {edt_s_str:>8s}  {c80_m_str:>8s} {c80_s_str:>8s}")

    if t30_errors:
        print(f"\nMean T30 error: {np.mean(t30_errors):.1f}%")

    # ── Save WAVs ────────────────────────────────────────────
    ir_obj = ImpulseResponse(ir_sim, fs_meas, f'Simulated {src_name}->{rec_name}')
    ir_obj.to_wav(os.path.join(OUT, 'bras_simulated.wav'))
    ir_obj.to_npz(os.path.join(OUT, 'bras_simulated.npz'))

    ir_meas_norm = ir_meas / max(np.max(np.abs(ir_meas)), 1e-30) * 0.9
    wavfile.write(os.path.join(OUT, 'bras_measured.wav'), fs_meas,
                  (ir_meas_norm * 32767).astype(np.int16))

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Waveform
    t_meas = np.arange(len(ir_meas)) / fs_meas
    axes[0].plot(t_meas * 1000, ir_meas / max(np.max(np.abs(ir_meas)), 1e-30),
                 'b-', lw=0.3, alpha=0.5, label='Measured')
    axes[0].plot(t_sim * 1000, ir_sim / max(np.max(np.abs(ir_sim)), 1e-30),
                 'r-', lw=0.3, alpha=0.5, label='Simulated')
    axes[0].set_xlabel('Time [ms]')
    axes[0].set_ylabel('Normalized')
    axes[0].set_title(f'BRAS Scene 9: {src_name}->{rec_name}')
    axes[0].legend()
    axes[0].set_xlim(0, 500)
    axes[0].grid(True, alpha=0.3)

    # T30 comparison bar chart
    x = np.arange(len(OCTAVE_BANDS))
    w = 0.35
    t30_m = [v if v is not None else 0 for v in meas_metrics['T30']]
    t30_s = [v if v is not None else 0 for v in sim_metrics['T30']]
    axes[1].bar(x - w/2, t30_m, w, label='Measured', color='steelblue')
    axes[1].bar(x + w/2, t30_s, w, label='Simulated', color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{f}' for f in OCTAVE_BANDS])
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('T30 [s]')
    axes[1].set_title('T30 per octave band')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # EDC comparison
    edc_meas = np.cumsum(ir_meas[::-1]**2)[::-1]
    edc_sim = np.cumsum(ir_sim[::-1]**2)[::-1]
    edc_meas_db = 10*np.log10(edc_meas/max(edc_meas[0], 1e-30) + 1e-30)
    edc_sim_db = 10*np.log10(edc_sim/max(edc_sim[0], 1e-30) + 1e-30)
    axes[2].plot(t_meas * 1000, edc_meas_db, 'b-', lw=1, label='Measured')
    axes[2].plot(t_sim * 1000, edc_sim_db, 'r-', lw=1, label='Simulated')
    axes[2].set_xlabel('Time [ms]')
    axes[2].set_ylabel('EDC [dB]')
    axes[2].set_title('Energy Decay Curve')
    axes[2].set_ylim(-60, 0)
    axes[2].set_xlim(0, 2000)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'bras_validation.png'), dpi=150)
    plt.close()

    # Save report
    report = {
        'room': '8.4 x 6.7 x 3.0 m',
        'source': src_name,
        'receiver': rec_name,
        'Ne': Ne, 'P': P, 'N': N,
        'octave_bands': OCTAVE_BANDS,
        'measured_T30': meas_metrics['T30'],
        'simulated_T30': sim_metrics['T30'],
        'measured_EDT': meas_metrics['EDT'],
        'simulated_EDT': sim_metrics['EDT'],
        'measured_C80': meas_metrics['C80'],
        'simulated_C80': sim_metrics['C80'],
    }
    import json
    with open(os.path.join(OUT, 'bras_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nOutput: {OUT}")
    print("  bras_validation.png")
    print("  bras_simulated.wav / .npz")
    print("  bras_measured.wav")
    print("  bras_report.json")


if __name__ == '__main__':
    main()
