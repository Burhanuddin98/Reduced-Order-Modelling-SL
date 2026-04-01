#!/usr/bin/env python
"""
Phase 3: BRAS CR2 Full-Bandwidth Validation

Compares our hybrid platform (modal ROM + axial modes + ray tracer + ISM)
against published BRAS CR2 measured values across octave bands.

BRAS CR2: Seminar room, 8.4 x 6.7 x 3.0 m (168.84 m^3)
Source: Dodecahedron loudspeaker
Reference: Aspock & Vorlander, "BRAS - Benchmark for Room Acoustical
           Simulation", TU Berlin, 2020.

Measured broadband values (dodecahedron source, averaged over receivers):
  T30  = 1.677 s
  EDT  = 1.203 s
  C80  = 1.80 dB
  D50  = 0.400

Published octave-band T30 (s) from BRAS CR2 round-robin:
  125 Hz:  1.88 s  (high variance between methods)
  250 Hz:  1.69 s
  500 Hz:  1.70 s
  1000 Hz: 1.65 s
  2000 Hz: 1.52 s
  4000 Hz: 1.29 s

Published octave-band EDT (s):
  125 Hz:  1.30 s
  250 Hz:  1.42 s
  500 Hz:  1.28 s
  1000 Hz: 1.10 s
  2000 Hz: 0.97 s
  4000 Hz: 0.81 s

Material assignments (BRAS CR2 documentation):
  Floor:   linoleum on concrete
  Ceiling: acoustic tiles (perforated metal + mineral wool)
  Walls:   painted concrete / plaster
  One wall: glass panels (windows)
  One wall: wooden door + plaster

Acceptance criteria (from PLAN.md):
  - T30 per octave band (250-2000 Hz): <10% error vs measured
  - Broadband T30: <10% error
  - C80: within 2 dB of measured
  - EDT: within 20% of measured

Run: python test_phase3_bras_fullband.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================================
# BRAS CR2 published measured values
# ===================================================================

MEASURED_BROADBAND = {
    'T30': 1.663,
    'EDT': 1.166,
    'C80': 2.8,
    'D50': 0.400,
}

# Octave-band T30 and EDT computed directly from BRAS CR2 measured WAV files
# (10 dodecahedron RIRs: 2 sources x 5 receivers, mean values)
MEASURED_OCTAVE_T30 = {
    # 125 Hz: insufficient decay range for reliable T30
    250:  1.746,
    500:  2.024,
    1000: 1.939,
    2000: 1.745,
    4000: 1.563,
}

MEASURED_OCTAVE_EDT = {
    250:  1.457,
    500:  1.988,
    1000: 1.909,
    2000: 1.720,
    4000: 1.585,
}

# BRAS CR2 per-surface absorption (third-octave band data, simplified to octave)
# These are approximate mid-band values from the BRAS documentation
BRAS_ALPHA_PER_BAND = {
    # fc:  {floor, ceiling, front, back, left (window), right (door)}
    125:  {'floor': 0.02, 'ceiling': 0.30, 'front': 0.01, 'back': 0.01,
           'left': 0.18, 'right': 0.04},
    250:  {'floor': 0.03, 'ceiling': 0.55, 'front': 0.02, 'back': 0.02,
           'left': 0.06, 'right': 0.04},
    500:  {'floor': 0.03, 'ceiling': 0.65, 'front': 0.02, 'back': 0.02,
           'left': 0.04, 'right': 0.05},
    1000: {'floor': 0.03, 'ceiling': 0.75, 'front': 0.03, 'back': 0.03,
           'left': 0.03, 'right': 0.05},
    2000: {'floor': 0.03, 'ceiling': 0.80, 'front': 0.04, 'back': 0.04,
           'left': 0.02, 'right': 0.05},
    4000: {'floor': 0.04, 'ceiling': 0.85, 'front': 0.05, 'back': 0.05,
           'left': 0.02, 'right': 0.05},
}


def compute_octave_band_metrics(ir, sr, fc):
    """Compute T30 and EDT for a single octave band."""
    from room_acoustics.acoustics_metrics import compute_t30, compute_edt
    from scipy.signal import butter, filtfilt

    nyq = sr / 2
    fl = fc / np.sqrt(2)
    fh = min(fc * np.sqrt(2), nyq * 0.95)

    if fl >= nyq * 0.95:
        return None, None

    b, a = butter(4, [fl / nyq, fh / nyq], btype='band')
    ir_band = filtfilt(b, a, ir)

    t30, r2 = compute_t30(ir_band, 1.0 / sr)
    edt, _ = compute_edt(ir_band, 1.0 / sr)

    return t30 if r2 > 0.8 else None, edt


def main():
    from room_acoustics.room import Room
    from room_acoustics.acoustics_metrics import impedance_to_alpha
    from room_acoustics.results_io import save_result

    print("=" * 70)
    print("  Phase 3: BRAS CR2 Full-Bandwidth Validation")
    print("  8.4 x 6.7 x 3.0 m, measured T30 = 1.677 s")
    print("=" * 70)

    # ============================================================
    # Build room with BRAS materials
    # ============================================================
    # Use mid-frequency absorption to set the FI impedance values
    # (our Room API uses frequency-independent impedance)
    Lx, Ly, Lz = 8.4, 6.7, 3.0
    rho_c = 1.2 * 343.0

    def alpha_to_Z(alpha):
        alpha = np.clip(alpha, 0.001, 0.999)
        R = np.sqrt(1.0 - alpha)
        return rho_c * (1 + R) / (1 - R)

    # Use 500 Hz absorption as the representative FI impedance
    alpha_500 = BRAS_ALPHA_PER_BAND[500]

    room = Room.from_box(Lx, Ly, Lz, P=4, ppw=6, f_target=400)

    # Set materials via impedance values (since we don't have exact
    # material names for BRAS, use impedance directly)
    # We need to work within the existing material system, so we'll
    # find the closest material match
    from room_acoustics.materials import MATERIALS

    def closest_material(target_alpha):
        """Find material with Z closest to target absorption."""
        target_Z = alpha_to_Z(target_alpha)
        best = min(MATERIALS.keys(),
                   key=lambda m: abs(MATERIALS[m]['Z'] - target_Z))
        return best

    mat_map = {
        'floor':   closest_material(alpha_500['floor']),
        'ceiling': closest_material(alpha_500['ceiling']),
        'front':   closest_material(alpha_500['front']),
        'back':    closest_material(alpha_500['back']),
        'left':    closest_material(alpha_500['left']),
        'right':   closest_material(alpha_500['right']),
    }

    print("\n  Material assignments (closest match to BRAS 500 Hz alpha):")
    for label, mat in mat_map.items():
        Z = MATERIALS[mat]['Z']
        alpha = 1.0 - abs((Z - rho_c) / (Z + rho_c))**2
        print(f"    {label:10s}: {mat:25s} (alpha={alpha:.3f}, "
              f"target={alpha_500[label]:.3f})")
        room.set_material(label, mat)

    # Build with enough modes to cover up to ~400 Hz
    print()
    room.build(n_modes=300)

    # ============================================================
    # Compute IR at two receiver positions
    # ============================================================
    source = (2.0, 3.35, 1.5)
    receivers = [
        (6.0, 2.0, 1.2),
        (6.0, 5.0, 1.2),
    ]

    all_results = []
    for i, rec in enumerate(receivers):
        print(f"\n{'='*70}")
        print(f"  Receiver {i+1}: {rec}")
        print(f"{'='*70}")

        ir = room.impulse_response(source, rec, T=3.0, n_rays=5000,
                                    max_bounces=200)
        ir.summary()

        # Octave-band analysis
        print(f"\n  Octave-band T30 comparison:")
        print(f"  {'Band':>6s}  {'Simulated':>10s}  {'Measured':>10s}  {'Error':>8s}")
        print(f"  {'-'*6:>6s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*8:>8s}")

        band_results = {}
        for fc in [250, 500, 1000, 2000, 4000]:
            t30_sim, edt_sim = compute_octave_band_metrics(ir.data, ir.sr, fc)
            t30_meas = MEASURED_OCTAVE_T30.get(fc)
            edt_meas = MEASURED_OCTAVE_EDT.get(fc)

            err = abs(t30_sim - t30_meas) / t30_meas * 100 if (
                t30_sim and t30_meas) else float('nan')

            t30_str = f"{t30_sim:.3f}s" if t30_sim else "N/A"
            meas_str = f"{t30_meas:.3f}s" if t30_meas else "N/A"
            err_str = f"{err:.1f}%" if not np.isnan(err) else "N/A"

            print(f"  {fc:6d}  {t30_str:>10s}  {meas_str:>10s}  {err_str:>8s}")

            band_results[fc] = {
                'sim_T30': float(t30_sim) if t30_sim else None,
                'sim_EDT': float(edt_sim) if edt_sim else None,
                'meas_T30': t30_meas,
                'meas_EDT': edt_meas,
                'err_T30_pct': float(err) if not np.isnan(err) else None,
            }

        # Broadband comparison
        print(f"\n  Broadband comparison:")
        print(f"    T30: {ir.T30:.3f}s vs {MEASURED_BROADBAND['T30']:.3f}s "
              f"({abs(ir.T30 - MEASURED_BROADBAND['T30'])/MEASURED_BROADBAND['T30']*100:.1f}%)")
        print(f"    EDT: {ir.EDT:.3f}s vs {MEASURED_BROADBAND['EDT']:.3f}s "
              f"({abs(ir.EDT - MEASURED_BROADBAND['EDT'])/MEASURED_BROADBAND['EDT']*100:.1f}%)")
        print(f"    C80: {ir.C80:.1f}dB vs {MEASURED_BROADBAND['C80']:.1f}dB "
              f"(delta={ir.C80 - MEASURED_BROADBAND['C80']:+.1f}dB)")

        all_results.append({
            'receiver': rec,
            'broadband': ir.metrics,
            'octave_bands': band_results,
        })

    # ============================================================
    # Acceptance criteria
    # ============================================================
    print(f"\n{'='*70}")
    print("  ACCEPTANCE CRITERIA")
    print(f"{'='*70}")

    # Average over receivers
    checks = []

    # Broadband T30
    avg_T30 = np.mean([r['broadband']['T30_s'] for r in all_results])
    err_T30 = abs(avg_T30 - MEASURED_BROADBAND['T30']) / MEASURED_BROADBAND['T30'] * 100
    ok = err_T30 < 10
    checks.append(('Broadband T30 <10%', ok, f"{err_T30:.1f}%"))

    # Broadband EDT
    avg_EDT = np.mean([r['broadband']['EDT_s'] for r in all_results])
    err_EDT = abs(avg_EDT - MEASURED_BROADBAND['EDT']) / MEASURED_BROADBAND['EDT'] * 100
    ok = err_EDT < 20
    checks.append(('Broadband EDT <20%', ok, f"{err_EDT:.1f}%"))

    # Broadband C80
    avg_C80 = np.mean([r['broadband']['C80_dB'] for r in all_results])
    delta_C80 = abs(avg_C80 - MEASURED_BROADBAND['C80'])
    ok = delta_C80 < 2
    checks.append(('C80 within 2 dB', ok, f"delta={delta_C80:.1f}dB"))

    # Per-band T30 (250-2000 Hz)
    for fc in [250, 500, 1000, 2000]:
        errs = []
        for r in all_results:
            b = r['octave_bands'].get(fc, {})
            if b.get('err_T30_pct') is not None:
                errs.append(b['err_T30_pct'])
        if errs:
            avg_err = np.mean(errs)
            ok = avg_err < 10
            checks.append((f'T30 @{fc}Hz <10%', ok, f"{avg_err:.1f}%"))
        else:
            checks.append((f'T30 @{fc}Hz <10%', False, "no data"))

    all_pass = all(c[1] for c in checks)
    for label, ok, detail in checks:
        status = 'PASS' if ok else 'FAIL'
        print(f"  {status}: {label} -- {detail}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    # ============================================================
    # Save results
    # ============================================================
    save_result('phase3_bras_fullband', {
        'measured_broadband': MEASURED_BROADBAND,
        'measured_octave_T30': MEASURED_OCTAVE_T30,
        'measured_octave_EDT': MEASURED_OCTAVE_EDT,
        'receivers': all_results,
        'acceptance': [{'name': c[0], 'pass': c[1], 'detail': c[2]}
                       for c in checks],
        'all_passed': all_pass,
        'material_map': mat_map,
    }, suite='phase3')

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 3: BRAS CR2 Full-Bandwidth Validation', fontweight='bold')

    bands = [125, 250, 500, 1000, 2000, 4000]
    x = np.arange(len(bands))
    width = 0.35

    # T30 per band
    meas_t30 = [MEASURED_OCTAVE_T30[fc] for fc in bands]
    sim_t30 = []
    for fc in bands:
        vals = [r['octave_bands'].get(fc, {}).get('sim_T30') for r in all_results]
        vals = [v for v in vals if v is not None]
        sim_t30.append(np.mean(vals) if vals else 0)

    axes[0].bar(x - width/2, meas_t30, width, label='Measured', color='#2196F3')
    axes[0].bar(x + width/2, sim_t30, width, label='Simulated', color='#F44336')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(fc) for fc in bands])
    axes[0].set_xlabel('Octave Band [Hz]')
    axes[0].set_ylabel('T30 [s]')
    axes[0].set_title('Octave-Band T30')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Error per band
    errs = []
    for fc in bands:
        vals = [r['octave_bands'].get(fc, {}).get('err_T30_pct') for r in all_results]
        vals = [v for v in vals if v is not None]
        errs.append(np.mean(vals) if vals else 0)

    colors = ['green' if e < 10 else 'orange' if e < 20 else 'red' for e in errs]
    axes[1].bar(x, errs, color=colors)
    axes[1].axhline(10, color='red', ls='--', alpha=0.5, label='10% target')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(fc) for fc in bands])
    axes[1].set_xlabel('Octave Band [Hz]')
    axes[1].set_ylabel('T30 Error [%]')
    axes[1].set_title('T30 Error vs Measured')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'phase3_bras_fullband.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    main()
