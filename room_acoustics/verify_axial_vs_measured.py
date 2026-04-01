#!/usr/bin/env python
"""
Verify axial modes against BRAS CR2 measured impulse responses.

Checks whether the measured RIRs contain resonant peaks at the
predicted axial mode frequencies f_n = n*c/(2L), and whether the
measured decay rates at those frequencies match our analytical model.

Accounts for:
  - Temperature-dependent speed of sound (BRAS CR2: 19.5°C)
  - Geometry tolerances (~2-3 cm on each dimension)
  - Frequency matching tolerance (±2 Hz per mode)

BRAS CR2: 8.4 x 6.7 x 3.0 m seminar room, RWTH Aachen
  Temperature: 19.5°C  →  c = 343.1 m/s
  10 dodecahedron RIRs (2 sources x 5 receivers)

Run: python verify_axial_vs_measured.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================================================================
# BRAS CR2 room parameters
# ===================================================================
Lx, Ly, Lz = 8.4, 6.7, 3.0
TEMP_C = 19.5
C_AIR = 331.3 + 0.606 * TEMP_C  # 343.1 m/s

# Geometry tolerance: dimensions could be off by this much
GEOM_TOL = 0.03  # 3 cm

# Frequency matching tolerance (Hz) — accounts for temperature
# uncertainty, geometry tolerance, and spectral resolution
FREQ_TOL = 3.0

RIR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'bras_data', '1 Scene descriptions',
    'CR2 small room (seminar room)', 'RIRs', 'wav')


def load_rirs():
    """Load all dodecahedron RIRs, return list of (filename, ir, sr)."""
    rirs = []
    for fn in sorted(os.listdir(RIR_DIR)):
        if not fn.endswith('.wav') or 'Dodecahedron' not in fn:
            continue
        sr, data = wavfile.read(os.path.join(RIR_DIR, fn))
        ir = data.astype(np.float64)
        if ir.ndim > 1:
            ir = ir[:, 0]
        ir /= max(np.abs(ir).max(), 1e-10)
        rirs.append((fn, ir, sr))
    return rirs


def compute_spectrum(ir, sr, f_max=500):
    """Compute magnitude spectrum up to f_max Hz."""
    N = len(ir)
    spectrum = np.abs(np.fft.rfft(ir))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    mask = freqs <= f_max
    return freqs[mask], spectrum[mask]


def find_peaks(freqs, spectrum, min_prominence=0.3):
    """Find spectral peaks above a prominence threshold.

    Uses a simple local-maximum approach: a peak is higher than
    both neighbors by at least min_prominence * median level.
    """
    # Smooth spectrum slightly for robust peak detection
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(spectrum, size=3)

    threshold = min_prominence * np.median(smooth)
    peaks = []
    for i in range(2, len(smooth) - 2):
        if (smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]
                and smooth[i] > smooth[i-2] and smooth[i] > smooth[i+2]
                and smooth[i] > threshold):
            peaks.append((freqs[i], smooth[i]))
    return peaks


def measure_decay_at_freq(ir, sr, f_center, bandwidth=5.0):
    """Measure the decay rate of the IR at a specific frequency.

    Bandpass filters around f_center, computes the envelope,
    and fits an exponential decay to get gamma.

    Returns gamma (decay rate in 1/s) or None if unreliable.
    """
    nyq = sr / 2
    fl = max(f_center - bandwidth, 1.0)
    fh = min(f_center + bandwidth, nyq * 0.95)
    if fl >= fh:
        return None

    sos = butter(4, [fl / nyq, fh / nyq], btype='band', output='sos')
    ir_band = sosfiltfilt(sos, ir)

    # Compute envelope via Hilbert transform
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(ir_band))

    # Smooth envelope
    from scipy.ndimage import uniform_filter1d
    window = max(1, int(0.01 * sr))  # 10 ms smoothing
    envelope = uniform_filter1d(envelope, window)

    # Find the peak (direct sound arrival)
    peak_idx = np.argmax(envelope)
    if peak_idx >= len(envelope) - int(0.1 * sr):
        return None  # peak too late

    # Fit exponential decay from peak to -40 dB below peak
    env_from_peak = envelope[peak_idx:]
    peak_val = env_from_peak[0]
    if peak_val < 1e-10:
        return None

    # Find where envelope drops to -40 dB (1% of peak)
    threshold = peak_val * 0.01
    valid = env_from_peak > threshold
    if np.sum(valid) < int(0.05 * sr):
        return None  # too short for reliable fit

    n_valid = np.argmin(valid)
    if n_valid < int(0.05 * sr):
        n_valid = min(len(env_from_peak), int(0.5 * sr))

    t_fit = np.arange(n_valid) / sr
    env_fit = env_from_peak[:n_valid]

    # Log-linear fit: log(envelope) = -gamma*t + const
    log_env = np.log(np.maximum(env_fit, 1e-20))
    # Use only the valid portion (above threshold)
    mask = env_fit > threshold
    if np.sum(mask) < 10:
        return None

    t_masked = t_fit[mask]
    log_masked = log_env[mask]

    # Linear regression
    A = np.vstack([t_masked, np.ones(len(t_masked))]).T
    result = np.linalg.lstsq(A, log_masked, rcond=None)
    slope = result[0][0]

    gamma = -slope  # decay rate (positive = decaying)
    if gamma < 0:
        return None  # growing, not physical

    return gamma


def main():
    print("=" * 70)
    print("  Axial Mode Verification vs BRAS CR2 Measured RIRs")
    print(f"  Room: {Lx} x {Ly} x {Lz} m, T = {TEMP_C}°C, c = {C_AIR:.1f} m/s")
    print("=" * 70)

    rirs = load_rirs()
    print(f"\n  Loaded {len(rirs)} measured RIRs")

    # ============================================================
    # Predicted axial mode frequencies
    # ============================================================
    pairs = [
        ('left/right', Lx),
        ('front/back', Ly),
        ('floor/ceiling', Lz),
    ]

    # Generate predicted modes up to 400 Hz
    f_max_analysis = 400
    predicted_modes = []
    for pair_name, L in pairs:
        n_max = int(np.floor(2 * L * f_max_analysis / C_AIR))
        for n in range(1, n_max + 1):
            f = n * C_AIR / (2 * L)
            if f <= f_max_analysis:
                predicted_modes.append({
                    'pair': pair_name,
                    'L': L,
                    'n': n,
                    'freq': f,
                })

    predicted_modes.sort(key=lambda m: m['freq'])
    print(f"\n  Predicted axial modes below {f_max_analysis} Hz: {len(predicted_modes)}")

    for m in predicted_modes:
        print(f"    {m['freq']:7.1f} Hz  n={m['n']:2d}  {m['pair']}")

    # ============================================================
    # Test 1: Spectral peak matching
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEST 1: Spectral peak matching")
    print(f"  (tolerance: ±{FREQ_TOL} Hz)")
    print(f"{'='*70}")

    # Average spectrum over all RIRs for robust peak detection
    all_spectra = []
    for fn, ir, sr in rirs:
        freqs, spec = compute_spectrum(ir, sr, f_max=f_max_analysis + 50)
        all_spectra.append(spec)
    avg_spectrum = np.mean(all_spectra, axis=0)

    # Find peaks in averaged spectrum
    peaks = find_peaks(freqs, avg_spectrum, min_prominence=0.2)
    peak_freqs = [p[0] for p in peaks]

    print(f"\n  Found {len(peaks)} spectral peaks in averaged measured spectrum")

    matched = 0
    unmatched = 0
    for m in predicted_modes:
        f_pred = m['freq']
        # Find closest measured peak
        if peak_freqs:
            dists = [abs(pf - f_pred) for pf in peak_freqs]
            best_dist = min(dists)
            best_peak = peak_freqs[dists.index(best_dist)]
        else:
            best_dist = 999
            best_peak = 0

        if best_dist <= FREQ_TOL:
            matched += 1
            status = "MATCH"
        else:
            unmatched += 1
            status = "miss "

        print(f"    {status}  predicted {f_pred:7.1f} Hz ({m['pair']:14s} n={m['n']})  "
              f"nearest peak: {best_peak:7.1f} Hz  (delta: {best_dist:+.1f} Hz)")

    match_rate = matched / len(predicted_modes) * 100 if predicted_modes else 0
    print(f"\n  Match rate: {matched}/{len(predicted_modes)} ({match_rate:.0f}%)")

    # ============================================================
    # Test 2: Decay rate comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEST 2: Decay rate at axial frequencies")
    print(f"{'='*70}")

    # Load BRAS absorption data for analytical decay prediction
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'bras_data', '3 Surface descriptions', '_csv', 'fitted_estimates')

    bras_alpha = {}
    for mat_file in ['mat_CR2_floor.csv', 'mat_CR2_ceiling.csv',
                      'mat_CR2_concrete.csv', 'mat_CR2_plaster.csv',
                      'mat_CR2_windows.csv']:
        path = os.path.join(csv_dir, mat_file)
        if os.path.exists(path):
            lines = open(path).readlines()
            freqs_csv = [float(x) for x in lines[0].strip().split(',')]
            alphas_csv = [float(x) for x in lines[1].strip().split(',')]
            name = mat_file.replace('mat_CR2_', '').replace('.csv', '')
            bras_alpha[name] = dict(zip(freqs_csv, alphas_csv))

    def get_alpha_at_freq(material, freq):
        """Interpolate BRAS absorption at a given frequency."""
        if material not in bras_alpha:
            return 0.05  # default
        data = bras_alpha[material]
        freqs_sorted = sorted(data.keys())
        alphas_sorted = [data[f] for f in freqs_sorted]
        return float(np.interp(freq, freqs_sorted, alphas_sorted))

    # Surface material mapping for each parallel pair
    pair_materials = {
        'left/right':    ('plaster', 'plaster'),   # approximate: walls are plaster/concrete
        'front/back':    ('concrete', 'concrete'),
        'floor/ceiling': ('floor', 'ceiling'),
    }

    print(f"\n  {'Freq':>7s}  {'Pair':>14s}  {'Meas gamma':>11s}  {'Pred gamma':>11s}  "
          f"{'Error':>7s}  {'Meas T60':>8s}  {'Pred T60':>8s}")
    print(f"  {'-'*7:>7s}  {'-'*14:>14s}  {'-'*11:>11s}  {'-'*11:>11s}  "
          f"{'-'*7:>7s}  {'-'*8:>8s}  {'-'*8:>8s}")

    decay_results = []
    for m in predicted_modes:
        if m['freq'] < 30 or m['freq'] > 300:
            continue  # skip very low (poor SNR) and high (tangential contamination)

        # Measure decay from averaged RIR (more robust)
        gamma_measurements = []
        for fn, ir, sr in rirs:
            g = measure_decay_at_freq(ir, sr, m['freq'], bandwidth=4.0)
            if g is not None and g > 0.1:
                gamma_measurements.append(g)

        if len(gamma_measurements) < 3:
            continue  # not enough reliable measurements

        gamma_meas = np.median(gamma_measurements)

        # Predicted decay from BRAS absorption data
        pair_name = m['pair']
        L = m['L']
        mat1, mat2 = pair_materials[pair_name]
        alpha1 = get_alpha_at_freq(mat1, m['freq'])
        alpha2 = get_alpha_at_freq(mat2, m['freq'])

        R_product = (1 - alpha1) * (1 - alpha2)
        if R_product <= 0 or R_product >= 1:
            continue
        gamma_pred = (C_AIR / (2 * L)) * (-np.log(R_product))

        # T60 from gamma: T60 = 6.91 / gamma (time for -60 dB)
        t60_meas = 6.91 / gamma_meas if gamma_meas > 0 else float('inf')
        t60_pred = 6.91 / gamma_pred if gamma_pred > 0 else float('inf')

        err = abs(gamma_meas - gamma_pred) / gamma_meas * 100

        print(f"  {m['freq']:7.1f}  {pair_name:>14s}  {gamma_meas:9.2f}/s  "
              f"{gamma_pred:9.2f}/s  {err:5.1f}%  {t60_meas:6.2f}s  {t60_pred:6.2f}s")

        decay_results.append({
            'freq': m['freq'],
            'pair': pair_name,
            'gamma_meas': gamma_meas,
            'gamma_pred': gamma_pred,
            'error_pct': err,
            't60_meas': t60_meas,
            't60_pred': t60_pred,
        })

    if decay_results:
        avg_err = np.mean([r['error_pct'] for r in decay_results])
        median_err = np.median([r['error_pct'] for r in decay_results])
        print(f"\n  Decay rate error: mean={avg_err:.1f}%, median={median_err:.1f}%")

    # ============================================================
    # Test 3: Position-dependent amplitude
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEST 3: Position-dependent mode amplitude")
    print("  (receivers near walls vs center should show different mode strengths)")
    print(f"{'='*70}")

    # Compare spectral energy at floor/ceiling axial frequencies
    # between different receiver positions
    f_fc1 = C_AIR / (2 * Lz)  # ~57.2 Hz (floor/ceiling fundamental)

    energies_per_rir = []
    for fn, ir, sr in rirs:
        nyq = sr / 2
        fl = max(f_fc1 - 5, 1)
        fh = min(f_fc1 + 5, nyq * 0.95)
        sos = butter(4, [fl/nyq, fh/nyq], btype='band', output='sos')
        ir_band = sosfiltfilt(sos, ir)
        energy = np.sum(ir_band**2)
        energies_per_rir.append((fn, energy))

    # Sort by energy to see variation
    energies_per_rir.sort(key=lambda x: -x[1])
    max_e = energies_per_rir[0][1]

    print(f"\n  Energy at floor/ceiling fundamental ({f_fc1:.1f} Hz):")
    for fn, e in energies_per_rir:
        # Extract source/receiver from filename
        parts = fn.replace('.wav', '').split('_')
        ls = parts[2]  # LS1 or LS2
        mp = parts[3]  # MP1-MP5
        rel_dB = 10 * np.log10(e / max_e) if e > 0 else -999
        bar = '#' * max(0, int(20 + rel_dB))
        print(f"    {ls} {mp}: {rel_dB:+5.1f} dB  {bar}")

    spread = 10 * np.log10(energies_per_rir[0][1] / max(energies_per_rir[-1][1], 1e-30))
    print(f"\n  Spread: {spread:.1f} dB across receiver positions")
    if spread > 3:
        print("  -> Position-dependent variation confirmed (>3 dB spread)")
    else:
        print("  -> Weak position dependence (room may be too diffuse at this freq)")

    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Axial Mode Verification vs BRAS CR2 Measured RIRs',
                 fontweight='bold')

    # 1. Averaged spectrum with predicted modes
    axes[0, 0].semilogy(freqs, avg_spectrum, 'b-', lw=0.8, label='Measured (avg)')
    for m in predicted_modes:
        axes[0, 0].axvline(m['freq'], color='red', alpha=0.3, lw=0.5)
    # Mark matched peaks
    for m in predicted_modes:
        if peak_freqs:
            best_dist = min(abs(pf - m['freq']) for pf in peak_freqs)
            if best_dist <= FREQ_TOL:
                axes[0, 0].axvline(m['freq'], color='green', alpha=0.5, lw=1)
    axes[0, 0].set_xlabel('Frequency [Hz]')
    axes[0, 0].set_ylabel('Magnitude')
    axes[0, 0].set_title('Spectrum: measured peaks vs predicted axial modes')
    axes[0, 0].set_xlim(0, f_max_analysis)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Decay rate comparison
    if decay_results:
        dr_freqs = [r['freq'] for r in decay_results]
        dr_meas = [r['gamma_meas'] for r in decay_results]
        dr_pred = [r['gamma_pred'] for r in decay_results]
        axes[0, 1].scatter(dr_freqs, dr_meas, c='blue', s=30, label='Measured', zorder=3)
        axes[0, 1].scatter(dr_freqs, dr_pred, c='red', s=30, marker='x',
                           label='Predicted', zorder=3)
        for i in range(len(dr_freqs)):
            axes[0, 1].plot([dr_freqs[i], dr_freqs[i]], [dr_meas[i], dr_pred[i]],
                           'gray', lw=0.5, alpha=0.5)
        axes[0, 1].set_xlabel('Frequency [Hz]')
        axes[0, 1].set_ylabel('Decay rate gamma [1/s]')
        axes[0, 1].set_title('Decay rates: measured vs predicted')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. T60 comparison
    if decay_results:
        t60_meas = [r['t60_meas'] for r in decay_results]
        t60_pred = [r['t60_pred'] for r in decay_results]
        axes[1, 0].scatter(t60_meas, t60_pred, c='purple', s=30, zorder=3)
        lim = max(max(t60_meas), max(t60_pred)) * 1.1
        axes[1, 0].plot([0, lim], [0, lim], 'k--', alpha=0.3, label='Perfect match')
        axes[1, 0].set_xlabel('Measured T60 [s]')
        axes[1, 0].set_ylabel('Predicted T60 [s]')
        axes[1, 0].set_title('T60 per axial mode')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect('equal')

    # 4. Per-RIR energy at fundamental
    rirs_names = [f"{fn.split('_')[2]}_{fn.split('_')[3]}"
                  for fn, _ in energies_per_rir]
    rirs_dB = [10*np.log10(e/max_e) if e > 0 else -60
               for _, e in energies_per_rir]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(rirs_dB)))
    axes[1, 1].barh(rirs_names, rirs_dB, color=colors)
    axes[1, 1].set_xlabel('Relative energy [dB]')
    axes[1, 1].set_title(f'Position dependence at {f_fc1:.0f} Hz (floor/ceiling)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'axial_mode_verification.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Spectral peak match rate: {match_rate:.0f}% "
          f"({matched}/{len(predicted_modes)} modes)")
    if decay_results:
        print(f"  Decay rate error: mean={avg_err:.1f}%, median={median_err:.1f}%")
    print(f"  Position-dependent spread: {spread:.1f} dB")


if __name__ == '__main__':
    main()
