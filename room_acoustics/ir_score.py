"""
IR quality scoring — comprehensive comparison of simulated vs measured IRs.

Produces a single score (0-100) from multiple weighted sub-metrics,
plus detailed per-metric breakdown for diagnostics. Designed to catch
both metric accuracy (T30, C80) and perceptual quality (spectral balance,
temporal envelope, early reflection pattern).

Usage:
    from room_acoustics.ir_score import score_ir, print_scorecard

    result = score_ir(ir_sim, ir_meas, sr=44100)
    print_scorecard(result)

All metrics are computed at full spectral resolution — no octave-band
limitations. Each sub-metric is normalized to 0-1 (1 = perfect match)
before weighted combination into the final score.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt, hilbert
from scipy.ndimage import uniform_filter1d


def score_ir(ir_sim, ir_meas, sr=44100):
    """
    Comprehensive comparison of simulated vs measured IR.

    Parameters
    ----------
    ir_sim, ir_meas : ndarray
        Simulated and measured impulse responses (same sample rate).
    sr : int
        Sample rate [Hz].

    Returns
    -------
    result : dict
        'score': float 0-100 (overall quality)
        'sub_scores': dict of metric_name -> {value, score_01, weight, detail}
    """
    # Align lengths
    n = min(len(ir_sim), len(ir_meas))
    sim = ir_sim[:n].astype(np.float64)
    meas = ir_meas[:n].astype(np.float64)
    nyq = sr / 2
    dt = 1.0 / sr

    results = {}

    # ===============================================================
    # 1. BROADBAND T30 (how long does the room ring?)
    # ===============================================================
    from .acoustics_metrics import compute_t30, all_metrics
    m_sim = all_metrics(sim, dt)
    m_meas = all_metrics(meas, dt)

    t30_sim = m_sim['T30_s']
    t30_meas = m_meas['T30_s']
    if t30_meas > 0 and not np.isnan(t30_sim):
        t30_err = abs(t30_sim - t30_meas) / t30_meas
        t30_score = max(0, 1 - t30_err / 0.2)  # 0% err = 1.0, 20% = 0.0
    else:
        t30_err = 1.0
        t30_score = 0.0

    results['broadband_T30'] = {
        'value': f'{t30_sim:.3f}s vs {t30_meas:.3f}s ({t30_err*100:.1f}%)',
        'score_01': t30_score,
        'weight': 0.10,
    }

    # ===============================================================
    # 2. PER-BAND T30 (frequency-dependent decay)
    # ===============================================================
    bands = [125, 250, 500, 1000, 2000, 4000]
    band_errors = []
    band_detail = []
    for fc in bands:
        fl = fc / np.sqrt(2)
        fh = min(fc * np.sqrt(2), nyq * 0.95)
        sos = butter(3, [fl / nyq, fh / nyq], btype='band', output='sos')
        t30_s, r2_s = compute_t30(sosfiltfilt(sos, sim), dt)
        t30_m, r2_m = compute_t30(sosfiltfilt(sos, meas), dt)
        if r2_s > 0.7 and r2_m > 0.7 and t30_m > 0:
            err = abs(t30_s - t30_m) / t30_m
            band_errors.append(err)
            band_detail.append(f'{fc}Hz: {t30_s:.3f} vs {t30_m:.3f} ({err*100:.1f}%)')
        else:
            band_detail.append(f'{fc}Hz: N/A')

    if band_errors:
        mean_band_err = np.mean(band_errors)
        band_score = max(0, 1 - mean_band_err / 0.2)
    else:
        band_score = 0.0

    results['per_band_T30'] = {
        'value': f'mean err {mean_band_err*100:.1f}%' if band_errors else 'N/A',
        'score_01': band_score,
        'weight': 0.10,
        'detail': band_detail,
    }

    # ===============================================================
    # 3. C80 (early-to-late energy ratio — perceptual clarity)
    # ===============================================================
    c80_sim = m_sim['C80_dB']
    c80_meas = m_meas['C80_dB']
    c80_diff = abs(c80_sim - c80_meas)
    c80_score = max(0, 1 - c80_diff / 6.0)  # 0 dB diff = 1.0, 6 dB = 0.0

    results['C80'] = {
        'value': f'{c80_sim:.1f} vs {c80_meas:.1f} dB (delta {c80_diff:.1f})',
        'score_01': c80_score,
        'weight': 0.10,
    }

    # ===============================================================
    # 4. SPECTRAL ENVELOPE (tonal balance — 1/3-octave)
    # ===============================================================
    f_thirds = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
    e_sim = []
    e_meas = []
    for fc in f_thirds:
        fl = fc / (2 ** (1 / 6))
        fh = min(fc * (2 ** (1 / 6)), nyq * 0.95)
        if fl >= nyq * 0.95:
            continue
        try:
            sos = butter(3, [fl / nyq, fh / nyq], btype='band', output='sos')
            e_sim.append(np.sum(sosfiltfilt(sos, sim) ** 2))
            e_meas.append(np.sum(sosfiltfilt(sos, meas) ** 2))
        except ValueError:
            continue

    e_sim = np.array(e_sim)
    e_meas = np.array(e_meas)
    valid = (e_sim > 1e-20) & (e_meas > 1e-20)

    if valid.any():
        ratio_dB = 10 * np.log10(e_sim[valid] / e_meas[valid])
        spectral_rms = np.sqrt(np.mean(ratio_dB ** 2))
        spectral_max = np.max(np.abs(ratio_dB))
        spectral_score = max(0, 1 - spectral_rms / 10.0)  # 0 dB = 1.0, 10 dB = 0.0

        # Find worst bands
        worst_idx = np.argsort(-np.abs(ratio_dB))[:3]
        valid_fc = [f_thirds[i] for i, v in enumerate(valid) if v]
        worst_detail = [f'{valid_fc[i]}Hz: {ratio_dB[i]:+.1f}dB' for i in worst_idx]
    else:
        spectral_rms = 99
        spectral_max = 99
        spectral_score = 0
        worst_detail = []

    results['spectral_balance'] = {
        'value': f'RMS {spectral_rms:.1f} dB, max {spectral_max:.1f} dB',
        'score_01': spectral_score,
        'weight': 0.20,
        'detail': worst_detail,
    }

    # ===============================================================
    # 5. TEMPORAL ENVELOPE per band (decay shape matching)
    # ===============================================================
    env_bands = [125, 250, 500, 1000, 2000, 4000]
    env_corrs = []
    env_detail = []
    w = max(1, int(0.005 * sr))

    for fc in env_bands:
        fl = fc / np.sqrt(2)
        fh = min(fc * np.sqrt(2), nyq * 0.95)
        try:
            sos = butter(3, [fl / nyq, fh / nyq], btype='band', output='sos')
            env_s = uniform_filter1d(np.abs(sosfiltfilt(sos, sim)), w)
            env_m = uniform_filter1d(np.abs(sosfiltfilt(sos, meas)), w)
            corr = np.corrcoef(env_s, env_m)[0, 1]
            if not np.isnan(corr):
                env_corrs.append(corr)
                env_detail.append(f'{fc}Hz: {corr:.4f}')
        except ValueError:
            pass

    if env_corrs:
        mean_env = np.mean(env_corrs)
        env_score = max(0, mean_env)  # correlation is already 0-1
    else:
        mean_env = 0
        env_score = 0

    results['envelope_correlation'] = {
        'value': f'mean {mean_env:.4f}',
        'score_01': env_score,
        'weight': 0.15,
        'detail': env_detail,
    }

    # ===============================================================
    # 6. EARLY REFLECTION PATTERN (first 80ms)
    # ===============================================================
    from scipy.signal import find_peaks
    n80 = int(0.08 * sr)
    early_m = np.abs(meas[:n80])
    early_s = np.abs(sim[:n80])

    thresh_m = 0.05 * np.max(early_m)
    thresh_s = 0.05 * np.max(early_s)
    peaks_m, _ = find_peaks(early_m, height=thresh_m, distance=int(0.001 * sr))
    peaks_s, _ = find_peaks(early_s, height=thresh_s, distance=int(0.001 * sr))

    matched = 0
    for pm in peaks_m:
        if len(peaks_s) > 0:
            dists = np.abs(peaks_s - pm)
            if np.min(dists) < int(0.002 * sr):  # within 2ms
                matched += 1

    if len(peaks_m) > 0:
        early_match = matched / len(peaks_m)
    else:
        early_match = 0

    results['early_reflections'] = {
        'value': f'{matched}/{len(peaks_m)} peaks matched within 2ms',
        'score_01': early_match,
        'weight': 0.10,
    }

    # ===============================================================
    # 7. SCHROEDER DECAY SHAPE (overall energy envelope)
    # ===============================================================
    from .acoustics_metrics import schroeder_decay
    t_s, d_s = schroeder_decay(sim, dt)
    t_m, d_m = schroeder_decay(meas, dt)

    # Resample to common time grid (100 points from 0 to min(T_sim, T_meas))
    t_max = min(t_s[-1], t_m[-1])
    t_common = np.linspace(0, t_max, 200)
    d_s_interp = np.interp(t_common, t_s, d_s)
    d_m_interp = np.interp(t_common, t_m, d_m)

    # RMS difference in dB
    decay_rms = np.sqrt(np.mean((d_s_interp - d_m_interp) ** 2))
    decay_score = max(0, 1 - decay_rms / 15.0)  # 0 dB = 1.0, 15 dB = 0.0

    results['schroeder_decay'] = {
        'value': f'RMS diff {decay_rms:.1f} dB',
        'score_01': decay_score,
        'weight': 0.15,
    }

    # ===============================================================
    # 8. TIME-VARYING SPECTRAL BALANCE (spectrogram match)
    # ===============================================================
    # Compare spectral envelopes at 3 time windows
    windows = [(0, 0.05), (0.05, 0.2), (0.2, 1.0)]
    tv_errors = []
    tv_detail = []

    for t_start, t_end in windows:
        n_s = int(t_start * sr)
        n_e = min(int(t_end * sr), n)
        if n_e <= n_s + 256:
            continue

        spec_s = np.abs(np.fft.rfft(sim[n_s:n_e]))
        spec_m = np.abs(np.fft.rfft(meas[n_s:n_e]))
        freqs_w = np.fft.rfftfreq(n_e - n_s, dt)
        fm = (freqs_w >= 50) & (freqs_w <= 8000)

        if fm.any():
            w_sm = max(1, fm.sum() // 50)
            ss = uniform_filter1d(spec_s[fm], w_sm)
            sm = uniform_filter1d(spec_m[fm], w_sm)
            valid_w = (ss > 1e-10) & (sm > 1e-10)
            if valid_w.any():
                ratio = 20 * np.log10(ss[valid_w] / sm[valid_w])
                rms_w = np.sqrt(np.mean(ratio ** 2))
                tv_errors.append(rms_w)
                tv_detail.append(f'{t_start*1000:.0f}-{t_end*1000:.0f}ms: {rms_w:.1f}dB RMS')

    if tv_errors:
        tv_mean = np.mean(tv_errors)
        tv_score = max(0, 1 - tv_mean / 15.0)
    else:
        tv_mean = 99
        tv_score = 0

    results['time_varying_spectrum'] = {
        'value': f'mean {tv_mean:.1f} dB RMS',
        'score_01': tv_score,
        'weight': 0.10,
        'detail': tv_detail,
    }

    # ===============================================================
    # OVERALL SCORE
    # ===============================================================
    total_weight = sum(r['weight'] for r in results.values())
    overall = sum(r['score_01'] * r['weight'] for r in results.values()) / total_weight
    score_100 = overall * 100

    return {
        'score': score_100,
        'sub_scores': results,
    }


def print_scorecard(result):
    """Print formatted scorecard."""
    print(f"\n{'='*65}")
    print(f"  IR QUALITY SCORE: {result['score']:.1f} / 100")
    print(f"{'='*65}")

    for name, r in result['sub_scores'].items():
        bar_len = int(r['score_01'] * 20)
        bar = '#' * bar_len + '.' * (20 - bar_len)
        weight_pct = int(r['weight'] * 100)
        print(f"\n  {name:<25s} [{bar}] {r['score_01']*100:5.1f}%  (w={weight_pct}%)")
        print(f"    {r['value']}")
        if 'detail' in r:
            for d in r['detail']:
                print(f"      {d}")

    print(f"\n{'='*65}")
