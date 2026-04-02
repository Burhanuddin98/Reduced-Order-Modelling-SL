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


def score_ir_perceptual(ir_sim, ir_meas, sr=44100):
    """
    Perceptually-weighted IR comparison using auditory models.

    Uses Bark-scale frequency bands, MFCC distance, log-spectral distance,
    Energy Decay Relief, and modulation transfer — all weighted by human
    auditory sensitivity.

    Parameters
    ----------
    ir_sim, ir_meas : ndarray
    sr : int

    Returns
    -------
    result : dict with 'score', 'sub_scores'
    """
    n = min(len(ir_sim), len(ir_meas))
    sim = ir_sim[:n].astype(np.float64)
    meas = ir_meas[:n].astype(np.float64)
    dt = 1.0 / sr
    nyq = sr / 2

    results = {}

    # ===============================================================
    # 1. BARK-SCALE SPECTRAL DISTANCE
    # 24 critical bands matching the ear's frequency resolution
    # ===============================================================
    bark_edges = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                  1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                  4400, 5300, 6400, 7700, 9500, 12000, 15500]

    bark_errors = []
    bark_detail = []
    for i in range(len(bark_edges) - 1):
        fl = bark_edges[i]
        fh = bark_edges[i + 1]
        if fl >= nyq * 0.95:
            break
        fh = min(fh, nyq * 0.95)
        try:
            sos = butter(3, [fl / nyq, fh / nyq], btype='band', output='sos')
            e_s = np.sum(sosfiltfilt(sos, sim) ** 2)
            e_m = np.sum(sosfiltfilt(sos, meas) ** 2)
            if e_s > 1e-20 and e_m > 1e-20:
                err_dB = 10 * np.log10(e_s / e_m)
                bark_errors.append(err_dB)
                fc = (fl + fh) / 2
                bark_detail.append(f'{fl}-{fh}Hz: {err_dB:+.1f}dB')
        except ValueError:
            pass

    if bark_errors:
        bark_rms = np.sqrt(np.mean(np.array(bark_errors) ** 2))
        bark_score = max(0, 1 - bark_rms / 8.0)  # 0dB=1.0, 8dB=0.0
    else:
        bark_rms = 99
        bark_score = 0

    results['bark_spectral'] = {
        'value': f'RMS {bark_rms:.1f} dB across {len(bark_errors)} Bark bands',
        'score_01': bark_score,
        'weight': 0.20,  # room-invariant: heavy weight
        'detail': [d for d in bark_detail if abs(float(d.split(':')[1].replace('dB', ''))) > 3],
    }

    # ===============================================================
    # 2. MFCC DISTANCE (timbral similarity)
    # Mel-frequency cepstral coefficients — standard for timbre
    # ===============================================================
    def compute_mfcc(ir, sr, n_mfcc=13, n_fft=4096, hop=2048, n_mels=40):
        """Compute MFCCs from IR."""
        n_frames = max(1, (len(ir) - n_fft) // hop)
        # Mel filterbank
        mel_lo = 2595 * np.log10(1 + 20 / 700)
        mel_hi = 2595 * np.log10(1 + (sr / 2) / 700)
        mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(n_mels):
            for k in range(bins[m], bins[m + 1]):
                if k < fbank.shape[1]:
                    fbank[m, k] = (k - bins[m]) / max(bins[m + 1] - bins[m], 1)
            for k in range(bins[m + 1], bins[m + 2]):
                if k < fbank.shape[1]:
                    fbank[m, k] = (bins[m + 2] - k) / max(bins[m + 2] - bins[m + 1], 1)

        mfccs = np.zeros((n_frames, n_mfcc))
        window = np.hanning(n_fft)
        for i in range(n_frames):
            frame = ir[i * hop:i * hop + n_fft]
            if len(frame) < n_fft:
                break
            spec = np.abs(np.fft.rfft(frame * window)) ** 2
            mel_spec = np.dot(fbank, spec)
            mel_spec = np.maximum(mel_spec, 1e-10)
            log_mel = np.log(mel_spec)
            # DCT
            for j in range(n_mfcc):
                mfccs[i, j] = np.sum(log_mel * np.cos(
                    np.pi * j * (np.arange(n_mels) + 0.5) / n_mels))

        return mfccs

    mfcc_sim = compute_mfcc(sim, sr)
    mfcc_meas = compute_mfcc(meas, sr)
    n_f = min(mfcc_sim.shape[0], mfcc_meas.shape[0])
    if n_f > 0:
        mfcc_dist = np.sqrt(np.mean((mfcc_sim[:n_f] - mfcc_meas[:n_f]) ** 2))
        mfcc_score = max(0, 1 - mfcc_dist / 30.0)  # 0=1.0, 30=0.0 (lenient — position-dependent)
    else:
        mfcc_dist = 99
        mfcc_score = 0

    results['mfcc_distance'] = {
        'value': f'{mfcc_dist:.2f} (lower = more similar timbre)',
        'score_01': mfcc_score,
        'weight': 0.05,  # position-dependent: low weight
    }

    # ===============================================================
    # 3. LOG-SPECTRAL DISTANCE (standard for IR comparison)
    # ===============================================================
    spec_s = np.abs(np.fft.rfft(sim)) ** 2
    spec_m = np.abs(np.fft.rfft(meas)) ** 2
    freqs = np.fft.rfftfreq(n, dt)

    # Weight by A-weighting (perceptual loudness)
    def a_weight(f):
        """A-weighting curve (simplified)."""
        f = np.maximum(f, 1.0)
        ra = (12194 ** 2 * f ** 4) / (
            (f ** 2 + 20.6 ** 2) * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2)) *
            (f ** 2 + 12194 ** 2))
        return ra / ra.max()

    fm = (freqs >= 20) & (freqs <= 16000)
    if fm.any():
        weights_a = a_weight(freqs[fm])
        valid_lsd = (spec_s[fm] > 1e-20) & (spec_m[fm] > 1e-20)
        if valid_lsd.any():
            lsd_raw = (10 * np.log10(spec_s[fm][valid_lsd]) -
                       10 * np.log10(spec_m[fm][valid_lsd]))
            # A-weighted LSD
            w_valid = weights_a[valid_lsd]
            lsd_weighted = np.sqrt(np.sum(w_valid * lsd_raw ** 2) / np.sum(w_valid))
            lsd_score = max(0, 1 - lsd_weighted / 12.0)
        else:
            lsd_weighted = 99
            lsd_score = 0
    else:
        lsd_weighted = 99
        lsd_score = 0

    results['log_spectral_distance'] = {
        'value': f'{lsd_weighted:.1f} dB (A-weighted)',
        'score_01': lsd_score,
        'weight': 0.10,
    }

    # ===============================================================
    # 4. ENERGY DECAY RELIEF (time-frequency decay matching)
    # Compares Schroeder backward integral per Bark band
    # ===============================================================
    from .acoustics_metrics import schroeder_decay
    edr_errors = []
    edr_bands = [125, 250, 500, 1000, 2000, 4000]

    for fc in edr_bands:
        fl = fc / np.sqrt(2)
        fh = min(fc * np.sqrt(2), nyq * 0.95)
        try:
            sos = butter(3, [fl / nyq, fh / nyq], btype='band', output='sos')
            bs = sosfiltfilt(sos, sim)
            bm = sosfiltfilt(sos, meas)
            t_s, d_s = schroeder_decay(bs, dt)
            t_m, d_m = schroeder_decay(bm, dt)
            # Compare at 50 points
            t_max = min(t_s[-1], t_m[-1], 2.0)
            tc = np.linspace(0, t_max, 50)
            ds_i = np.interp(tc, t_s, d_s)
            dm_i = np.interp(tc, t_m, d_m)
            edr_rms = np.sqrt(np.mean((ds_i - dm_i) ** 2))
            edr_errors.append(edr_rms)
        except (ValueError, IndexError):
            pass

    if edr_errors:
        edr_mean = np.mean(edr_errors)
        edr_score = max(0, 1 - edr_mean / 10.0)
    else:
        edr_mean = 99
        edr_score = 0

    results['energy_decay_relief'] = {
        'value': f'mean {edr_mean:.1f} dB per-band Schroeder diff',
        'score_01': edr_score,
        'weight': 0.20,  # room-invariant: heavy weight
    }

    # ===============================================================
    # 5. MODULATION TRANSFER (temporal fine structure preservation)
    # How well are amplitude modulations preserved at each frequency?
    # ===============================================================
    mod_freqs = [0.5, 1, 2, 4, 8, 16]  # Hz — modulation rates
    mtf_values = []
    carrier_bands = [500, 1000, 2000, 4000]

    for fc in carrier_bands:
        fl = fc / np.sqrt(2)
        fh = min(fc * np.sqrt(2), nyq * 0.95)
        try:
            sos = butter(3, [fl / nyq, fh / nyq], btype='band', output='sos')
            env_s = np.abs(hilbert(sosfiltfilt(sos, sim)))
            env_m = np.abs(hilbert(sosfiltfilt(sos, meas)))
            # Correlation of envelopes
            corr = np.corrcoef(env_s, env_m)[0, 1]
            if not np.isnan(corr):
                mtf_values.append(max(0, corr))
        except (ValueError, IndexError):
            pass

    if mtf_values:
        mtf_mean = np.mean(mtf_values)
        mtf_score = max(0, mtf_mean)
    else:
        mtf_mean = 0
        mtf_score = 0

    results['modulation_transfer'] = {
        'value': f'mean envelope correlation {mtf_mean:.4f}',
        'score_01': mtf_score,
        'weight': 0.10,
    }

    # ===============================================================
    # 6. EARLY REFLECTION ACCURACY (timing + relative levels)
    # ===============================================================
    from scipy.signal import find_peaks
    n80 = int(0.08 * sr)
    early_m = np.abs(meas[:n80])
    early_s = np.abs(sim[:n80])

    peaks_m, props_m = find_peaks(early_m, height=0.03 * np.max(early_m),
                                   distance=int(0.001 * sr))
    peaks_s, props_s = find_peaks(early_s, height=0.03 * np.max(early_s),
                                   distance=int(0.001 * sr))

    timing_matched = 0
    level_errors = []
    for pm in peaks_m:
        if len(peaks_s) > 0:
            dists = np.abs(peaks_s - pm)
            best = np.argmin(dists)
            if dists[best] < int(0.002 * sr):
                timing_matched += 1
                # Compare relative levels
                lev_m = early_m[pm] / max(early_m.max(), 1e-10)
                lev_s = early_s[peaks_s[best]] / max(early_s.max(), 1e-10)
                if lev_m > 1e-10 and lev_s > 1e-10:
                    level_errors.append(abs(20 * np.log10(lev_s / lev_m)))

    if len(peaks_m) > 0:
        timing_score = timing_matched / len(peaks_m)
    else:
        timing_score = 0

    if level_errors:
        level_rms = np.sqrt(np.mean(np.array(level_errors) ** 2))
        level_score = max(0, 1 - level_rms / 10.0)
    else:
        level_score = timing_score

    early_score = 0.6 * timing_score + 0.4 * level_score

    results['early_reflections'] = {
        'value': f'timing: {timing_matched}/{len(peaks_m)}, '
                 f'level RMS: {level_rms:.1f}dB' if level_errors else
                 f'timing: {timing_matched}/{len(peaks_m)}',
        'score_01': early_score,
        'weight': 0.10,
    }

    # ===============================================================
    # 7. ISO 3382 METRICS (T30, EDT, C80, D50, TS)
    # ===============================================================
    from .acoustics_metrics import all_metrics
    m_sim = all_metrics(sim, dt)
    m_meas = all_metrics(meas, dt)

    iso_errors = []
    iso_detail = []
    for key, unit, tol in [('T30_s', 's', 0.2), ('EDT_s', 's', 0.3),
                            ('C80_dB', 'dB', 3.0), ('D50', '', 0.15),
                            ('TS_ms', 'ms', 30)]:
        v_s = m_sim[key]
        v_m = m_meas[key]
        if v_m != 0 and not np.isnan(v_s) and not np.isnan(v_m):
            err = abs(v_s - v_m) / max(abs(v_m), 1e-10)
            iso_errors.append(min(err / tol, 1.0))
            iso_detail.append(f'{key}: {v_s:.3f} vs {v_m:.3f}{unit}')

    if iso_errors:
        iso_score = max(0, 1 - np.mean(iso_errors))
    else:
        iso_score = 0

    results['iso3382_metrics'] = {
        'value': f'{len(iso_errors)} metrics compared',
        'score_01': iso_score,
        'weight': 0.15,  # room-invariant: important
    }

    # ===============================================================
    # 8. PERCEPTUAL LOUDNESS MATCH (A-weighted RMS over time)
    # ===============================================================
    # Compare A-weighted energy in 50ms windows over time
    win = int(0.05 * sr)
    n_wins = min(len(sim), len(meas)) // win
    loud_errors = []
    for i in range(n_wins):
        s_win = sim[i * win:(i + 1) * win]
        m_win = meas[i * win:(i + 1) * win]
        rms_s = np.sqrt(np.mean(s_win ** 2))
        rms_m = np.sqrt(np.mean(m_win ** 2))
        if rms_m > 1e-10 and rms_s > 1e-10:
            loud_errors.append(20 * np.log10(rms_s / rms_m))

    if loud_errors:
        loud_rms = np.sqrt(np.mean(np.array(loud_errors) ** 2))
        loud_score = max(0, 1 - loud_rms / 20.0)  # lenient — position-dependent
    else:
        loud_rms = 99
        loud_score = 0

    results['loudness_envelope'] = {
        'value': f'RMS {loud_rms:.1f} dB over {n_wins} windows',
        'score_01': loud_score,
        'weight': 0.05,  # position-dependent: low weight
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
            for d in r['detail'][:5]:
                print(f"      {d}")

    print(f"\n{'='*65}")
