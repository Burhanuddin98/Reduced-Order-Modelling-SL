"""Octave-band ISO 3382 metrics from impulse response."""

import numpy as np
from scipy.signal import butter, sosfilt


def octave_band_filter(signal, fs, center_freq):
    """Apply octave-band bandpass filter centered at center_freq."""
    f_low = center_freq / np.sqrt(2)
    f_high = center_freq * np.sqrt(2)
    nyq = fs / 2
    # Clamp to valid range
    f_low = max(f_low, 20)
    f_high = min(f_high, nyq * 0.95)
    if f_low >= f_high:
        return np.zeros_like(signal)
    sos = butter(4, [f_low / nyq, f_high / nyq], btype='band', output='sos')
    return sosfilt(sos, signal)


def _edc(signal):
    """Schroeder energy decay curve."""
    s2 = signal ** 2
    return np.cumsum(s2[::-1])[::-1]


def _decay_time(edc_db, t, start_db, end_db):
    """Find reverberation time from EDC between two dB levels."""
    try:
        i_start = np.where(edc_db <= start_db)[0][0]
        i_end = np.where(edc_db <= end_db)[0][0]
        if i_end <= i_start:
            return float('nan')
        slope = (edc_db[i_end] - edc_db[i_start]) / (t[i_end] - t[i_start])
        if slope >= 0:
            return float('nan')
        return -60.0 / slope
    except (IndexError, ZeroDivisionError):
        return float('nan')


def band_metrics(signal, fs):
    """Compute ISO 3382 metrics for a single (filtered) IR signal.

    Returns dict with T30, T20, EDT, C80, D50, TS.
    """
    t = np.arange(len(signal)) / fs
    edc = _edc(signal)
    edc_max = max(edc[0], 1e-30)
    edc_db = 10 * np.log10(edc / edc_max + 1e-30)

    T30 = _decay_time(edc_db, t, -5, -35)
    T20 = _decay_time(edc_db, t, -5, -25)
    EDT = _decay_time(edc_db, t, 0, -10)

    s2 = signal ** 2
    total_energy = max(np.sum(s2), 1e-30)

    # C80
    n80 = min(int(0.08 * fs), len(s2))
    early = np.sum(s2[:n80])
    late = max(np.sum(s2[n80:]), 1e-30)
    C80 = 10 * np.log10(early / late)

    # D50
    n50 = min(int(0.05 * fs), len(s2))
    D50 = np.sum(s2[:n50]) / total_energy

    # TS (centre time)
    TS = np.sum(t * s2) / total_energy

    return dict(T30=T30, T20=T20, EDT=EDT, C80=C80, D50=D50, TS=TS)


def octave_band_metrics(signal, fs, center_freqs=None):
    """Compute ISO 3382 metrics per octave band.

    Parameters
    ----------
    signal : ndarray — impulse response
    fs : int — sample rate
    center_freqs : list — octave band centers [Hz].
        Default: [125, 250, 500, 1000, 2000]

    Returns
    -------
    dict with keys T30, T20, EDT, C80, D50, TS — each a list
    of values per octave band.
    """
    if center_freqs is None:
        center_freqs = [125, 250, 500, 1000, 2000]

    result = {k: [] for k in ['T30', 'T20', 'EDT', 'C80', 'D50', 'TS']}

    for fc in center_freqs:
        filtered = octave_band_filter(signal, fs, fc)
        m = band_metrics(filtered, fs)
        for k in result:
            result[k].append(round(m[k], 4) if np.isfinite(m[k]) else None)

    return result
