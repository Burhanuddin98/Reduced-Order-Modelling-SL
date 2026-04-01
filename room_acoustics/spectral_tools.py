"""
Audio and spectral analysis tools for IR comparison and validation.

Provides WAV loading, spectrogram computation, spectral peak extraction,
and side-by-side comparison of simulated vs measured impulse responses.

Usage:
    from room_acoustics.spectral_tools import (
        load_wav, compute_spectrogram, compare_irs, extract_peaks)

    # Load and compare
    ir_meas = load_wav("measured.wav")
    ir_sim = load_wav("simulated.wav")
    compare_irs(ir_meas, ir_sim, sr=44100, output="comparison.png")
"""

import numpy as np


def load_wav(path, normalize=True):
    """
    Load a WAV file as float64 array.

    Parameters
    ----------
    path : str
        Path to WAV file.
    normalize : bool
        Normalize to [-1, 1] range.

    Returns
    -------
    data : ndarray (n_samples,), float64
    sr : int
        Sample rate.
    """
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data[:, 0]  # mono
    if normalize and np.max(np.abs(data)) > 0:
        data /= np.max(np.abs(data))
    return data, sr


def compute_spectrogram(ir, sr, n_fft=2048, hop=512, f_max=8000):
    """
    Compute magnitude spectrogram of an impulse response.

    Parameters
    ----------
    ir : ndarray (n_samples,)
    sr : int
        Sample rate.
    n_fft : int
        FFT window size.
    hop : int
        Hop size between frames.
    f_max : float
        Maximum frequency to include [Hz].

    Returns
    -------
    S_dB : ndarray (n_freq, n_frames), float
        Magnitude spectrogram in dB (relative to peak).
    freqs : ndarray (n_freq,)
        Frequency axis [Hz].
    times : ndarray (n_frames,)
        Time axis [s].
    """
    n_samples = len(ir)
    n_frames = max(1, (n_samples - n_fft) // hop + 1)

    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    f_mask = freqs <= f_max
    freqs = freqs[f_mask]

    S = np.zeros((len(freqs), n_frames))
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        if end > n_samples:
            break
        frame = ir[start:end] * window
        spectrum = np.abs(np.fft.rfft(frame))
        S[:, i] = spectrum[f_mask]
        times[i] = (start + n_fft // 2) / sr

    # Convert to dB
    S_max = np.max(S)
    if S_max > 0:
        S_dB = 20 * np.log10(np.maximum(S / S_max, 1e-10))
    else:
        S_dB = np.full_like(S, -100.0)

    return S_dB, freqs, times


def extract_peaks(ir, sr, f_min=20, f_max=8000, n_peaks=10, min_prominence_dB=6):
    """
    Extract dominant spectral peaks from an impulse response.

    Parameters
    ----------
    ir : ndarray
    sr : int
    f_min, f_max : float
        Frequency range [Hz].
    n_peaks : int
        Maximum number of peaks to return.
    min_prominence_dB : float
        Minimum prominence above local median [dB].

    Returns
    -------
    peaks : list of (frequency_Hz, magnitude_dB)
        Sorted by magnitude (strongest first).
    """
    N = len(ir)
    spectrum = np.abs(np.fft.rfft(ir))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)

    # Frequency range mask
    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[mask]
    spectrum = spectrum[mask]

    if len(spectrum) < 5:
        return []

    # Convert to dB
    spec_max = np.max(spectrum)
    if spec_max <= 0:
        return []
    spec_dB = 20 * np.log10(np.maximum(spectrum / spec_max, 1e-10))

    # Local median for prominence calculation
    from scipy.ndimage import median_filter
    median = median_filter(spec_dB, size=max(5, len(spec_dB) // 50))

    # Find peaks: local maxima above prominence threshold
    peaks = []
    for i in range(2, len(spec_dB) - 2):
        if (spec_dB[i] > spec_dB[i - 1] and spec_dB[i] > spec_dB[i + 1]
                and spec_dB[i] > spec_dB[i - 2] and spec_dB[i] > spec_dB[i + 2]):
            prominence = spec_dB[i] - median[i]
            if prominence >= min_prominence_dB:
                # Quadratic interpolation for sub-bin accuracy
                y0, y1, y2 = spec_dB[i - 1], spec_dB[i], spec_dB[i + 1]
                d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if abs(y0 - 2 * y1 + y2) > 1e-10 else 0
                f_interp = freqs[i] + d * (freqs[1] - freqs[0])
                peaks.append((f_interp, spec_dB[i]))

    # Sort by magnitude (strongest first), take top n_peaks
    peaks.sort(key=lambda p: -p[1])
    return peaks[:n_peaks]


def compare_irs(ir_meas, ir_sim, sr, output=None, title=None):
    """
    Compare measured and simulated impulse responses.

    Produces a 4-panel plot: waveforms, Schroeder decay, spectra,
    and octave-band T30 comparison.

    Parameters
    ----------
    ir_meas, ir_sim : ndarray
        Measured and simulated IRs (same sample rate).
    sr : int
        Sample rate.
    output : str or None
        Path to save PNG. If None, displays interactively.
    title : str or None
        Plot title.
    """
    import matplotlib
    if output:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import butter, sosfiltfilt

    try:
        from .acoustics_metrics import compute_t30, schroeder_decay
    except ImportError:
        from room_acoustics.acoustics_metrics import compute_t30, schroeder_decay

    dt = 1.0 / sr
    n = min(len(ir_meas), len(ir_sim))
    ir_meas = ir_meas[:n]
    ir_sim = ir_sim[:n]
    t = np.arange(n) / sr

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if title:
        fig.suptitle(title, fontweight='bold')

    # 1. Waveforms
    axes[0, 0].plot(t * 1000, ir_meas, 'b-', lw=0.5, alpha=0.7, label='Measured')
    axes[0, 0].plot(t * 1000, ir_sim, 'r-', lw=0.5, alpha=0.7, label='Simulated')
    axes[0, 0].set_xlabel('Time [ms]')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Impulse Response')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Schroeder decay
    t_m, d_m = schroeder_decay(ir_meas, dt)
    t_s, d_s = schroeder_decay(ir_sim, dt)
    axes[0, 1].plot(t_m * 1000, d_m, 'b-', lw=1, label='Measured')
    axes[0, 1].plot(t_s * 1000, d_s, 'r-', lw=1, label='Simulated')
    axes[0, 1].axhline(-60, color='gray', ls='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time [ms]')
    axes[0, 1].set_ylabel('Energy [dB]')
    axes[0, 1].set_title('Schroeder Decay')
    axes[0, 1].set_ylim(-80, 5)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Magnitude spectra
    spec_m = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(ir_meas)), 1e-10))
    spec_s = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(ir_sim)), 1e-10))
    freqs = np.fft.rfftfreq(n, dt)
    f_mask = freqs <= 8000
    axes[1, 0].plot(freqs[f_mask], spec_m[f_mask], 'b-', lw=0.5, alpha=0.7, label='Measured')
    axes[1, 0].plot(freqs[f_mask], spec_s[f_mask], 'r-', lw=0.5, alpha=0.7, label='Simulated')
    axes[1, 0].set_xlabel('Frequency [Hz]')
    axes[1, 0].set_ylabel('Magnitude [dB]')
    axes[1, 0].set_title('Spectrum')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Octave-band T30 comparison
    bands = [250, 500, 1000, 2000, 4000]
    nyq = sr / 2
    t30_meas = []
    t30_sim = []
    for fc in bands:
        fl = fc / np.sqrt(2)
        fh = min(fc * np.sqrt(2), nyq * 0.95)
        sos = butter(4, [fl / nyq, fh / nyq], btype='band', output='sos')

        band_m = sosfiltfilt(sos, ir_meas)
        band_s = sosfiltfilt(sos, ir_sim)

        t30_m, r2_m = compute_t30(band_m, dt)
        t30_s, r2_s = compute_t30(band_s, dt)
        t30_meas.append(t30_m if r2_m > 0.7 else 0)
        t30_sim.append(t30_s if r2_s > 0.7 else 0)

    x = np.arange(len(bands))
    w = 0.35
    axes[1, 1].bar(x - w / 2, t30_meas, w, label='Measured', color='#2196F3')
    axes[1, 1].bar(x + w / 2, t30_sim, w, label='Simulated', color='#F44336')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(fc) for fc in bands])
    axes[1, 1].set_xlabel('Octave Band [Hz]')
    axes[1, 1].set_ylabel('T30 [s]')
    axes[1, 1].set_title('Octave-Band T30')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output}")
    else:
        plt.show()
