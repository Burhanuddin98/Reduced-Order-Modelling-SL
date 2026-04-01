"""
Diffuse reverberant tail — fills spectral gaps between modal resonances.

Real rooms have continuous spectral energy from thousands of overlapping
modes plus surface scattering. The axial/analytical modes capture the
coherent resonant peaks, but the gaps between them need diffuse energy.

This module synthesizes a noise-based reverberant tail with:
  - Per-octave-band exponential decay (from Eyring + air absorption)
  - Frequency-dependent absorption from MaterialFunction
  - Proper energy balance relative to the modal component
  - Smooth onset (avoids double-counting ISM early reflections)

The result sounds like natural late reverberation — not tonal resonances,
not white noise, but properly shaped decaying energy in each frequency band.

Usage:
    from room_acoustics.diffuse_tail import synthesize_diffuse_tail

    ir_diffuse = synthesize_diffuse_tail(
        volume=168.84, surface_area=203.16,
        materials=mats, T=3.0, sr=44100)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from .material_function import MaterialFunction, air_absorption_coefficient


def synthesize_diffuse_tail(volume, surface_area, materials,
                            T=3.0, sr=44100,
                            humidity=50.0, temperature=20.0,
                            onset_delay_ms=10, onset_fade_ms=30,
                            seed=42):
    """
    Synthesize diffuse reverberant tail shaped by room acoustics.

    Generates band-filtered noise with per-band exponential decay
    matching the Eyring RT60 at each frequency. The result fills
    the spectral gaps between coherent modal resonances.

    Parameters
    ----------
    volume : float
        Room volume [m^3].
    surface_area : float
        Total surface area [m^2].
    materials : dict of label -> MaterialFunction
        Per-surface absorption functions.
    T : float
        Duration [seconds].
    sr : int
        Sample rate [Hz].
    humidity, temperature : float
        For air absorption.
    onset_delay_ms : float
        Delay before diffuse tail starts [ms]. Should be after
        the direct sound and first reflections.
    onset_fade_ms : float
        Fade-in duration [ms] for smooth onset.
    seed : int
        Random seed for reproducible noise.

    Returns
    -------
    ir_diffuse : ndarray (n_samples,)
        Diffuse reverberant tail.
    """
    c = 343.0
    V = volume
    S = surface_area
    n_samples = int(T * sr)
    nyq = sr / 2
    rng = np.random.RandomState(seed)

    # Time vector
    t = np.arange(n_samples, dtype=np.float64) / sr

    # Onset window: silence → fade-in → full
    n_delay = int(onset_delay_ms * sr / 1000)
    n_fade = int(onset_fade_ms * sr / 1000)
    onset = np.zeros(n_samples)
    if n_delay + n_fade < n_samples:
        onset[n_delay:n_delay + n_fade] = np.linspace(0, 1, n_fade)
        onset[n_delay + n_fade:] = 1.0
    else:
        onset[n_delay:] = 1.0

    # Mean absorption from all materials at representative frequencies
    def mean_alpha(f):
        alphas = []
        for label, mat in materials.items():
            if isinstance(mat, MaterialFunction):
                alphas.append(mat(f))
            else:
                alphas.append(0.05)
        return np.mean(alphas) if alphas else 0.05

    # Generate base noise (white, full bandwidth)
    noise = rng.randn(n_samples)

    # Process per octave band: filter noise, apply band-specific decay
    ir_diffuse = np.zeros(n_samples, dtype=np.float64)

    # Octave bands from 62.5 Hz to 16 kHz
    band_centers = [62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    for fc in band_centers:
        fl = fc / np.sqrt(2)
        fh = fc * np.sqrt(2)

        # Skip bands outside Nyquist
        if fl >= nyq * 0.95:
            continue
        fh = min(fh, nyq * 0.95)

        # Band-pass filter the noise
        try:
            sos = butter(3, [fl / nyq, fh / nyq], btype='band', output='sos')
            band_noise = sosfiltfilt(sos, noise)
        except ValueError:
            continue

        # Compute decay rate for this band
        alpha_band = np.clip(mean_alpha(fc), 0.001, 0.999)

        # Eyring decay
        gamma_eyring = -c * S / (4 * V) * np.log(1 - alpha_band)

        # Air absorption at band center
        m_air = float(air_absorption_coefficient(
            np.array([fc]), humidity, temperature)[0])
        gamma = gamma_eyring + m_air * c

        # Exponential decay envelope
        envelope = np.exp(-gamma * t)

        # Apply envelope and onset window
        ir_diffuse += band_noise * envelope * onset

    return ir_diffuse
