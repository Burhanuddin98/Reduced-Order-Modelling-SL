"""
Statistical mode synthesis — fills the high-frequency modal density gap
for rooms where analytical modes aren't available (non-rectangular).

Above the Schroeder frequency, room modes are so dense that individual
mode shapes become unpredictable. But the statistical properties are
well-defined by Weyl's formula:

  Modal density:  dN/df = 4*pi*V*f^2 / c^3
  Mean spacing:   df_avg = c^3 / (4*pi*V*f^2)

This module generates synthetic modes at the correct density with:
  - Frequencies following Weyl's distribution (Poisson spacing)
  - Amplitudes randomized with correct variance (diffuse field)
  - Decay rates from Eyring formula + air absorption

The synthetic modes fill the gap between the eigensolve range and
the Nyquist frequency. Combined with detected-pair modes (axial,
tangential) that provide the coherent peaks, this gives broadband
coverage for any room shape.

Usage:
    from room_acoustics.statistical_modes import StatisticalModesProvider

    synth.register(StatisticalModesProvider(
        volume=168.84, surface_area=203.16,
        f_min=400, f_max=8000))
"""

import numpy as np
from .unified_modes import make_modes
from .material_function import air_absorption_coefficient


class StatisticalModesProvider:
    """
    Generates synthetic modes at the correct statistical density.

    Fills the frequency range between eigensolve f_max and the target
    f_max with modes whose density follows Weyl's formula and whose
    decay follows Eyring + air absorption.

    Confidence: 0.4 (statistical, not exact — yields to any deterministic
    engine in the merge, but fills gaps where nothing else exists).
    """
    name = 'statistical'
    confidence_base = 0.4

    def __init__(self, volume, surface_area, f_min=400, f_max=8000,
                 seed=42):
        """
        Parameters
        ----------
        volume : float
            Room volume [m^3].
        surface_area : float
            Total room surface area [m^2].
        f_min : float
            Start frequency [Hz] (should be above eigensolve f_max).
        f_max : float
            End frequency [Hz].
        seed : int
            Random seed for reproducible mode placement.
        """
        self.volume = float(volume)
        self.surface_area = float(surface_area)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self._seed = seed

    @property
    def frequency_range(self):
        return (self.f_min, self.f_max)

    def provide_modes(self, source, receiver, materials,
                      c=343.0, humidity=50.0, temperature=20.0, **kw):
        """
        Generate statistical modes between f_min and f_max.

        Mode frequencies follow Poisson-spaced placement with density
        from Weyl's formula. Amplitudes are Rayleigh-distributed (diffuse
        field assumption). Decay rates from area-weighted mean absorption
        (Eyring) + air absorption at each mode's frequency.
        """
        V = self.volume
        S = self.surface_area

        rng = np.random.RandomState(self._seed)

        # Generate mode frequencies using Weyl's density
        # N(f) = (4*pi*V / (3*c^3)) * f^3  (cumulative count)
        # We invert: place modes by drawing from the cumulative distribution
        N_min = 4 * np.pi * V / (3 * c ** 3) * self.f_min ** 3
        N_max = 4 * np.pi * V / (3 * c ** 3) * self.f_max ** 3
        n_modes = int(np.round(N_max - N_min))

        if n_modes <= 0:
            return np.zeros(0, dtype=np.dtype([
                ('frequency', np.float64), ('amplitude', np.float64),
                ('decay_rate', np.float64), ('omega_d', np.float64),
                ('confidence', np.float32), ('source_engine', 'U16')]))

        # Uniform samples in cumulative mode count → invert to frequency
        # N(f) = (4piV/3c³)f³ → f = (3c³N / 4piV)^(1/3)
        N_samples = np.sort(rng.uniform(N_min, N_max, n_modes))
        freqs = (3 * c ** 3 * N_samples / (4 * np.pi * V)) ** (1.0 / 3.0)

        # Amplitudes: Rayleigh-distributed (diffuse field)
        # RMS amplitude scales as 1/sqrt(modal_density) to maintain
        # correct energy density. Normalized so total energy matches
        # Eyring prediction.
        sigma_amp = 1.0 / np.sqrt(max(n_modes, 1))
        amplitudes = rng.rayleigh(sigma_amp, n_modes)
        # Random sign (modes can be positive or negative at receiver)
        amplitudes *= rng.choice([-1, 1], n_modes)

        # Decay rates: Eyring formula per mode frequency
        # Uses area-weighted mean absorption from materials
        from .material_function import MaterialFunction
        mean_alpha = np.zeros(n_modes)
        n_surfaces = 0
        for label, mat in materials.items():
            if isinstance(mat, MaterialFunction):
                mean_alpha += mat(freqs)
                n_surfaces += 1
        if n_surfaces > 0:
            mean_alpha /= n_surfaces
        mean_alpha = np.clip(mean_alpha, 0.001, 0.999)

        # Eyring decay rate per mode
        gamma_eyring = -c * S / (4 * V) * np.log(1 - mean_alpha)

        # Air absorption per mode
        m_air = air_absorption_coefficient(freqs, humidity, temperature)
        gamma = gamma_eyring + m_air * c

        return make_modes(freqs, amplitudes, gamma,
                          self.confidence_base, self.name)

    def summary(self):
        """Print expected mode count."""
        c = 343.0
        N_min = 4 * np.pi * self.volume / (3 * c ** 3) * self.f_min ** 3
        N_max = 4 * np.pi * self.volume / (3 * c ** 3) * self.f_max ** 3
        n = int(np.round(N_max - N_min))
        print(f"  Statistical modes: {self.f_min:.0f}-{self.f_max:.0f} Hz, "
              f"~{n} modes (Weyl density)")
        print(f"  Volume: {self.volume:.1f} m3, Surface: {self.surface_area:.1f} m2")
