"""
Frequency-dependent material absorption functions.

Materials are defined as absorption functions α(f) that can be
evaluated at any frequency. Input can be:
  - A scalar (frequency-independent)
  - Octave-band values (6 points)
  - Third-octave-band values (31 points)
  - Arbitrary (frequency, alpha) pairs
  - A callable function

Internally stored as interpolation tables, evaluated at any resolution.
This is the correct representation — octave bands are a measurement
convention, not a physical limitation.

Usage:
    # From scalar
    mat = MaterialFunction.from_scalar(0.05)

    # From octave-band data
    mat = MaterialFunction.from_bands(
        {125: 0.10, 250: 0.15, 500: 0.20, 1000: 0.30, 2000: 0.40, 4000: 0.50})

    # From BRAS CSV
    mat = MaterialFunction.from_csv('mat_CR2_ceiling.csv')

    # Evaluate at any frequency
    alpha_at_440 = mat(440.0)
    alpha_array = mat(np.array([100, 200, 500, 1000, 4000]))

    # Convert to impedance
    Z_at_440 = mat.impedance(440.0)
"""

import numpy as np


class MaterialFunction:
    """
    Frequency-dependent absorption coefficient α(f).

    Stores (frequency, alpha) pairs and interpolates log-linearly
    for evaluation at any frequency. Extrapolates flat beyond the
    defined range.
    """

    def __init__(self, freqs, alphas, name='unknown', scattering=None):
        """
        Parameters
        ----------
        freqs : array-like
            Frequency points [Hz], must be sorted ascending.
        alphas : array-like
            Absorption coefficients at each frequency, in [0, 1].
        name : str
            Human-readable material name.
        scattering : array-like or None
            Scattering coefficients at each frequency. If None, defaults to 0.05.
        """
        self.freqs = np.asarray(freqs, dtype=float)
        self.alphas = np.clip(np.asarray(alphas, dtype=float), 0.001, 0.999)
        self.name = name

        if scattering is not None:
            self.scattering = np.clip(np.asarray(scattering, dtype=float), 0.0, 1.0)
        else:
            self.scattering = np.full_like(self.freqs, 0.05)

        # Use log-frequency for interpolation (more physical spacing)
        self._log_freqs = np.log10(np.maximum(self.freqs, 1.0))

    def __call__(self, f):
        """Evaluate α(f) at arbitrary frequency or frequency array."""
        f = np.asarray(f, dtype=float)
        scalar = f.ndim == 0
        f = np.atleast_1d(f)
        log_f = np.log10(np.maximum(f, 1.0))
        result = np.interp(log_f, self._log_freqs, self.alphas)
        return float(result[0]) if scalar else result

    def scatter(self, f):
        """Evaluate scattering coefficient at arbitrary frequency."""
        f = np.asarray(f, dtype=float)
        scalar = f.ndim == 0
        f = np.atleast_1d(f)
        log_f = np.log10(np.maximum(f, 1.0))
        result = np.interp(log_f, self._log_freqs, self.scattering)
        return float(result[0]) if scalar else result

    def impedance(self, f, rho_c=411.6):
        """Convert α(f) to normal-incidence impedance Z(f)."""
        alpha = self(f)
        alpha = np.clip(alpha, 0.001, 0.999)
        R = np.sqrt(1.0 - alpha)
        return rho_c * (1 + R) / (1 - R)

    @classmethod
    def from_scalar(cls, alpha, name='constant'):
        """Frequency-independent material."""
        return cls([20, 20000], [alpha, alpha], name=name)

    @classmethod
    def from_bands(cls, band_dict, name='band_data'):
        """
        From dict of {frequency_hz: alpha}.
        Works with octave bands, third-octave bands, or any spacing.
        """
        freqs = sorted(band_dict.keys())
        alphas = [band_dict[f] for f in freqs]
        return cls(freqs, alphas, name=name)

    @classmethod
    def from_csv(cls, path, name=None):
        """
        Load from BRAS-format CSV (3 lines: frequencies, alphas, scattering).
        """
        import os
        lines = open(path).readlines()
        freqs = [float(x) for x in lines[0].strip().split(',')]
        alphas = [float(x) for x in lines[1].strip().split(',')]
        scattering = None
        if len(lines) > 2:
            scattering = [float(x) for x in lines[2].strip().split(',')]
        if name is None:
            name = os.path.basename(path).replace('.csv', '')
        return cls(freqs, alphas, name=name, scattering=scattering)

    @classmethod
    def from_impedance_scalar(cls, Z, name='constant_Z'):
        """From frequency-independent impedance."""
        rho_c = 411.6
        R = (Z - rho_c) / (Z + rho_c)
        alpha = 1.0 - R ** 2
        return cls.from_scalar(alpha, name=name)

    def with_structural_absorption(self, surface_density=10.0, cavity_depth=0.08,
                                    damping=0.05, f_blend=200.0):
        """
        Add structural/membrane absorption below f_blend.

        Real walls vibrate — plasterboard on studs has a membrane resonance
        where absorption spikes. This blends a mass-spring-damper model
        with the existing alpha(f) data below f_blend Hz.

        Parameters
        ----------
        surface_density : float
            Wall surface density [kg/m^2]. Plasterboard: 8-12, concrete: 200+.
        cavity_depth : float
            Air cavity behind the surface [m]. Stud wall: 0.05-0.1m.
        damping : float
            Structural damping factor (0-1). Typical: 0.03-0.1.
        f_blend : float
            Frequency below which the structural model is blended in [Hz].

        Returns
        -------
        MaterialFunction with structural absorption added below f_blend.
        """
        rho = 1.2  # air density
        c = 343.0
        rho_c = rho * c

        # Mass-spring resonance: f_res = c/(2*pi) * sqrt(rho/(m_s * d))
        m_s = surface_density
        d = max(cavity_depth, 0.001)
        K = rho * c ** 2 / d  # air spring stiffness
        f_res = np.sqrt(K / m_s) / (2 * np.pi)

        # Structural impedance: Z_wall(f) for mass-spring-damper
        # At resonance, Z drops → absorption peaks
        new_freqs = np.unique(np.concatenate([
            self.freqs,
            np.linspace(20, f_blend, 50)
        ]))
        new_freqs.sort()

        new_alphas = self(new_freqs).copy()

        for i, f in enumerate(new_freqs):
            if f > f_blend:
                break
            omega = 2 * np.pi * max(f, 1.0)
            omega_res = 2 * np.pi * f_res

            # Impedance magnitude of mass-spring-damper
            Z_mass = omega * m_s  # mass term
            Z_spring = K / omega  # spring term
            Z_damp = damping * 2 * m_s * omega_res  # damping term
            Z_wall = np.sqrt((Z_mass - Z_spring) ** 2 + Z_damp ** 2)

            # Absorption from wall impedance vs air impedance
            R = (Z_wall - rho_c) / (Z_wall + rho_c)
            alpha_struct = 1.0 - R ** 2

            # Blend: smooth transition from structural to measured
            blend = 1.0 - np.clip((f - f_blend * 0.5) / (f_blend * 0.5), 0, 1)
            new_alphas[i] = blend * max(alpha_struct, new_alphas[i]) + (1 - blend) * new_alphas[i]

        new_alphas = np.clip(new_alphas, 0.001, 0.999)
        return MaterialFunction(new_freqs, new_alphas,
                                name=f"{self.name}_structural",
                                scattering=self.scatter(new_freqs))

    def __repr__(self):
        return (f"MaterialFunction('{self.name}', {len(self.freqs)} points, "
                f"alpha=[{self.alphas.min():.3f}, {self.alphas.max():.3f}])")


def air_absorption_coefficient(f, humidity=50.0, temperature=20.0):
    """
    Air absorption coefficient (ISO 9613-1 simplified).

    Parameters
    ----------
    f : float or array
        Frequency [Hz].
    humidity : float
        Relative humidity [%].
    temperature : float
        Temperature [degrees C].

    Returns
    -------
    m : float or array
        Absorption coefficient [Nepers/m].
        Multiply by c to get decay rate [1/s].
    """
    f = np.asarray(f, dtype=float)
    # Simplified ISO 9613-1 formula
    m = 5.5e-4 * (50.0 / max(humidity, 10.0)) * (f / 1000.0) ** 1.7
    return m


def compute_modal_decay_spectral(weights, material_functions, eigenfrequencies,
                                  c=343.0, humidity=50.0, temperature=20.0):
    """
    Compute per-mode decay rates using frequency-dependent absorption
    and air absorption.

    Each mode's decay rate uses the material absorption evaluated at
    that mode's specific eigenfrequency, plus air absorption at that
    frequency. No band binning.

    Parameters
    ----------
    weights : dict of label -> array (n_modes,)
        Per-surface modal coupling weights from precompute_surface_gamma_weights().
    material_functions : dict of label -> MaterialFunction
        Absorption function per surface.
    eigenfrequencies : array (n_modes,)
        Eigenfrequencies in Hz.
    c : float
        Speed of sound.
    humidity : float
        Relative humidity [%]. Default 50%.
    temperature : float
        Temperature [degrees C]. Default 20.

    Returns
    -------
    gamma : array (n_modes,)
        Decay rate per mode (wall absorption + air absorption).
    """
    rho_c = 1.2 * c
    labels = list(weights.keys())
    n_modes = len(weights[labels[0]])
    gamma = np.zeros(n_modes)

    # Wall absorption: per-surface, per-mode
    for label in labels:
        if label not in material_functions:
            continue
        mat = material_functions[label]
        Z_per_mode = mat.impedance(eigenfrequencies, rho_c=rho_c)
        gamma += weights[label] / Z_per_mode

    # Air absorption: volumetric, frequency-dependent
    m_air = air_absorption_coefficient(eigenfrequencies, humidity, temperature)
    gamma += m_air * c

    return gamma
