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

    def __repr__(self):
        return (f"MaterialFunction('{self.name}', {len(self.freqs)} points, "
                f"alpha=[{self.alphas.min():.3f}, {self.alphas.max():.3f}])")


def compute_modal_decay_spectral(weights, material_functions, eigenfrequencies,
                                  c=343.0):
    """
    Compute per-mode decay rates using frequency-dependent absorption.

    Each mode's decay rate uses the material absorption evaluated at
    that mode's specific eigenfrequency — no band binning.

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

    Returns
    -------
    gamma : array (n_modes,)
        Decay rate per mode.
    """
    rho_c = 1.2 * c
    labels = list(weights.keys())
    n_modes = len(weights[labels[0]])
    gamma = np.zeros(n_modes)

    for label in labels:
        if label not in material_functions:
            continue
        mat = material_functions[label]
        # Evaluate absorption at each mode's eigenfrequency
        Z_per_mode = mat.impedance(eigenfrequencies, rho_c=rho_c)
        gamma += weights[label] / Z_per_mode

    return gamma
