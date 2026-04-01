"""
Analytical room modes — exact modal solution for rectangular rooms.

Computes ALL room modes (axial, tangential, oblique) analytically.
No mesh, no eigensolve, no frequency limit, no DOF scaling.

  Axial modes (1D):     f_{n,0,0}  — between one pair of parallel walls
  Tangential modes (2D): f_{n,m,0}  — between two pairs of walls
  Oblique modes (3D):    f_{n,m,l}  — all three pairs of walls

Eigenfrequencies:
  f_{n,m,l} = c/2 * sqrt((n/Lx)^2 + (m/Ly)^2 + (l/Lz)^2)

Mode shapes:
  phi_{n,m,l}(x,y,z) = cos(n*pi*x/Lx) * cos(m*pi*y/Ly) * cos(l*pi*z/Lz)

Decay rates:
  gamma_{n,m,l} = sum over 6 surfaces of (per-surface contribution)
                + air absorption at f_{n,m,l}

This replaces both the eigensolve-based modal ROM and the axial mode
engine for box rooms, covering 20 Hz to 20 kHz with exact physics.

Usage:
    from room_acoustics.analytical_modes import AnalyticalRoomModes

    modes = AnalyticalRoomModes(8.4, 6.7, 3.0, f_max=8000)
    ir = modes.synthesize_ir(source, receiver, materials, T=3.0)
"""

import numpy as np
from .material_function import MaterialFunction, air_absorption_coefficient


class AnalyticalRoomModes:
    """
    Full analytical modal solution for a rectangular room.

    Enumerates all (n, m, l) mode triplets up to f_max, computes
    eigenfrequencies, mode shapes at source/receiver, and decay
    rates from per-surface frequency-dependent absorption.
    """

    def __init__(self, Lx, Ly, Lz, f_max=8000, c=343.0):
        """
        Parameters
        ----------
        Lx, Ly, Lz : float
            Room dimensions [m].
        f_max : float
            Maximum frequency [Hz]. Modes above this are excluded.
        c : float
            Speed of sound [m/s].
        """
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.c = float(c)
        self.f_max = float(f_max)
        self.volume = Lx * Ly * Lz
        self.surface_area = 2 * (Lx * Ly + Lx * Lz + Ly * Lz)

        # Surface areas for decay rate calculation
        self._surface_areas = {
            'left': Ly * Lz, 'right': Ly * Lz,     # x = 0, x = Lx
            'front': Lx * Lz, 'back': Lx * Lz,     # y = 0, y = Ly
            'floor': Lx * Ly, 'ceiling': Lx * Ly,   # z = 0, z = Lz
        }

        # Enumerate all modes
        self._enumerate_modes()

    def _enumerate_modes(self):
        """Find all (n, m, l) with f_{n,m,l} <= f_max."""
        Lx, Ly, Lz, c = self.Lx, self.Ly, self.Lz, self.c

        n_max = int(np.ceil(2 * Lx * self.f_max / c))
        m_max = int(np.ceil(2 * Ly * self.f_max / c))
        l_max = int(np.ceil(2 * Lz * self.f_max / c))

        modes_n = []
        modes_m = []
        modes_l = []
        modes_f = []
        modes_type = []  # 'axial', 'tangential', 'oblique'

        for n in range(0, n_max + 1):
            for m in range(0, m_max + 1):
                for l in range(0, l_max + 1):
                    if n == 0 and m == 0 and l == 0:
                        continue  # DC mode, skip

                    f = c / 2.0 * np.sqrt(
                        (n / Lx) ** 2 + (m / Ly) ** 2 + (l / Lz) ** 2)

                    if f > self.f_max:
                        continue

                    # Classify mode type
                    n_zero = (n == 0) + (m == 0) + (l == 0)
                    if n_zero == 2:
                        mtype = 'axial'
                    elif n_zero == 1:
                        mtype = 'tangential'
                    else:
                        mtype = 'oblique'

                    modes_n.append(n)
                    modes_m.append(m)
                    modes_l.append(l)
                    modes_f.append(f)
                    modes_type.append(mtype)

        # Sort by frequency
        order = np.argsort(modes_f)
        self.modes_n = np.array(modes_n, dtype=int)[order]
        self.modes_m = np.array(modes_m, dtype=int)[order]
        self.modes_l = np.array(modes_l, dtype=int)[order]
        self.modes_f = np.array(modes_f, dtype=float)[order]
        self.modes_type = [modes_type[i] for i in order]
        self.n_modes = len(self.modes_f)

        # Count by type
        self.n_axial = sum(1 for t in self.modes_type if t == 'axial')
        self.n_tangential = sum(1 for t in self.modes_type if t == 'tangential')
        self.n_oblique = sum(1 for t in self.modes_type if t == 'oblique')

    def mode_shape(self, x, y, z):
        """
        Evaluate all mode shapes at a point.

        phi_{n,m,l}(x,y,z) = cos(n*pi*x/Lx) * cos(m*pi*y/Ly) * cos(l*pi*z/Lz)

        Parameters
        ----------
        x, y, z : float
            Position in room [m].

        Returns
        -------
        phi : ndarray (n_modes,)
            Mode shape amplitude at (x, y, z) for each mode.
        """
        phi_x = np.cos(self.modes_n * np.pi * x / self.Lx)
        phi_y = np.cos(self.modes_m * np.pi * y / self.Ly)
        phi_z = np.cos(self.modes_l * np.pi * z / self.Lz)
        return phi_x * phi_y * phi_z

    def compute_decay_rates(self, materials, humidity=50.0, temperature=20.0):
        """
        Compute per-mode decay rates from frequency-dependent materials.

        For each mode, the decay rate is the sum of contributions from
        all 6 surfaces, weighted by how much the mode couples to each
        surface, plus air absorption at the mode's frequency.

        The per-surface contribution for mode (n,m,l) depends on the
        mode's behavior at that surface:
        - A mode with n=0 has constant pressure along x → no energy
          exchange with the x-walls → zero contribution from left/right
        - A mode with n>0 has cos(n*pi*x/Lx) → exchanges energy with
          both x-walls → contribution proportional to alpha(f)

        Exact formula (from Kuttruff, Room Acoustics):
          gamma_{n,m,l} = c * sum_surfaces( alpha_s(f) * A_s / (4V) * epsilon_s )

        where epsilon_s accounts for which mode indices are nonzero:
          For x-walls: epsilon = 2 if n>0, else 0
          For y-walls: epsilon = 2 if m>0, else 0
          For z-walls: epsilon = 2 if l>0, else 0

        Parameters
        ----------
        materials : dict of label -> MaterialFunction
            'floor', 'ceiling', 'left', 'right', 'front', 'back'.
        humidity : float
            Relative humidity [%].
        temperature : float
            Temperature [degrees C].

        Returns
        -------
        gamma : ndarray (n_modes,)
            Decay rate per mode [1/s].
        """
        c = self.c
        V = self.volume

        # Per-surface alpha evaluated at each mode's frequency
        gamma = np.zeros(self.n_modes)

        # Surface pairs and which mode index activates them
        surface_config = [
            # (label, area, mode_index_array)
            ('left',    self._surface_areas['left'],    self.modes_n),
            ('right',   self._surface_areas['right'],   self.modes_n),
            ('front',   self._surface_areas['front'],   self.modes_m),
            ('back',    self._surface_areas['back'],    self.modes_m),
            ('floor',   self._surface_areas['floor'],   self.modes_l),
            ('ceiling', self._surface_areas['ceiling'], self.modes_l),
        ]

        for label, area, mode_idx in surface_config:
            mat = materials.get(label)
            if mat is None:
                continue

            # Alpha at each mode's frequency
            alpha = mat(self.modes_f)  # (n_modes,)

            # Epsilon: mode couples to this surface only if the
            # corresponding index is nonzero
            epsilon = np.where(mode_idx > 0, 2.0, 0.0)

            # Contribution: c * alpha * A / (4V) * epsilon
            gamma += c * alpha * area / (4 * V) * epsilon

        # Air absorption
        m_air = air_absorption_coefficient(self.modes_f, humidity, temperature)
        gamma += m_air * c

        return gamma

    def synthesize_ir(self, source, receiver, materials,
                      T=3.0, sr=44100,
                      humidity=50.0, temperature=20.0,
                      f_min=None):
        """
        Synthesize impulse response using all analytical modes.

        Parameters
        ----------
        source : (x, y, z)
            Source position [m].
        receiver : (x, y, z)
            Receiver position [m].
        materials : dict of label -> MaterialFunction
            Per-surface absorption functions.
        T : float
            IR duration [seconds].
        sr : int
            Output sample rate [Hz].
        humidity, temperature : float
            For air absorption.
        f_min : float or None
            Minimum frequency to include (skip modes below this).

        Returns
        -------
        ir : ndarray (n_samples,)
            Impulse response.
        mode_info : dict
            Diagnostics: mode counts, frequency range, etc.
        """
        n_samples = int(T * sr)
        t = np.arange(n_samples, dtype=np.float64) / sr
        ir = np.zeros(n_samples, dtype=np.float64)

        # Mode shapes at source and receiver
        sx, sy, sz = source
        rx, ry, rz = receiver
        phi_src = self.mode_shape(sx, sy, sz)
        phi_rec = self.mode_shape(rx, ry, rz)

        # Amplitudes: product of source and receiver coupling
        # Normalization: 8/V for oblique, 4/V for tangential, 2/V for axial
        norm = np.ones(self.n_modes)
        for i in range(self.n_modes):
            n_nonzero = (self.modes_n[i] > 0) + (self.modes_m[i] > 0) + (self.modes_l[i] > 0)
            norm[i] = 2.0 ** n_nonzero / self.volume

        amplitudes = phi_src * phi_rec * norm

        # Decay rates
        gamma = self.compute_decay_rates(materials, humidity, temperature)

        # Angular frequencies
        omega = 2.0 * np.pi * self.modes_f

        # Frequency filter
        if f_min is not None:
            mask = self.modes_f >= f_min
        else:
            mask = np.ones(self.n_modes, dtype=bool)

        # Synthesize: per-mode loop with early termination
        n_active = 0
        for i in range(self.n_modes):
            if not mask[i]:
                continue
            if abs(amplitudes[i]) < 1e-20:
                continue

            g = gamma[i]
            w = omega[i]

            # Early termination: skip if decayed below -80 dB
            if g > 0:
                t_80dB = 80.0 * np.log(10) / (20.0 * g)
                n_cut = min(int(t_80dB * sr) + 1, n_samples)
            else:
                n_cut = n_samples

            if n_cut < 10:
                continue

            # Damped oscillation
            omega_d = np.sqrt(max(w ** 2 - g ** 2, 0))
            if omega_d > 0:
                ir[:n_cut] += (amplitudes[i] * np.exp(-g * t[:n_cut])
                               * np.cos(omega_d * t[:n_cut]))
            elif w > 0:
                ir[:n_cut] += amplitudes[i] * np.exp(-g * t[:n_cut])

            n_active += 1

        mode_info = {
            'n_modes_total': self.n_modes,
            'n_modes_active': n_active,
            'n_axial': self.n_axial,
            'n_tangential': self.n_tangential,
            'n_oblique': self.n_oblique,
            'f_min': float(self.modes_f[0]) if self.n_modes > 0 else 0,
            'f_max': float(self.modes_f[-1]) if self.n_modes > 0 else 0,
        }

        return ir, mode_info

    def summary(self):
        """Print mode count summary."""
        print(f"  Room: {self.Lx} x {self.Ly} x {self.Lz} m")
        print(f"  f_max: {self.f_max:.0f} Hz")
        print(f"  Total modes: {self.n_modes}")
        print(f"    Axial:      {self.n_axial}")
        print(f"    Tangential: {self.n_tangential}")
        print(f"    Oblique:    {self.n_oblique}")
        if self.n_modes > 0:
            print(f"  Frequency range: {self.modes_f[0]:.1f} - "
                  f"{self.modes_f[-1]:.1f} Hz")
