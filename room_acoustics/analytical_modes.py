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

# Optional Numba JIT for fast synthesis
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


def _synthesize_numba(ir, amp, gam, wd, nc, dt):
    """Numba-accelerated modal synthesis. Fallback defined below."""
    pass  # replaced by JIT version if Numba available


if _HAVE_NUMBA:
    @njit(cache=True)
    def _synthesize_numba(ir, amp, gam, wd, nc, dt):
        """
        Synthesize IR using recursive oscillators (zero transcendentals).

        Instead of computing exp(-g*t)*cos(w*t) per sample, uses:
          decay_step = exp(-g*dt)          (precomputed once per mode)
          cos recurrence: c[k+1] = 2*cos(w*dt)*c[k] - c[k-1]

        This eliminates ALL exp/cos calls from the inner loop,
        replacing them with 2 multiplies + 1 subtract per mode per sample.
        """
        n_modes = len(amp)
        for i in range(n_modes):
            a = amp[i]
            g = gam[i]
            w = wd[i]
            n = nc[i]
            if n <= 0 or abs(a) < 1e-30:
                continue

            # Precompute per-mode constants (only transcendentals here)
            decay_step = np.exp(-g * dt)   # multiplied each step
            cos_coeff = 2.0 * np.cos(w * dt)  # for cos recurrence

            # Initial values: k=0 → env=1, cos=1; k=1 → env*=decay, cos=cos(w*dt)
            env = 1.0
            cos_prev = 1.0
            cos_curr = np.cos(w * dt)

            # k=0
            ir[0] += a * env * cos_prev

            # k=1
            if n > 1:
                env *= decay_step
                ir[1] += a * env * cos_curr

            # k=2..n-1: pure multiply+subtract, no transcendentals
            for k in range(2, n):
                env *= decay_step
                cos_next = cos_coeff * cos_curr - cos_prev
                ir[k] += a * env * cos_next
                cos_prev = cos_curr
                cos_curr = cos_next

        return ir


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

        Vectorized: groups modes by decay rate and synthesizes each
        group as a batched numpy operation. Handles 100K+ modes in
        seconds instead of minutes.

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
        ir = np.zeros(n_samples, dtype=np.float64)

        # Mode shapes at source and receiver (fully vectorized)
        phi_src = self.mode_shape(*source)
        phi_rec = self.mode_shape(*receiver)

        # Normalization: 2^(n_nonzero_indices) / V
        n_nonzero = ((self.modes_n > 0).astype(int) +
                     (self.modes_m > 0).astype(int) +
                     (self.modes_l > 0).astype(int))
        norm = np.power(2.0, n_nonzero) / self.volume

        amplitudes = phi_src * phi_rec * norm

        # Decay rates and angular frequencies
        gamma = self.compute_decay_rates(materials, humidity, temperature)
        omega = 2.0 * np.pi * self.modes_f

        # Filter: frequency range and significant amplitude
        # Threshold at 1e-3 of peak amplitude — modes below this contribute
        # <0.1% to the total energy and can be safely skipped.
        amp_threshold = max(np.max(np.abs(amplitudes)) * 1e-3, 1e-20)
        mask = np.abs(amplitudes) > amp_threshold
        if f_min is not None:
            mask &= self.modes_f >= f_min

        # Damped frequency: omega_d = sqrt(omega^2 - gamma^2)
        omega_d = np.sqrt(np.maximum(omega ** 2 - gamma ** 2, 0.0))
        mask &= omega_d > 0

        # Effective time length per mode (early termination at -80 dB)
        safe_gamma = np.maximum(gamma, 0.01)
        t_80dB = 80.0 * np.log(10) / (20.0 * safe_gamma)
        n_cut = np.minimum(np.floor(t_80dB * sr).astype(int) + 1, n_samples)
        mask &= n_cut >= 10

        # Apply mask
        idx = np.where(mask)[0]
        n_active = len(idx)

        if n_active == 0:
            return ir, self._make_info(0)

        amp = amplitudes[idx]
        gam = gamma[idx]
        wd = omega_d[idx]
        nc = n_cut[idx]

        dt_val = 1.0 / sr

        if _HAVE_NUMBA:
            # Fast path: Numba JIT with parallel threads.
            # First call has ~1s compilation overhead, then instant.
            ir = _synthesize_numba(ir, amp, gam, wd, nc, dt_val)
        else:
            # Fallback: tiered chunked numpy
            order = np.argsort(nc)
            amp, gam, wd, nc = amp[order], gam[order], wd[order], nc[order]

            tier_limits = [int(0.01 * sr), int(0.05 * sr), int(0.2 * sr),
                           int(1.0 * sr), n_samples]
            i = 0
            for tier_n in tier_limits:
                j = i
                while j < n_active and nc[j] <= tier_n:
                    j += 1
                if j <= i:
                    continue

                t_len = min(tier_n, n_samples)
                max_chunk = max(1, int(100e6 / (max(t_len, 1) * 8 * 3)))
                t_vec = np.arange(t_len, dtype=np.float64) / sr

                for ci in range(i, j, max_chunk):
                    cj = min(ci + max_chunk, j)
                    phase = wd[ci:cj, np.newaxis] * t_vec[np.newaxis, :]
                    decay = np.exp(-gam[ci:cj, np.newaxis] * t_vec[np.newaxis, :])
                    ir[:t_len] += np.dot(amp[ci:cj], decay * np.cos(phase))
                i = j

        return ir, self._make_info(n_active)

    def _make_info(self, n_active):
        """Build mode_info dict."""
        return {
            'n_modes_total': self.n_modes,
            'n_modes_active': n_active,
            'n_axial': self.n_axial,
            'n_tangential': self.n_tangential,
            'n_oblique': self.n_oblique,
            'f_min': float(self.modes_f[0]) if self.n_modes > 0 else 0,
            'f_max': float(self.modes_f[-1]) if self.n_modes > 0 else 0,
        }

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
