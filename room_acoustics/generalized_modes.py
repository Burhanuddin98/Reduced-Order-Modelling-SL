"""
Generalized analytical modes for arbitrary room geometry.

Extends the analytical mode approach to non-rectangular rooms by
detecting parallel surface pairs and computing:
  - Axial modes (1D) between each parallel pair
  - Tangential modes (2D) between perpendicular pairs of parallel pairs
  - Oblique modes (3D) between mutually perpendicular triplets

For rectangular rooms, this reproduces the exact analytical solution.
For non-rectangular rooms with some parallel surfaces (most real rooms),
it captures the dominant modal structure without eigensolve or mesh.

The eigensolve-based modal ROM can still be used for low frequencies
where the mesh resolves the modes — this module fills in the high
frequencies where the mesh can't reach.

Usage:
    from room_acoustics.generalized_modes import GeneralizedModes

    # From detected parallel surfaces (any geometry)
    gm = GeneralizedModes.from_parallel_pairs(pairs, volume, surface_area)
    ir, info = gm.synthesize_ir(source, receiver, materials, T=3.0)

    # Shortcut for box rooms (uses AnalyticalRoomModes internally)
    gm = GeneralizedModes.from_box(8.4, 6.7, 3.0, f_max=8000)
"""

import numpy as np
from .material_function import MaterialFunction, air_absorption_coefficient


class GeneralizedModes:
    """
    Analytical modes for rooms with detected parallel surface pairs.

    Finds perpendicular relationships between pairs to construct
    tangential (2D) and oblique (3D) modes in addition to axial (1D).
    """

    def __init__(self):
        self.axial_modes = []      # list of mode dicts
        self.tangential_modes = []
        self.oblique_modes = []
        self.all_modes = []        # combined, sorted by frequency
        self.volume = 0
        self.surface_area = 0

    @classmethod
    def from_box(cls, Lx, Ly, Lz, f_max=8000, c=343.0):
        """Create from box room dimensions (exact analytical solution)."""
        from .analytical_modes import AnalyticalRoomModes
        arm = AnalyticalRoomModes(Lx, Ly, Lz, f_max=f_max, c=c)

        gm = cls()
        gm.volume = arm.volume
        gm.surface_area = arm.surface_area
        gm._arm = arm  # store for direct use
        gm._is_box = True
        gm._dimensions = (Lx, Ly, Lz)
        gm._c = c
        gm._f_max = f_max
        return gm

    @classmethod
    def from_parallel_pairs(cls, pairs, volume, surface_area,
                            f_max=8000, c=343.0):
        """
        Create from detected parallel surface pairs (any geometry).

        Parameters
        ----------
        pairs : list of ParallelPair
            From detect_parallel_surfaces() or detect_parallel_surfaces_box().
        volume : float
            Room volume [m^3].
        surface_area : float
            Total room surface area [m^2].
        f_max : float
            Maximum frequency [Hz].
        c : float
            Speed of sound [m/s].
        """
        gm = cls()
        gm.volume = volume
        gm.surface_area = surface_area
        gm._is_box = False
        gm._pairs = pairs
        gm._c = c
        gm._f_max = f_max

        # Find perpendicular relationships between pairs
        gm._perpendicular_pairs = _find_perpendicular_pairs(pairs)
        gm._perpendicular_triplets = _find_perpendicular_triplets(
            pairs, gm._perpendicular_pairs)

        # Enumerate modes
        gm._enumerate_generalized_modes()
        return gm

    def _enumerate_generalized_modes(self):
        """Enumerate axial, tangential, and oblique modes from pairs."""
        c = self._c
        f_max = self._f_max
        pairs = self._pairs

        # Axial modes (1D): between each parallel pair
        for pair in pairs:
            L = pair.distance
            n_max = int(np.floor(2 * L * f_max / c))
            for n in range(1, n_max + 1):
                f = n * c / (2 * L)
                if f <= f_max:
                    self.axial_modes.append({
                        'type': 'axial',
                        'freq': f,
                        'indices': (n,),
                        'pairs': (pair,),
                        'distances': (L,),
                    })

        # Tangential modes (2D): between perpendicular pair combinations
        for (pair_a, pair_b) in self._perpendicular_pairs:
            La = pair_a.distance
            Lb = pair_b.distance
            na_max = int(np.ceil(2 * La * f_max / c))
            nb_max = int(np.ceil(2 * Lb * f_max / c))

            for na in range(1, na_max + 1):
                for nb in range(1, nb_max + 1):
                    f = c / 2 * np.sqrt((na / La) ** 2 + (nb / Lb) ** 2)
                    if f <= f_max:
                        self.tangential_modes.append({
                            'type': 'tangential',
                            'freq': f,
                            'indices': (na, nb),
                            'pairs': (pair_a, pair_b),
                            'distances': (La, Lb),
                        })

        # Oblique modes (3D): between perpendicular triplets
        for (pair_a, pair_b, pair_c) in self._perpendicular_triplets:
            La = pair_a.distance
            Lb = pair_b.distance
            Lc = pair_c.distance
            na_max = int(np.ceil(2 * La * f_max / c))
            nb_max = int(np.ceil(2 * Lb * f_max / c))
            nc_max = int(np.ceil(2 * Lc * f_max / c))

            for na in range(1, na_max + 1):
                for nb in range(1, nb_max + 1):
                    for nc in range(1, nc_max + 1):
                        f = c / 2 * np.sqrt(
                            (na / La) ** 2 + (nb / Lb) ** 2 + (nc / Lc) ** 2)
                        if f <= f_max:
                            self.oblique_modes.append({
                                'type': 'oblique',
                                'freq': f,
                                'indices': (na, nb, nc),
                                'pairs': (pair_a, pair_b, pair_c),
                                'distances': (La, Lb, Lc),
                            })

        # Combine and sort
        self.all_modes = sorted(
            self.axial_modes + self.tangential_modes + self.oblique_modes,
            key=lambda m: m['freq'])

    def synthesize_ir(self, source, receiver, materials,
                      T=3.0, sr=44100,
                      humidity=50.0, temperature=20.0,
                      f_min=None):
        """
        Synthesize IR using all detected modes.

        For box rooms, delegates to AnalyticalRoomModes (exact).
        For general rooms, uses the generalized pair-based modes.
        """
        if hasattr(self, '_is_box') and self._is_box:
            return self._arm.synthesize_ir(
                source, receiver, materials,
                T=T, sr=sr, humidity=humidity, temperature=temperature,
                f_min=f_min)

        return self._synthesize_generalized(
            source, receiver, materials, T, sr, humidity, temperature, f_min)

    def _synthesize_generalized(self, source, receiver, materials,
                                T, sr, humidity, temperature, f_min):
        """Synthesize IR from generalized (non-box) modes."""
        c = self._c
        V = self.volume
        S = self.surface_area
        n_samples = int(T * sr)
        t = np.arange(n_samples, dtype=np.float64) / sr
        ir = np.zeros(n_samples, dtype=np.float64)

        src = np.asarray(source, dtype=float)
        rec = np.asarray(receiver, dtype=float)

        n_active = 0

        for mode in self.all_modes:
            f = mode['freq']
            if f_min is not None and f < f_min:
                continue

            # Source and receiver coupling
            amplitude = 1.0
            for i, pair in enumerate(mode['pairs']):
                n_idx = mode['indices'][i]
                L = mode['distances'][i]
                n_dir = pair.normal

                x_src = np.dot(src - pair.centroid_1, n_dir)
                x_rec = np.dot(rec - pair.centroid_1, n_dir)

                x_src = np.clip(x_src, 0.0, L)
                x_rec = np.clip(x_rec, 0.0, L)

                amplitude *= (np.cos(n_idx * np.pi * x_src / L) *
                              np.cos(n_idx * np.pi * x_rec / L))

            if abs(amplitude) < 1e-20:
                continue

            # Normalization: 2^(n_dims) / V
            n_dims = len(mode['pairs'])
            amplitude *= 2.0 ** n_dims / V

            # Decay rate: per-surface contributions
            gamma = 0.0
            for i, pair in enumerate(mode['pairs']):
                n_idx = mode['indices'][i]
                if n_idx == 0:
                    continue

                for label in [pair.label_1, pair.label_2]:
                    mat = materials.get(label)
                    if mat is None:
                        continue
                    if isinstance(mat, MaterialFunction):
                        alpha = mat(f)
                    else:
                        alpha = 0.05
                    A_surface = pair.overlap_area
                    gamma += c * alpha * A_surface / (4 * V) * 2.0

            # Air absorption
            m_air = float(air_absorption_coefficient(np.array([f]),
                                                      humidity, temperature)[0])
            gamma += m_air * c

            if gamma <= 0:
                continue

            # Synthesize
            omega = 2 * np.pi * f
            t_80dB = 80.0 * np.log(10) / (20.0 * gamma)
            n_cut = min(int(t_80dB * sr) + 1, n_samples)
            if n_cut < 10:
                continue

            omega_d = np.sqrt(max(omega ** 2 - gamma ** 2, 0))
            if omega_d > 0:
                ir[:n_cut] += (amplitude * np.exp(-gamma * t[:n_cut])
                               * np.cos(omega_d * t[:n_cut]))

            n_active += 1

        mode_info = {
            'n_modes_total': len(self.all_modes),
            'n_modes_active': n_active,
            'n_axial': len(self.axial_modes),
            'n_tangential': len(self.tangential_modes),
            'n_oblique': len(self.oblique_modes),
        }

        return ir, mode_info

    def summary(self):
        """Print mode statistics."""
        if hasattr(self, '_is_box') and self._is_box:
            self._arm.summary()
            return

        print(f"  Generalized modes (f_max={self._f_max:.0f} Hz):")
        print(f"    Parallel pairs: {len(self._pairs)}")
        print(f"    Perpendicular pair combos: {len(self._perpendicular_pairs)}")
        print(f"    Perpendicular triplets: {len(self._perpendicular_triplets)}")
        print(f"    Axial modes:      {len(self.axial_modes)}")
        print(f"    Tangential modes: {len(self.tangential_modes)}")
        print(f"    Oblique modes:    {len(self.oblique_modes)}")
        print(f"    Total:            {len(self.all_modes)}")


def _find_perpendicular_pairs(pairs):
    """
    Find pairs of parallel-surface-pairs that are perpendicular to each other.

    Two pairs are perpendicular if their normals are orthogonal
    (dot product ≈ 0).

    Returns list of (pair_a, pair_b) tuples.
    """
    perp_pairs = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dot = abs(np.dot(pairs[i].normal, pairs[j].normal))
            if dot < 0.1:  # nearly perpendicular (< ~6 degrees from 90)
                perp_pairs.append((pairs[i], pairs[j]))
    return perp_pairs


def _find_perpendicular_triplets(pairs, perp_pairs):
    """
    Find triplets of mutually perpendicular parallel-surface-pairs.

    Three pairs are mutually perpendicular if each pair of normals
    is orthogonal.

    Returns list of (pair_a, pair_b, pair_c) tuples.
    """
    triplets = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if abs(np.dot(pairs[i].normal, pairs[j].normal)) >= 0.1:
                continue
            for k in range(j + 1, len(pairs)):
                if (abs(np.dot(pairs[i].normal, pairs[k].normal)) < 0.1 and
                        abs(np.dot(pairs[j].normal, pairs[k].normal)) < 0.1):
                    triplets.append((pairs[i], pairs[j], pairs[k]))
    return triplets
