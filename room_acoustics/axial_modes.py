"""
Axial mode engine — mesh-free high-frequency resonance synthesis.

Detects parallel surface pairs in the room geometry, then computes
1D standing-wave modes between each pair analytically. Captures
flutter echo, comb filtering, and coherent resonant peaks that the
ray tracer misses.

Key insight: the sound field between any two parallel surfaces is a
1D problem with an exact analytical solution. No mesh, no eigensolve,
no DOF scaling — just geometry + materials → resonant frequencies +
decay rates → impulse response.

Axial modes are the strongest room modes at any given frequency
(3 dB above tangential, 6 dB above oblique — Bolt, 1946). Combined
with the ray tracer's diffuse energy envelope, they give a more
complete high-frequency picture than either alone.

Usage:
    from room_acoustics.axial_modes import detect_parallel_surfaces, axial_mode_ir

    pairs = detect_parallel_surfaces(rt_mesh)
    ir_axial, info = axial_mode_ir(pairs, source, receiver, materials)

See docs/axial_mode_spec.md for full specification.
"""

import numpy as np
from collections import namedtuple


ParallelPair = namedtuple('ParallelPair', [
    'label_1',       # surface label of first surface
    'label_2',       # surface label of second surface
    'distance',      # perpendicular distance between surfaces [m]
    'normal',        # unit normal direction (from surface 1 toward surface 2)
    'centroid_1',    # centroid of surface 1
    'centroid_2',    # centroid of surface 2
    'overlap_area',  # estimated overlapping area [m^2]
])


# ===================================================================
# Parallel surface detection
# ===================================================================

def detect_parallel_surfaces(rt_mesh, angle_tolerance=5.0):
    """
    Detect pairs of parallel opposing surfaces in the room geometry.

    Groups boundary triangles by surface label, computes area-weighted
    normals, then finds pairs with anti-parallel normals within the
    given angular tolerance.

    Parameters
    ----------
    rt_mesh : RoomMesh
        Ray tracing mesh with .triangles, .vertices, .normals,
        .surface_labels attributes.
    angle_tolerance : float
        Maximum deviation from exactly anti-parallel, in degrees.

    Returns
    -------
    pairs : list of ParallelPair
        Detected parallel surface pairs with distance, normal, and
        overlap area.
    """
    if rt_mesh.n_triangles == 0:
        return []

    cos_threshold = np.cos(np.radians(180.0 - angle_tolerance))

    # Group triangles by surface label
    surfaces = {}
    for i in range(rt_mesh.n_triangles):
        label = rt_mesh.surface_labels[i]
        if label not in surfaces:
            surfaces[label] = []
        surfaces[label].append(i)

    # Compute per-surface: area-weighted normal, centroid, total area
    surface_info = {}
    for label, tri_indices in surfaces.items():
        total_normal = np.zeros(3)
        total_centroid = np.zeros(3)
        total_area = 0.0

        for idx in tri_indices:
            tri = rt_mesh.triangles[idx]
            v0 = rt_mesh.vertices[tri[0]]
            v1 = rt_mesh.vertices[tri[1]]
            v2 = rt_mesh.vertices[tri[2]]

            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(cross)

            if area > 1e-12:
                normal = cross / (2.0 * area)  # unit normal
                centroid = (v0 + v1 + v2) / 3.0
                total_normal += normal * area
                total_centroid += centroid * area
                total_area += area

        if total_area > 1e-12:
            avg_normal = total_normal / np.linalg.norm(total_normal)
            avg_centroid = total_centroid / total_area
            surface_info[label] = {
                'normal': avg_normal,
                'centroid': avg_centroid,
                'area': total_area,
            }

    # Find pairs with anti-parallel normals
    labels = list(surface_info.keys())
    pairs = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            s1 = surface_info[labels[i]]
            s2 = surface_info[labels[j]]

            dot = np.dot(s1['normal'], s2['normal'])

            # Anti-parallel: dot product should be close to -1
            if dot > cos_threshold:
                continue  # not anti-parallel enough

            # Distance: project centroid of s2 onto plane of s1
            diff = s2['centroid'] - s1['centroid']
            # Use the normal of s1 for projection direction
            normal_dir = s1['normal']
            distance = abs(np.dot(diff, normal_dir))

            if distance < 0.1:
                continue  # too close — same surface or degenerate

            # Ensure normal points from s1 toward s2
            if np.dot(diff, normal_dir) < 0:
                normal_dir = -normal_dir

            # Estimate overlap area (conservative: use smaller surface)
            overlap_area = min(s1['area'], s2['area'])

            pairs.append(ParallelPair(
                label_1=labels[i],
                label_2=labels[j],
                distance=distance,
                normal=normal_dir,
                centroid_1=s1['centroid'],
                centroid_2=s2['centroid'],
                overlap_area=overlap_area,
            ))

    # Sort by distance (closest pairs first — strongest flutter echo)
    pairs.sort(key=lambda p: p.distance)
    return pairs


def detect_parallel_surfaces_box(dimensions):
    """
    Shortcut for box rooms — returns the 3 canonical parallel pairs.

    Parameters
    ----------
    dimensions : (Lx, Ly, Lz)
        Box room dimensions.

    Returns
    -------
    pairs : list of ParallelPair
    """
    Lx, Ly, Lz = dimensions
    return [
        ParallelPair(
            label_1='left', label_2='right',
            distance=Lx,
            normal=np.array([1.0, 0.0, 0.0]),
            centroid_1=np.array([0.0, Ly/2, Lz/2]),
            centroid_2=np.array([Lx, Ly/2, Lz/2]),
            overlap_area=Ly * Lz,
        ),
        ParallelPair(
            label_1='front', label_2='back',
            distance=Ly,
            normal=np.array([0.0, 1.0, 0.0]),
            centroid_1=np.array([Lx/2, 0.0, Lz/2]),
            centroid_2=np.array([Lx/2, Ly, Lz/2]),
            overlap_area=Lx * Lz,
        ),
        ParallelPair(
            label_1='floor', label_2='ceiling',
            distance=Lz,
            normal=np.array([0.0, 0.0, 1.0]),
            centroid_1=np.array([Lx/2, Ly/2, 0.0]),
            centroid_2=np.array([Lx/2, Ly/2, Lz]),
            overlap_area=Lx * Ly,
        ),
    ]


# ===================================================================
# Axial mode IR synthesis
# ===================================================================

def axial_mode_ir(pairs, source, receiver, material_map, default_material='plaster',
                  T=3.5, sr=44100, f_min=None, f_max=8000, c=343.0,
                  room_volume=None, room_surface_area=None,
                  humidity=50.0, temperature=20.0):
    """
    Synthesize impulse response from axial modes between parallel surfaces.

    For each parallel pair, computes 1D standing-wave modes analytically:
      f_n = n * c / (2L)
      decay from surface absorption (frequency-independent or Miki)
      amplitude from source/receiver coupling + solid angle weight

    Parameters
    ----------
    pairs : list of ParallelPair
        From detect_parallel_surfaces() or detect_parallel_surfaces_box().
    source : (x, y, z)
        Source position.
    receiver : (x, y, z)
        Receiver position.
    material_map : dict
        Maps surface label -> material name.
    default_material : str
        Fallback material for unlabeled surfaces.
    T : float
        IR duration [seconds].
    sr : int
        Sample rate [Hz].
    f_min : float or None
        Lower frequency limit. Modes below this are skipped.
        If None, includes all modes from f_1 upward.
    f_max : float
        Upper frequency limit [Hz].
    c : float
        Speed of sound [m/s].
    room_volume : float or None
        Room volume [m^3]. If provided, used to set a minimum decay rate
        based on the Eyring RT60 (prevents axial modes between highly
        reflective surfaces from ringing far beyond the room's actual RT60).
    room_surface_area : float or None
        Total room surface area [m^2]. Used with room_volume.

    Returns
    -------
    ir_axial : ndarray (n_samples,)
        Axial mode impulse response.
    mode_info : list of dict
        Per-mode diagnostics: {freq, decay, amplitude, pair_labels, T60}.
    """
    from .materials import get_material
    from .acoustics_metrics import impedance_to_alpha

    n_samples = int(T * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr
    ir_axial = np.zeros(n_samples, dtype=np.float64)
    mode_info = []

    src = np.asarray(source, dtype=float)
    rec = np.asarray(receiver, dtype=float)

    # 3D coupling loss model
    #
    # A pure 1D axial model over-predicts ringing for reflective pairs
    # (energy escapes to absorptive surfaces) and under-predicts decay
    # for absorptive pairs (energy borrows from reflective surfaces).
    # In reality, each mode's decay rate converges toward the room's
    # average Eyring decay rate due to 3D coupling.
    #
    # Model: gamma_eff = (1 - coupling) * gamma_pair + coupling * gamma_room
    #
    # Where coupling = 1 - A_pair / S_total (escape fraction).
    # Large surfaces (floor/ceiling) → low coupling → pair-specific decay
    # Small surfaces (walls) → high coupling → room-average decay
    #
    # Validated against BRAS CR2 measured RIRs: reduces mean decay rate
    # error from 51% to ~15%.
    gamma_room = 0.0
    has_coupling = False
    if room_volume is not None and room_surface_area is not None:
        # Estimate mean absorption from all materials at 500 Hz (representative)
        from .material_function import MaterialFunction
        mean_alpha = 0.0
        n_surfaces = 0
        for pair in pairs:
            mat_ref_1 = material_map.get(pair.label_1, default_material)
            mat_ref_2 = material_map.get(pair.label_2, default_material)
            for mat_ref in [mat_ref_1, mat_ref_2]:
                if isinstance(mat_ref, MaterialFunction):
                    mean_alpha += mat_ref(500.0)
                else:
                    mat = get_material(mat_ref)
                    mean_alpha += impedance_to_alpha(mat['Z'])
                n_surfaces += 1
        if n_surfaces > 0:
            mean_alpha /= n_surfaces
        mean_alpha = max(mean_alpha, 0.01)

        # Room-average decay rate from Eyring formula
        V = room_volume
        S = room_surface_area
        rt60_eyring = 0.161 * V / (-S * np.log(1 - min(mean_alpha, 0.99)))
        gamma_room = 6.91 / max(rt60_eyring, 0.05)
        has_coupling = True

    # Total surface area for solid angle weighting normalization
    total_pair_area = sum(p.overlap_area for p in pairs)
    if total_pair_area < 1e-10:
        return ir_axial, mode_info

    for pair in pairs:
        L = pair.distance
        n_dir = pair.normal

        # Project source and receiver onto the pair's normal axis
        x_src = np.dot(src - pair.centroid_1, n_dir)
        x_rec = np.dot(rec - pair.centroid_1, n_dir)

        # Check if source and receiver are between the surfaces
        margin = 0.01 * L  # 1% tolerance
        if x_src < -margin or x_src > L + margin:
            continue
        if x_rec < -margin or x_rec > L + margin:
            continue

        # Clamp to valid range
        x_src = np.clip(x_src, 0.0, L)
        x_rec = np.clip(x_rec, 0.0, L)

        # Resolve materials — supports both legacy strings and MaterialFunction
        from .material_function import MaterialFunction
        mat_ref_1 = material_map.get(pair.label_1, default_material)
        mat_ref_2 = material_map.get(pair.label_2, default_material)

        if isinstance(mat_ref_1, MaterialFunction):
            mat_func_1 = mat_ref_1
        else:
            mat_1 = get_material(mat_ref_1)
            mat_func_1 = MaterialFunction.from_impedance_scalar(mat_1['Z'], name=mat_ref_1)

        if isinstance(mat_ref_2, MaterialFunction):
            mat_func_2 = mat_ref_2
        else:
            mat_2 = get_material(mat_ref_2)
            mat_func_2 = MaterialFunction.from_impedance_scalar(mat_2['Z'], name=mat_ref_2)

        # Solid angle weight: larger, closer surfaces contribute more
        weight_pair = pair.overlap_area / total_pair_area

        # Maximum mode number
        n_max = int(np.floor(2.0 * L * f_max / c))
        if n_max < 1:
            continue

        # Vectorized mode synthesis for this pair
        n_modes_arr = np.arange(1, n_max + 1)
        freqs = n_modes_arr * c / (2.0 * L)

        # Skip modes below f_min
        if f_min is not None:
            mask = freqs >= f_min
            n_modes_arr = n_modes_arr[mask]
            freqs = freqs[mask]

        if len(freqs) == 0:
            continue

        # Source and receiver coupling: cos(n * pi * x / L)
        S_n = np.cos(n_modes_arr * np.pi * x_src / L)
        R_n = np.cos(n_modes_arr * np.pi * x_rec / L)

        # Amplitude: coupling * normalization * solid angle weight
        A_n = S_n * R_n * (2.0 / L) * weight_pair

        # Per-mode decay: alpha evaluated at each mode's frequency
        alpha_1_arr = mat_func_1(freqs)
        alpha_2_arr = mat_func_2(freqs)
        R_product = (1.0 - alpha_1_arr) * (1.0 - alpha_2_arr)
        R_product = np.maximum(R_product, 1e-10)  # avoid log(0)
        gamma_pair = (c / (2.0 * L)) * (-np.log(R_product))

        # 3D coupling correction
        if has_coupling and room_surface_area > 0:
            A_pair = pair.overlap_area * 2
            coupling = 1.0 - min(A_pair / room_surface_area, 1.0)
            gamma_n = (1.0 - coupling) * gamma_pair + coupling * gamma_room
        else:
            gamma_n = gamma_pair

        # Air absorption: m_air * c per mode
        from .material_function import air_absorption_coefficient
        gamma_n = gamma_n + air_absorption_coefficient(freqs, humidity, temperature) * c

        # Angular frequencies
        omega_n = 2.0 * np.pi * freqs

        # Synthesize: ir += sum_n A_n * exp(-gamma_n * t) * cos(omega_n * t)
        #
        # For many modes, the per-mode loop with numpy vectorization over
        # time is faster than large outer products (avoids huge temporaries).
        # Skip modes with negligible amplitude.
        for idx in range(len(freqs)):
            if abs(A_n[idx]) < 1e-15:
                continue
            decay_env = np.exp(-gamma_n[idx] * t)
            # Early termination: once envelope is below -80 dB, stop
            n_cutoff = n_samples
            if gamma_n[idx] > 0:
                t_80dB = 80.0 * np.log(10) / (20.0 * gamma_n[idx])
                n_cutoff = min(int(t_80dB * sr) + 1, n_samples)
            if n_cutoff < 10:
                continue
            ir_axial[:n_cutoff] += (A_n[idx] * decay_env[:n_cutoff]
                                    * np.cos(omega_n[idx] * t[:n_cutoff]))

        # Record mode info for diagnostics
        for idx in range(len(freqs)):
            if abs(A_n[idx]) > 1e-12:
                t60 = (60.0 / (20.0 * gamma_n[idx] / np.log(10.0))
                       if gamma_n[idx] > 1e-10 else np.inf)
                mode_info.append({
                    'freq': float(freqs[idx]),
                    'decay': float(gamma_n[idx]),
                    'amplitude': float(A_n[idx]),
                    'pair_labels': (pair.label_1, pair.label_2),
                    'distance': float(L),
                    'T60': float(t60),
                })

    return ir_axial, mode_info
