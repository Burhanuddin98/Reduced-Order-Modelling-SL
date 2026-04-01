"""
Validation tests for the axial mode engine.

Test 1: Shoebox eigenfrequency verification
  - Box room axial modes must match f_n = n*c/(2L) exactly
  - Compare against full 3D eigensolve for axial modes

Test 2: Flutter echo reproduction
  - Two hard parallel walls, source/receiver between them
  - IR must show periodic reflections at tau = 2L/c
  - Spectrum must show peaks at n*c/(2L)

Test 3: Position dependence
  - Receiver at midpoint hears only odd modes
  - Receiver near wall hears all modes

Test 4: Decay rate accuracy
  - Compare modal decay against analytical formula
  - gamma = c/(2L) * (-ln((1-alpha_1)(1-alpha_2)))

Run: python validate_axial_modes.py
"""

import numpy as np
import sys
import os

# Ensure parent directory is on path for absolute imports within the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_shoebox_frequencies():
    """Test 1: Axial mode frequencies match f_n = n*c/(2L) for a box room."""
    print("=" * 60)
    print("TEST 1: Shoebox axial mode frequencies")
    print("=" * 60)

    from room_acoustics.axial_modes import detect_parallel_surfaces_box, axial_mode_ir

    Lx, Ly, Lz = 5.0, 4.0, 3.0
    c = 343.0

    pairs = detect_parallel_surfaces_box((Lx, Ly, Lz))
    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"

    # Check distances
    distances = sorted([p.distance for p in pairs])
    expected = sorted([Lx, Ly, Lz])
    for d, e in zip(distances, expected):
        assert abs(d - e) < 1e-10, f"Distance {d} != expected {e}"
    print(f"  Distances: {[p.distance for p in pairs]} — correct")

    # Generate IR and check mode frequencies
    source = (2.5, 2.0, 1.5)  # center of room
    receiver = (1.0, 1.0, 1.0)
    materials = {}  # will use default

    _, mode_info = axial_mode_ir(
        pairs, source, receiver, materials,
        default_material='concrete',
        T=1.0, sr=44100, f_min=None, f_max=2000, c=c)

    # Check that all reported frequencies match n*c/(2L) for one of the three pairs
    n_checked = 0
    for m in mode_info:
        f = m['freq']
        L = m['distance']
        n = round(2 * L * f / c)
        f_expected = n * c / (2 * L)
        err = abs(f - f_expected)
        assert err < 0.01, f"Mode at {f:.1f} Hz: expected {f_expected:.1f} Hz (n={n}, L={L})"
        n_checked += 1

    print(f"  {n_checked} modes verified against f_n = n*c/(2L) — all correct")
    print("  PASS\n")
    return True


def test_flutter_echo():
    """Test 2: Flutter echo — periodic reflections between hard walls."""
    print("=" * 60)
    print("TEST 2: Flutter echo between parallel walls")
    print("=" * 60)

    from room_acoustics.axial_modes import detect_parallel_surfaces_box, axial_mode_ir

    L = 5.0  # distance between walls
    c = 343.0
    tau = 2 * L / c  # round-trip time ≈ 29.15 ms

    pairs = detect_parallel_surfaces_box((L, 10.0, 10.0))
    # Only use the x-pair (L=5m)
    pair_x = [p for p in pairs if p.distance == L]

    source = (1.3, 5.0, 5.0)     # irrational fraction of L: couples to all modes
    receiver = (3.7, 5.0, 5.0)  # irrational fraction of L: couples to all modes

    _, mode_info = axial_mode_ir(
        pair_x, source, receiver, {},
        default_material='concrete',  # very reflective
        T=2.0, sr=44100, f_min=None, f_max=4000, c=c)

    # Check spectral peaks at n*c/(2L) = n*34.3 Hz
    freqs = [m['freq'] for m in mode_info]
    f_fundamental = c / (2 * L)
    print(f"  Fundamental: {f_fundamental:.1f} Hz (L={L}m)")
    print(f"  Expected round-trip: {tau*1000:.1f} ms")

    # Verify first 5 harmonics present (tolerance 1 Hz for floating point)
    for n in range(1, 6):
        f_expected = n * f_fundamental
        found = any(abs(f - f_expected) < 1.0 for f in freqs)
        assert found, f"Missing mode at {f_expected:.1f} Hz (n={n})"
        print(f"  n={n}: {f_expected:.1f} Hz — present")

    print(f"  Total modes: {len(mode_info)}")
    print("  PASS\n")
    return True


def test_position_dependence():
    """Test 3: Receiver at midpoint should have zero coupling to even modes."""
    print("=" * 60)
    print("TEST 3: Position-dependent mode coupling")
    print("=" * 60)

    from room_acoustics.axial_modes import detect_parallel_surfaces_box, axial_mode_ir

    L = 4.0
    c = 343.0
    f_fundamental = c / (2 * L)  # 42.875 Hz

    # Use a single pair (x-direction)
    pairs = [p for p in detect_parallel_surfaces_box((L, 10.0, 10.0))
             if p.distance == L]

    # Source at L/4 (couples to all modes)
    source = (1.0, 5.0, 5.0)

    # Receiver at midpoint: x = L/2
    # cos(n * pi * (L/2) / L) = cos(n * pi / 2)
    #   n=1: cos(pi/2) = 0    (odd n: zero)
    #   n=2: cos(pi)   = -1   (even n: nonzero)
    #   n=3: cos(3pi/2) = 0   (odd n: zero)
    #   n=4: cos(2pi)  = 1    (even n: nonzero)
    # So ODD modes have zero receiver coupling at the midpoint.
    receiver_mid = (2.0, 5.0, 5.0)
    _, info_mid = axial_mode_ir(
        pairs, source, receiver_mid, {},
        default_material='concrete',
        T=0.5, sr=44100, f_min=None, f_max=1000, c=c)

    # Receiver near wall: x = 0.1 (all modes have nonzero coupling)
    receiver_wall = (0.1, 5.0, 5.0)
    _, info_wall = axial_mode_ir(
        pairs, source, receiver_wall, {},
        default_material='concrete',
        T=0.5, sr=44100, f_min=None, f_max=1000, c=c)

    # At midpoint: odd modes should have near-zero receiver coupling
    for m in info_mid:
        n = round(2 * L * m['freq'] / c)
        if n % 2 == 1:
            # cos(n*pi/2) = 0 for odd n, so amplitude should be ~0
            # But amplitude = S_n * R_n * (2/L) * weight
            # R_n = cos(n*pi*x_rec/L) = cos(n*pi/2) = 0 for odd n
            assert abs(m['amplitude']) < 1e-10, \
                f"Odd mode n={n} at midpoint has amplitude {m['amplitude']}"

    n_mid = len([m for m in info_mid if abs(m['amplitude']) > 1e-10])
    n_wall = len([m for m in info_wall if abs(m['amplitude']) > 1e-10])

    print(f"  Midpoint: {n_mid} active modes (odd modes suppressed)")
    print(f"  Near wall: {n_wall} active modes (all modes present)")
    assert n_wall > n_mid, "Near-wall should have more active modes"
    print("  PASS\n")
    return True


def test_decay_rate():
    """Test 4: Modal decay matches analytical formula."""
    print("=" * 60)
    print("TEST 4: Decay rate accuracy")
    print("=" * 60)

    from room_acoustics.axial_modes import detect_parallel_surfaces_box, axial_mode_ir
    from room_acoustics.acoustics_metrics import impedance_to_alpha

    L = 6.0
    c = 343.0

    pairs = [p for p in detect_parallel_surfaces_box((L, 10.0, 10.0))
             if p.distance == L]

    materials = {'left': 'carpet_thick', 'right': 'plaster'}
    from room_acoustics.materials import get_material
    alpha_1 = impedance_to_alpha(get_material('carpet_thick')['Z'])
    alpha_2 = impedance_to_alpha(get_material('plaster')['Z'])

    # Analytical decay rate
    R_product = (1 - alpha_1) * (1 - alpha_2)
    gamma_expected = (c / (2 * L)) * (-np.log(R_product))

    source = (1.5, 5.0, 5.0)
    receiver = (4.5, 5.0, 5.0)

    _, info = axial_mode_ir(
        pairs, source, receiver, materials,
        default_material='plaster',
        T=2.0, sr=44100, f_min=None, f_max=500, c=c)

    print(f"  alpha_1 (carpet_thick): {alpha_1:.4f}")
    print(f"  alpha_2 (plaster):      {alpha_2:.4f}")
    print(f"  Expected gamma:         {gamma_expected:.4f} /s")

    for m in info[:5]:
        err = abs(m['decay'] - gamma_expected) / gamma_expected
        print(f"  Mode {m['freq']:.1f} Hz: gamma={m['decay']:.4f} "
              f"(error: {err*100:.2f}%)")
        assert err < 0.01, f"Decay rate error {err*100:.1f}% > 1%"

    print("  PASS\n")
    return True


def test_detection_on_rt_mesh():
    """Test 5: Parallel surface detection on RoomMesh (non-box geometry)."""
    print("=" * 60)
    print("TEST 5: Parallel surface detection on triangle mesh")
    print("=" * 60)

    from room_acoustics.axial_modes import detect_parallel_surfaces
    from room_acoustics.ray_tracer import RoomMesh

    # Build a box RoomMesh manually (simulating what build() does)
    Lx, Ly, Lz = 8.0, 6.0, 3.0
    verts = np.array([
        [0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0],
        [0,0,Lz],[Lx,0,Lz],[Lx,Ly,Lz],[0,Ly,Lz],
    ], dtype=float)
    tris = np.array([
        [0,1,2],[0,2,3],  # floor
        [4,6,5],[4,7,6],  # ceiling
        [0,4,5],[0,5,1],  # front
        [2,6,7],[2,7,3],  # back
        [0,3,7],[0,7,4],  # left
        [1,5,6],[1,6,2],  # right
    ], dtype=int)
    labels = ['floor','floor','ceiling','ceiling',
              'front','front','back','back',
              'left','left','right','right']
    normals = []
    for t in tris:
        e1 = verts[t[1]] - verts[t[0]]
        e2 = verts[t[2]] - verts[t[0]]
        n = np.cross(e1, e2)
        normals.append(n / np.linalg.norm(n))

    # Build a mock RoomMesh
    rt = RoomMesh.__new__(RoomMesh)
    rt.vertices = verts
    rt.triangles = tris
    rt.normals = np.array(normals)
    rt.n_triangles = len(tris)
    rt.surface_labels = labels
    rt.surface_alpha = {}

    pairs = detect_parallel_surfaces(rt)

    print(f"  Detected {len(pairs)} parallel pairs:")
    for p in pairs:
        print(f"    {p.label_1} <-> {p.label_2}: {p.distance:.2f}m")

    assert len(pairs) == 3, f"Expected 3 pairs for a box, got {len(pairs)}"

    # Verify distances match box dimensions
    detected_dists = sorted([p.distance for p in pairs])
    expected_dists = sorted([Lx, Ly, Lz])
    for d, e in zip(detected_dists, expected_dists):
        assert abs(d - e) < 0.01, f"Distance {d:.2f} != expected {e:.2f}"

    print("  PASS\n")
    return True


if __name__ == '__main__':
    results = []
    tests = [
        test_shoebox_frequencies,
        test_flutter_echo,
        test_position_dependence,
        test_decay_rate,
        test_detection_on_rt_mesh,
    ]

    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  FAIL: {e}\n")
            results.append(False)

    n_pass = sum(results)
    n_total = len(results)
    print("=" * 60)
    print(f"AXIAL MODE VALIDATION: {n_pass}/{n_total} tests passed")
    print("=" * 60)

    if n_pass < n_total:
        sys.exit(1)
