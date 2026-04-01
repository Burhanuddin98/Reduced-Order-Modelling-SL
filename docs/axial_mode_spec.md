# Axial Mode Engine — Design Specification

**Last updated:** 2026-04-01
**Status:** Proposed — not yet implemented
**Target file:** `room_acoustics/axial_modes.py`

---

## Motivation

The hybrid platform's weakest link is high-frequency accuracy: 24-33% T30 error at 2-4 kHz. The ray tracer captures diffuse energy decay but misses **coherent resonant structure** — the specific frequencies that ring out between parallel surfaces.

This is audible as:
- **Flutter echo** — the distinct repetitive "tat-tat-tat" between hard parallel walls
- **Frequency coloring** — a metallic or ringy character at specific pitches
- **Comb filtering** — notches and peaks at multiples of c/(2L)

These phenomena are dominated by **axial modes** between parallel surface pairs. The key insight: a 3D room's axial behavior between any pair of parallel surfaces is a **1D problem** with an analytical solution. No mesh needed, no eigensolve, no DOF scaling. The answer is exact and instantaneous.

---

## Physics

### Axial modes between two parallel surfaces

For two parallel surfaces separated by distance `L`:

```
Resonant frequencies:  f_n = n * c / (2L)    for n = 1, 2, 3, ...
Mode shapes:           phi_n(x) = cos(n * pi * x / L)
```

where `x` is the perpendicular distance from surface 1 to the observation point.

### Modal decay

Each axial mode decays based on the absorption of both surfaces:

```
Frequency-independent (FI):
  gamma_n = (c / 2L) * (-ln((1 - alpha_1)(1 - alpha_2)))

  where alpha_1, alpha_2 = absorption coefficients of the two surfaces

Frequency-dependent (Miki):
  gamma_n = (c / 2L) * (-ln((1 - alpha_1(f_n))(1 - alpha_2(f_n))))

  where alpha(f) comes from Miki impedance model per surface material
```

### Source and receiver coupling

```
Source coupling:     S_n = cos(n * pi * x_src / L)
Receiver coupling:   R_n = cos(n * pi * x_rec / L)
Amplitude:           A_n = S_n * R_n * (2 / L)

IR contribution per mode:
  p_n(t) = A_n * exp(-gamma_n * t) * cos(2 * pi * f_n * t)
```

### Solid angle weighting

Not all surface pairs contribute equally. A pair of large, close surfaces dominates over small, distant ones. Weight each pair's contribution by the solid angle subtended by the surfaces as seen from the receiver:

```
weight_pair ≈ min(A_1, A_2) / (4 * pi * L^2)

where A_1, A_2 = areas of the two surfaces
```

This prevents small surface patches from over-contributing.

---

## Algorithm

### Step 1: Parallel surface detection

Given the room's boundary triangles (already extracted for ray tracing), detect pairs of surfaces with opposing parallel normals.

```
Input:  boundary triangles with normals and surface labels
Output: list of (surface_1, surface_2, distance L, normal direction)

Algorithm:
  1. Group triangles by surface label
  2. For each surface, compute area-weighted average normal
  3. For each pair of surfaces (i, j):
     a. Check if normals are anti-parallel: dot(n_i, n_j) < -cos(tolerance)
        tolerance ≈ 5 degrees (0.996 dot product threshold)
     b. Compute distance L:
        - Project centroid of surface j onto plane of surface i
        - L = distance along the shared normal direction
     c. Verify surfaces actually overlap (project onto shared plane,
        check intersection area > 0)
     d. Record: {pair: (i,j), L: distance, normal: direction,
                 area: overlap area, mat_1: material_i, mat_2: material_j}
```

For shoebox rooms this trivially finds 3 pairs (floor/ceiling, left/right, front/back). For arbitrary geometry it finds any parallel opposing surfaces, including partial ones.

### Step 2: Receiver/source positioning

For each detected pair:

```
x_rec = dot(receiver_pos - centroid_surface_1, normal)   [0 to L]
x_src = dot(source_pos - centroid_surface_1, normal)     [0 to L]

If x_rec < 0 or x_rec > L: receiver is not between this pair → skip
If x_src < 0 or x_src > L: source is not between this pair → skip
```

### Step 3: Axial mode synthesis

For each valid pair, compute axial modes up to `f_max`:

```
n_max = floor(2 * L * f_max / c)

For n = 1 to n_max:
  f_n = n * c / (2L)
  alpha_1 = get_absorption(material_1, f_n)   # from materials.py
  alpha_2 = get_absorption(material_2, f_n)
  gamma_n = (c / (2L)) * (-ln((1-alpha_1)(1-alpha_2)))
  S_n = cos(n * pi * x_src / L)
  R_n = cos(n * pi * x_rec / L)
  A_n = S_n * R_n * (2 / L) * weight_pair

  ir_axial += A_n * exp(-gamma * t) * cos(2*pi*f_n * t)
```

### Step 4: Integration into hybrid pipeline

```
Current:  ir_total = ir_low (modal ROM) + ir_high (ray tracer) + ir_ism

Proposed: ir_total = ir_low (modal ROM)
                   + ir_axial (axial modes, f_cross to f_max)
                   + ir_diffuse (ray tracer, high-passed)
                   + ir_ism (early reflections)

The axial modes fill in the coherent resonant peaks that the ray tracer
misses. The ray tracer still provides the diffuse energy envelope.

Blending:
  - ir_axial: band-pass filtered [f_cross, f_max]
  - ir_diffuse: remains as-is (ray tracer output)
  - Level matching: normalize axial contribution so total energy
    matches expected Sabine/Eyring decay at each octave band
```

---

## Interface

```python
# axial_modes.py

def detect_parallel_surfaces(mesh, boundary_data, angle_tolerance=5.0):
    """
    Detect pairs of parallel opposing surfaces in the room geometry.

    Args:
        mesh: room mesh with boundary triangles and normals
        boundary_data: dict of surface_label → triangle list
        angle_tolerance: max angle deviation from parallel (degrees)

    Returns:
        list of ParallelPair namedtuples:
          {label_1, label_2, distance, normal, overlap_area, mat_1, mat_2}
    """

def axial_mode_ir(pairs, source, receiver, materials, T=3.5, sr=44100,
                  f_min=None, f_max=8000, c=343.0):
    """
    Synthesize IR from axial modes between parallel surface pairs.

    Args:
        pairs: output of detect_parallel_surfaces()
        source: (x, y, z) source position
        receiver: (x, y, z) receiver position
        materials: dict of surface_label → material_name
        T: IR duration [seconds]
        sr: sample rate [Hz]
        f_min: lower frequency limit (default: use f_cross from room)
        f_max: upper frequency limit [Hz]
        c: speed of sound [m/s]

    Returns:
        ir_axial: numpy array, axial mode impulse response
        mode_info: list of {freq, decay, amplitude, pair} per mode
    """
```

### Integration in Room API (room.py)

```python
# In Room.impulse_response():

# After build, detect parallel surfaces (one-time, cached)
if self._parallel_pairs is None:
    self._parallel_pairs = detect_parallel_surfaces(
        self.mesh, self.boundary_data)

# During IR query:
ir_axial = axial_mode_ir(
    self._parallel_pairs, source, receiver,
    self.materials, T=T, sr=sr, f_min=f_cross, f_max=f_max)
```

---

## Validation plan

### Test 1: Shoebox verification
- Compute axial modes for a box room analytically
- Compare against eigenfrequencies from the full 3D eigensolve
- Axial frequencies must match exactly: f_n = n*c/(2L)
- Axial decay rates must match modal ROM decay rates for axial modes

### Test 2: Flutter echo reproduction
- Two hard parallel walls (Z = 100,000), 5 m apart
- Source and receiver at different positions between walls
- Verify: IR shows clear periodic reflections at tau = 2L/c = 29 ms
- Verify: spectral peaks at 34.3, 68.6, 102.9, ... Hz

### Test 3: Hybrid improvement on BRAS CR2
- Run full hybrid with and without axial modes
- Compare octave-band T30 at 1-4 kHz vs measured
- Target: reduce high-freq T30 error from 24-33% toward <15%

### Test 4: Non-shoebox room
- L-shaped room (has some parallel surfaces, not all)
- Verify detection finds the correct pairs
- Verify axial modes don't contribute for non-parallel surfaces
- Check that total IR is physically plausible

---

## Computational cost

```
Detection: O(S^2) where S = number of surfaces (one-time, negligible)
Synthesis: O(n_pairs * n_modes * n_samples)
  Typical: 3 pairs * 200 modes * 154K samples = ~92M multiply-adds
  Time: <10 ms (pure numpy vectorized)

Total overhead: negligible compared to eigensolve or ray tracing
```

---

## Scope and limitations

**What this captures:**
- Axial modes (1D standing waves between parallel surfaces)
- Flutter echo (periodic reflections)
- Frequency-dependent decay per mode (via Miki model)
- Position-dependent response (source and receiver coupling)

**What this does NOT capture:**
- Tangential modes (2D: involving two pairs of parallel surfaces simultaneously)
- Oblique modes (3D: involving three pairs)
- Diffraction around edges or openings
- Non-specular scattering
- Coupling between axial mode sets from different pairs

**Why axial modes are sufficient for the high-frequency gap:**
Axial modes are the strongest room modes at any given frequency. Tangential modes are 3 dB weaker, oblique modes 6 dB weaker (Bolt, 1946). The ray tracer already captures the diffuse field — the axial modes add only the coherent resonant peaks that the ray tracer smears out. The combination should be more accurate than either alone.
