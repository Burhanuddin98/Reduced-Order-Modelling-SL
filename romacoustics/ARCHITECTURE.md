# romacoustics v0.2 — Architecture Plan

## Current state (v0.1)

Works for box rooms with uniform boundaries. Validated, pip-installable, clean API. But:
- Box geometry only (structured Kronecker mesh)
- Same material on every wall
- No error estimation
- Weeks parameters are magic numbers

## Target state (v0.2)

A CHORAS dev can load a Gmsh mesh of an arbitrary room, assign different materials to each surface, build a parametric ROM, query it with error bounds, and get ISO 3382 metrics.

```python
from romacoustics import Room

room = Room.from_gmsh('concert_hall.msh')
room.set_source(4.0, 3.0, 1.5)
room.set_receiver(2.0, 5.0, 1.2)
room.set_material('stage_floor', 'wood_panel')
room.set_material('walls', 'plaster')
room.set_material('ceiling', 'acoustic_panel')
room.set_material('seats', 'upholstered_seats')

ir = room.solve(f_max=1000, t_max=1.0)
print(ir.T30, ir.C80)

rom = room.build_rom(material_variations={
    'ceiling': ['acoustic_panel', 'foam', 'plaster'],
    'walls': ['plaster', 'concrete', 'curtain_heavy'],
})
# 3^2 = 9 combinations, each instant
ir2 = rom.solve(ceiling='foam', walls='curtain_heavy')
print(f'T30={ir2.T30:.2f}s, error_bound={ir2.error_bound:.2e}')
```

---

## What needs to change

### 1. Per-surface boundary mass (the critical missing piece)

Currently `assemble_2d/3d` returns one `B_total` diagonal. For per-surface materials, we need `B_labels`: a dict mapping surface label → diagonal boundary mass vector.

**Source:** The existing `romac/affine.py` already does this for structured meshes via `ops['B_edges']`. For unstructured meshes, need to build from boundary face data.

**Implementation:**
- `assemble_2d` and `assemble_3d` return `B_labels = {'bottom': diag, 'top': diag, ...}` alongside `B_total`
- For unstructured: group boundary faces by their Gmsh physical group label
- Affine decomposition: `Br = sum_label (c²ρ/Z_label) * B_label`
- ROM projection: project each `B_label` separately → `Br_r_label`
- Online: assemble `Br_r = sum (1/Z_label) * Br_r_label` with new Z values

**Effort:** Small. The math exists, just needs wiring.

### 2. Gmsh mesh import

Need to read `.msh` files and build SEM operators on the resulting mesh.

**Source:** `room_acoustics/gmsh_tet_import.py` reads `.msh` files. `room_acoustics/unstructured_sem.py` assembles operators on quad/hex meshes. `room_acoustics/tet_sem.py` does tets.

**Implementation:**
- `Room.from_gmsh(path)` — auto-detect element type (quad/hex/tet), build mesh, assemble operators
- Surface labels come from Gmsh physical groups → feed into per-surface boundary mass
- Require `gmsh` as optional dependency (not needed for box rooms)

**Effort:** Medium. The pieces exist but need gluing + testing on real meshes.

### 3. Material database

**Source:** `room_acoustics/materials.py` has 22 materials with impedance Z.

**Implementation:**
- `room.set_material('wall_label', 'plaster')` → looks up Z from database
- `room.set_material('wall_label', Z=8000)` → raw impedance
- `room.set_material('wall_label', sigma_flow=10000, d_mat=0.05)` → Miki model
- For ROM: parametrize over material choices per surface

**Effort:** Small. Mostly API design.

### 4. Error estimator

The standard RBM residual-based error estimator:

```
error_bound = ||r(u_rb)|| / alpha_LB(s)
```

where `r` is the FOM residual evaluated at the ROM solution, and `alpha_LB` is a lower bound on the coercivity constant (Successive Constraint Method or simplified).

**Implementation:**
- Pre-compute `||f||` and projection of residual operators during offline phase
- Online: compute residual norm from reduced quantities (no FOM solve needed)
- Return `ir.error_bound` alongside the solution

**Effort:** Medium. Standard RBM technique but needs careful implementation.

### 5. Automatic Weeks parameters

Instead of magic numbers, auto-tune (σ, b) by:
- Start with heuristic: σ = 10, b = 500×f_max
- Verify coefficient decay: if last Laguerre coefficient > 1e-6 × first, increase Ns
- If user provides a TD reference, optimize via grid search (existing `optimize_weeks_params`)

**Effort:** Small. Heuristic is good enough for most cases.

---

## File structure (v0.2)

```
romacoustics/
  __init__.py          Room, ImpulseResponse exports
  room.py              Room class (high-level API)
  ir.py                ImpulseResponse (metrics, WAV, plots)
  sem.py               Structured mesh (box 2D/3D, Kronecker)
  unstructured.py      Unstructured quad/hex/tet mesh + assembly
  gmsh_io.py           Gmsh .msh import, surface label extraction
  solver.py            Laplace FOM, Weeks ILT, Miki model
  rom.py               SVD basis, affine decomposition, ROM projection, error estimator
  materials.py         22-material database + per-surface assignment
  geometry.py          Parametric room generators (rect, L-shape, T-shape)
```

## Dependencies

**Required:** numpy, scipy, matplotlib
**Optional:** gmsh (for arbitrary geometry), numba (for tet assembly speedup)

## Priority order

1. Per-surface boundary mass + material database (unlocks real rooms)
2. Gmsh mesh import (unlocks arbitrary geometry)
3. Error estimator (unlocks trustworthy ROM queries)
4. Auto Weeks parameters (quality of life)

Items 1-2 make the package useful for CHORAS. Item 3 makes it trustworthy. Item 4 makes it pleasant.

## What we are NOT doing

- GPU acceleration (premature — scipy is fine for the target use case)
- Time-domain solver (the Laplace approach is the whole point)
- Scattering / diffraction (beyond the scope of this method)
- Arbitrary source/receiver directivity
- Parallelism (each FOM solve is independent — user can trivially parallelize with multiprocessing)
