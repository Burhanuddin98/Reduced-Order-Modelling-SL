# Reduced Order Modelling — Architecture Reference

**Last updated:** 2026-04-01

---

## System overview

Hybrid room acoustics engine: **Modal ROM** (low-frequency wave simulation) + **ISM/Ray Tracer** (high-frequency geometric acoustics), blended into a broadband impulse response. Python frontend + C/CUDA backend.

```
                         +-----------------+
                         |    Room API     |  room.py
                         |  (orchestrator) |
                         +--------+--------+
                                  |
      +------------+--------------+--------------+------------+
      |            |              |              |             |
+-----v------+ +--v---------+ +-v----------+ +-v---------+ +-v---------+
| Modal ROM  | | Axial Modes| | Ray Tracer | |    ISM    | |  Metrics  |
| 0-f_cross  | | f_cross to | | f_cross to | |  early    | | ISO 3382  |
| modal_rom  | | f_max      | | Nyquist    | | reflect.  | | acoustics |
| .py        | | axial_modes| | ray_tracer | | image_src | | _metrics  |
| (analytical| | .py        | | .py/.c     | | .py       | | .py       |
|  eigenmode)| | (analytical| | (stochastic| | (shoebox  | |           |
+-----+------+ |  1D pairs) | |  energy)   | |  only)    | +-----------+
      |         +--+---------+ +-+----------+ +-+---------+
      |            |             |              |
      +------------+-------------+--------------+
                                 |
                      +----------v----------+
                      |  Crossover Blend    |
                      |  ir_low + ir_axial  |
                      |  + ir_diffuse       |
                      |  + ir_ism           |
                      +---------------------+
```

---

## Module map

### Layer 1: User API

| File | LOC | Role |
|------|-----|------|
| `room.py` | 654 | **Main entry point.** `Room` class: geometry → mesh → build → IR. `ImpulseResponse` class: metrics, WAV export, auralization |

### Layer 2: Solvers

| File | LOC | Role |
|------|-----|------|
| `solvers.py` | 1085 | FOM/ROM time-domain solvers (p-v, p-Phi), PSD/modified-PSD basis construction, Schur stabilization, Miki impedance |
| `modal_rom.py` | 170 | Eigenmode computation + analytical per-mode time evolution (no time-stepping) |
| `freq_domain.py` | 565 | Helmholtz frequency-domain solver + greedy ROM basis |
| `laplace_domain.py` | 437 | Laplace-domain (s = sigma + j*omega) solver, frequency-dependent BCs |
| `lanczos_rom.py` | 260 | Lanczos eigenvalue solver with UMFPACK factorization reuse |

### Layer 3: Spatial discretization

| File | LOC | Role |
|------|-----|------|
| `sem.py` | 498 | Structured mesh (RectMesh2D, BoxMesh3D), GLL quadrature, Kronecker assembly |
| `unstructured_sem.py` | 991 | Unstructured quad/hex mesh, isoparametric mapping, element-by-element assembly |
| `tet_sem.py` | 815 | P=2 tetrahedral elements, Gmsh import, HRZ mass lumping, optional Numba JIT |
| `geometry.py` | 368 | Room geometry definitions, Gmsh interface for automated meshing |
| `gmsh_tet_import.py` | 271 | Gmsh .msh file parser, surface mesh extraction |

### Layer 4: Unified modal synthesis + geometric acoustics

| File | LOC | Role |
|------|-----|------|
| `unified_modes.py` | 645 | **Core orchestrator.** Provider registry, merge/dedup, single-pass synthesis (Numba JIT) |
| `analytical_modes.py` | ~300 | Exact box room modes (axial+tangential+oblique), Kuttruff decay, Numba recursive oscillator |
| `generalized_modes.py` | ~280 | Non-box room modes via perpendicular parallel pair detection |
| `axial_modes.py` | ~400 | Parallel surface detection + 1D axial mode synthesis with 3D coupling loss |
| `statistical_modes.py` | ~150 | Weyl-density statistical mode fill for irregular rooms above eigensolve range |
| `image_source.py` | 322 | Image Source Method for shoebox rooms (early reflections + Eyring diffuse tail) |
| `ray_tracer.py` | 287 | Python stochastic ray tracer (Moller-Trumbore intersection, scattering) |

### Layer 5: Physics / materials

| File | LOC | Role |
|------|-----|------|
| `materials.py` | 229 | 22-material database, Miki impedance model, per-surface assignment |
| `impedance_fit.py` | 218 | Vector fitting for frequency-dependent material curves |

### Layer 6: Output / metrics

| File | LOC | Role |
|------|-----|------|
| `acoustics_metrics.py` | 292 | ISO 3382: T30, T20, EDT, C80, D50, TS via Schroeder backward integration |
| `visualize.py` | 406 | Pressure field plots, energy decay curves, T30 maps |
| `results_io.py` | 67 | JSON export of metrics and metadata |

### Layer 7: C/CUDA backend

| File | LOC | Role |
|------|-----|------|
| `engine/src/eigensolve.c` | 379 | Shift-invert Lanczos + UMFPACK (100x faster than scipy ARPACK) |
| `engine/src/ray_tracer.c` | 261 | C ray tracer (17x faster than Python), Moller-Trumbore |
| `engine/src/skp_reader.c` | 233 | SketchUp SKP format parser |
| `engine/include/room_engine.h` | 145 | C API: room lifecycle, IR computation, metrics, WAV |
| `solver_core/helmholtz_gpu.cu` | ~300 | CUDA Helmholtz solver (cuSOLVER + cuSPARSE) |
| `solver_core/helmholtz_py.py` | 244 | ctypes wrapper for GPU solver |
| `solver_core/helmholtz_umfpack.c` | ~200 | UMFPACK sparse frequency sweep |

### Validation

| File | LOC | Role |
|------|-----|------|
| `validate.py` | 773 | 9 tests: rectangular rooms vs analytical modal expansion, L-shaped Dauge benchmark |
| `validate_3d.py` | 174 | 3D box eigenfrequencies vs formula |
| `validate_eigenfrequencies.py` | 453 | Eigenfrequency accuracy, Dauge L-shape singular corner |
| `validate_unstructured.py` | 394 | 2D unstructured quad mesh validation |
| `validate_unstructured_3d.py` | 328 | 3D extruded hex validation |
| `validate_tet_3d.py` | 497 | Tet P=2 convergence study, HRZ lumping validation |
| `test_bras_cr2.py` | 347 | BRAS benchmark: 8.4x6.7x3.0 m seminar room, T30/EDT/C80 vs measured |
| `test_materials.py` | 206 | Material library + Miki model tests |

---

## Data flow

### Build pipeline (one-time, expensive)

```
Room.from_box/geo/stl/polygon()
  |
  v
_build_mesh()
  +-- from_box:    BoxMesh3D(Lx,Ly,Lz,Nex,Ney,Nez,P) → structured GLL grid
  +-- from_geo:    Gmsh API → TetMesh3D(nodes, tets, boundary_data)
  +-- from_stl:    Gmsh API → TetMesh3D(nodes, tets, boundary_data)
  +-- from_polygon: RoomGeometry → UnstructuredHexMesh3D (2D extruded)
  |
  v
_assemble()
  +-- structured:    Kronecker products (M_z ⊗ M_y ⊗ M_x, etc.)
  +-- unstructured:  element-by-element loop (isoparametric Jacobian)
  +-- tet:           element-by-element + HRZ mass lumping
  |
  → Output: {M_diag, M_inv, S, B_total, Sx, Sy, [Sz]}
  |
  v
compute_room_modes(ops, n_modes)
  → scipy eigsh (shift-invert Lanczos)
  → eigenvalues (n_modes,), eigenvectors (N, n_modes), frequencies [Hz]
```

### Query pipeline (fast, per source/receiver pair)

```
Room.impulse_response(source, receiver)
  |
  +--[1] Modal ROM (0 to f_cross Hz) ─────────────────────────+
  |   modal_ir(eigenvalues, eigenvectors, bc_params)           |
  |     per mode i:                                            |
  |       A_i = source_coupling * receiver_coupling            |
  |       gamma_i = modal decay rate (from impedance)          |
  |       omega_d = sqrt(omega_i^2 - gamma_i^2)               |
  |       p_i(t) = A_i * exp(-gamma_i*t) * cos(omega_d*t)     |
  |     → ir_modal = sum over modes                            |
  |     → low-pass filter at f_cross                           |
  |     → ir_low                                               |
  |                                                            |
  +--[2] Axial Modes (f_cross to f_max) ──────────────────────+
  |   axial_mode_ir(parallel_pairs, source, receiver, ...)     |
  |     per parallel surface pair:                             |
  |       detect distance L, receiver/source positions         |
  |       per mode n: f_n = n*c/(2L), analytical decay+amp    |
  |     → ir_axial (coherent resonant peaks + flutter echo)    |
  |     → band-pass filter [f_cross, f_max]                    |
  |                                                            |
  +--[3] Ray Tracer (f_cross to Nyquist) ─────────────────────+
  |   _ray_trace_c(source, receiver, n_rays, max_bounces, T)  |
  |     → C DLL (or Python fallback)                           |
  |     → reflectogram (energy vs time bins)                   |
  |     → reflectogram_to_ir (modulate noise by envelope)      |
  |     → high-pass filter at f_cross                          |
  |     → ir_diffuse (level-matched to ir_low)                 |
  |                                                            |
  +--[4] ISM (early reflections, box rooms only) ─────────────+
  |   image_sources_shoebox(Lx, Ly, Lz, src, rec, ...)        |
  |     → mirror sources, distance attenuation, absorption     |
  |     → ir_ism → high-pass + fade window → ir_ism_early      |
  |                                                            |
  v                                                            v
  ir_total = ir_low + ir_axial + ir_diffuse + ir_ism_early
  |
  v
  ImpulseResponse(ir_total, sr)
    .T30, .EDT, .C80, .D50, .TS   (ISO 3382 metrics)
    .save_wav(path)                (WAV export)
    .auralize(dry_audio, output)   (convolution)
```

---

## Operator assembly

### Governing equations (p-Phi formulation)

```
dp/dt   = rho * c^2 * M^{-1} * S * Phi  + boundary terms
dPhi/dt = -(1/rho) * p
```

Energy `E = (rho/2)|grad(Phi)|^2 + (1/(2*rho*c^2))*p^2` is exactly conserved for PR boundaries.

### Assembled operators

| Operator | Size | Storage | What it represents |
|----------|------|---------|-------------------|
| M_diag | (N,) | dense diagonal | Mass matrix (GLL quadrature weights * Jacobian) |
| M_inv | (N,) | dense diagonal | Inverse mass (1/M_diag) |
| S | (N,N) | sparse CSR | Stiffness (Laplacian weak form) |
| B_total | (N,) | dense diagonal | Boundary mass (nonzero only at boundary nodes) |
| Sx, Sy, Sz | (N,N) | sparse CSR | Directional gradient operators |

### Element types

| Type | Order | Nodes/element | Best for |
|------|-------|---------------|----------|
| Hex (structured) | P=4 | 125 | Box rooms (fastest — Kronecker assembly) |
| Quad (unstructured) | P=4 | 25 | 2D arbitrary shapes |
| Hex (extruded) | P=4 | 125 | Extruded 2D polygons |
| Tet | P=2 | 10 | Arbitrary 3D geometry (STL/OBJ/Gmsh) |

---

## Boundary conditions

| Type | Physics | Implementation |
|------|---------|----------------|
| PR (perfectly reflecting) | Rigid walls, zero energy loss | dp/dt += 0 at boundary (no boundary term) |
| FI (frequency-independent) | Fixed absorption at all frequencies | dp/dt += -(rho*c^2/Z) * B * p (damping proportional to Z) |
| LR (locally reacting) | Frequency-dependent absorption (Miki model) | ADE: auxiliary variables march alongside wave eq |

### Miki impedance model
```
Z_c = rho*c * (1 + 0.0699*X^{-0.632} - i*0.1071*X^{-0.618})
k_c = (2*pi*f/c) * (1 + 0.1093*X^{-0.618} - i*0.1597*X^{-0.683})
Z_s = -i * Z_c / tan(k_c * d)

where X = f/sigma (frequency / flow resistivity)
```

---

## ROM methods

### 1. Modal ROM (primary — production-ready)

Eigenmodes of the generalized problem `S*phi = lambda*M*phi`. Each mode has:
- Eigenfrequency: `f_i = sqrt(lambda_i) * c / (2*pi)`
- Modal decay (FI): `gamma_i = (rho*c^2/2) * phi_i^T * (B/Z) * phi_i`
- Analytical solution: `p_i(t) = A_i * exp(-gamma_i*t) * cos(omega_d_i*t + phase_i)`

No time-stepping required. Zero numerical dispersion. 0.2s for 3.5s IR.

### 1b. Unified Modal Synthesis (unified_modes.py)

Plug-and-play architecture where multiple engines contribute modes as `(frequency, amplitude, decay_rate)` tuples to a shared list. Confidence-based merge deduplicates overlapping modes. Single synthesis pass via Numba JIT recursive oscillator (2.9s for 37K modes). No crossover filters.

Available providers:
- **AnalyticalModesProvider** (confidence 1.0): exact box room modes (axial+tangential+oblique)
- **ModalROMProvider** (confidence 0.95): eigensolve modes with exact decay
- **GeneralizedModesProvider** (confidence 0.75): non-box modes from perpendicular pairs
- **AxialModesProvider** (confidence 0.65): 1D parallel surface modes with 3D coupling
- **StatisticalModesProvider** (confidence 0.4): Weyl-density fill for irregular rooms
- ISM early reflections added separately (discrete arrivals, not modal)

Adding a new engine: implement `provide_modes()` → register with `synth.register()`. That's it.

### 2. PSD basis ROM (time-domain, structure-preserving)

Proper Symplectic Decomposition on [p | Phi | p_b] snapshots from FOM. Reduced system evolved via RK4 with propagator matrix + Schur eigenvalue stabilization.

Known issue: propagator breaks symplectic structure (see AUDIT.md).

### 3. Frequency-domain ROM (Helmholtz)

Solve `(S - omega^2*M + i*omega*C)*p = f` at training frequencies. Build POD basis from solutions. ROM evaluates at new frequencies via reduced system (2r x 2r dense solve).

38,000x speedup per frequency. But basis needs 60-80+ vectors for highly resonant rooms.

### 4. Laplace-domain ROM (experimental)

Same as frequency-domain but with `s = sigma + i*omega` (sigma > 0). Better conditioned at resonances. Not fully implemented yet — see AUDIT.md for analysis.

---

## Material database

22 materials in `materials.py`. Key entries:

| Material | Z (N*s/m^3) | sigma (N*s/m^4) | Typical use |
|----------|-------------|-----------------|-------------|
| concrete | 50,000 | 20,000 | Walls, floors |
| plaster | 20,000 | 15,000 | Walls, ceilings |
| carpet_thick | 1,000 | 10,000 | Floors |
| acoustic_panel | 800 | 12,000 | Absorption treatment |
| glass | 100,000 | — | Windows |
| wood_panel | 5,000 | 8,000 | Stages, paneling |

---

## ISO 3382 metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| T30 | Schroeder decay, -5 to -35 dB, extrapolate to -60 dB | Reverberation time |
| T20 | Schroeder decay, -5 to -25 dB, extrapolate to -60 dB | Reverberation time (shorter window) |
| EDT | Schroeder decay, 0 to -10 dB, extrapolate to -60 dB | Early decay time |
| C80 | 10*log10(E_0-80ms / E_80ms+) | Clarity (music) |
| D50 | E_0-50ms / E_total | Definition (speech) |
| TS | energy-weighted mean arrival time | Center time |

---

## Threading and parallelism

| Component | Threading model |
|-----------|----------------|
| Python (room_acoustics) | Single-threaded (scipy sparse ops use BLAS internally) |
| scipy eigsh | Multi-threaded LAPACK |
| C eigensolve | Single-threaded (UMFPACK sequential LU) |
| C ray tracer | Single-threaded loop (vectorizable) |
| CUDA Helmholtz | Fully GPU-parallel (cuSOLVER + cuSPARSE) |

---

## Constants

```
C_AIR    = 343.0   m/s       Speed of sound
RHO_AIR  = 1.2     kg/m^3    Air density
rho*c    = 411.6   N*s/m^3   Impedance of air
CFL      = 0.15              Stability limit for RK4
dt       <= 0.15 * h / (c * P^2)
eps_pod  = 1e-8              SVD truncation tolerance
ppw      = 6-10              Points per wavelength (mesh quality)
```
