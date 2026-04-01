# Reduced Order Modelling — Signal Chain

**Last updated:** 2026-04-01

---

## Full pipeline (build + query)

```
                    BUILD (one-time, seconds to minutes)
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  1. GEOMETRY DEFINITION                                 │
  │     Room.from_box(Lx, Ly, Lz)      → structured hex    │
  │     Room.from_geo("room.geo")       → tet via Gmsh     │
  │     Room.from_stl("room.stl")       → tet via Gmsh     │
  │     Room.from_polygon(verts, Lz)    → extruded hex     │
  │                                                         │
  │  2. MESHING                                             │
  │     Structured:    GLL grid, (P+1)^3 nodes/element     │
  │     Unstructured:  Gmsh quad/hex, isoparametric map    │
  │     Tetrahedral:   Gmsh tet (P=2, 10-node), HRZ lump  │
  │                                                         │
  │  3. OPERATOR ASSEMBLY                                   │
  │     Mass M (diagonal), Stiffness S (sparse), Boundary B│
  │     Gradient operators Sx, Sy, Sz (sparse)             │
  │     Structured: Kronecker products (instant)            │
  │     Unstructured: element-by-element loop               │
  │     Tet: element loop + optional Numba JIT              │
  │                                                         │
  │  4. EIGENSOLVE                                          │
  │     Generalized eigenproblem: S*phi = lambda*M*phi      │
  │     Method: shift-invert Lanczos (scipy eigsh or C)     │
  │     Output: n_modes eigenvalues + eigenvectors          │
  │     Typical: 200-400 modes, covering 0 to ~400 Hz      │
  │                                                         │
  │  5. MATERIAL ASSIGNMENT                                 │
  │     Per-surface material from 22-entry database         │
  │     Frequency-independent Z or Miki model Z(f)          │
  │     Mapped to boundary nodes via surface labels         │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

                    QUERY (fast, per source/receiver)
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  6. MODAL ROM (0 to f_cross Hz)                         │
  │     Source coupling: a_i = Gaussian(x_src) . phi_i      │
  │     Receiver coupling: phi_i(x_rec)                     │
  │     Modal decay: gamma_i from boundary impedance        │
  │     Analytical: p_i(t) = A_i * e^{-gamma*t} * cos(w*t) │
  │     Sum all modes → ir_modal                            │
  │     Low-pass filter at f_cross → ir_low                 │
  │                                                         │
  │  7. AXIAL MODES (f_cross to f_max)                       │
  │     Detect parallel surface pairs (cached after build)  │
  │     Per pair: distance L, source/receiver positions     │
  │     Per mode n: f_n = n*c/(2L), decay from materials   │
  │     Analytical: A_n * exp(-gamma*t) * cos(2*pi*f_n*t)  │
  │     Sum over all pairs and modes → ir_axial             │
  │     Band-pass filter [f_cross, f_max]                   │
  │     Captures: flutter echo, coherent resonant peaks     │
  │                                                         │
  │  8. RAY TRACER (f_cross to Nyquist)                     │
  │     Launch n_rays from source (random directions)       │
  │     Moller-Trumbore triangle intersection               │
  │     Energy loss: E *= (1 - alpha) per bounce            │
  │     Specular + diffuse scattering at walls              │
  │     Record energy at receiver → reflectogram            │
  │     Modulate noise by envelope → ir_ray                 │
  │     High-pass filter at f_cross → ir_diffuse            │
  │     Captures: diffuse energy envelope, late decay       │
  │                                                         │
  │  9. ISM (early reflections, shoebox only)               │
  │     Mirror sources across walls (up to max_order)       │
  │     Distance attenuation + absorption per bounce        │
  │     → ir_ism → high-pass + fade → ir_ism_early          │
  │                                                         │
  │  10. CROSSOVER BLEND                                    │
  │     ir_total = ir_low + ir_axial + ir_diffuse           │
  │               + ir_ism_early                            │
  │     Level matching between components                   │
  │                                                         │
  │  11. METRICS (ISO 3382)                                  │
  │      Schroeder backward integration → decay curve       │
  │      Linear regression → T30, T20, EDT                  │
  │      Energy ratios → C80, D50, TS                       │
  │                                                         │
  │  12. OUTPUT                                             │
  │      ImpulseResponse object with .T30, .EDT, .C80       │
  │      .save_wav(path) → WAV file                         │
  │      .auralize(dry_audio, output) → convolved audio     │
  │      JSON export via results_io                         │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

---

## Alternative solver paths

### Frequency-domain (Helmholtz)
```
For each frequency omega:
  Build system: A(omega) = S - omega^2*M + i*omega*C
  Solve: A(omega) * p = f
  Record: H(omega) = p[receiver]

IFFT → time-domain IR
```
Used for: frequency sweeps, parametric material studies, ROM basis training.

### Time-domain FOM (full order model)
```
State vector: [p; Phi]  (2N unknowns)
Time integration: RK4, dt <= CFL * h / (c * P^2)
Per step: sparse matvec S @ state

BC types:
  PR: no boundary term (energy constant)
  FI: damping term -(rho*c^2/Z) * B * p
  LR: ADE auxiliary variables alongside main system
```
Used for: snapshot generation (PSD basis training), reference solutions.

### Time-domain ROM (reduced order)
```
Project onto PSD basis: [p; Phi] ≈ Psi * z  (2r unknowns, r << N)
Propagator: P = I + dt*A_r + ... (dense, precomputed)
Per step: z_new = P @ z  (dense matvec, microseconds)
Reconstruct: p = Psi * z

Stabilization: Schur decomposition, clip |eigenvalues| <= 1
```
Used for: fast time-domain simulation when analytical modal solution isn't applicable.

---

## Crossover design

```
  Eigensolve      Analytical     Axial       Statistical     ISM
  (exact modes)   (box exact)    (parallel)  (Weyl density)  (early refl)
  0-f_mesh        0-f_max        any room    any room        box rooms
  conf=0.95       conf=1.0       conf=0.65   conf=0.4        non-modal
       |               |             |             |              |
       +-------+-------+------+------+             |              |
               |              |                    |              |
         Unified Mode List (merge by confidence)   |              |
               |                                   |              |
         Single synthesis pass (Numba JIT)         |              |
               |                                   |              |
               +------- + add ISM separately ------+              |
                                    |                             |
                              broadband IR

f_cross ≈ 400 Hz (Schroeder frequency of typical room)
```

The crossover frequency is approximately the Schroeder frequency:
```
f_s = 2000 * sqrt(T60 / V)

where T60 = reverberation time [s], V = room volume [m^3]
```

Below f_s: room modes dominate (wave behavior, modal ROM accurate).
Above f_s: diffuse field (geometric behavior, ray tracer accurate).

The unified modal synthesis replaces crossover-filter blending. Each engine
contributes modes as (frequency, amplitude, decay_rate) tuples. Confidence-based
merge deduplicates overlapping modes — higher-confidence engines win. One synthesis
pass produces the IR via Numba JIT recursive oscillator (no exp/cos calls in inner loop).

For box rooms: analytical modes provide exact coverage at any frequency.
For irregular rooms: eigensolve (low freq) + axial (parallel surfaces) + statistical
(Weyl density fill) combine to cover the full bandwidth.
ISM early reflections are added separately (discrete arrivals, not standing waves).
