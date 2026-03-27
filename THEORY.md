# Room Acoustics Simulation Engine — Complete Theory Guide

This document explains everything this codebase does, from the physics to the code, in plain language. You can hand this to Claude or any AI assistant and they will understand the full system.

---

## 1. What This Project Does

You clap your hands in a room. Sound waves travel outward, bounce off walls, and eventually die out. The pattern of those reflections — how loud, how fast they decay, which frequencies survive — is called the room's **impulse response (IR)**.

This project simulates that process numerically. You define a room shape, choose wall materials, place a sound source, and the code computes the impulse response at any listener position. The result is a pressure-vs-time signal that captures everything about how that room sounds.

The core innovation is **Model Order Reduction (ROM)**. A full simulation solves for pressure at tens of thousands of spatial points. The ROM compresses this to ~20-60 variables, giving 100-500x speedup while keeping the answer nearly identical.

---

## 2. The Physics: How Sound Travels in a Room

### 2.1 The Wave Equation

Sound is small pressure fluctuations in air. Two physical laws govern it:

- **Conservation of mass**: pressure changes cause air to move
- **Conservation of momentum**: pressure gradients push air around

Written as equations:

```
dp/dt = -rho * c^2 * div(v)       (pressure changes when air compresses/expands)
dv/dt = -(1/rho) * grad(p)        (air accelerates toward lower pressure)
```

where:
- `p(x,t)` = pressure at position x, time t [Pascals]
- `v(x,t)` = air velocity [m/s]
- `rho` = 1.2 kg/m^3 (air density)
- `c` = 343 m/s (speed of sound)

This is the **pressure-velocity (p-v)** formulation.

### 2.2 Why We Use the Pressure-Potential (p-Phi) Formulation

Instead of tracking velocity directly, define a "velocity potential" Phi where `v = -(1/rho) * grad(Phi)`. The equations become:

```
dp/dt   = rho * c^2 * M^{-1} * S * Phi    (+ boundary terms)
dPhi/dt = -(1/rho) * p
```

This looks more abstract but has a critical property: **energy conservation is built into the math**. The total acoustic energy:

```
E = (kinetic) + (potential) = (rho/2) * |grad(Phi)|^2 + (1/(2*rho*c^2)) * p^2
```

is exactly constant when walls are rigid. This matters because when we compress the model (ROM), formulations that conserve energy stay stable forever. Formulations that don't can blow up after a few milliseconds.

### 2.3 What Happens at Walls

Three types of wall conditions, from simplest to most realistic:

**Perfectly Reflecting (PR)** — rigid walls, no sound absorbed. Sound bounces forever. Energy is constant. This is the "bathroom tile" extreme.

**Frequency-Independent Impedance (FI)** — walls absorb a fixed fraction of energy at all frequencies. Controlled by impedance Z (units: N*s/m^3). Low Z = more absorption (carpet), high Z = less absorption (concrete). The boundary condition is `p = Z * v_normal`.

**Locally Reacting (LR)** — absorption depends on frequency. Real materials do this: a thick carpet absorbs high frequencies much more than low frequencies. Modeled using the Miki empirical formula, which gives complex impedance Z(f) as a function of material properties (flow resistivity sigma, thickness d). Since Z(f) is frequency-dependent but we simulate in the time domain, we use **Auxiliary Differential Equations (ADE)** — a trick that converts the frequency-domain impedance into a set of coupled ODEs that march in time alongside the wave equation.

---

## 3. Spatial Discretization: Turning Continuous Equations into Computable Ones

### 3.1 The Idea

The pressure field p(x,y,z,t) is a continuous function — it has a value at every point in space. Computers can't store infinity. So we sample the field at a finite set of points (nodes) and track the pressure at those nodes over time.

The room is divided into small elements (like tiles covering a floor). Within each element, the pressure is represented as a polynomial. The nodes are the points where we know the pressure exactly; between nodes, we interpolate.

### 3.2 Two Types of Elements

**Hexahedral (hex) elements** — 3D bricks. Each element is a deformed cube with 8 corners. Inside, we place (P+1)^3 nodes in a regular grid pattern using Gauss-Lobatto-Legendre (GLL) points. For P=4, that's 125 nodes per element. The huge advantage: the mass matrix is diagonal (each node has its own mass, no coupling), so time-stepping is cheap — just divide by a number, no linear system to solve.

**Tetrahedral (tet) elements** — 3D triangular pyramids with 4 corners. For P=2 (quadratic), each tet has 10 nodes (4 vertices + 6 edge midpoints). Tets can mesh any geometry — Gmsh generates them automatically for arbitrary shapes. The disadvantage: the mass matrix isn't naturally diagonal, so we use "lumping" (an approximation that makes it diagonal at the cost of some accuracy).

### 3.3 The Operators

After discretization, the continuous equations become matrix equations. The key matrices:

- **Mass matrix M** — diagonal array, one weight per node. Represents "how much space does this node control." For hex elements, this is exact (GLL quadrature). For tet elements, it's approximate (HRZ lumping).

- **Stiffness matrix S** — sparse matrix. Entry S[i,j] represents how much pressure at node j contributes to the Laplacian at node i. This encodes the room geometry — the shape of the room lives inside S. For a 25,000-DOF mesh, S might have 500,000 nonzero entries out of 625 million possible.

- **Boundary mass matrix B** — diagonal, nonzero only at wall nodes. Represents the wall surface area associated with each boundary node. Used for absorbing boundary conditions.

The solvers only need these three things: `M_diag`, `S`, `B_total`. They don't care whether the mesh is hex or tet, structured or unstructured. This is why we can swap mesh types without changing the solver code.

### 3.4 How Many Nodes Do You Need?

Rule of thumb: you need about 6-10 nodes per shortest wavelength. At 500 Hz, wavelength = 343/500 = 0.686 m. With P=4 hex elements, element size h ~ 0.3 m works. With P=2 tets, you need h ~ 0.1-0.15 m (smaller because lower polynomial order is less accurate per node).

The number of DOFs scales as (room_volume / h^3). A 4x3x2 m room at h=0.2: N ~ 24,000 DOFs. A 20x15x4 m lecture hall at h=0.1: N ~ 1,200,000 DOFs.

---

## 4. Time Integration: Stepping Forward in Time

### 4.1 RK4

We use the classical 4th-order Runge-Kutta method. Given the current state [p, Phi] at time t, compute the state at t + dt:

```
k1 = f(state)
k2 = f(state + dt/2 * k1)
k3 = f(state + dt/2 * k2)
k4 = f(state + dt * k3)
state_new = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

Each `f()` evaluation requires one sparse matrix-vector product (S * Phi), which is the dominant cost.

### 4.2 The CFL Condition

The time step dt must be small enough for stability:

```
dt <= CFL * h / (c * P^2)
```

CFL ~ 0.15 for RK4. For P=4 hex with h=0.19: dt ~ 1e-5 seconds. For P=2 tet with h=0.2: dt ~ 1.5e-5 seconds. A 40 ms simulation requires ~2,700 time steps.

### 4.3 The Propagator Matrix (ROM speedup trick)

For a linear system dp/dt = A*p, the RK4 scheme is equivalent to:

```
state_{n+1} = P * state_n
```

where P is the "propagator matrix":

```
P = I + dt*A + (dt*A)^2/2 + (dt*A)^3/6 + (dt*A)^4/24
```

This is just the Taylor expansion of the matrix exponential. For the ROM, A is tiny (e.g., 120x120 for Nrb=60), so P is precomputed once and each time step is a single small matrix-vector multiply. This is why the ROM is so fast.

---

## 5. Model Order Reduction (ROM)

### 5.1 The Key Insight

Run the full simulation once (the FOM). Collect "snapshots" — the pressure field at many time instants. These snapshots are vectors of length N (thousands of entries), but they don't span the full N-dimensional space. They lie on or near a low-dimensional subspace.

Think of it like video compression: each frame of a video has millions of pixels, but most frames look similar. You can describe the video using a small set of "key frames" and their combinations. The ROM does the same thing with pressure fields.

### 5.2 Building the Basis (Offline Phase)

1. Run the FOM, save pressure snapshots at many time steps: p(t_1), p(t_2), ..., p(t_K)
2. Stack them into a matrix (N rows x K columns)
3. Compute the SVD (Singular Value Decomposition): Snapshot = U * Sigma * V^T
4. The columns of U are the "basis vectors" — the dominant spatial patterns
5. Keep only the first r columns (where r << N), corresponding to the largest singular values
6. These r vectors form the reduced basis Psi (N x r matrix)

The truncation criterion: keep enough modes to capture (1 - epsilon) of the total energy, where epsilon ~ 1e-8. Typically r = 20-100 modes suffice.

### 5.3 Two Basis Types

**POD** (Proper Orthogonal Decomposition) — standard approach, separate bases for p, u, v. Does not preserve energy conservation.

**PSD** (Proper Symplectic Decomposition) — combines p and Phi snapshots, uses a single basis for both. Preserves the Hamiltonian (energy-conserving) structure. This is what we use for the p-Phi formulation.

**Modified PSD** — also includes boundary pressure snapshots in the SVD. This enriches the basis with modes that capture how energy leaves through absorbing walls, improving ROM stability for FI and LR boundaries.

**DC mode enrichment** — explicitly adds the constant-pressure mode (uniform pressure everywhere) to the basis. Without this, absorbing-wall ROMs can develop a DC offset that drifts because the basis can't represent it.

### 5.4 Solving in the Reduced Space (Online Phase)

Project the full operators onto the reduced basis:

```
K1_r = Psi^T * (rho*c^2 * M^{-1} * S) * Psi     (r x r matrix)
K3_r = Psi^T * (boundary_operator) * Psi           (r x r matrix, FI only)
```

Now instead of evolving N unknowns, evolve r unknowns:

```
d(a_p)/dt   = K1_r * a_Phi + K3_r * a_p
d(a_Phi)/dt = -(1/rho) * a_p
```

where a_p and a_Phi are the r-dimensional coefficient vectors.

Cost comparison per time step:
- FOM: sparse matvec on N-dimensional vectors (N = 25,000 → ~500,000 operations)
- ROM: dense matvec on 2r-dimensional vector (r = 40 → 6,400 operations)

That's why the ROM is 100-500x faster.

### 5.5 Eigenvalue Stabilization (Schur decomposition)

The RK4 propagator P should have all eigenvalues inside or on the unit circle (|lambda| <= 1). Floating-point errors can push some slightly outside, causing exponential blow-up.

**The fix**: decompose P using the Schur decomposition (P = Q * T * Q^H, where Q is unitary). The eigenvalues sit on the diagonal of T. Clip any |lambda| > 1 to exactly 1. Reconstruct P. Since Q is unitary (condition number = 1), this is numerically rock-solid — unlike the eigendecomposition approach (P = V * D * V^{-1}) where V can be ill-conditioned.

For PR (conservative): force all |lambda| = 1 (energy conserved exactly).
For FI (dissipative): clip |lambda| > 1 to 1 (energy can only decrease).

---

## 6. Mesh Generation: From Room Shape to Computable Mesh

### 6.1 Structured Meshes (Rectangular Rooms Only)

For rectangular rooms, the mesh is a regular grid. Operators are assembled using Kronecker products of 1D operators — very fast, very accurate. This is the `RectMesh2D` / `BoxMesh3D` path.

### 6.2 Unstructured Quad/Hex Meshes (Arbitrary 2D Floor Plans)

For non-rectangular rooms (L-shapes, T-shapes, rooms with columns), we use Gmsh to generate a quad mesh of the floor plan, then either:
- Use it directly for 2D simulation (`UnstructuredQuadMesh2D`)
- Extrude it vertically into hex elements for 3D simulation (`UnstructuredHexMesh3D`)

The extrusion approach covers most real rooms — they're floor plans with a ceiling height. Assembly is element-by-element with isoparametric mapping (computing the Jacobian at each quadrature point to account for element distortion).

### 6.3 Tetrahedral Meshes (Any 3D Geometry)

For truly arbitrary geometries, Gmsh generates tetrahedral meshes. P=2 (quadratic, 10-node tets) gives O(h^2) convergence. The mass matrix uses HRZ lumping (Hinton-Rock-Zienkiewicz) to stay diagonal and positive.

The solvers don't know or care which mesh type produced the operators. They just see M_diag, S, and B_total.

---

## 7. Validation: How We Know the Results Are Correct

### 7.1 Analytical Comparison (Rectangular Room)

For a rigid rectangular room, the exact eigenfrequencies are known:

```
f_{m,n,l} = (c/2) * sqrt((m/Lx)^2 + (n/Ly)^2 + (l/Lz)^2)
```

Our hex SEM matches these to 14 digits (machine precision) for low modes.

### 7.2 Dauge Benchmark (L-Shaped Domain)

The Neumann eigenvalues of the Laplacian on an L-shaped domain have been computed to 11+ digits by Monique Dauge's group using specially refined meshes. Our unstructured quad SEM matches these:

- Mode 1 (singular corner eigenfunction): error 5.7e-05
- Modes 3-4 (exact pi^2): error 1e-14 (machine precision)
- Regular modes: error 1e-8

The singular mode converges slower because the 270-degree reentrant corner creates a singularity in the eigenfunction that polynomials can't resolve efficiently.

### 7.3 Tet Convergence Study

For P=2 tets on a unit cube, eigenvalue errors converge at rate O(h^2):

| Element size h | DOFs | Mode 1 error |
|---------------|------|-------------|
| 0.40 | 325 | 3.4% |
| 0.25 | 945 | 1.5% |
| 0.15 | 4,035 | 0.51% |
| 0.10 | 11,193 | 0.24% |

### 7.4 Energy Conservation

For PR (rigid walls), the total acoustic energy must be constant. Our best result: drift of 6.3e-08 over thousands of time steps (hex), 4.9e-06 (P=2 tet on refined mesh). This confirms the operators are mathematically consistent.

For FI (absorbing walls), energy must decay monotonically. Verified on every test case.

### 7.5 Cross-Validation

The extruded hex mesh produces identical results to the Kronecker-product structured mesh when run on a box geometry: relative error 1.17e-14. This proves the unstructured assembly computes the same operators as the fast structured path.

---

## 8. What the Code Actually Contains

```
room_acoustics/
  sem.py                  — Structured mesh (RectMesh2D, BoxMesh3D) + Kronecker assembly
  unstructured_sem.py     — Unstructured quad/hex mesh + element-by-element assembly
  tet_sem.py              — P=2 tet mesh + assembly (any 3D geometry)
  geometry.py             — Room shape definitions + Gmsh quad meshing + extrusion
  gmsh_tet_import.py      — Gmsh tet meshing + .msh file import
  solvers.py              — FOM + ROM solvers (p-v, p-Phi), boundary conditions, basis builders
  results_io.py           — JSON data export for validation results
  visualize.py            — Pressure field visualizations
  validate.py             — 2D validation suite (9 tests)
  validate_3d.py          — 3D structured validation
  validate_unstructured.py     — 2D unstructured validation
  validate_unstructured_3d.py  — 3D extruded validation
  validate_eigenfrequencies.py — Eigenfrequency validation vs analytical + Dauge
  validate_tet_3d.py      — Tet element validation suite
```

### Key Design Principle

The codebase has a strict separation:
- **Mesh layer** produces: N_dof, coordinates, boundary nodes
- **Assembly layer** produces: {M_diag, M_inv, S, B_total}
- **Solver/ROM layer** consumes only the above — never touches elements

This means you can add a new element type without changing a single line in the solvers.

---

## 9. Performance Summary

| Configuration | N_dof | FOM time | ROM speedup | ROM error |
|--------------|-------|----------|-------------|-----------|
| 2D rect (P=4 hex) | 3,825 | 1.2s | 83x | 2e-3 |
| 2D rect large (P=4 hex) | 10,293 | 3.5s | 351x | 3e-1 to 1e-3 |
| 3D cube (P=4 hex) | 35,937 | 71s | 583x | 3e-4 |
| 3D L-shape hex (extruded) | 11,097 | 9.8s | 744x | 8e-4 |
| 3D L-shape tet (P=2, h=0.2) | 25,410 | 7.7s | 476x | 1e-2 |

---

## 10. Current Limitations

### 10.1 Assembly Speed

The unstructured/tet assembly loops over elements in pure Python. For meshes above ~50K DOFs, assembly takes minutes. Fix: Numba JIT compilation of the element loop (same code, 50-100x faster).

### 10.2 GPU Support

The FOM solver supports CuPy for GPU-accelerated sparse matvecs, but requires matching CuPy and CUDA toolkit versions. When it works, GPU gives 10-50x FOM speedup for large meshes.

### 10.3 Frequency Range

SEM resolution is fixed at mesh creation. Higher frequencies need finer meshes (more DOFs). For full-bandwidth auralization (20 Hz - 20 kHz), a hybrid approach is needed: wave-based (SEM) for low frequencies, geometric acoustics (ray tracing) for high frequencies.

### 10.4 Tet Accuracy vs Hex

P=2 tets have O(h^2) convergence. P=4 hexes have spectral convergence. For the same accuracy, tets need a much finer mesh. The tradeoff: tets mesh anything, hexes are more accurate per DOF.

---

## 11. References

1. **Bonthu et al. (2026)** — Stable MOR for Time-Domain Room Acoustics
2. **Sampedro Llopis et al. (2022)** — Reduced Basis Methods for Room Acoustics
3. **Miki (1990)** — Acoustic properties of porous materials (impedance model)
4. **Gustavsen & Semlyen (1999)** — Vector fitting (frequency-to-time-domain conversion)
5. **Patera (1984)** — Spectral element method foundations
6. **Keast (1986)** — Quadrature rules for tetrahedra
7. **Dauge** — Maxwell eigenvalue benchmark (Neumann eigenvalues on L-shaped domain), https://perso.univ-rennes1.fr/monique.dauge/benchmax.html
