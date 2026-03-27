# Room Acoustics with Model Order Reduction — Theory Guide

## 1. The Physical Problem

Sound in a room obeys the **acoustic wave equation**. When a sound source fires (a clap, a speaker impulse), pressure waves radiate outward, bounce off walls, and eventually decay. Simulating this process gives us an **impulse response (IR)** — the DNA of a room's acoustics. Convolve any audio with the IR and you hear what that audio sounds like *in that room*.

The challenge: a full numerical simulation (the **Full Order Model**, FOM) solves for pressure at thousands or millions of spatial points, thousands of times per second. This is accurate but slow. **Model Order Reduction (ROM)** compresses the problem from thousands of unknowns down to tens, achieving 10-100x speedup while preserving accuracy.

---

## 2. Governing Equations

### 2.1 The Linearized Acoustic Equations

Starting from conservation of mass and momentum for small perturbations in air:

```
∂p/∂t + ρc² ∇·v = 0          (mass conservation)
∂v/∂t + (1/ρ) ∇p = 0          (momentum conservation)
```

where:
- **p(x,t)** — acoustic pressure [Pa]
- **v(x,t)** — particle velocity vector [m/s]
- **ρ = 1.2 kg/m³** — air density
- **c = 343 m/s** — speed of sound

This is the **pressure-velocity (p-v)** formulation, also called the Linearized Euler Equations.

### 2.2 The Pressure-Potential (p-Φ) Formulation

Define a **velocity potential** Φ such that v = −(1/ρ) ∇Φ. The equations become:

```
∂p/∂t  = ρc² M⁻¹ S Φ  +  boundary terms
∂Φ/∂t  = −(1/ρ) p
```

where S is the stiffness (Laplacian) operator and M is the mass operator.

**Why bother?** This formulation has **Hamiltonian structure** — the total acoustic energy

```
H = (ρ/2) ∫|∇Φ|² dx  +  (1/2ρc²) ∫ p² dx
      ╰─ kinetic ─╯       ╰─ potential ─╯
```

is exactly conserved for rigid walls. This structure is critical for building stable reduced models (Section 6).

---

## 3. Spatial Discretization: Spectral Element Method (SEM)

### 3.1 Why SEM?

The Spectral Element Method combines the geometric flexibility of finite elements with the exponential convergence of spectral methods. For room acoustics, this means:
- **High accuracy per DOF** — fewer unknowns needed compared to standard FEM or FDTD
- **Diagonal mass matrix** — via GLL quadrature, enabling explicit time stepping without solving linear systems
- **Tensor-product structure** — Kronecker products make 2D/3D assembly from 1D building blocks

### 3.2 GLL Quadrature

The foundation is **Gauss-Lobatto-Legendre (GLL)** quadrature on the reference interval [−1, 1].

For polynomial order P, we get P+1 nodes {ξ₀, ξ₁, ..., ξ_P} and weights {w₀, w₁, ..., w_P}. The nodes are:
- ξ₀ = −1 and ξ_P = +1 (endpoints always included — this is the "Lobatto" part)
- Interior nodes = roots of P'_N(ξ), the derivative of the Legendre polynomial

The weights satisfy: ∫₋₁¹ f(ξ) dξ ≈ Σ wᵢ f(ξᵢ), exact for polynomials up to degree 2P−1.

**Key property:** Because GLL nodes include endpoints, neighboring elements share boundary nodes automatically — no separate "gluing" step needed.

### 3.3 1-D Element Operators

On each element of physical width h, we build three operators:

**Mass matrix M** (diagonal, from GLL quadrature):
```
M[i] = (h/2) wᵢ
```
This is the "lumped" mass — diagonal because GLL quadrature uses the same points as the basis functions. This is what makes explicit time stepping cheap.

**Stiffness matrix K** (from the Laplacian ∇²):
```
K = D^T W D    where D[i,j] = l'_j(ξᵢ)  (derivative of Lagrange basis)
                      W = diag(w)
```
Scaled by the Jacobian: K_physical = (2/h) K_reference.

**Gradient matrix G** (maps potential to velocity):
```
G[i,j] = ∫ l'_j(ξ) lᵢ(ξ) dξ
```
Computed with **exact Gauss quadrature** (not GLL) to ensure the discrete identity S_x^T M⁻¹ S_x ≈ S holds. This is important: if the gradient isn't computed accurately, the p-v and p-Φ formulations disagree, and the Hamiltonian structure breaks.

### 3.4 2-D Assembly via Kronecker Products

For a rectangular domain [0,Lx] × [0,Ly] with Nex × Ney elements:

The 2-D mass is simply the outer product of 1-D masses:
```
M_2D = Mʸ ⊗ Mˣ     (diagonal — fast!)
```

The 2-D Laplacian stiffness decomposes as:
```
S = (Mʸ ⊗ Kˣ) + (Kʸ ⊗ Mˣ)
```
Read this as: "differentiate twice in x (with y mass weighting) plus differentiate twice in y (with x mass weighting)."

The gradient operators:
```
Sₓ = Mʸ ⊗ Gˣ       (gradient in x)
Sᵧ = Gʸ ⊗ Mˣ       (gradient in y)
```

### 3.5 3-D Assembly (Triple Kronecker)

For a box [0,Lx] × [0,Ly] × [0,Lz]:
```
M_3D = M_z ⊗ M_y ⊗ M_x
S    = (M_z ⊗ M_y ⊗ K_x) + (M_z ⊗ K_y ⊗ M_x) + (K_z ⊗ M_y ⊗ M_x)
```

The Kronecker structure means we never form the full N×N matrices explicitly in the assembly — we build them from small (P+1)×(P+1) blocks. For a mesh with N = 36,000 DOFs (the 3D test case), the stiffness matrix S has ~3.8M nonzeros stored as a sparse CSR matrix.

### 3.6 Resolution Rule

To resolve waves up to frequency f_max, we need roughly PPW (points per wavelength) nodes per shortest wavelength λ_min = c/f_max:

```
element size h ≈ (λ_min / PPW) × P
```

With P=4 and PPW=10: for f_max=700 Hz, h ≈ 0.196 m → about 21×11 elements for a 4×2 m room (N ≈ 3,700 DOFs). For f_max=1000 Hz in 3D: N ≈ 36,000.

---

## 4. Boundary Conditions

### 4.1 Perfectly Reflecting (PR) — Rigid Walls

The simplest case: v·n = 0 at walls (no normal velocity). No energy leaves the system. In the p-Φ formulation, this requires no explicit boundary term — it's the natural (Neumann) boundary condition.

Energy is exactly conserved: H(t) = H(0) for all time.

### 4.2 Frequency-Independent Impedance (FI)

Real walls absorb some sound. The simplest absorption model relates pressure to normal velocity at the wall:

```
p = Z · vₙ     at the boundary
```

where Z [N·s/m³] is the wall impedance. Low Z = more absorption, high Z = more reflection (Z → ∞ recovers rigid walls).

In the FOM, this adds a damping term:
```
∂p/∂t += −(ρc²/Z) M⁻¹ B p     (B = boundary mass matrix)
```

Energy now decays monotonically: dH/dt ≤ 0.

### 4.3 Locally Reacting, Frequency-Dependent (LR)

Real materials have impedance that varies with frequency — a carpet absorbs high frequencies much more than low frequencies. The **Miki model** (1990) gives the surface impedance of a porous absorber backed by a rigid wall:

```
Z_c = ρc [1 + 0.0699(f/σ)^{-0.632} − j·0.1071(f/σ)^{-0.618}]
k_c = (2πf/c) [1 + 0.1093(f/σ)^{-0.618} − j·0.1597(f/σ)^{-0.683}]
Z_s = −jZ_c / tan(k_c · d)
```

Parameters:
- σ_mat — flow resistivity of the material [N·s/m⁴] (e.g., 10,000 for dense fiberglass)
- d_mat — material thickness [m]

**The problem:** Z_s(f) is a complex function of frequency, but our time-domain solver works step-by-step in time. We can't just multiply by Z_s — that's a frequency-domain operation.

**The solution: Auxiliary Differential Equations (ADE)**

We approximate the surface admittance Y_s(ω) = 1/Z_s(ω) as a sum of rational functions (poles):

```
Y_s(jω) ≈ Y_∞ + Σₖ Aₖ / (λₖ + jω)
```

Each pole becomes an ODE (an "accumulator" variable φₖ at each boundary node):
```
dφₖ/dt = −λₖ φₖ + p_boundary
```

The normal velocity at the wall is then:
```
vₙ = Y_∞ · p + Σₖ Aₖ · φₖ
```

This converts the frequency-dependent boundary into a set of coupled ODEs that march in time alongside the wave equation. The pole locations and residues {λₖ, Aₖ} are found by **vector fitting** (Gustavsen-Semlyen algorithm), which iteratively relocates poles to minimize the least-squares fit to Y_s(ω).

---

## 5. Time Integration: RK4

All solvers use the classical 4th-order Runge-Kutta method. For state vector **q** = [p, Φ] (or [p, u, v]):

```
k₁ = f(qⁿ)
k₂ = f(qⁿ + ½Δt k₁)
k₃ = f(qⁿ + ½Δt k₂)
k₄ = f(qⁿ + Δt k₃)
qⁿ⁺¹ = qⁿ + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

**CFL condition** (stability limit):
```
Δt ≤ CFL · h / (c · P²)
```

With CFL ≈ 0.15-0.2 for RK4 with SEM. Smaller P² in the denominator reflects the fact that higher-order elements have tighter stability constraints.

Each RK4 step requires 4 evaluations of the right-hand side, each involving a sparse matrix-vector product (S·Φ or Sₓ·p). This is the dominant cost — and what we want the ROM to eliminate.

---

## 6. Model Order Reduction (ROM)

### 6.1 The Core Idea

After running the FOM, we have **snapshots**: the pressure field p(x, tₖ) sampled at many time steps. These snapshots live in a high-dimensional space (N = thousands of DOFs) but typically lie on or near a much lower-dimensional manifold.

**ROM in one sentence:** Find a small basis {ψ₁, ..., ψᵣ} (r << N) that spans the snapshot data, then solve the wave equation projected onto that basis.

Instead of evolving N unknowns, we evolve r ≈ 10-100 unknowns:
```
p(x, t) ≈ Σᵢ aᵢ(t) ψᵢ(x)     →     p ≈ Ψ a
```

The FOM system **q̇ = A q** becomes the ROM system **ȧ = Aᵣ a** where Aᵣ = Ψᵀ A Ψ is a tiny r×r matrix.

### 6.2 Basis Construction: POD vs PSD

**POD (Proper Orthogonal Decomposition)** — Standard approach for p-v:
1. Collect snapshots into a matrix: columns = [p(t₁), p(t₂), ...]
2. Compute SVD: snapshot matrix = U Σ Vᵀ
3. Keep the first r columns of U (corresponding to largest singular values)
4. Truncation criterion: retain enough modes to capture (1 − ε) of the total "energy" (sum of σ²)

Separate bases are built for p, u, and v. This works but **does not preserve Hamiltonian structure**.

**PSD (Proper Symplectic Decomposition)** — For p-Φ:
1. Combine p and Φ snapshots: columns = [p(t₁)...p(tₙ), Φ(t₁)...Φ(tₙ)]
2. SVD of the combined matrix → single basis Ψ_H
3. Use **the same basis** for both p and Φ (the "cotangent lift")

This preserves the symplectic (Hamiltonian) structure: if the FOM conserves energy, so does the ROM. This is why p-Φ + PSD is preferred over p-v + POD for long-time stability.

**Modified PSD (Boundary-Energy Enrichment)** — For absorbing walls:

Standard PSD misses the boundary behavior because boundary pressure is a small fraction of the total field. The fix: include boundary pressure snapshots in the SVD:
```
combined = [p snapshots | Φ snapshots | p_boundary snapshots]
```

This enriches the basis with modes that capture how energy leaves through the walls, improving ROM stability for FI and LR boundaries.

### 6.3 ROM Online Phase

Once the basis Ψ is built, the ROM solve is cheap. The **reduced operators** are precomputed:

```
K₁ʳ = Ψᵀ (ρc² M⁻¹ S) Ψ          (r × r matrix)
K₂  = −1/ρ                         (scalar)
K₃ʳ = Ψᵀ (boundary operator) Ψ    (r × r, only for FI)
```

Then time-stepping is just r×r matrix-vector products instead of N×N sparse matvecs.

**Cost comparison per time step:**
- FOM: O(nnz) ≈ O(N · (2P+1)²) — sparse matvec
- ROM: O(r²) — dense matvec on tiny system

For N=36,000 and r=40: the ROM step is ~900x cheaper per step. After accounting for overhead (basis projection, operator construction), the total speedup is 10-100x.

### 6.4 RK4 Propagator Matrix (3D ROM)

For a **linear** ODE ȧ = Aᵣ a, the RK4 scheme reduces to:
```
aⁿ⁺¹ = P · aⁿ
```

where P is the **propagator matrix**:
```
P = I + ΔtA + (ΔtA)²/2 + (ΔtA)³/6 + (ΔtA)⁴/24
```

This is just the degree-4 Taylor expansion of the matrix exponential e^{ΔtA}.

**Key advantage:** P is a 2r × 2r dense matrix computed once. Each time step is then a single matrix-vector multiply — no RK4 stages, no 4× overhead.

### 6.5 Eigenvalue Stabilization

The propagator P should have **spectral radius ≤ 1** (all eigenvalues inside or on the unit circle). If any |λᵢ| > 1, that mode grows exponentially and the simulation blows up.

For PR (conservative): all |λᵢ| should equal exactly 1. In practice, floating-point errors push some slightly outside. Fix: project all eigenvalues onto the unit circle.
```
λᵢ → λᵢ / |λᵢ|
```

For FI (dissipative): |λᵢ| should be ≤ 1. Fix: clamp any |λᵢ| > 1 to exactly 1, preserving the phase (damping direction).

The stabilized propagator is reconstructed: P_stable = V · diag(λ_fixed) · V⁻¹.

---

## 7. Analytical Validation (Rigid Rectangular Room)

For a rectangular room with perfectly rigid walls, the exact solution is known via modal expansion:

```
p(x, y, t) = Σ_{m,n} A_{mn} · cos(mπx/Lx) · cos(nπy/Ly) · cos(ω_{mn} t)
```

where:
- ω_{mn} = cπ √((m/Lx)² + (n/Ly)²) — the modal frequencies
- A_{mn} — coefficients determined by the initial condition (Gaussian pulse)

The (m,n) = (0,0) mode is the DC component (uniform pressure). Higher modes correspond to standing wave patterns. For a 4×2 m room, the first few modal frequencies are:

| Mode (m,n) | Frequency [Hz] |
|-----------|----------------|
| (1,0)     | 42.9           |
| (0,1)     | 85.8           |
| (1,1)     | 95.9           |
| (2,0)     | 85.8           |
| (2,1)     | 121.2          |

This analytical solution is used to verify the FOM: if the numerical IR matches the modal expansion to within ~0.03% relative error, the spatial discretization and time integrator are working correctly.

---

## 8. Energy Diagnostics

Total acoustic energy in the discrete system:

**p-Φ formulation:**
```
H = (ρ/2) Φᵀ S Φ  +  (1/2ρc²) pᵀ M p
```

**p-v formulation:**
```
H = (ρ/2)(uᵀMu + vᵀMv)  +  (1/2ρc²) pᵀ M p
```

Expected behavior:
- **PR:** H = constant (energy conservation — the gold standard test)
- **FI:** H decays monotonically (energy leaves through walls)
- **LR:** H decays faster for thicker/softer materials (more absorption)

If energy grows in any case, something is wrong (numerical instability, incorrect boundary implementation, or a broken ROM basis).

---

## 9. Speedup Analysis

The ROM speedup depends on the ratio N/r and the time integration cost:

| Component | FOM cost | ROM cost |
|-----------|----------|----------|
| Per time step | O(nnz) sparse matvec | O(r²) dense matvec |
| Total steps | Nt | Nt (same Δt) |
| Offline (one-time) | — | O(N · Nt_snap) SVD + O(N · r · nnz) projection |

**When does ROM pay off?**
- The offline cost (FOM snapshots + SVD + operator projection) must be amortized. If you only need one simulation, ROM is slower. ROM shines when you run **many simulations** with different sources, receivers, or boundary conditions on the same geometry.
- The online speedup scales as ~(nnz / r²). For N = 36,000 (3D, 1kHz), nnz ≈ 3.8M, and r = 40: theoretical speedup ≈ 2,375×. Measured: ~100× (overhead from Python, memory access, non-optimized loops).
- For small 2D problems (N < 5,000), the FOM is already fast enough that ROM overhead dominates. Speedup: only 5-7×.

---

## 10. Known Limitations and Open Problems

### 10.1 ROM Stability for Absorbing Boundaries

The standard PSD basis preserves the symplectic (energy-conserving) structure of the p-Φ formulation. But for absorbing walls (FI, LR), the system is **dissipative**, not conservative. The ROM must also preserve the dissipative structure — otherwise energy can grow in the reduced model even though it decays in the FOM.

The modified PSD basis (boundary enrichment) partially addresses this by including boundary pressure in the snapshot matrix. However, for long simulations (T > 100ms), the ROM can still drift. A fully port-Hamiltonian Galerkin projection that separately preserves the conservative and dissipative parts is the proper fix (this is the main contribution of Bonthu et al. 2026).

### 10.2 Geometry Limitations

The current implementation uses structured rectangular/box meshes only. Real rooms have irregular shapes, columns, furniture, etc. Extending to unstructured meshes requires:
- Replacing Kronecker-product assembly with element-by-element assembly
- Handling non-conforming interfaces
- More complex boundary node identification

### 10.3 Frequency Range

The SEM resolution is fixed at mesh creation time. To resolve higher frequencies, you need more elements (larger N), which makes the FOM slower — but also makes the ROM more valuable (bigger N/r ratio → bigger speedup).

The practical upper limit is set by memory: 3D problems at 2kHz with PPW=10 need N > 500,000 DOFs, and snapshot storage for ROM basis construction becomes the bottleneck (~10+ GB for 2000 snapshots).

---

## 11. References

1. **Bonthu et al. (2026)** — "Stable Model Order Reduction for Time-Domain Room Acoustics." Introduces the modified p-Φ formulation with boundary-energy enrichment for stable ROMs with absorbing walls.

2. **Sampedro Llopis et al. (2022)** — "Reduced Basis Methods for Numerical Room Acoustic Simulations with Parametrized Boundaries." POD/PSD for room acoustics with frequency-dependent impedance.

3. **Miki (1990)** — "Acoustical properties of porous materials — Modifications of Delany-Bazley models." Empirical impedance model for fibrous absorbers.

4. **Gustavsen & Semlyen (1999)** — "Rational approximation of frequency domain responses by vector fitting." The algorithm used to convert Z_s(f) into time-domain ADE poles.

5. **Patera (1984)** — "A spectral element method for fluid dynamics." Foundation of the SEM approach.
