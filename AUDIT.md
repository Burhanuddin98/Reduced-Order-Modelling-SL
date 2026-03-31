# Audit: What We Built vs What The Papers Describe

## Bonthu et al. (2026) — Structure-Preserving MOR

### What the paper does
1. **p-v AND p-Φ formulations** — compares both, shows p-Φ is 3.5x faster
2. **Structure-preserving (symplectic) projection** — PSD cotangent lift basis preserves Hamiltonian structure
3. **Stability analysis** — eigenvalues of reduced operator must stay in left half-plane
4. **Modified p-Φ formulation** — adds boundary pressure equation (p_b_t = K3 * v_n) to capture boundary energy
5. **Combined snapshot matrix** — includes p, Φ, AND p_b snapshots for modified PSD basis
6. **SEM on triangular elements, P=4** — unstructured mesh, 178 elements, 1509 DOFs
7. **4m × 2m domain** — same as our test case
8. **PR, FI, LR boundary conditions** — all three
9. **100x speedup** — their claimed ROM speedup
10. **ADE system for LR** — reduced separately with its own basis Ψ_a

### What we built correctly
- p-Φ formulation ✓
- PSD cotangent lift basis ✓
- Modified PSD with boundary pressure enrichment ✓
- PR, FI, LR boundary conditions ✓
- SEM spatial discretization ✓
- RK4 time integration ✓
- 100-500x speedup ✓

### What we built WRONG or DIFFERENTLY
1. **We use QUAD/HEX elements, not triangles** — Bonthu uses unstructured triangular P=4 elements. Our SEM uses tensor-product GLL on quads/hexes. Different element type, same polynomial order.

2. **Our gradient operator G is different** — Bonthu's K operators (K1-K5) come from the weak form with integration by parts. Our operators are assembled differently (Kronecker products for structured, element-by-element for unstructured). The MATHEMATICAL operators should be equivalent, but the assembly path is different.

3. **We never properly implemented the modified p-Φ formulation for the ROM** — Bonthu adds an equation for p_b (boundary pressure) that evolves in time. We store p_b snapshots and include them in the SVD, but we don't solve the additional equation. The paper says the key is: "the reduced system can still be solved using the original, unmodified formulation while leveraging the modified reduced basis." We DO this.

4. **We added eigenvalue stabilization (Schur) which is NOT in the paper** — Bonthu's approach is structure-preserving by construction, so eigenvalue stabilization shouldn't be needed. We added it because our ROM was unstable — which suggests our structure preservation is broken somewhere.

5. **Our propagator matrix approach is NOT in the paper** — We precompute P = I + dtA + ... and multiply per step. The paper uses standard RK4 on the reduced system. Our propagator works for linear systems but is NOT structure-preserving (it doesn't maintain the symplectic structure of the Hamiltonian). This might be why we needed Schur stabilization.

6. **The modal ROM and frequency-domain solver are NOT in either paper** — These are our own additions, not from the Bonthu or Sampedro Llopis methodology.

## Sampedro Llopis et al. (2022) — RBM in Laplace Domain

### What the paper does
1. **Laplace domain, NOT time domain** — solves (s²M + c²S + sc²ρ/Z_s M_C)p = s*p0*M
2. **Parametric ROM** — boundary condition Z_s is the parameter
3. **SEM spatial discretization** — same as Bonthu
4. **Greedy basis construction** — a posteriori error estimator drives basis enrichment
5. **Split into real/imaginary** — solves 2N real system instead of N complex
6. **Frequency-dependent BCs** — via ADE in Laplace domain
7. **1000x speedup in 3D** — for parametric boundary sweeps
8. **Stable by construction** — Laplace domain avoids time-stepping instability entirely

### What we attempted
- Frequency-domain Helmholtz solver ✓ (but using ω not s = σ + iω)
- Greedy basis construction ✓ (but it doesn't converge — too few basis vectors for too many modes)
- UMFPACK for fast solves ✓

### What we got wrong
1. **We used Helmholtz (ω), not Laplace (s = σ + iω)** — The Laplace domain has a real part σ > 0 that acts as damping, making the system better conditioned. Our Helmholtz at real ω hits resonance singularities. This is WHY the greedy ROM fails — the system is nearly singular near resonances.

2. **We didn't split into real/imaginary** — Sampedro Llopis converts to a 2N real system. This avoids complex arithmetic and makes the system symmetric positive definite (or close to it), enabling Cholesky instead of LU.

3. **We didn't use the Laplace→time reconstruction** — The paper uses Weeks method or numerical inverse Laplace transform to get the time-domain IR from the Laplace-domain solution.

## Critical Findings

### The propagator matrix breaks symplectic structure
Our P = I + dtA + dt²A²/2 + ... is a polynomial approximation of e^{dtA}. For the p-Φ system where A has the block structure [[0, K1], [K2, 0]] (for PR) or [[K3, K1], [K2, 0]] (for FI), the matrix exponential preserves symplectic structure ONLY if the time integrator is symplectic. RK4 is NOT symplectic. Our propagator inherits RK4's non-symplectic nature.

This is why:
- PR ROM works fine for short times but drifts
- FI ROM needs Schur stabilization
- Long-time stability requires eigenvalue correction

The paper avoids this by using the structure-preserving formulation that makes RK4 stable ENOUGH (the modified PSD basis inherits the Lyapunov structure from the boundary energy).

### The Laplace domain is fundamentally better for ROM
Sampedro Llopis avoids all time-stepping issues by working in the Laplace domain. The system is well-conditioned (σ > 0 provides regularization), the ROM basis needs far fewer vectors (because the Laplace transform smears out resonances), and stability is guaranteed.

Our attempt to do frequency-domain ROM at real ω failed because we hit resonance singularities that the Laplace approach avoids.

### The modal ROM is an accidental rediscovery
Our modal ROM (eigenmode basis + analytical time evolution) is mathematically related to the Laplace-domain approach — the eigenvalues of M^{-1}S are the poles of the Laplace-domain transfer function. Our "analytical time evolution" is equivalent to the inverse Laplace transform of the pole expansion. We accidentally rediscovered the connection between eigenmode analysis and Laplace-domain ROM.

## What Should We Do

1. **Implement the Laplace-domain approach properly** — use s = σ + iω with σ > 0, split into 2N real system, use the greedy basis with proper error estimator. This is Sampedro Llopis's method and it works.

2. **Keep the modal ROM** — it's valid and gives excellent results at low frequencies. It's essentially the Laplace-domain approach with explicit pole identification.

3. **Fix the time-domain ROM** — either use a symplectic integrator (Störmer-Verlet) instead of RK4, or accept that the propagator needs Schur stabilization (which is what we do).

4. **The hybrid approach is sound** — modal/Laplace for low frequencies, ISM for high frequencies. Both papers support this architecture.
