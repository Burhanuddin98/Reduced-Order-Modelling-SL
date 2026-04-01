# Reduced Order Modelling — Research References

**Last updated:** 2026-04-01

---

## Primary papers (our implementation is based on these)

### Bonthu et al. (2026) — Structure-Preserving MOR
- **Method:** p-Phi formulation with PSD (Proper Symplectic Decomposition) basis
- **Key innovation:** cotangent lift preserves Hamiltonian structure → ROM stability
- **Modified formulation:** boundary pressure equation (p_b) enriches basis for absorbing walls
- **Elements:** unstructured triangular P=4 SEM
- **Result:** 100x speedup, energy-preserving ROM
- **Our status:** implemented (with quad/hex elements instead of triangles). See AUDIT.md for deviations.

### Sampedro Llopis et al. (2022) — RBM in Laplace Domain
- **Method:** Laplace domain (s = sigma + i*omega) instead of time domain
- **Key innovation:** sigma > 0 regularizes the system → avoids resonance singularities
- **Basis:** greedy enrichment with a posteriori error estimator
- **Parametric:** boundary impedance Z_s is the parameter
- **Result:** 1000x speedup in 3D for parametric boundary sweeps
- **Our status:** partially implemented (Helmholtz at real omega, not full Laplace). See AUDIT.md.

---

## Validation benchmarks

### BRAS CR2
- 8.4 x 6.7 x 3.0 m seminar room at TU Berlin
- Measured dodecahedron source RIRs (10 positions)
- Published absorption coefficients per surface (31 third-octave bands)
- Our best result: 0.6% T30 error at 250 Hz (modal ROM)

### Dauge L-shape benchmark
- L-shaped domain with re-entrant corner (singular eigenfunctions)
- Published eigenfrequencies to high precision
- Tests mesh convergence near geometric singularities

### ARD test room
- Non-rectangular STL geometry
- Used for pipeline validation (assembly, eigensolve, IR generation)
- No measured RIR data available

---

## Key concepts

### Spectral Element Method (SEM)
- High-order finite elements with GLL (Gauss-Lobatto-Legendre) quadrature
- Diagonal mass matrix → explicit time-stepping without linear solve
- Exponential convergence with polynomial order P
- Points per wavelength (ppw): 6-10 for P=4

### Model Order Reduction (ROM)
- Project full system (N DOFs) onto reduced basis (r << N DOFs)
- Methods: POD (data-driven), PSD (structure-preserving), greedy (error-driven)
- Offline/online decomposition: expensive build once, cheap evaluation many times

### ISO 3382 (Room acoustics measurements)
- T30: reverberation time from -5 to -35 dB Schroeder decay
- EDT: early decay time from 0 to -10 dB
- C80: clarity index (energy ratio, 80 ms boundary)
- D50: definition (energy ratio, 50 ms boundary)
- TS: center time (energy-weighted mean arrival)

---

## Useful resources

- SuiteSparse (UMFPACK, CHOLMOD): https://people.engr.tamu.edu/davis/suitesparse.html
- Gmsh (mesh generator): https://gmsh.info/
- BRAS database: https://depositonce.tu-berlin.de/items/d8dc3651-58b5-4f0f-beac-cb27a78f0a43
- Hesthaven & Rozza — Certified Reduced Basis Methods (textbook for greedy RBM)
