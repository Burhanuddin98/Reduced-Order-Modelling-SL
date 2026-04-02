# Reduced Order Modelling — Improvement Plan & Roadmap

**Last updated:** 2026-04-01

---

## Current metrics

### Modal ROM — BRAS CR2 (8.4 x 6.7 x 3.0 m, 250 Hz)
| Metric | Error | Status |
|--------|-------|--------|
| T30 | 0.6% | Production-ready |
| EDT | 2.5% | Production-ready |
| Synthesis time | 0.2s / 3.5s IR | Fast |

### Hybrid platform — BRAS CR2 (250-1000 Hz)
| Metric | Error | Status |
|--------|-------|--------|
| T30 | 1.7-6% | Good |
| EDT | 4.2-4.5% | Good |
| Total time | ~31s | Acceptable |

### Phase 3 first run — BRAS CR2 (74K DOFs, 200 modes)
| Metric | Error | Status |
|--------|-------|--------|
| Broadband T30 | 3.0% | PASS |
| T30 @250 Hz | 20% | FAIL |
| T30 @500 Hz | 29% | FAIL |
| T30 @1-2 kHz | 43-67% | FAIL |
| EDT | 28% | FAIL |
| C80 | 2.7 dB delta | FAIL (borderline) |

### Known weaknesses
| Issue | Error | Root cause |
|-------|-------|------------|
| Octave-band T30 rises with frequency | 20-67% | FI impedance: single alpha per surface, no freq-dep absorption |
| Freq-domain ROM at resonances | 63-192% | Only 12 basis vectors, misses peaks |
| Non-shoebox validation | Limited | Only box + STL import tested for T30 |

---

## The key question

**Can we get <10% T30 error across 250-4000 Hz on rooms that aren't shoeboxes?**

Everything else is packaging. This question drives the roadmap.

---

## Active roadmap

### Phase 1: Frequency-domain solver speed [DONE]
- UMFPACK symbolic factorization reuse: 27x speedup
- Status: `splu` path working, cholmod available as fallback

### Phase 2: Frequency-domain ROM accuracy [IN PROGRESS]
- Problem: greedy basis with 12 vectors misses resonances
- Solution: adaptive greedy training with residual-based enrichment
- Target: <1% error at all frequencies 20-500 Hz with 30-60 basis vectors
- Risk: medium (well-established algorithm, tricky near singularities)

### Phase 2b: Unified modal synthesis [DONE]
- Axial mode engine: parallel surface detection, 3D coupling loss, air absorption
- Verified vs BRAS measured: 88% spectral match, 28% decay error (coupling model)
- Full analytical modes: axial + tangential + oblique for box rooms (Kuttruff decay)
- Generalized modes: non-box rooms via perpendicular pair detection
- Statistical modes: Weyl-density fill for irregular rooms
- Unified pipeline: provider registry, confidence-based merge, single-pass Numba JIT synthesis
- Best result (axial+statistical): broadband T30 9.3%, 250Hz 21%, 500Hz 12%, 1kHz 9%
- Spec: `docs/axial_mode_spec.md`

### Phase 3: BRAS CR2 full-bandwidth validation [IN PROGRESS]
- First run complete (74K DOFs, 200 modes, 5000 rays)
- **Broadband T30: 3.0% error — PASS**
- Octave-band T30: 20-67% error — FAIL (T30 rises with frequency)
- Root cause: FI impedance uses single alpha per surface; real materials
  absorb more at higher frequencies. Ray tracer + axial modes need
  frequency-dependent absorption.
- **Next:** Per-surface absorption calibration via simulation-measurement residuals.
  Unlike Eyring inversion (assumes homogeneous field, gives room-mean alpha),
  this uses the position-dependent simulation at known source/receiver pairs
  to extract per-surface absorption. The modal ROM gives spatially resolved
  decay that depends on which surfaces each mode couples to — different
  receiver positions weight different surfaces differently. Minimizing the
  per-position, per-band T30 residual between sim and measurement gives
  per-surface, per-band alpha that Eyring cannot resolve.
- BRAS data downloaded: 10 measured RIRs, fitted absorption CSVs
- Accept: T30 <10% per octave band (250-2000 Hz), C80 within 2 dB

### Phase 4: Non-shoebox validation [NOT STARTED]
- Test A: BRAS Scene 11 (auditorium) — T30 within 15% of measured
- Test B: ARD test room (STL import) — eigenfrequencies plausible, T30 reasonable
- Test C: L-shaped room — modal and freq-domain ROM agree within 5%
- Risk: high for Scene 11 (no exact geometry)

### Phase 5: Auralization + Room API polish [NOT STARTED]
- Convolve IR with dry audio → auralized WAV output
- Sweep receivers for T30 heatmaps
- Material comparison (A/B testing)
- Only build after Phase 3 passes

### Phase 6: CUDA solver optimization [NOT STARTED]
- Option A: scikit-sparse CHOLMOD (easiest, 5-10x speedup)
- Option B: Custom CUDA analyze-factor-solve phases
- Option C: Batched GPU solve for multiple frequencies
- Target: <0.5s per frequency at N=23K, <5s at N=100K

### Phase 7: Interactive demo + product [NOT STARTED]
- Source/receiver sweep in <1 second
- Material optimization: "minimize RT60 below 2s"
- Cloud deployment for large rooms
- Real-time VR integration

---

## Decision points

### After Phase 3:
- **Pass (<10% T30 at 250-2000 Hz):** proceed to Phase 4 (non-shoebox)
- **Fail:** investigate — finer mesh? different BC model? more training?

### After Phase 4:
- **Pass (non-shoebox works):** proceed to Phase 5 (product layer)
- **Fail:** pivot to hybrid-only architecture (modal low + ray high), accept geometry limitations

### After Phase 6:
- **Pass (<5s/freq at N=100K):** full-bandwidth on large rooms is feasible
- **Fail:** stick with smaller meshes + ROM, accept f_max limitation

---

## Perceptual IR quality [IN PROGRESS]

### Achieved
- Spectral ISM: per-bounce frequency-dependent filtering + scattering smear
- Diffuse tail: 1/3-octave decaying noise, measured-calibrated decay rates
- Tonal balance: 4.8 dB RMS error (within natural position variation 2.8-6.2 dB)
- Early reflection matching: 100% peak alignment (49/49 within 1ms)

### Remaining
- **80 Hz excess (+14 dB)**: rigid-wall ISM overestimates low-frequency energy
  - Root cause: real walls vibrate (membrane absorption, structural coupling)
  - Fix: wall impedance model for frequencies below ~200 Hz
  - Nonlinear structural coupling (amplitude-dependent absorption at resonance)
- **C80 mismatch** (-1.2 vs 1.8 dB): early/late energy balance needs tuning
- **Late tail spectral shape**: diffuse noise lacks spatial coherence of real reverberation

---

## Known technical debt

1. **Propagator matrix breaks symplectic structure** — Schur stabilization is a workaround, not a fix. Proper solution: symplectic integrator (Stormer-Verlet) or Laplace domain
2. **Laplace-domain solver incomplete** — should use s = sigma + i*omega (not just omega). Sampedro Llopis approach not fully implemented
3. **Python assembly bottleneck** — unstructured/tet assembly single-threaded. Numba JIT available but not enabled everywhere
4. **No requirements.txt** — dependencies installed manually
5. **Precompiled DLLs checked in** — should have CI/CD build pipeline
6. **CUDA solver slower than scipy for N<50K** — no symbolic factorization reuse in GPU path

---

## Success criteria

### Minimum viable product
- Load STL → materials → T30/EDT/C80 within 15% of measurement
- Total: <5 min first run, <1s for parameter changes
- Rooms up to ~2000 m^3

### Competitive product
- T30 within 5% across 250-4000 Hz
- Full auralization
- Source/receiver sweep in real-time
- Arbitrary geometry up to ~10,000 m^3

### Game changer
- All of the above on a laptop GPU
- Material optimization
- Real-time VR integration
