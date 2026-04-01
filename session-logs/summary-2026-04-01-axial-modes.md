# Session Summary: 2026-04-01 — Axial Mode Engine → Full Analytical Modes

## Overview
Major session: set up collaboration infrastructure, built the axial mode engine (Michael's parallel surface idea), verified against measured RIRs, added air absorption, built calibration tools, rewrote modal-clouds components, and extended to full analytical room modes (axial + tangential + oblique).

## Branch: feature/axial-modes (18 commits)

### Core new modules
| Module | LOC | Purpose |
|--------|-----|---------|
| axial_modes.py | ~400 | Parallel surface detection + 1D axial mode synthesis |
| analytical_modes.py | ~230 | Full analytical modal solution for box rooms (all mode types) |
| generalized_modes.py | ~280 | Extended modes for non-rectangular rooms |
| material_function.py | ~200 | Frequency-dependent alpha(f) at arbitrary resolution |
| material_catalog.py | ~250 | 28 catalog materials as MaterialFunction objects |
| calibrate_absorption.py | ~640 | Per-surface absorption calibration from measured RIRs |
| calibrate_spectral.py | ~330 | Spectral calibration with regularization |
| fdtd.py | ~350 | 27-pt Laplacian FDTD solver (CPU/GPU) |
| voxelize.py | ~280 | Box/STL voxelization, boundary detection |
| spectral_tools.py | ~250 | WAV comparison, spectrogram, peak extraction |

### Infrastructure
- CLAUDE.md, docs/architecture.md, docs/signal_chain.md, docs/improvement_plan.md
- docs/research_references.md, docs/axial_mode_spec.md
- session-logs/ directory

### BRAS data (bras_data/, gitignored)
- CR2, CR3, CR4 scene ZIPs downloaded (2.2 GB total)
- Surface absorption CSVs extracted (31 third-octave bands per surface)
- 10 measured dodecahedron RIRs extracted

## Key findings

### Engine comparison (head-to-head vs BRAS measured)
| Engine | Broadband T30 err | 500 Hz | 1000 Hz | 2000 Hz |
|--------|:-:|:-:|:-:|:-:|
| Axial modes only | **2.5%** | **6.5%** | **2.4%** | **8.8%** |
| Current hybrid | 31.8% | 4.6% | 1.6% | 4.6% |
| Full analytical (145K modes) | 44.9% | 55% | 21% | 22% |
| Modal ROM (100 modes) | 42.6% | 99% | 99% | 100% |
| Ray tracer | 47.1% | 25% | 39% | 54% |

### Why axial modes alone beat the full solution
The 3D coupling model (gamma_eff = (1-coupling)*gamma_pair + coupling*gamma_room) acts as implicit calibration, pulling decay rates toward the measured room average. The full analytical modes are more physically correct but more sensitive to absorption input values.

### Architecture principle established
Materials are continuous alpha(f) functions, never octave bands internally. Each mode evaluates absorption at its specific eigenfrequency. Octave bands are measurement/validation output only.

## What works
- Axial mode frequencies match measured: 88% (35/40 within ±3 Hz)
- Decay rates with coupling model: 28% mean error (down from 51%)
- Position dependence: 16.4 dB spread confirmed across receivers
- Air absorption (ISO 9613-1): reduces high-freq T30 error by 10-20 pp
- MaterialFunction + per-mode spectral decay: correct architecture
- Per-surface weight decomposition: microsecond material recalculation

## What needs work
- **Absorption calibration**: BRAS fitted values are calibrated for different simulator; need calibration with our modal physics
- **Full analytical modes**: correct physics (Kuttruff) but needs right alpha values
- **Synthesis speed**: 107K modes × 132K samples = 79s in Python loop; needs vectorization
- **Hybrid blend**: level-matching between engines is diluting good results

## Session continued into 2026-04-02

### Additional modules
- **unified_modes.py** (645 LOC): plug-and-play provider architecture, merge/dedup, single-pass synthesis
- **analytical_modes.py**: full box room modes (axial+tangential+oblique), Numba JIT (2.9s for 37K modes)
- **generalized_modes.py**: non-box rooms via perpendicular pair detection

### Numba JIT recursive oscillator
Eliminates exp/cos from inner loop — pure multiply+subtract per sample.
47s → 2.9s (16x speedup) at 2 kHz, 346s → 26s at 4 kHz.

### Calibration results (analytical modes + catalog priors, 5 iters)
- Broadband T30: **1.7% error** — excellent
- 500 Hz: 14% error — reasonable  
- 250 Hz: 133% error — Kuttruff decay underestimates at low freq
- 1000-2000 Hz: 28-30% — needs more absorption data refinement

### Key finding: hybrid eigensolve + analytical is the answer
- Eigensolve modal ROM: exact at low freq (0.6% T30 at 250 Hz) but mesh-limited
- Analytical modes: exact frequencies at any freq, but Kuttruff decay is statistical
- Combined via unified pipeline: eigensolve dominates 0-400 Hz, analytical fills 400-8000 Hz
- The unified_modes.py architecture supports this with confidence-based merge

### Total commits on feature/axial-modes: 26
658c71d through 1eada0a

## Next session priorities
1. **Run eigensolve + analytical hybrid** via unified pipeline on finer mesh
2. **Calibrate per-surface absorption** with the hybrid (eigensolve low + analytical high)
3. **Phase 3 validation** with calibrated hybrid
4. **Phase 2 (other PC)**: frequency-domain ROM greedy basis
5. **CUDA**: port Numba kernel to CuPy for GPU target device

## Commits
658c71d, 846327d, 0e5d2d7, db7fa0c, 0b352e0, e563949, 4e149f6, 523eaf3,
c5a3404, 648090e, 25b0256, 22abc56, 0272442, 2a68f83, 39b6609, 2b3f9cb,
bcbb8ac, 25cf323, ec3907f
