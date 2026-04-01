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

## Next session priorities
1. **Vectorize synthesis** — chunked numpy for 100K+ modes
2. **Calibrate with full analytical modes** — use catalog priors + regularization
3. **Integrate ISM** for early reflections with analytical modes for late field
4. **Phase 3 re-test** with calibrated full-spectrum approach
5. **Phase 2 (other PC)**: frequency-domain ROM greedy basis

## Commits
658c71d, 846327d, 0e5d2d7, db7fa0c, 0b352e0, e563949, 4e149f6, 523eaf3,
c5a3404, 648090e, 25b0256, 22abc56, 0272442, 2a68f83, 39b6609, 2b3f9cb,
bcbb8ac, 25cf323, ec3907f
