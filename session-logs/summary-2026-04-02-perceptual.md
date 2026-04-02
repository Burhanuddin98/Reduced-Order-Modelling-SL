# Session Summary: 2026-04-02 — Perceptual IR Quality + Structural Acoustics

## Overview
Continued from 2026-04-01. Focused on perceptual IR quality: built spectral ISM, diffuse tail, comparison tools. Iterated through v1-v8 of simulated IR. Received GPU calibration results from other PC. Identified low-frequency structural acoustics as the next frontier.

## GPU Calibration Results (from other PC)
- Broadband T30: **1.0% error** (1.647s vs 1.663s measured)
- 500 Hz: **4.6%**, 1000 Hz: **6.6%**, 2000 Hz: **7.5%** — all passing
- 250 Hz: 16% (needs eigensolve), 4000 Hz: 24% (air absorption model)
- Catalog materials were already correct — calibration barely moved scale factors
- 63 evals in 1810s (27s/eval on GPU vs 160s on CPU)

## New Modules
| Module | Purpose |
|--------|---------|
| ism_spectral.py | Frequency-dependent ISM with per-bounce spectral filtering + scattering smear |
| diffuse_tail.py | Per-third-octave decaying noise with measured-calibrated decay rates |
| cuda_synthesis.py | CUDA GPU kernel for recursive oscillator synthesis |
| statistical_modes.py | Weyl-density mode fill for irregular rooms |
| unified_modes.py | Plug-and-play provider architecture with GPU auto-dispatch |
| analytical_modes.py | Full box room modes (A+T+O) with Numba JIT |
| generalized_modes.py | Non-box modes from perpendicular pair detection |

## IR Version Progression
| Version | Architecture | Sound Quality | Issue |
|---------|-------------|--------------|-------|
| v1-v2 | Axial modes only | Tunnel/tube sound | Only axial resonances, no broadband energy |
| v3 | ISM + axial + diffuse | Better but axial dominant | Axial modes +10-20dB too loud |
| v4 | ISM + axial(10%) + diffuse | Much better | T30 too short (wrong decay rates) |
| v5 | Spectral ISM + diffuse | Natural spectrum | Pre-signal from sosfiltfilt, T30 short |
| v6 | + measured-calibrated decay | T30 matching | Pre-signal, band jumps |
| v7 | + causal filters, 1/3-oct | Clean, good decay | Still some spectral issues |
| v8 | + scattering smear | Best yet | Low-freq excess at 80-200 Hz |

## Perceptual Metrics Developed
- 1/3-octave energy RMS difference: **4.8 dB** (sim vs measured)
- Natural position variation: **2.8-6.2 dB** (measured vs measured)
- Our sim error is within natural variation except at 80 Hz (+14 dB)
- Band envelope correlation: 0.84 mean (0.64 at 125 Hz, 0.97 at 4 kHz)
- Early reflection peak matching: 100% (49/49 within 1ms)

## Key Finding: Low-Frequency Structural Acoustics
The 80-200 Hz excess (+14 dB at 80 Hz) is caused by rigid-wall ISM assumption.
Real walls vibrate at low frequencies:
- Plasterboard resonance: ~50-150 Hz (membrane absorption)
- Floor/ceiling structural coupling: energy transmitted through structure
- Nonlinear behavior: amplitude-dependent absorption at resonance
- These effects are NOT captured by alpha(f) alone — need structural impedance model

## Architecture Status
```
Unified Modal Synthesis (unified_modes.py)
  ├── AnalyticalModesProvider (box, conf=1.0)
  ├── ModalROMProvider (eigensolve, conf=0.95)
  ├── AxialModesProvider (parallel surfaces, conf=0.65)
  ├── StatisticalModesProvider (Weyl fill, conf=0.4)
  └── GPU auto-dispatch: CUDA → Numba → numpy

IR Synthesis Pipeline:
  ISM spectral (early reflections, freq-dep, scatter-smeared)
  + Modes (resonant structure via unified pipeline)
  + Diffuse tail (1/3-oct decaying noise, measured-calibrated)
```

## Next Session Priorities
1. **Low-frequency structural model** — wall impedance / membrane absorption below 200 Hz
2. **Eigensolve + analytical hybrid** on finer mesh for 250 Hz accuracy
3. **Non-box room validation** — CR3/CR4 with generalized modes + statistical fill
4. **Merge to main** — feature/axial-modes is stable, 30+ commits
