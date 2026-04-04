# Session Summary — 2026-04-04 — Sampedro Replication + CHORAS Package

## Overview
Completed 3D Sampedro replication, FOM validation, built romacoustics pip package v0.2 with CHORAS interface, researched CHORAS architecture in detail.

## Commits this session
- `120af0a` Add WAV + NPZ exports for all Sampedro IRs
- `533476d` Colab-ready paper reproduction (all figures)
- Multiple Colab fixes (GPU, import, idempotent setup)
- `310b4a2` romacoustics v0.1.0: pip-installable package
- `017946c` Professional README with pipeline diagram
- `5fa9fb1` romacoustics v0.2: CHORAS-ready with per-surface materials

## Key results

### 3D Sampedro replication (GPU GMRES)
- N=35,937 (Ne=8, P=4), freq-dependent Miki boundaries
- d=0.05m: Nrb=16, speedup 6,750x, error 0.8%
- d=0.15m: Nrb=16, speedup 9,202x, error 0.8%
- 100% GPU solves, zero CPU fallback
- All WAVs + NPZs saved to results/sampedro_wav/

### FOM validation
- Eigenfrequencies: 9/10 match analytical within 2.3 Hz
- Laplace vs TD (FI): relative error 6.85e-4
- Three-way comparison (Laplace vs TD vs Analytical) confirms correctness

### romacoustics v0.2 package
New modules: choras_interface.py, materials.py, metrics.py, unstructured.py
- Per-surface boundary mass (B_labels) in both 2D and 3D assembly
- 22-material database with absorption↔impedance conversion
- Octave-band ISO 3382 metrics (T30, T20, EDT, C80, D50, TS)
- CHORAS solver backend (rom_method entry point)
- P1 tet mesh with per-surface boundary mass

### CHORAS research findings
- CHORAS platform is still prototype — no unified material database
- Three solvers use three different absorption formats
- edg-acoustics: multi-pole TDIBC (needs MATLAB vector fitting)
- acousticDE: octave-band alpha CSV
- Our romacoustics: impedance Z (with alpha conversion)
- No BRAS integration exists anywhere
- Entry point: one function, one JSON file in/out

## Next session
- BRAS Scene 9 validation: load measured RIRs, assign per-surface materials, compare octave-band T30/EDT/C80
- Test with real Gmsh mesh (not just box)
- Write contribution proposal for TU Eindhoven team
