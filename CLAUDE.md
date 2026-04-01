# Reduced Order Modelling — Claude Code Instructions

## What is this project?
Room acoustics simulator using Model Order Reduction (ROM) to compress full-scale wave simulations from thousands of unknowns to tens, achieving 100-500x speedup. Computes impulse responses and ISO 3382 metrics (T30, EDT, C80, D50, TS) for arbitrary room geometries. Python + C/CUDA hybrid architecture.

## Project structure

```
room_acoustics/                 Core Python package (~9,200 LOC)
  room.py                      High-level Room API, hybrid engine orchestration
  solvers.py                   FOM/ROM solvers, p-v/p-Phi formulations, RK4
  sem.py                       Structured mesh (rect/box), Kronecker assembly
  unstructured_sem.py          Unstructured quad/hex mesh, element-by-element
  tet_sem.py                   Tetrahedral elements P=2, arbitrary 3D geometry
  geometry.py                  Room shapes, Gmsh meshing interface
  gmsh_tet_import.py           Tet mesh import from Gmsh .msh files
  modal_rom.py                 Eigenmode-based ROM, analytical time evolution
  freq_domain.py               Helmholtz frequency-domain solver + greedy ROM
  laplace_domain.py            Laplace-domain (s-domain) solver
  lanczos_rom.py               Lanczos eigenvalue solver with UMFPACK
  impedance_fit.py             Vector fitting for freq-dependent materials (Miki)
  image_source.py              ISM for early/late reflections
  ray_tracer.py                Python ray tracer for high-frequency content
  axial_modes.py               Parallel surface detection + 1D axial mode synthesis
  material_function.py         Frequency-dependent absorption functions alpha(f) at any resolution
  calibrate_absorption.py      Per-surface absorption calibration from measured RIRs
  materials.py                 22-material database, per-surface assignment (FI legacy)
  acoustics_metrics.py         ISO 3382 metrics (T30, T20, EDT, C80, D50, TS)
  visualize.py                 Pressure field plots, energy decay curves
  results_io.py                JSON result export
  validate.py                  9 validation tests (rectangular, L-shaped)
  validate_3d.py               3D box validation
  validate_eigenfrequencies.py Eigenfrequency validation vs analytical
  validate_unstructured.py     2D unstructured mesh validation
  validate_unstructured_3d.py  3D extruded hex validation
  validate_tet_3d.py           Tet element validation suite
  test_bras_cr2.py             BRAS benchmark (8.4x6.7x3.0 m seminar room)
  test_materials.py            Material library tests

solver_core/                   GPU-accelerated Helmholtz solver
  CMakeLists.txt               CUDA build (RTX 2060, SM 7.5)
  helmholtz.h                  C header for GPU solver
  helmholtz_gpu.cu             CUDA kernel (cuSOLVER + cuSPARSE)
  helmholtz_py.py              ctypes wrapper, numpy interface
  helmholtz_umfpack.c          UMFPACK sparse solver, frequency sweep
  helmholtz.dll                Precompiled Windows binary
  build.bat / build_umfpack.bat

engine/                        C room acoustics engine
  include/room_engine.h        C API: mesh, operators, modes, IR, metrics, WAV
  src/eigensolve.c             Shift-invert Lanczos + UMFPACK
  src/ray_tracer.c             Moller-Trumbore ray-triangle intersection (17x faster than Python)
  src/skp_reader.c/cpp         SketchUp SKP format parser
  CMakeLists.txt               UMFPACK eigensolve build
  build.bat / build_rt.bat / build_skp.bat

results/                       Output: JSON metrics, WAV IRs, PNG visualizations

docs/
  THEORY.md                    Full governing equations, SEM assembly, ROM derivation
  PLAN.md                      Development roadmap (7 phases, decision points)
  AUDIT.md                     Code audit vs Bonthu/Sampedro Llopis papers
  READ.me                      User guide + physics overview
```

## Signal chain (hybrid architecture)

```
Room Geometry -> Gmsh Mesh (quad/hex/tet)
  -> SEM Assembly (M, S, B operators)
  -> Eigensolve (Lanczos + UMFPACK)
  -> Modal ROM (0-f_cross Hz): per-mode analytical time evolution
  -> Axial Modes (f_cross-f_max): 1D analytical between parallel surfaces
  -> Ray Tracer (f_cross-Nyquist): stochastic diffuse energy
  -> ISM (early reflections): image source + Eyring diffuse tail
  -> Crossover Blend -> Broadband IR
  -> ISO 3382 Metrics (T30, EDT, C80, D50, TS)
  -> WAV / JSON / Visualization
```

## Key documentation

```
CLAUDE.md                       This file — conventions, metrics, checklist
THEORY.md                       Full governing equations, SEM, ROM derivation
PLAN.md                         Development roadmap (7 phases, decision points)
AUDIT.md                        Code audit vs Bonthu/Sampedro Llopis papers
docs/architecture.md            Complete system reference (KEEP UP TO DATE with every push)
docs/signal_chain.md            Build + query pipeline spec
docs/improvement_plan.md        Active roadmap + metrics (KEEP UP TO DATE)
docs/research_references.md     Papers, benchmarks, key concepts
docs/axial_mode_spec.md         Axial mode engine design spec (parallel surface detection + 1D modes)
```

## Starting a new session

1. `git pull --rebase origin main`
2. **Read this file** (CLAUDE.md) — conventions and current metrics
3. **Read the latest session summary**: `session-logs/` (most recent date)
4. **Check the roadmap**: `docs/improvement_plan.md` for remaining work
5. **Ask the user** what they want to work on

## Dependencies

**Python 3.8+:**
- scipy, numpy (sparse linear algebra, eigensolvers)
- gmsh (Python API, automated meshing)
- matplotlib (visualization)
- scikit-sparse (optional, CHOLMOD for faster sparse solves)
- numba (optional, JIT acceleration for tet assembly)

**C/CUDA (optional, for performance-critical paths):**
- CMake 3.18+
- CUDA 11.8+ with cuSOLVER + cuSPARSE (GPU solver)
- UMFPACK / SuiteSparse (sparse direct solver)

## Running

```python
from room_acoustics.room import Room

room = Room(width=8.4, depth=6.7, height=3.0)
room.set_material("all", "plaster")
room.set_source([4.0, 3.0, 1.5])
room.set_receiver([2.0, 5.0, 1.2])
room.solve(method="hybrid", f_max=1000)
room.plot_metrics()
```

**Validation tests:**
```bash
cd room_acoustics
python validate.py              # Main 9-test suite
python validate_eigenfrequencies.py
python test_bras_cr2.py         # BRAS benchmark
```

## Current metrics (2026-04-01)

### Broadband (BRAS CR2, 74K DOFs, hybrid platform)
- **T30 error**: 3.0% — PASS
- **EDT error**: 28% — needs freq-dep absorption
- **C80 delta**: 2.7 dB — borderline

### Modal ROM (BRAS CR2, 250 Hz)
- **T30 error**: 0.6%
- **EDT error**: 2.5%
- **Synthesis time**: 0.2s for 3.5s IR

### Axial mode engine (verified vs measured RIRs)
- **Spectral peak match**: 88% (35/40 predicted modes within ±3 Hz)
- **Decay rate (coupling model)**: 28% mean error, 24% median
- **Position dependence**: 16.4 dB spread confirmed

### Known limitations
- Octave-band T30 rises with frequency (20-67% error) — FI impedance lacks freq-dep absorption
- Freq-domain ROM: 12 basis vectors gave 63-192% error at resonances
- Non-shoebox rooms: limited validation beyond rectangular boxes

## Building C/CUDA components

From respective directories:
- **GPU Helmholtz**: `solver_core/build.bat`
- **UMFPACK Helmholtz**: `solver_core/build_umfpack.bat`
- **C eigensolve**: `engine/build.bat`
- **C ray tracer**: `engine/build_rt.bat`
- **SketchUp reader**: `engine/build_skp.bat`

Precompiled DLLs are checked into the repo for Windows.

## Key references

- **Bonthu et al. (2026)** — Structure-preserving MOR (p-Phi + PSD basis + boundary enrichment)
- **Sampedro Llopis et al. (2022)** — Laplace-domain parametric ROM
- **BRAS CR2** — 8.4x6.7x3.0 m seminar room benchmark with measured RIRs

---

## Pre-push checklist

Before every `git push`, verify:

1. **Validation passes** — run `python validate.py` (no regressions)
2. **Update metrics** — if accuracy changed, update this file's metrics section AND `docs/improvement_plan.md`
3. **Write session summary** — `session-logs/summary-YYYY-MM-DD-topic.md`
4. **Update architecture** — if modules, data flow, or pipeline changed: `docs/architecture.md`
5. **Update PLAN.md** — if roadmap status changed
6. **Update THEORY.md** — if governing equations or formulations changed

## Session summary format

Write one summary per working session at `session-logs/summary-YYYY-MM-DD-topic.md`.

Include:
- Session overview (1-2 sentences)
- Commits this session (hash + description)
- Key changes (what and why)
- Current metrics (if changed)
- Known issues or next steps

## What NOT to do
- Don't commit large binary files (WAVs, result PNGs) without checking they're needed
- Don't modify precompiled DLLs without rebuilding from source
- Don't use hardcoded absolute paths (use relative paths or auto-detection)
- Don't break the Room API interface without updating all validation scripts

## Claude Code conventions (for all developers)

- **Session summaries are mandatory** — write one per working session before the final push
- **No long sleeps** — when waiting for processes, check periodically with short intervals
- **Terse responses** — keep explanations concise, no trailing summaries
- **Validation before push** — run validate.py before every push, no exceptions
- **Theory docs stay current** — if you change a formulation, update THEORY.md

## Multi-machine workflow

- Always `git pull --rebase` before starting work and before pushing
- Both developers push to `main` — communicate via session summaries
- For large changes (multi-day refactors), use a short-lived feature branch and merge when done
- If Claude Code memory is empty on a new machine, all conventions are in this file
