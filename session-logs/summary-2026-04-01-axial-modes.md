# Session Summary: 2026-04-01 — Axial Mode Engine + Project Documentation

## Overview
Set up the project for Claude Code collaboration (CLAUDE.md, architecture, signal chain, improvement plan, research references). Designed and implemented the axial mode engine for mesh-free high-frequency resonance synthesis. Integrated into the hybrid IR pipeline as Engine 2.

## Commits this session
- (pending) Initial project documentation: CLAUDE.md, docs/architecture.md, docs/signal_chain.md, docs/improvement_plan.md, docs/research_references.md
- (pending) Axial mode engine: axial_modes.py, integration into room.py, validation tests, spec doc

## Key changes

### Documentation (new)
- **CLAUDE.md** — conventions, metrics, pre-push checklist, session workflow (mirrors ATTuner structure)
- **docs/architecture.md** — full module map, data flow, operators, element types, ROM methods, threading
- **docs/signal_chain.md** — build + query pipeline spec with crossover design
- **docs/improvement_plan.md** — 7-phase roadmap with metrics, decision points, tech debt
- **docs/research_references.md** — Bonthu, Sampedro Llopis, BRAS, key concepts
- **session-logs/** directory created

### Axial mode engine (new)
- **room_acoustics/axial_modes.py** (~220 LOC):
  - `detect_parallel_surfaces(rt_mesh)` — groups boundary triangles by label, area-weighted normals, finds anti-parallel pairs
  - `detect_parallel_surfaces_box(dimensions)` — shortcut for box rooms
  - `axial_mode_ir(pairs, source, receiver, materials, ...)` — vectorized 1D mode synthesis with analytical decay, source/receiver coupling, solid angle weighting, RT60-based minimum decay rate
- **docs/axial_mode_spec.md** — full design specification
- **room_acoustics/validate_axial_modes.py** — 5 validation tests, all passing

### Room API integration
- **room_acoustics/room.py**:
  - `build()` detects parallel surfaces (cached)
  - `impulse_response()` now runs 4-engine blend: modal ROM + axial modes + ray tracer + ISM
  - Level matching: axial modes scaled to ray tracer energy level
  - RT60-clamped decay prevents unphysical ringing

## Validation results
- 5/5 unit tests passing (frequencies, flutter echo, position dependence, decay rate, mesh detection)
- A/B comparison on BRAS CR2 (coarse mesh): axial modes add +0.1-0.4 dB per octave band without distorting T30/EDT/C80
- Verified against BRAS CR2 measured WAVs:
  - Spectral peak match: 88% (35/40 predicted modes found within ±3 Hz)
  - Decay rate (1D model): 51% mean error
  - Decay rate (coupling model): 28% mean, 24% median error
  - Position dependence: 16.4 dB spread across receivers confirmed

## 3D coupling loss model
Blends pair-specific and room-average decay:
  gamma_eff = (1 - coupling) * gamma_pair + coupling * gamma_room
  coupling  = 1 - A_pair / S_total
Reduces decay rate error from 51% → 28%. Best for low-order modes (<150 Hz).

## BRAS data downloaded (bras_data/, gitignored)
- CR2 seminar room: 10 dodecahedron RIRs, surface absorption CSVs
- CR3 chamber music hall: ready for Phase 4
- CR4 auditorium: ready for Phase 4
- Documentation PDF

## Measured ground truth (from BRAS CR2 WAVs)
- Broadband: T30=1.663s, EDT=1.166s, C80=2.8dB
- 250 Hz: T30=1.746s, 500 Hz: T30=2.024s, 1kHz: T30=1.939s
- 2 kHz: T30=1.745s, 4 kHz: T30=1.563s

## Current metrics (unchanged from 2026-03-31)
- Modal ROM: 0.6% T30 error on BRAS CR2 at 250 Hz
- Hybrid: 1.7-6% T30 error at 250-1000 Hz

## Phase 3 first run results
- Broadband T30: **3.0% error — PASS**
- Octave-band T30 (250-4000 Hz): 20-67% error — FAIL
- Root cause: FI impedance gives single alpha per surface; real materials absorb more at higher freq
- Ray tracer + axial modes need frequency-dependent absorption per octave band

## Commits this session (feature/axial-modes branch)
- 658c71d: Axial mode engine + project documentation
- 846327d: Phase 3 BRAS test + measured ground truth from WAVs
- 0e5d2d7: Axial mode verification vs BRAS measured RIRs (88% spectral match)
- db7fa0c: 3D coupling loss model (decay error 51% → 28%)
- 0b352e0: Housekeeping: session summary, spec doc, verification plot
- e563949: Phase 3 first run results

## Next steps
- **Per-band absorption calibration**: inverse problem — measure per-band RT from WAVs → infer alpha per surface per band. This replaces the FI single-alpha model for high-frequency engines.
- **Phase 2** (other PC): frequency-domain ROM greedy basis enrichment
