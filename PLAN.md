# RoomGUI Acoustics Engine — Implementation Plan

**Location:** `c:\RoomGUI\ROM`
**Repository:** https://github.com/Burhanuddin98/Reduced_Order_Modelling

---

## What We Have (Built and Validated)

| Component | File | Status |
|-----------|------|--------|
| SEM assembly (hex, P=4) | `room_acoustics/sem.py` | Validated — eigenfrequencies match analytical |
| SEM assembly (quad, unstructured) | `room_acoustics/unstructured_sem.py` | Validated — 2D and 3D extruded |
| SEM assembly (tet, P=2, Numba) | `room_acoustics/tet_sem.py` | Validated — arbitrary geometry |
| Gmsh mesh generation | `room_acoustics/geometry.py`, `gmsh_tet_import.py` | STL/OBJ import working |
| Time-domain FOM (p-Φ, RK4) | `room_acoustics/solvers.py` | Working, GPU via CuPy |
| Snapshot ROM (POD/PSD) | `room_acoustics/solvers.py` | 100-500x speedup |
| Modal ROM (eigenmodes) | `room_acoustics/modal_rom.py` | 0.6% T30 error at 250 Hz |
| Image source method | `room_acoustics/image_source.py` | Shoebox rooms, with scattering |
| Material database | `room_acoustics/materials.py` | 22 materials, per-surface assignment |
| ISO 3382 metrics | `room_acoustics/acoustics_metrics.py` | T30, EDT, C80, D50, TS |
| Impedance fitting | `room_acoustics/impedance_fit.py` | Miki-to-Sabine via Paris integration |
| Freq-domain Helmholtz | `room_acoustics/freq_domain.py` | Working, ROM gives 38,000x speedup |
| CUDA sparse solver | `solver_core/helmholtz_gpu.cu` | Compiled, running on RTX 2060 |
| BRAS validation | `results/bras_cr2_*.json` | Tested against measured IRs |

## Best Results So Far

| Test | Method | Error |
|------|--------|-------|
| BRAS CR2 @ 250 Hz | Modal ROM | **0.6%** |
| BRAS CR2 @ 500 Hz | Hybrid (modal + ISM + scatter) | **5.4%** |
| BRAS CR2 @ 1000 Hz | Hybrid | **1.7%** |
| BRAS CR2 broadband T30 | Hybrid | **9.0%** |
| Eigenfrequencies (Dauge benchmark) | SEM hex | **machine precision** |
| ROM speedup | Snapshot-based | **500x** |
| ROM speedup | Frequency-domain | **38,000x** |

---

## The Product: What It Should Be

A room acoustics engine where a user:
1. Loads a 3D room geometry (STL/OBJ/polygon+extrusion)
2. Assigns materials to surfaces from a database
3. Places source and receiver positions
4. Gets an impulse response with T30/EDT/C80/D50
5. Can listen to the room (auralization)
6. Can change materials/positions and get instant results

---

## Implementation Plan

### Phase 1: Unified Room API (1-2 days)

**Goal:** One class that ties everything together.

```python
from roomgui import Room

room = Room.from_stl("concert_hall.stl", h_target=0.3)
room.set_material("floor", "wood_floor")
room.set_material("ceiling", "acoustic_panel")
room.set_material_default("plaster")

room.build()  # mesh + operators + eigenmodes

ir = room.impulse_response(source=(3,7,1.5), receiver=(15,7,1.2))
print(f"T30={ir.T30:.2f}s, EDT={ir.EDT:.2f}s, C80={ir.C80:.1f}dB")
ir.save_wav("hall_ir.wav")
```

**New file:** `room_acoustics/room.py`
- `Room` class wrapping mesh, operators, eigenmodes, materials
- `Room.from_stl()`, `Room.from_obj()`, `Room.from_polygon()`
- `Room.set_material()`, `Room.set_material_default()`
- `Room.build()` — mesh + assemble + eigensolve (cached)
- `Room.impulse_response()` — returns `ImpulseResponse` object
- `ImpulseResponse` — holds IR data, computes metrics on demand, saves WAV

**Changes to existing files:** None. This is a wrapper layer.

### Phase 2: Auralization (half day)

**Goal:** Convolve IR with dry audio, write WAV.

**New file:** `room_acoustics/auralize.py`
- `convolve(ir, audio_path) -> ndarray`
- `save_wav(ir, path, sr=44100)`
- `play(ir, audio_path)` — optional, if sounddevice is installed

**Dependencies:** scipy.io.wavfile (already used), numpy (already used)

### Phase 3: Optimize CUDA Solver (2-3 days)

**Goal:** 10-50x speedup on frequency-domain solves.

**File:** `solver_core/helmholtz_gpu.cu`

Three optimizations:
1. **Symbolic factorization reuse** — `cusolverSpZcsrlsvchol` with
   analysis phase done once, numeric factorization per frequency
2. **Batch solve** — upload all frequency shifts at once, solve in
   a loop on GPU without host-device sync per frequency
3. **Receiver extraction on GPU** — don't transfer full solution,
   just extract `x[rec_idx]` on device

Expected result: N=23K from 200s/freq → 2-5s/freq.
N=3K from 3.7s/freq → 0.05s/freq.

### Phase 4: Full-Bandwidth Frequency-Domain ROM (2-3 days)

**Goal:** Accurate IR across 20-4000 Hz from a single method.

**File:** Update `room_acoustics/freq_domain.py`

1. **Adaptive training frequency selection** — place training points
   near resonances (detected from the transfer function magnitude)
2. **Greedy basis enrichment** — add basis vectors where the ROM
   error is largest (standard RBM approach from Sampedro Llopis)
3. **Error estimator** — cheap residual-based estimate of ROM error
   at untrained frequencies, used to drive the greedy algorithm
4. **Full H(ω) → IR pipeline** — solve at 2000+ frequencies via
   ROM, IFFT, apply air absorption in frequency domain

Expected result: T30 within 5-10% across 125-4000 Hz octave bands,
computed in <10 seconds after training.

### Phase 5: Validation Suite (1-2 days)

**Goal:** Automated comparison against BRAS measurements.

**File:** `room_acoustics/validate_bras.py`

1. Run engine on BRAS Scene 9 (seminar room) — compare T30, EDT,
   C80, D50 per octave band against all 10 measured RIRs
2. Run on BRAS Scene 11 (auditorium) — second room validation
3. Generate comparison plots + JSON data
4. Pass/fail criteria: T30 within 10% per octave band (250-2000 Hz)

### Phase 6: Complex Geometry Support (1-2 days)

**Goal:** Run on real-world geometries without manual setup.

1. **Auto-surface labeling** — Gmsh labels surfaces by angle/area;
   map to floor/ceiling/walls automatically based on normal direction
2. **Mesh quality checker** — warn about degenerate elements
3. **Default material heuristics** — "room with carpet" presets
4. **Test on ARD room, ballroom STL, car cabin**

### Phase 7: Interactive Mode (2-3 days)

**Goal:** Change source/receiver position and hear the result instantly.

The eigenmodes/ROM basis are precomputed. Changing source or receiver
only requires:
- Modal ROM: recompute modal amplitudes (dot product, <1ms)
- Freq-domain ROM: update RHS projection (dot product, <1ms)

This enables:
- **Source/receiver sweep** — evaluate 1000 positions in 1 second
- **Real-time material adjustment** — change wall materials,
  recompute modal decay rates, new IR in 0.2 seconds
- **RT60 map** — T30 at every point in the room, visualized as
  a heatmap

---

## Architecture

```
roomgui/
├── room_acoustics/          ← Python engine
│   ├── room.py              ← NEW: unified Room API
│   ├── auralize.py          ← NEW: audio convolution + WAV
│   ├── sem.py               ← structured hex assembly
│   ├── unstructured_sem.py  ← unstructured hex/quad assembly
│   ├── tet_sem.py           ← tet assembly (Numba)
│   ├── geometry.py          ← Gmsh mesh generation
│   ├── gmsh_tet_import.py   ← STL/OBJ import
│   ├── solvers.py           ← time-domain FOM + ROM
│   ├── modal_rom.py         ← eigenmode-based ROM
│   ├── freq_domain.py       ← frequency-domain solver + ROM
│   ├── image_source.py      ← ISM for early reflections
│   ├── materials.py         ← material database
│   ├── acoustics_metrics.py ← ISO 3382 metrics
│   ├── impedance_fit.py     ← Miki parameter fitting
│   ├── results_io.py        ← JSON data export
│   └── visualize.py         ← pressure field plots
├── solver_core/             ← CUDA C engine
│   ├── helmholtz.h          ← C API
│   ├── helmholtz_gpu.cu     ← CUDA solver
│   ├── helmholtz_py.py      ← Python ctypes wrapper
│   ├── build.bat            ← Windows build script
│   └── CMakeLists.txt       ← CMake build
├── results/                 ← validation data (JSON + PNG)
├── THEORY.md                ← complete theory documentation
├── PLAN.md                  ← this file
└── README.md
```

---

## Priority Order

1. **Phase 1** (Room API) — makes everything usable
2. **Phase 2** (Auralization) — makes it tangible (you can hear it)
3. **Phase 3** (CUDA optimization) — makes full-bandwidth practical
4. **Phase 4** (Freq-domain ROM) — the core differentiator
5. **Phase 5** (Validation) — proves it works
6. **Phase 6** (Complex geometry) — real-world applicability
7. **Phase 7** (Interactive) — the product experience

---

## What Makes This Different

**vs ODEON/CATT:** Wave-based accuracy at low frequencies (room modes,
diffraction) that geometric acoustics cannot capture. Plus ROM for
instant parameter sweeps.

**vs Treble:** Same wave-based approach but with modal ROM for
dispersion-free reverberation and instant source/receiver changes.
Treble requires cloud HPC; this runs on a laptop with an RTX GPU.

**vs COMSOL:** Purpose-built for room acoustics, not general PDE.
Material database, ISO 3382 metrics, auralization built in.
ROM for real-time parameter exploration.

**The unique combination:** SEM spatial accuracy + modal ROM for
dispersion-free T30 + frequency-domain ROM for full bandwidth +
CUDA acceleration + simple Python API. Nobody else has all of these
in one package.
