# RoomGUI Acoustics Engine — Implementation Plan

**Location:** `c:\RoomGUI\ROM`
**Repository:** https://github.com/Burhanuddin98/Reduced_Order_Modelling

---

## Current State (Honest Assessment)

### What works
- SEM spatial discretization: hex (P=4), quad, tet (P=2) — eigenfrequencies match published benchmarks to machine precision
- Modal ROM: 0.6% T30 error at 250 Hz on BRAS CR2. Zero numerical dispersion. 0.2s synthesis for 3.5s IR
- Hybrid (modal + ISM + scatter): 1.7-6% error at 250-1000 Hz on BRAS CR2
- Frequency-domain Helmholtz solver: working, ROM gives 38,000x speedup per frequency
- CUDA solver: compiled, running on RTX 2060, correct results
- Material database: 22 materials, per-surface assignment
- Tet assembly: Numba-accelerated, 184K DOFs in 10 seconds
- STL/OBJ import via Gmsh: tested on ARD test room

### What doesn't work yet
- **Frequency-domain ROM basis quality**: 12 basis vectors gave 63-192% error at resonances. Not enough training points
- **High-frequency octave-band T30**: 24-33% error at 2-4 kHz (scattering model too simple)
- **Non-shoebox validation**: zero rooms tested with modal ROM or freq-domain solver beyond rectangular boxes
- **CUDA solver speed**: slower than scipy for N<50K (no symbolic factorization reuse)
- **Broadband IR from single method**: no method currently gives accurate full-bandwidth results alone
- **Auralization**: no audio output capability

### The one question that matters
**Can we get <10% T30 error across 250-4000 Hz octave bands on rooms that aren't shoeboxes?**

Everything else is packaging. This question is unanswered.

---

## Phase 1: Fix the Frequency-Domain Solver Speed (Day 1)

### Problem
scipy's `spsolve` redoes full LU factorization at every frequency (2.8s at N=3K, 200s at N=23K). The sparsity pattern is identical — only the diagonal changes.

### Solution
Use `scipy.sparse.linalg.splu` to pre-analyze the sparsity pattern. For each frequency, build the shifted matrix and call `splu` which reuses the symbolic analysis internally when the pattern matches.

Better approach: use `sksparse.cholmod` (SuiteSparse Python wrapper) which explicitly supports analyze-once, factor-many. For symmetric positive definite systems (which ours is when we add a small imaginary shift), Cholesky is 2-3x faster than LU.

### Files to change
- `room_acoustics/freq_domain.py`: replace `spsolve` loop with `splu` or cholmod

### Acceptance test
- N=3K: <0.1s per frequency (currently 2.8s)
- N=23K: <2s per frequency (currently 200s)
- Results match scipy `spsolve` to machine precision

### Risk
Low. This is well-understood engineering. sksparse may need installing (`pip install scikit-sparse`), which requires SuiteSparse C library. Fallback: use `splu` from scipy directly.

---

## Phase 2: Frequency-Domain ROM Accuracy (Days 2-4)

### Problem
The ROM basis from 15 training frequencies missed resonances. 12 basis vectors can't represent the sharp peaks in H(ω) near room modes.

### Root cause
Room modes are narrow peaks in the transfer function. A basis trained at frequencies between the peaks misses the peak shape entirely. The ROM interpolates smoothly where the true response has spikes.

### Solution: Adaptive greedy training
1. Start with 10-20 logarithmically spaced training frequencies
2. Build initial ROM basis (SVD of solution snapshots)
3. Evaluate ROM at 500+ test frequencies
4. At each test frequency, compute the **residual** `r = f - A_h * x_rom` (cheap — one sparse matvec)
5. Find the frequency where the residual is largest
6. Solve the FOM at that frequency, add the solution to the snapshot set
7. Rebuild the basis (incremental SVD or just re-SVD)
8. Repeat until max residual < tolerance

This is the standard greedy RBM algorithm from Hesthaven & Rozza. It automatically places training points at resonances.

### Files to change
- `room_acoustics/freq_domain.py`: add `build_greedy_rom()` function

### Acceptance test
- ROM H(ω) matches FOM H(ω) to <1% at all frequencies 20-500 Hz
- Basis size: 30-60 vectors (from ~50 FOM solves)
- Full sweep (500 frequencies): <0.1s after training

### Risk
Medium. The residual estimator needs careful implementation. The basis might grow too large for highly resonant rooms. But the algorithm is well-established in the RBM literature.

---

## Phase 3: Validate on BRAS CR2 Full Bandwidth (Days 4-5)

### What to do
1. Train the greedy ROM on BRAS CR2 (shoebox, N=23K, P=4 hex)
2. Evaluate H(ω) at 20-4000 Hz with 1 Hz resolution
3. IFFT to get broadband IR
4. Compute octave-band T30 at 125, 250, 500, 1000, 2000, 4000 Hz
5. Compare against all 10 measured dodecahedron RIRs from BRAS
6. Use frequency-dependent Z(ω) from BRAS fitted absorption data
7. Include air absorption (ISO 9613-1 at BRAS temperature/humidity)

### Materials (from BRAS CSVs)
Use the 31 third-octave-band absorption values directly. For each frequency in the sweep, interpolate alpha(f) per surface, convert to Z(f), build the system matrix. The frequency-domain solver handles this naturally — Z changes at each frequency.

### Acceptance criteria
- T30 per octave band (250-2000 Hz): <10% error vs measured mean
- Broadband T30: <10% error
- C80: within 2 dB of measured
- EDT: within 20% of measured

### What this proves
The frequency-domain ROM with frequency-dependent materials can match real measurements on a real room. This is the core validation.

### Risk
Medium-high. The 250 Hz band should work (modal ROM already proved this at 0.6%). Higher bands depend on:
- Whether the mesh (N=23K) resolves 2-4 kHz (it doesn't — need N=100K+)
- Whether the ROM basis captures high-frequency resonance structure
- Whether the impedance BC model is accurate at high frequencies

If N=23K doesn't resolve high frequencies, run on N=100K+ mesh (feasible if Phase 1 brings per-frequency solve to <2s).

---

## Phase 4: Validate on Non-Shoebox Room (Days 5-7)

### Why this matters
Every result so far is on rectangular rooms. The claim "works on arbitrary geometry" is untested for acoustic accuracy. Assembly and eigenfrequency checks pass on L-shapes and STL imports, but T30 has never been validated on non-rectangular geometry.

### What to do

**Test A: BRAS Scene 11 (auditorium)**
- Already have measured RIRs (downloaded)
- Approximate geometry as shoebox with estimated dimensions
- Run frequency-domain ROM, compare T30
- This tests the method on a larger room with different materials

**Test B: ARD test room (STL import)**
- Already have the mesh (9K tets, 15K DOFs)
- No measured data — but can compare eigenfrequencies against analytical estimates
- Run frequency-domain ROM, check if T30 is physically plausible
- This tests the pipeline on a true arbitrary geometry

**Test C: L-shaped room (extruded)**
- Non-rectangular but we control the geometry exactly
- No measured data — use Sabine/Eyring as reference
- Run both modal ROM and frequency-domain ROM, compare
- This tests whether the two ROM approaches agree on a non-trivial geometry

### Acceptance criteria
- Scene 11: T30 within 15% of measured (we had 24% with modal ROM + guessed materials)
- ARD room: eigenfrequencies physically plausible, T30 in reasonable range
- L-shape: modal and freq-domain ROM agree within 5%

### Risk
High for Scene 11 (we don't have exact geometry, only estimated dimensions). Medium for others.

---

## Phase 5: Auralization + Room API (Days 7-8)

### Only build this after Phase 3 passes

**Auralization** (`room_acoustics/auralize.py`):
```python
def convolve_ir(ir, audio_path, output_path, sr=44100):
    """Convolve impulse response with audio, write WAV."""

def save_ir_wav(ir, path, sr=44100):
    """Save IR as WAV file."""
```

**Room API** (`room_acoustics/room.py`):
```python
class Room:
    @classmethod
    def from_stl(cls, path, h_target=0.3, P=4): ...

    @classmethod
    def from_box(cls, Lx, Ly, Lz, Nex, Ney, Nez, P=4): ...

    def set_material(self, surface_label, material_name): ...
    def build(self): ...  # mesh + operators + ROM training
    def impulse_response(self, source, receiver): ...
    def t30(self, source, receiver): ...
    def sweep_receivers(self, source, receivers): ...  # batch
```

### Risk
Low. This is wrapping working code. The hard part is already done (if Phase 3 passes).

---

## Phase 6: CUDA Solver Optimization (Days 8-10)

### Only do this if scipy is too slow for the target mesh size

Three approaches, in order of effort:

**A. scikit-sparse CHOLMOD (easiest)**
```
pip install scikit-sparse
```
SuiteSparse's CHOLMOD supports analyze-once, factor-many natively. 5-10x speedup over scipy spsolve. Python, no CUDA needed.

**B. Custom CUDA kernel (medium)**
Rewrite `helmholtz_gpu.cu` to use `cusolverSpXcsrluAnalysis` + `cusolverSpDcsrluFactor` + `cusolverSpDcsrluSolve` (the three-phase API). Symbolic analysis once, numeric factor per frequency.

**C. Batched GPU solve (hardest, biggest payoff)**
Solve multiple frequencies simultaneously on GPU. cuSOLVER doesn't support this directly, but we can:
- Use cuBLAS batched dense solve on the ROM system (r×r, trivially parallel)
- Or use MAGMA library for batched sparse solves

### Acceptance test
- N=23K: <0.5s per frequency
- N=100K: <5s per frequency
- ROM training (50 frequencies): <5 minutes total

---

## Phase 7: Interactive Mode + Product Demo (Days 10-12)

### Only after everything above works

**Source/receiver sweep:**
- Precompute eigenmodes or ROM basis (one-time)
- Evaluate at 1000 receiver positions in <1 second
- Generate T30 heatmap of the room

**Material comparison:**
- Change material on one surface
- Recompute T30 in <0.5 seconds
- A/B comparison: "carpet vs hardwood floor"

**Demo script:**
```python
room = Room.from_stl("concert_hall.stl")
room.set_material_default("plaster")
room.build()

# Sweep receivers
grid = room.receiver_grid(height=1.2, spacing=0.5)
t30_map = room.sweep_t30(source=(3,7,1.5), receivers=grid)
t30_map.plot()  # heatmap
t30_map.save("t30_map.png")

# A/B comparison
room.set_material("ceiling", "acoustic_panel")
ir_treated = room.impulse_response((3,7,1.5), (15,7,1.2))
ir_treated.save_wav("treated.wav", audio="speech.wav")

room.set_material("ceiling", "concrete")
ir_untreated = room.impulse_response((3,7,1.5), (15,7,1.2))
ir_untreated.save_wav("untreated.wav", audio="speech.wav")
```

---

## Decision Points

### After Phase 3:
- If BRAS CR2 hits <10% across 250-2000 Hz → proceed to Phase 4
- If not → investigate why. Options: finer mesh, different BC model, more ROM training

### After Phase 4:
- If non-shoebox rooms work → proceed to Phase 5 (product layer)
- If not → the method has fundamental limitations for complex geometry. Pivot to hybrid (modal low + ray tracing high) as the product architecture

### After Phase 6:
- If CUDA gives <5s per frequency at N=100K → full-bandwidth on large rooms is feasible
- If not → stick with smaller meshes + ROM, accept f_max limitation

---

## What Success Looks Like

**Minimum viable product:**
- Load STL → assign materials → get T30/EDT/C80 within 15% of measurement
- Total time: <5 minutes first run, <1 second for parameter changes
- Works on rooms up to ~2000 m³

**Competitive product:**
- T30 within 5% across 250-4000 Hz
- Full auralization (listen to the room)
- Source/receiver sweep in real-time
- Works on arbitrary geometry up to ~10,000 m³
- Total time: <10 minutes first run, instant parameter changes

**Game changer:**
- All of the above + runs on a laptop GPU
- Cloud deployment for larger rooms
- Material optimization: "what ceiling treatment minimizes RT60 below 2s?"
- Real-time VR integration
