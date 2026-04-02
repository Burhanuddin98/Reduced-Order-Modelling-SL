# GPU Quickstart — Running on the CUDA Machine

## Setup

```bash
# Get the feature branch
git fetch origin
git checkout feature/axial-modes
git pull

# Install dependencies
pip install numpy scipy numba cupy-cuda12x matplotlib
# (adjust cupy-cuda12x to match your CUDA version: cupy-cuda11x, cupy-cuda12x, etc.)
```

## Full FDTD IR on GPU (the key experiment)

This is the most important test — a physically complete IR from the wave equation,
no noise, no energy matching hacks. Should produce natural C80 and correct
early/late balance directly from the physics.

```bash
cd room_acoustics
python -c "
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('.'))))
sys.path.insert(0, '..')
from room_acoustics.voxelize import voxelize_box, find_boundary_voxels
from room_acoustics.fdtd import FDTDSolver
from room_acoustics.material_function import MaterialFunction
from room_acoustics.acoustics_metrics import all_metrics
from room_acoustics.ir_score import score_ir_perceptual, print_scorecard
from room_acoustics.spectral_tools import load_wav, compare_irs
from scipy.signal import resample
import scipy.io.wavfile as wavfile
import numpy as np

csv_dir = '../bras_data/3 Surface descriptions/_csv/fitted_estimates'
mats = {
    'floor':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_floor.csv').with_structural_absorption(200, 0.3, 0.05),
    'ceiling': MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_ceiling.csv').with_structural_absorption(5, 0.3, 0.1),
    'front':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_concrete.csv').with_structural_absorption(200, 0.1, 0.03),
    'back':    MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_concrete.csv').with_structural_absorption(200, 0.1, 0.03),
    'left':    MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_windows.csv').with_structural_absorption(8, 0.02, 0.08),
    'right':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_plaster.csv').with_structural_absorption(10, 0.08, 0.05),
}

Lx, Ly, Lz = 8.4, 6.7, 3.0
src = (2.0, 3.35, 1.5); rec = (6.0, 2.0, 1.2)
dx = 0.05  # 5cm voxels -> f_max ~3.4 kHz

print('Voxelizing...')
air, origin, _ = voxelize_box(Lx, Ly, Lz, dx=dx, padding=3)
print(f'  Grid: {air.shape}, air: {air.sum()}')

# Per-voxel absorption
bndy_ijk, bndy_normals = find_boundary_voxels(air)
alpha_3d = np.zeros(air.shape, dtype=np.float32)
ox, oy, oz = origin
for idx in range(len(bndy_ijk)):
    i, j, k = bndy_ijk[idx]
    normal = bndy_normals[idx]
    if abs(normal[2]) > 0.5:
        mat = mats['floor'] if normal[2] < 0 else mats['ceiling']
    elif abs(normal[0]) > 0.5:
        mat = mats['left'] if normal[0] < 0 else mats['right']
    else:
        mat = mats['front'] if normal[1] < 0 else mats['back']
    alpha_3d[i, j, k] = mat(500.0)

# USE GPU
print('Creating FDTD solver (GPU)...')
solver = FDTDSolver(air, dx=dx, c=343.0, CFL=0.4, use_gpu=True)
solver.set_materials(alpha_3d=alpha_3d)

src_ijk = (int(round((src[0]-ox)/dx)), int(round((src[1]-oy)/dx)), int(round((src[2]-oz)/dx)))
rec_ijk = (int(round((rec[0]-ox)/dx)), int(round((rec[1]-oy)/dx)), int(round((rec[2]-oz)/dx)))
print(f'  fs={solver.fs:.0f}Hz, GPU={solver.use_gpu}')

print('Running FDTD (3.5s)...')
t0 = time.perf_counter()
ir_fdtd, fs_fdtd = solver.impulse_response(src_ijk, rec_ijk, duration=3.5, warmup_steps=32)
dt_run = time.perf_counter() - t0
print(f'  Done: {dt_run:.1f}s, {len(ir_fdtd)} samples at {fs_fdtd}Hz')

# Resample to 44100 Hz
ir_44k = resample(ir_fdtd, int(len(ir_fdtd) * 44100 / fs_fdtd)).astype(np.float64)
sr = 44100

# Save
ir_out = ir_44k.astype(np.float32)
ir_out = ir_out / max(abs(ir_out).max(), 1e-10) * 0.95
wavfile.write('../results/IR_fdtd_gpu_full.wav', sr, ir_out)
print(f'  Saved: results/IR_fdtd_gpu_full.wav')

# Score against measured
rir_dir = '../bras_data/1 Scene descriptions/CR2 small room (seminar room)/RIRs/wav'
ir_meas, _ = load_wav(f'{rir_dir}/CR2_RIR_LS1_MP1_Dodecahedron.wav')
n = min(len(ir_44k), len(ir_meas))

m = all_metrics(ir_44k[:n], 1/sr)
print(f'\\nMetrics: T30={m[\"T30_s\"]:.3f}s C80={m[\"C80_dB\"]:.1f}dB EDT={m[\"EDT_s\"]:.3f}s')
print(f'Measured: T30=1.663s C80=1.8dB EDT=1.166s')

r = score_ir_perceptual(ir_44k[:n], ir_meas[:n], sr)
print_scorecard(r)

compare_irs(ir_meas[:n], ir_44k[:n], sr=sr,
    output='../results/fdtd_gpu_vs_measured.png',
    title='Full FDTD (GPU, dx=0.05m, 3.5s) vs Measured')
"
```

**Expected on RTX 2060:** ~3-5 seconds for full 3.5s IR
**Expected on CPU:** ~6-10 minutes

This is the ground truth test: if FDTD gives correct C80 and good perceptual score,
it proves the wave equation approach works and GPU makes it practical.

## Quick test: GPU synthesis

```bash
cd room_acoustics
python -c "
from cuda_synthesis import has_gpu, synthesize_gpu
import numpy as np, time

print(f'GPU available: {has_gpu()}')

# Benchmark: 37K modes, 3s IR at 44.1kHz
n_modes = 37000
n_samples = 132300
amp = np.random.randn(n_modes) * 0.001
gam = np.random.uniform(1, 50, n_modes)
wd = np.random.uniform(100, 12000, n_modes).astype(np.float64)
nc = np.full(n_modes, 30000, dtype=np.int64)

# Warmup (CUDA kernel compilation)
synthesize_gpu(amp[:10], gam[:10], wd[:10], nc[:10], 1000, 1/44100)

# Benchmark
t0 = time.perf_counter()
ir = synthesize_gpu(amp, gam, wd, nc, n_samples, 1/44100)
dt = time.perf_counter() - t0
print(f'{n_modes} modes x {n_samples} samples: {dt:.3f}s')
print(f'Peak: {abs(ir).max():.6f}')
"
```

Expected: ~10-50ms on RTX 2060 (vs 2.9s CPU Numba).

## Run calibration on GPU

```bash
python -c "
import sys; sys.path.insert(0, '..')
from room_acoustics.unified_modes import UnifiedModalSynthesizer, AxialModesProvider
from room_acoustics.statistical_modes import StatisticalModesProvider
from room_acoustics.axial_modes import detect_parallel_surfaces_box
from room_acoustics.material_function import MaterialFunction
from room_acoustics.material_catalog import get_catalog_material
from room_acoustics.acoustics_metrics import compute_t30
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import minimize
import numpy as np, time

# Measured BRAS CR2 reference
measured = {250: 1.745, 500: 2.024, 1000: 1.939, 2000: 1.745}
bands = list(measured.keys())

# Catalog priors
catalog = {
    'floor':   get_catalog_material('linoleum'),
    'ceiling': get_catalog_material('acoustic_tile_perforated'),
    'front':   get_catalog_material('concrete_painted'),
    'back':    get_catalog_material('concrete_painted'),
    'left':    get_catalog_material('glass_window'),
    'right':   get_catalog_material('plaster_on_masonry'),
}
labels = list(catalog.keys())

Lx, Ly, Lz = 8.4, 6.7, 3.0
V = Lx*Ly*Lz; S = 2*(Lx*Ly+Lx*Lz+Ly*Lz)
pairs = detect_parallel_surfaces_box((Lx, Ly, Lz))

def make_mats(scale_vec):
    mats = {}
    for i, label in enumerate(labels):
        base = catalog[label]
        s = np.clip(scale_vec[i], 0.3, 3.0)
        mats[label] = MaterialFunction(base.freqs, np.clip(base.alphas*s, 0.001, 0.999))
    return mats

def objective(scale_vec):
    mats = make_mats(scale_vec)
    synth = UnifiedModalSynthesizer()
    synth.register(AxialModesProvider(pairs, V, S))
    synth.register(StatisticalModesProvider(V, S, f_min=20, f_max=4000))
    ir_obj, _ = synth.impulse_response(
        (2,3.35,1.5), (6,2,1.2), mats, T=3.0, humidity=41.7, temperature=19.5)
    
    err = 0.0
    sr=44100; nyq=sr/2
    for fc in bands:
        fl=fc/np.sqrt(2); fh=min(fc*np.sqrt(2),nyq*0.95)
        sos=butter(4,[fl/nyq,fh/nyq],btype='band',output='sos')
        t30,r2=compute_t30(sosfiltfilt(sos,ir_obj.data),1/sr)
        if r2>0.7 and not np.isnan(t30):
            err += (t30-measured[fc])**2
        else:
            err += measured[fc]**2
    err += 0.05*np.sum(np.log(scale_vec)**2)*len(bands)
    return err

# Run
print('Calibrating (20 iterations)...')
t0 = time.perf_counter()
result = minimize(objective, np.ones(len(labels)), method='L-BFGS-B',
                  bounds=[(0.3,3.0)]*len(labels),
                  options={'maxiter': 20, 'ftol': 1e-6})
print(f'Done: {time.perf_counter()-t0:.0f}s, {result.nfev} evals')

for i,label in enumerate(labels):
    print(f'  {label:10s}: x{result.x[i]:.2f} (alpha@500={catalog[label](500)*result.x[i]:.3f})')

# Validate
mats_cal = make_mats(result.x)
synth = UnifiedModalSynthesizer()
synth.register(AxialModesProvider(pairs, V, S))
synth.register(StatisticalModesProvider(V, S, f_min=20, f_max=4000))
ir_cal, _ = synth.impulse_response(
    (2,3.35,1.5), (6,2,1.2), mats_cal, T=3.0, humidity=41.7, temperature=19.5)

sr=44100; nyq=sr/2
print(f'\\nBroadband: T30={ir_cal.T30:.3f}s (err={abs(ir_cal.T30-1.663)/1.663*100:.1f}%)')
for fc in bands:
    fl=fc/np.sqrt(2); fh=min(fc*np.sqrt(2),nyq*0.95)
    sos=butter(4,[fl/nyq,fh/nyq],btype='band',output='sos')
    t30,r2=compute_t30(sosfiltfilt(sos,ir_cal.data),1/sr)
    err=abs(t30-measured[fc])/measured[fc]*100 if not np.isnan(t30) and r2>0.7 else float('nan')
    print(f'  {fc}Hz: {t30:.3f}s err={err:.1f}%')
"
```

## Architecture overview

```
Providers (any combination):
  ModalROMProvider    → eigensolve modes (0-f_max, exact decay)
  AnalyticalProvider  → box room modes (0-f_max, Kuttruff decay)
  AxialProvider       → parallel surface modes (any room)
  StatisticalProvider → Weyl density fill (any room)
  [your engine here]  → just implement provide_modes()

         ↓ all produce (freq, amplitude, decay_rate) tuples

  merge_modes() → deduplicate by confidence

         ↓

  synthesize_ir() → Numba CPU or CUDA GPU kernel
                    (recursive oscillator, zero exp/cos in inner loop)

         ↓ + ISM early reflections

  ImpulseResponse → T30, EDT, C80, save_wav()
```

## Current best results (uncalibrated, axial+statistical)

| Band | Simulated | Measured | Error |
|------|-----------|----------|-------|
| Broadband | 1.509s | 1.663s | 9.3% |
| 250 Hz | 2.115s | 1.745s | 21% |
| 500 Hz | 1.789s | 2.024s | 12% |
| 1000 Hz | 1.757s | 1.939s | 9.4% |
| 2000 Hz | 1.467s | 1.745s | 16% |

Calibration should bring these below 10% across all bands.

## Files to know

| File | What |
|------|------|
| `room_acoustics/cuda_synthesis.py` | CUDA kernel + GPU/CPU dispatch |
| `room_acoustics/unified_modes.py` | Provider registry + merge + synthesis |
| `room_acoustics/analytical_modes.py` | Box room modes + Numba JIT |
| `room_acoustics/statistical_modes.py` | Weyl density fill |
| `room_acoustics/axial_modes.py` | Parallel surface modes |
| `room_acoustics/material_catalog.py` | 28 catalog materials |
| `room_acoustics/material_function.py` | alpha(f) at any resolution |
| `bras_data/` | Measured RIRs (gitignored, download from BRAS) |

## BRAS data download (if not present)

```bash
mkdir -p bras_data && cd bras_data
curl -L -o 1_scene_descriptions-CR2.zip "https://depositonce.tu-berlin.de/bitstreams/53c3cf64-3547-4aa6-946b-1b4755729f2a/download"
curl -L -o 3_surface_descriptions.zip "https://depositonce.tu-berlin.de/bitstreams/b2970524-fb10-482a-ab14-f07da5ad7615/download"
unzip 1_scene_descriptions-CR2.zip -d .
unzip 3_surface_descriptions.zip -d .
```
