#!/usr/bin/env python
"""Test 4 approaches for late field energy to fix C80."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from room_acoustics.material_function import MaterialFunction
from room_acoustics.ism_spectral import ism_spectral
from room_acoustics.image_source import image_sources_shoebox
from room_acoustics.spectral_tools import load_wav
from room_acoustics.ir_score import score_ir_perceptual
from room_acoustics.acoustics_metrics import all_metrics
from room_acoustics.analytical_modes import _synthesize_numba
from scipy.signal import butter, sosfilt
import scipy.io.wavfile as wavfile

# JIT warmup
_synthesize_numba(np.zeros(100, dtype=np.float64),
    np.array([1.0]), np.array([5.0]), np.array([600.0]),
    np.array([100], dtype=np.int64), 1.0/44100)

csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'bras_data', '3 Surface descriptions', '_csv', 'fitted_estimates')
rir_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'bras_data', '1 Scene descriptions', 'CR2 small room (seminar room)', 'RIRs', 'wav')

mats = {
    'floor':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_floor.csv').with_structural_absorption(200, 0.3, 0.05),
    'ceiling': MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_ceiling.csv').with_structural_absorption(5, 0.3, 0.1),
    'front':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_concrete.csv').with_structural_absorption(200, 0.1, 0.03),
    'back':    MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_concrete.csv').with_structural_absorption(200, 0.1, 0.03),
    'left':    MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_windows.csv').with_structural_absorption(8, 0.02, 0.08),
    'right':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_plaster.csv').with_structural_absorption(10, 0.08, 0.05),
}

alpha_w = {}
for wall, label in {'x0':'left','x1':'right','y0':'front','y1':'back','z0':'floor','z1':'ceiling'}.items():
    alpha_w[wall] = mats[label](500.0)

Lx, Ly, Lz = 8.4, 6.7, 3.0
src = (2.0, 3.35, 1.5); rec = (6.0, 2.0, 1.2)
V = Lx*Ly*Lz; S = 2*(Lx*Ly+Lx*Lz+Ly*Lz)
sr = 44100; T = 3.5

ir_meas, _ = load_wav(f'{rir_dir}/CR2_RIR_LS1_MP1_Dodecahedron.wav')
pk_m = np.max(np.abs(ir_meas[:int(0.005*sr)]))

# Shared spectral ISM
print("Spectral ISM (order 30)...")
ir_ism30 = ism_spectral(Lx, Ly, Lz, src, rec, mats,
    max_order=30, sr=sr, T=T, humidity=41.7, temperature=19.5, n_filt=64)
pk_i = np.max(np.abs(ir_ism30[:int(0.005*sr)]))
if pk_i > 1e-30:
    ir_ism30 *= pk_m / pk_i

n = min(len(ir_ism30), len(ir_meas))
n80 = int(0.08 * sr)
n_fade = int(0.02 * sr)
n_s = int(0.15 * sr); n_e = int(0.5 * sr)
rms_m = np.sqrt(np.mean(ir_meas[n_s:n_e]**2))

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# ============================================================
# APPROACH 1: Higher ISM order (50, scalar alpha)
# ============================================================
print("\n--- Approach 1: ISM order 50 (scalar) ---")
t0 = time.perf_counter()
ir_ism50, _ = image_sources_shoebox(Lx, Ly, Lz, src, rec,
    max_order=50, alpha_walls=alpha_w, sr=sr, T=T)
pk50 = np.max(np.abs(ir_ism50[:int(0.005*sr)]))
if pk50 > 1e-30:
    ir_ism50 *= pk_m / pk50
dt1 = time.perf_counter() - t0

ir_a1 = ir_ism50[:n].copy()
m1 = all_metrics(ir_a1, 1/sr)
print(f"  Time: {dt1:.1f}s, T30={m1['T30_s']:.3f}s C80={m1['C80_dB']:.1f}dB")

# ============================================================
# APPROACH 2: Spectral ISM(10) + scalar ISM(50) blended
# ============================================================
print("\n--- Approach 2: Spectral ISM early + scalar ISM(50) late ---")
t0 = time.perf_counter()
xf = np.ones(n)
xf[n80:n80+n_fade] = np.linspace(1, 0, min(n_fade, n-n80))
if n80+n_fade < n:
    xf[n80+n_fade:] = 0
ir_a2 = ir_ism30[:n] * xf + ir_ism50[:n] * (1 - xf)
dt2 = time.perf_counter() - t0 + dt1

m2 = all_metrics(ir_a2, 1/sr)
print(f"  Time: {dt2:.1f}s, T30={m2['T30_s']:.3f}s C80={m2['C80_dB']:.1f}dB")

# ============================================================
# APPROACH 3: Spectral ISM(30) + ray tracer
# ============================================================
print("\n--- Approach 3: Spectral ISM(30) + ray tracer ---")
t0 = time.perf_counter()
from room_acoustics.ray_tracer import trace_rays, reflectogram_to_ir, RoomMesh

verts = np.array([[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0],
                   [0,0,Lz],[Lx,0,Lz],[Lx,Ly,Lz],[0,Ly,Lz]], dtype=float)
tris_arr = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                      [2,6,7],[2,7,3],[0,3,7],[0,7,4],[1,5,6],[1,6,2]], dtype=int)
rt = RoomMesh.__new__(RoomMesh)
rt.vertices = verts
rt.triangles = tris_arr
rt.n_triangles = 12
norms = []
for t_idx in tris_arr:
    e1 = verts[t_idx[1]] - verts[t_idx[0]]
    e2 = verts[t_idx[2]] - verts[t_idx[0]]
    nn = np.cross(e1, e2)
    norms.append(nn / np.linalg.norm(nn))
rt.normals = np.array(norms)
rt.surface_labels = ['floor','floor','ceiling','ceiling','front','front',
                      'back','back','left','left','right','right']
rt.surface_alpha = {l: mats[l](1000.0) for l in mats}

reflecto, _ = trace_rays(rt, src, rec, n_rays=5000, max_order=200,
                          capture_radius=0.3, scatter_coeff=0.1, T=T)
ir_ray = reflectogram_to_ir(reflecto, sr)
rms_r = np.sqrt(np.mean(ir_ray[n_s:min(n_e, len(ir_ray))]**2))
if rms_r > 1e-30:
    ir_ray *= rms_m / rms_r

ray_win = np.zeros(n)
ray_win[n80:n80+n_fade] = np.linspace(0, 1, min(n_fade, n-n80))
if n80+n_fade < n:
    ray_win[n80+n_fade:] = 1.0

ir_a3 = ir_ism30[:n].copy()
nr = min(len(ir_ray), n)
ir_a3[:nr] += ir_ray[:nr] * ray_win[:nr]
dt3 = time.perf_counter() - t0

m3 = all_metrics(ir_a3, 1/sr)
print(f"  Time: {dt3:.1f}s, T30={m3['T30_s']:.3f}s C80={m3['C80_dB']:.1f}dB")

# ============================================================
# APPROACH 4: Spectral ISM(30) + axial modes
# ============================================================
print("\n--- Approach 4: Spectral ISM(30) + axial modes ---")
t0 = time.perf_counter()
from room_acoustics.axial_modes import axial_mode_ir, detect_parallel_surfaces_box

pairs = detect_parallel_surfaces_box((Lx, Ly, Lz))
ir_axial, info = axial_mode_ir(pairs, src, rec, mats, mats['front'],
    T=T, sr=sr, f_min=None, f_max=8000, c=343.0,
    room_volume=V, room_surface_area=S, humidity=41.7, temperature=19.5)

rms_ax = np.sqrt(np.mean(ir_axial[n_s:min(n_e, len(ir_axial))]**2))
if rms_ax > 1e-30:
    ir_axial *= rms_m / rms_ax

ir_a4 = ir_ism30[:n].copy()
na = min(len(ir_axial), n)
ir_a4[:na] += ir_axial[:na] * ray_win[:na]
dt4 = time.perf_counter() - t0

m4 = all_metrics(ir_a4, 1/sr)
print(f"  Time: {dt4:.1f}s, {len(info)} modes, T30={m4['T30_s']:.3f}s C80={m4['C80_dB']:.1f}dB")

# ============================================================
# SCORE ALL
# ============================================================
print("\n" + "="*70)
print("  SCORING")
print("="*70)

approaches = [
    ("1: ISM(50) scalar", ir_a1, dt1),
    ("2: Spectral+scalar blend", ir_a2, dt2),
    ("3: ISM(30)+ray tracer", ir_a3, dt3),
    ("4: ISM(30)+axial modes", ir_a4, dt4),
]

best_score = 0
best_ir = None
best_name = ""

for name, ir, dt in approaches:
    ns = min(len(ir), len(ir_meas))
    r = score_ir_perceptual(ir[:ns], ir_meas[:ns], sr)
    m = all_metrics(ir[:ns], 1/sr)
    print(f"\n  {name}:")
    print(f"    Score: {r['score']:.1f}/100  T30={m['T30_s']:.3f}s  C80={m['C80_dB']:.1f}dB  Time={dt:.1f}s")
    for key in ['bark_spectral', 'energy_decay_relief', 'early_reflections', 'iso3382_metrics', 'modulation_transfer']:
        ss = r['sub_scores'][key]
        print(f"    {key}: {ss['score_01']*100:.0f}%")
    if r['score'] > best_score:
        best_score = r['score']
        best_ir = ir[:ns]
        best_name = name

print(f"\n  BEST: {best_name} ({best_score:.1f}/100)")

ir_out = best_ir.astype(np.float32)
ir_out = ir_out / max(abs(ir_out).max(), 1e-10) * 0.95
wavfile.write(f'{results_dir}/IR_simulated_BRAS_CR2_v13.wav', sr, ir_out)
print(f"  Saved: v13")
