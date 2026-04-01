#!/usr/bin/env python
"""
Engine-by-engine comparison against BRAS CR2 measured RIRs.

Runs each engine independently, measures what it gets right and wrong,
then identifies the optimal combination strategy.

Engines:
  1. Modal ROM (eigenmodes, analytical decay)
  2. Axial modes (1D parallel surface modes)
  3. FDTD (full wave equation on voxel grid)
  4. Ray tracer (stochastic geometric)
  5. ISM (image source, shoebox)

For each: broadband T30, per-band T30, spectral shape, early/late energy.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile


def load_measured_reference():
    """Load first BRAS CR2 measured RIR as reference."""
    rir_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'bras_data', '1 Scene descriptions',
        'CR2 small room (seminar room)', 'RIRs', 'wav')

    fn = 'CR2_RIR_LS1_MP1_Dodecahedron.wav'
    sr, data = wavfile.read(os.path.join(rir_dir, fn))
    ir = data.astype(np.float64)
    if ir.ndim > 1:
        ir = ir[:, 0]
    ir /= max(np.abs(ir).max(), 1e-10)
    return ir, sr


def band_t30(ir, sr, fc):
    """Compute T30 in one octave band."""
    from room_acoustics.acoustics_metrics import compute_t30
    nyq = sr / 2
    fl = fc / np.sqrt(2)
    fh = min(fc * np.sqrt(2), nyq * 0.95)
    if fl >= nyq * 0.95:
        return None
    sos = butter(4, [fl / nyq, fh / nyq], btype='band', output='sos')
    ir_band = sosfiltfilt(sos, ir)
    t30, r2 = compute_t30(ir_band, 1.0 / sr)
    return t30 if r2 > 0.7 and not np.isnan(t30) else None


def band_energy(ir, sr, fc):
    """Compute energy in one octave band."""
    nyq = sr / 2
    fl = fc / np.sqrt(2)
    fh = min(fc * np.sqrt(2), nyq * 0.95)
    if fl >= nyq * 0.95:
        return 0
    sos = butter(4, [fl / nyq, fh / nyq], btype='band', output='sos')
    ir_band = sosfiltfilt(sos, ir)
    return float(np.sum(ir_band ** 2))


def early_late_ratio(ir, sr, t_split_ms=80):
    """Compute early-to-late energy ratio (like C80)."""
    n_split = int(t_split_ms * sr / 1000)
    e_early = np.sum(ir[:n_split] ** 2)
    e_late = np.sum(ir[n_split:] ** 2)
    if e_late > 0:
        return 10 * np.log10(e_early / e_late)
    return float('inf')


def spectral_shape(ir, sr, f_max=4000):
    """Compute normalized spectral shape."""
    spectrum = np.abs(np.fft.rfft(ir))
    freqs = np.fft.rfftfreq(len(ir), 1.0 / sr)
    mask = freqs <= f_max
    spec = spectrum[mask]
    if spec.max() > 0:
        spec /= spec.max()
    return freqs[mask], spec


def analyze_engine(name, ir, sr, ref_ir, ref_sr, bands):
    """Analyze one engine's IR against reference."""
    from room_acoustics.acoustics_metrics import all_metrics

    # Resample if needed
    if sr != ref_sr:
        from scipy.signal import resample
        ir = resample(ir, int(len(ir) * ref_sr / sr))
        sr = ref_sr

    # Trim/pad to same length
    n = min(len(ir), len(ref_ir))
    ir = ir[:n]
    ref = ref_ir[:n]

    # Normalize
    if np.max(np.abs(ir)) > 0:
        ir = ir / np.max(np.abs(ir))

    metrics = all_metrics(ir, 1.0 / sr)

    result = {
        'name': name,
        'T30': metrics['T30_s'],
        'EDT': metrics['EDT_s'],
        'C80': metrics['C80_dB'],
        'band_t30': {},
        'band_energy': {},
        'early_late': early_late_ratio(ir, sr),
    }

    for fc in bands:
        result['band_t30'][fc] = band_t30(ir, sr, fc)
        result['band_energy'][fc] = band_energy(ir, sr, fc)

    return result, ir


def main():
    from room_acoustics.material_function import MaterialFunction

    print("=" * 70)
    print("  Engine Comparison: BRAS CR2 (8.4 x 6.7 x 3.0 m)")
    print("=" * 70)

    # Load reference
    ref_ir, ref_sr = load_measured_reference()
    print(f"\n  Reference: {len(ref_ir)} samples @ {ref_sr} Hz "
          f"({len(ref_ir)/ref_sr:.2f}s)")

    bands = [250, 500, 1000, 2000, 4000]

    # Reference metrics
    ref_result, _ = analyze_engine('Measured', ref_ir, ref_sr, ref_ir, ref_sr, bands)

    Lx, Ly, Lz = 8.4, 6.7, 3.0
    source = (2.0, 3.35, 1.5)
    receiver = (6.0, 2.0, 1.2)
    sr = 44100
    dt = 1.0 / sr
    T = 3.0

    # Load BRAS materials
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'bras_data', '3 Surface descriptions', '_csv', 'fitted_estimates')
    mat_funcs = {
        'floor':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_floor.csv'),
        'ceiling': MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_ceiling.csv'),
        'front':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_concrete.csv'),
        'back':    MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_concrete.csv'),
        'left':    MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_windows.csv'),
        'right':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_plaster.csv'),
    }

    results = [ref_result]

    # ================================================================
    # ENGINE 1: Modal ROM only
    # ================================================================
    print("\n--- ENGINE 1: Modal ROM ---")
    t0 = time.perf_counter()

    from room_acoustics.room import Room
    room = Room.from_box(Lx, Ly, Lz, P=4, ppw=4, f_target=250)
    for label, mat in mat_funcs.items():
        room.set_material(label, mat)
    room.humidity = 41.7
    room.temperature = 19.5
    room.build(n_modes=100)

    # Modal ROM only (no axial, no ray tracer)
    from room_acoustics.calibrate_absorption import (
        precompute_surface_gamma_weights, synthesize_ir_fast)
    from room_acoustics.material_function import compute_modal_decay_spectral

    weights = precompute_surface_gamma_weights(
        room.mesh, room.ops, room._eigenvectors, room._dimensions)
    gamma = compute_modal_decay_spectral(
        weights, mat_funcs, room._frequencies, c=343.0,
        humidity=41.7, temperature=19.5)
    omega = np.sqrt(np.maximum(room._eigenvalues, 0)) * 343.0
    M = room.ops['M_diag']
    r2 = ((room.mesh.x - source[0])**2 + (room.mesh.y - source[1])**2 +
          (room.mesh.z - source[2])**2)
    p0 = np.exp(-r2 / 0.3**2)
    modal_amps = room._eigenvectors.T @ (M * p0)
    rec_idx = room.mesh.nearest_node(*receiver)

    ir_modal = synthesize_ir_fast(
        room._eigenvalues, room._eigenvectors, omega,
        modal_amps, gamma, rec_idx, dt, T)[:int(T * sr)]

    t_modal = time.perf_counter() - t0
    print(f"  Time: {t_modal:.1f}s, f_max={room._frequencies[-1]:.0f}Hz, "
          f"{len(room._eigenvalues)} modes")

    r, ir1 = analyze_engine('Modal ROM', ir_modal, sr, ref_ir, ref_sr, bands)
    results.append(r)

    # ================================================================
    # ENGINE 2: Axial modes only
    # ================================================================
    print("\n--- ENGINE 2: Axial Modes ---")
    t0 = time.perf_counter()

    from room_acoustics.axial_modes import detect_parallel_surfaces_box, axial_mode_ir
    pairs = detect_parallel_surfaces_box((Lx, Ly, Lz))
    ir_axial, axial_info = axial_mode_ir(
        pairs, source, receiver, mat_funcs, mat_funcs['front'],
        T=T, sr=sr, f_min=None, f_max=8000, c=343.0,
        room_volume=Lx*Ly*Lz, room_surface_area=2*(Lx*Ly+Lx*Lz+Ly*Lz),
        humidity=41.7, temperature=19.5)

    t_axial = time.perf_counter() - t0
    print(f"  Time: {t_axial:.1f}s, {len(axial_info)} modes")

    r, ir2 = analyze_engine('Axial Modes', ir_axial, sr, ref_ir, ref_sr, bands)
    results.append(r)

    # ================================================================
    # ENGINE 3: FDTD
    # ================================================================
    print("\n--- ENGINE 3: FDTD ---")
    t0 = time.perf_counter()

    from room_acoustics.voxelize import voxelize_box, find_boundary_voxels
    from room_acoustics.fdtd import FDTDSolver

    # Coarse grid for speed (dx=0.1m → f_max ~1.7 kHz)
    air, origin, vdx = voxelize_box(Lx, Ly, Lz, dx=0.1, padding=3)
    solver = FDTDSolver(air, dx=vdx, c=343.0, CFL=0.4)

    # Set boundary absorption from materials (simplified: use mean alpha)
    bndy_ijk, bndy_normals = find_boundary_voxels(air)
    alpha_3d = np.zeros(air.shape, dtype=np.float32)
    mean_alpha = np.mean([mat_funcs[l](500.0) for l in mat_funcs])
    for (i, j, k) in bndy_ijk:
        alpha_3d[i, j, k] = mean_alpha
    solver.set_materials(alpha_3d=alpha_3d)

    # Convert source/receiver to voxel indices
    ox, oy, oz = origin
    src_ijk = (int(round((source[0] - ox) / vdx)),
               int(round((source[1] - oy) / vdx)),
               int(round((source[2] - oz) / vdx)))
    rec_ijk = (int(round((receiver[0] - ox) / vdx)),
               int(round((receiver[1] - oy) / vdx)),
               int(round((receiver[2] - oz) / vdx)))

    ir_fdtd, fs_fdtd = solver.impulse_response(
        src_ijk, rec_ijk, duration=min(T, 0.5))  # short for speed

    t_fdtd = time.perf_counter() - t0
    print(f"  Time: {t_fdtd:.1f}s, grid={air.shape}, fs={fs_fdtd}Hz, "
          f"{len(ir_fdtd)} samples")

    r, ir3 = analyze_engine('FDTD', ir_fdtd, fs_fdtd, ref_ir, ref_sr, bands)
    results.append(r)

    # ================================================================
    # ENGINE 4: Ray Tracer only
    # ================================================================
    print("\n--- ENGINE 4: Ray Tracer ---")
    t0 = time.perf_counter()

    from room_acoustics.ray_tracer import trace_rays, reflectogram_to_ir, RoomMesh
    from room_acoustics.acoustics_metrics import impedance_to_alpha

    rt_mesh = room._rt_mesh
    for label in ['floor', 'ceiling', 'front', 'back', 'left', 'right']:
        rt_mesh.set_alpha(label, mat_funcs[label](1000.0))

    reflecto, _ = trace_rays(rt_mesh, source, receiver,
                              n_rays=5000, max_order=200,
                              capture_radius=0.3, scatter_coeff=0.1, T=T)
    ir_ray = reflectogram_to_ir(reflecto, sr)

    t_ray = time.perf_counter() - t0
    print(f"  Time: {t_ray:.1f}s, {5000} rays")

    r, ir4 = analyze_engine('Ray Tracer', ir_ray, sr, ref_ir, ref_sr, bands)
    results.append(r)

    # ================================================================
    # ENGINE 5: ISM only
    # ================================================================
    print("\n--- ENGINE 5: ISM ---")
    t0 = time.perf_counter()

    from room_acoustics.image_source import image_sources_shoebox
    alpha_walls = {}
    wall_map = {'x0': 'left', 'x1': 'right', 'y0': 'front',
                'y1': 'back', 'z0': 'floor', 'z1': 'ceiling'}
    for wall, label in wall_map.items():
        alpha_walls[wall] = mat_funcs[label](500.0)

    ir_ism, _ = image_sources_shoebox(
        Lx, Ly, Lz, source, receiver,
        max_order=20, alpha_walls=alpha_walls, sr=sr, T=T)

    t_ism = time.perf_counter() - t0
    print(f"  Time: {t_ism:.1f}s")

    r, ir5 = analyze_engine('ISM', ir_ism, sr, ref_ir, ref_sr, bands)
    results.append(r)

    # ================================================================
    # ENGINE 6: Current hybrid (modal + axial + ray + ISM)
    # ================================================================
    print("\n--- ENGINE 6: Current Hybrid ---")
    t0 = time.perf_counter()

    ir_hybrid_obj = room.impulse_response(source, receiver, T=T, n_rays=3000)
    ir_hybrid = ir_hybrid_obj.data

    t_hybrid = time.perf_counter() - t0

    r, ir6 = analyze_engine('Hybrid', ir_hybrid, sr, ref_ir, ref_sr, bands)
    results.append(r)

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*90}")
    print("  COMPARISON: per-engine metrics vs measured")
    print(f"{'='*90}")

    print(f"\n  {'Engine':<16s} {'T30':>7s} {'EDT':>7s} {'C80':>7s} {'Time':>7s}")
    print(f"  {'-'*16:<16s} {'-'*7:>7s} {'-'*7:>7s} {'-'*7:>7s} {'-'*7:>7s}")
    times = [0, t_modal, t_axial, t_fdtd, t_ray, t_ism, t_hybrid]
    for i, r in enumerate(results):
        t_str = f"{times[i]:.1f}s" if i > 0 else ""
        t30_str = f"{r['T30']:.3f}" if r['T30'] and not np.isnan(r['T30']) else "N/A"
        edt_str = f"{r['EDT']:.3f}" if r['EDT'] and not np.isnan(r['EDT']) else "N/A"
        c80_str = f"{r['C80']:.1f}" if r['C80'] and not np.isnan(r['C80']) else "N/A"
        print(f"  {r['name']:<16s} {t30_str:>7s} {edt_str:>7s} {c80_str:>7s} {t_str:>7s}")

    print(f"\n  Octave-band T30:")
    print(f"  {'Engine':<16s}" + "".join(f" {fc:>7d}" for fc in bands))
    print(f"  {'-'*16:<16s}" + "".join(f" {'-'*7:>7s}" for _ in bands))
    for r in results:
        row = f"  {r['name']:<16s}"
        for fc in bands:
            t30 = r['band_t30'].get(fc)
            row += f" {t30:7.3f}" if t30 and not np.isnan(t30) else "     N/A"
        print(row)

    # Error vs measured
    print(f"\n  T30 error vs measured (%):")
    print(f"  {'Engine':<16s}" + "".join(f" {fc:>7d}" for fc in bands) + "  broad")
    print(f"  {'-'*16:<16s}" + "".join(f" {'-'*7:>7s}" for _ in bands) + "  -----")
    ref_t30 = ref_result['T30']
    for r in results[1:]:
        row = f"  {r['name']:<16s}"
        for fc in bands:
            t30_r = ref_result['band_t30'].get(fc)
            t30_s = r['band_t30'].get(fc)
            if t30_r and t30_s and not np.isnan(t30_r) and not np.isnan(t30_s):
                err = abs(t30_s - t30_r) / t30_r * 100
                row += f" {err:6.1f}%"
            else:
                row += "     N/A"
        # Broadband
        if r['T30'] and ref_t30 and not np.isnan(r['T30']):
            err_b = abs(r['T30'] - ref_t30) / ref_t30 * 100
            row += f"  {err_b:4.1f}%"
        print(row)

    # Energy distribution
    print(f"\n  Octave-band energy (dB relative to measured):")
    print(f"  {'Engine':<16s}" + "".join(f" {fc:>7d}" for fc in bands))
    ref_energy = {fc: ref_result['band_energy'].get(fc, 0) for fc in bands}
    for r in results[1:]:
        row = f"  {r['name']:<16s}"
        for fc in bands:
            e_r = ref_energy.get(fc, 0)
            e_s = r['band_energy'].get(fc, 0)
            if e_r > 0 and e_s > 0:
                delta = 10 * np.log10(e_s / e_r)
                row += f" {delta:+6.1f}d"
            else:
                row += "     N/A"
        print(row)

    # ================================================================
    # ANALYSIS: What each engine does right/wrong
    # ================================================================
    print(f"\n{'='*90}")
    print("  ANALYSIS: Engine strengths and weaknesses")
    print(f"{'='*90}")

    analysis = []
    for r in results[1:]:
        strengths = []
        weaknesses = []

        # Broadband T30
        if r['T30'] and ref_t30 and not np.isnan(r['T30']):
            err = abs(r['T30'] - ref_t30) / ref_t30 * 100
            if err < 10:
                strengths.append(f"broadband T30 ({err:.1f}%)")
            else:
                weaknesses.append(f"broadband T30 ({err:.1f}%)")

        # Per-band
        for fc in bands:
            t30_r = ref_result['band_t30'].get(fc)
            t30_s = r['band_t30'].get(fc)
            if t30_r and t30_s and not np.isnan(t30_r) and not np.isnan(t30_s):
                err = abs(t30_s - t30_r) / t30_r * 100
                if err < 15:
                    strengths.append(f"{fc}Hz T30 ({err:.0f}%)")
                elif err > 50:
                    weaknesses.append(f"{fc}Hz T30 ({err:.0f}%)")

        print(f"\n  {r['name']}:")
        print(f"    Strengths:  {', '.join(strengths) if strengths else 'none'}")
        print(f"    Weaknesses: {', '.join(weaknesses) if weaknesses else 'none'}")
        analysis.append({'name': r['name'], 'strengths': strengths, 'weaknesses': weaknesses})


if __name__ == '__main__':
    main()
