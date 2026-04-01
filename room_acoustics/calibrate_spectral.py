#!/usr/bin/env python
"""
Spectral absorption calibration from measured impulse responses.

Optimizes a per-surface scaling factor applied to baseline absorption
curves (e.g., BRAS CSV data) so that simulated per-mode T30 matches
measured T30 across all octave bands simultaneously.

Each surface gets one scalar: alpha_calibrated(f) = scale_s * alpha_baseline(f).
This preserves the spectral shape while adjusting the overall level.

6 parameters total (one per surface), evaluated across all bands and
positions in a single objective. Uses analytical T30 estimation — no
IR synthesis in the optimization loop.

Usage:
    python calibrate_spectral.py

Or from code:
    from calibrate_spectral import calibrate_spectral
    scales, result = calibrate_spectral(room, baseline_mats, rir_dir)
"""

import sys, os
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calibrate_spectral(room, baseline_mats, rir_dir,
                       bands=(250, 500, 1000, 2000),
                       surface_labels=None, verbose=True):
    """
    Calibrate per-surface scaling factors on baseline absorption curves.

    Parameters
    ----------
    room : Room (already built with surface weights)
    baseline_mats : dict of label -> MaterialFunction
        Baseline absorption curves (e.g., from BRAS CSVs).
    rir_dir : str
        Path to measured dodecahedron WAV directory.
    bands : tuple
        Octave-band center frequencies for T30 comparison.
    surface_labels : list or None
        Surfaces to calibrate. Default: all 6 box faces.

    Returns
    -------
    scales : dict of label -> float
        Optimal scaling factor per surface.
    calibrated_mats : dict of label -> MaterialFunction
        New MaterialFunction objects with scaled absorption.
    """
    from room_acoustics.material_function import MaterialFunction, compute_modal_decay_spectral
    from room_acoustics.calibrate_absorption import (
        precompute_surface_gamma_weights, load_measured_rirs,
        compute_band_t30, AxialModeCache)

    if surface_labels is None:
        surface_labels = ['floor', 'ceiling', 'front', 'back', 'left', 'right']

    Lx, Ly, Lz = room._dimensions
    V = Lx * Ly * Lz
    S = 2 * (Lx * Ly + Lx * Lz + Ly * Lz)
    sr = room.sr
    c = 343.0

    # Load measured RIRs and compute per-band T30
    rirs = load_measured_rirs(rir_dir)
    if verbose:
        print(f"  Loaded {len(rirs)} measured RIRs")

    measured_t30 = {}
    for fc in bands:
        vals = []
        for fn, ir, sr_m in rirs:
            t30 = compute_band_t30(ir, sr_m, fc)
            if t30 is not None:
                vals.append(t30)
        measured_t30[fc] = vals
        if verbose and vals:
            print(f"  {fc:5d} Hz: measured T30 = {np.mean(vals):.3f}s "
                  f"(std={np.std(vals):.3f}, n={len(vals)})")

    # Precompute surface weights
    weights = precompute_surface_gamma_weights(
        room.mesh, room.ops, room._eigenvectors, room._dimensions)

    # Source/receiver configs
    source_positions = {
        'LS1': (2.0, 3.35, 1.5),
        'LS2': (4.5, 4.0, 1.5),
    }
    receiver_map = {
        'MP1': (6.0, 1.5, 1.2),
        'MP2': (6.0, 3.0, 1.2),
        'MP3': (6.0, 4.5, 1.2),
        'MP4': (3.0, 1.5, 1.2),
        'MP5': (3.0, 5.0, 1.2),
    }

    # Modal amplitudes per source
    M = room.ops['M_diag']
    modal_amps = {}
    for ls_name, src_pos in source_positions.items():
        r2 = ((room.mesh.x - src_pos[0]) ** 2 +
              (room.mesh.y - src_pos[1]) ** 2 +
              (room.mesh.z - src_pos[2]) ** 2)
        p0 = np.exp(-r2 / 0.3 ** 2)
        modal_amps[ls_name] = room._eigenvectors.T @ (M * p0)

    eigen_freqs = room._frequencies
    omega = np.sqrt(np.maximum(room._eigenvalues, 0)) * c

    # Parse RIR configs
    rir_configs = []
    for fn, ir_meas, sr_m in rirs:
        parts = fn.replace('.wav', '').split('_')
        ls, mp = parts[2], parts[3]
        rec_pos = receiver_map.get(mp, (4.0, 3.0, 1.2))
        rec_idx = room.mesh.nearest_node(*rec_pos)
        rir_configs.append({
            'fn': fn, 'ir_meas': ir_meas, 'sr': sr_m,
            'ls': ls, 'mp': mp, 'rec_idx': rec_idx,
            'modal_amps': modal_amps[ls],
        })

    # Precompute axial mode caches
    axial_caches = {}
    for ls_name, src_pos in source_positions.items():
        for mp_name, rec_pos in receiver_map.items():
            key = f"{ls_name}_{mp_name}"
            axial_caches[key] = AxialModeCache(
                room._parallel_pairs, src_pos, rec_pos,
                f_min=0, f_max=4000, c=c, sr=sr)

    # Precompute band masks for eigenmodes
    band_eigen_masks = {}
    for fc in bands:
        fl = fc / np.sqrt(2)
        fh = fc * np.sqrt(2)
        band_eigen_masks[fc] = (eigen_freqs >= fl) & (eigen_freqs <= fh)

    def make_scaled_mats(scale_vec):
        """Create MaterialFunction dict with scaled absorption."""
        scaled = {}
        for i, label in enumerate(surface_labels):
            base = baseline_mats[label]
            s = np.clip(scale_vec[i], 0.3, 3.0)
            new_alphas = np.clip(base.alphas * s, 0.001, 0.999)
            scaled[label] = MaterialFunction(
                base.freqs, new_alphas, name=f"{base.name}_x{s:.2f}")
        return scaled

    # Regularization: penalize deviation from scale=1 (baseline values)
    # lambda controls tradeoff: higher = stay closer to baseline
    reg_lambda = 0.5  # moderate regularization

    def objective(scale_vec):
        """Total T30 residual + regularization penalty."""
        scaled_mats = make_scaled_mats(scale_vec)

        # Compute spectral modal decay (with air absorption)
        gamma = compute_modal_decay_spectral(
            weights, scaled_mats, eigen_freqs, c=c,
            humidity=getattr(room, 'humidity', 50.0),
            temperature=getattr(room, 'temperature', 20.0))

        # Room-average gamma for axial coupling
        mean_alpha = np.mean([scaled_mats[l](500.0) for l in surface_labels])
        rt60_room = 0.161 * V / (-S * np.log(1 - min(max(mean_alpha, 0.01), 0.99)))
        gamma_room = 6.91 / max(rt60_room, 0.05)

        total_err = 0.0
        n_comparisons = 0

        for fc in bands:
            fl = fc / np.sqrt(2)
            fh = fc * np.sqrt(2)
            meas_vals = measured_t30.get(fc, [])
            if not meas_vals:
                continue

            eigen_mask = band_eigen_masks[fc]

            for k, cfg in enumerate(rir_configs):
                if k >= len(meas_vals):
                    continue

                # Collect modes in this band
                energies = []
                gammas = []

                # Eigenmodes
                phi_rec = room._eigenvectors[cfg['rec_idx'], :]
                for idx in np.where(eigen_mask)[0]:
                    A = cfg['modal_amps'][idx] * phi_rec[idx]
                    if abs(A) > 1e-30:
                        energies.append(A ** 2)
                        gammas.append(gamma[idx])

                # Axial modes
                cache_key = f"{cfg['ls']}_{cfg['mp']}"
                cache = axial_caches.get(cache_key)
                if cache:
                    for m in cache.modes:
                        if fl <= m['freq'] <= fh:
                            l1, l2 = m['pair_labels']
                            a1 = np.clip(scaled_mats.get(l1, baseline_mats.get(l1))(m['freq']),
                                         0.001, 0.999)
                            a2 = np.clip(scaled_mats.get(l2, baseline_mats.get(l2))(m['freq']),
                                         0.001, 0.999)
                            R = (1 - a1) * (1 - a2)
                            if R <= 0:
                                continue
                            gp = (c / (2 * m['L'])) * (-np.log(R))
                            coup = 1.0 - min(m['A_pair'] / S, 1.0) if S > 0 else 0
                            g = (1 - coup) * gp + coup * gamma_room
                            energies.append(m['amplitude'] ** 2)
                            gammas.append(g)

                if not energies:
                    total_err += meas_vals[k] ** 2
                    n_comparisons += 1
                    continue

                gamma_eff = np.sum(np.array(energies) * np.array(gammas)) / np.sum(energies)
                t30_sim = 6.91 / max(gamma_eff, 0.01)
                total_err += (t30_sim - meas_vals[k]) ** 2
                n_comparisons += 1

        # Regularization: penalize log-deviation from scale=1
        # Using log-scale so that x2 and x0.5 are penalized equally
        reg = reg_lambda * np.sum(np.log(scale_vec) ** 2) * n_comparisons
        return total_err + reg

    # Optimize: scale factors starting at 1.0
    x0 = np.ones(len(surface_labels))
    bounds = [(0.3, 3.0)] * len(surface_labels)  # max 3x deviation from baseline

    if verbose:
        print(f"\n  Optimizing {len(surface_labels)} scale factors "
              f"across {len(bands)} bands, {len(rir_configs)} positions...")
        err0 = objective(x0)
        print(f"  Initial error: {np.sqrt(err0 / max(len(rir_configs)*len(bands), 1)):.4f}s RMS")

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-8})

    scales = {label: result.x[i] for i, label in enumerate(surface_labels)}
    calibrated_mats = make_scaled_mats(result.x)

    if verbose:
        err_final = np.sqrt(result.fun / max(len(rir_configs) * len(bands), 1))
        print(f"  Final RMS T30 error: {err_final:.4f}s")
        print(f"\n  Scale factors:")
        for label in surface_labels:
            base_500 = baseline_mats[label](500.0)
            cal_500 = calibrated_mats[label](500.0)
            print(f"    {label:10s}: x{scales[label]:.3f}  "
                  f"(alpha@500Hz: {base_500:.4f} -> {cal_500:.4f})")

    return scales, calibrated_mats


if __name__ == '__main__':
    from room_acoustics.room import Room
    from room_acoustics.material_function import MaterialFunction
    from room_acoustics.acoustics_metrics import compute_t30
    from scipy.signal import butter, sosfiltfilt

    print("=" * 70)
    print("  Spectral Absorption Calibration — BRAS CR2")
    print("=" * 70)

    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'bras_data', '3 Surface descriptions', '_csv', 'fitted_estimates')
    rir_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'bras_data', '1 Scene descriptions',
        'CR2 small room (seminar room)', 'RIRs', 'wav')

    baseline_mats = {
        'floor':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_floor.csv'),
        'ceiling': MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_ceiling.csv'),
        'front':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_concrete.csv'),
        'back':    MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_concrete.csv'),
        'left':    MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_windows.csv'),
        'right':   MaterialFunction.from_csv(f'{csv_dir}/mat_CR2_plaster.csv'),
    }

    # Build room
    room = Room.from_box(8.4, 6.7, 3.0, P=4, ppw=4, f_target=250)
    for label, mat in baseline_mats.items():
        room.set_material(label, mat)
    room.build(n_modes=100)

    # Calibrate
    scales, calibrated_mats = calibrate_spectral(
        room, baseline_mats, rir_dir, bands=(250, 500, 1000, 2000))

    # Test with calibrated materials
    print(f"\n{'='*70}")
    print("  Testing with calibrated materials")
    print(f"{'='*70}")

    for label, mat in calibrated_mats.items():
        room.set_material(label, mat)
    # Recompute spectral decay (materials changed)
    room._surface_weights = None  # force rebuild
    from room_acoustics.calibrate_absorption import precompute_surface_gamma_weights
    room._surface_weights = precompute_surface_gamma_weights(
        room.mesh, room.ops, room._eigenvectors, room._dimensions)

    ir = room.impulse_response((2.0, 3.35, 1.5), (6.0, 2.0, 1.2), T=3.0, n_rays=3000)
    ir.summary()

    measured = {250: 1.746, 500: 2.024, 1000: 1.939, 2000: 1.745, 4000: 1.563}
    sr = ir.sr
    nyq = sr / 2
    print('\nOctave-band T30 (calibrated):')
    for fc in [250, 500, 1000, 2000, 4000]:
        fl = fc / np.sqrt(2)
        fh = min(fc * np.sqrt(2), nyq * 0.95)
        sos = butter(4, [fl / nyq, fh / nyq], btype='band', output='sos')
        ir_band = sosfiltfilt(sos, ir.data)
        t30, r2 = compute_t30(ir_band, 1.0 / sr)
        m = measured[fc]
        err = abs(t30 - m) / m * 100 if not np.isnan(t30) and r2 > 0.7 else float('nan')
        t_str = f'{t30:.3f}s' if not np.isnan(t30) else 'N/A'
        print(f'  {fc:6d} Hz: sim={t_str:>8s}  meas={m:.3f}s  err={err:5.1f}%')

    print(f'\nBroadband: T30={ir.T30:.3f}s (meas=1.663s, '
          f'err={abs(ir.T30 - 1.663) / 1.663 * 100:.1f}%)')
