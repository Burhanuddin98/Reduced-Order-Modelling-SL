#!/usr/bin/env python
"""
Per-surface absorption calibration from measured impulse responses.

Uses the modal ROM's analytical decay formula to fit per-surface,
per-band absorption coefficients by minimizing the residual between
simulated and measured T30 at multiple receiver positions.

Key insight: the modal decay rate decomposes by surface:
  gamma_i = 0.5 * rho*c^2 * sum_s( phi_i^T * (B_s / Z_s) * phi_i )

Each surface's contribution is a precomputed quadratic form of the
eigenvector. Changing Z_s only requires rescaling — no re-eigensolve.
This makes the calibration loop run in milliseconds per evaluation.

Unlike Eyring inversion (homogeneous field → room-mean alpha), this
method exploits the spatial resolution of the modal ROM: different
receivers weight different surfaces differently, allowing per-surface
absorption to be resolved.

Usage:
    from room_acoustics.calibrate_absorption import calibrate_from_rirs

    alpha_per_surface_per_band = calibrate_from_rirs(
        room, measured_rir_paths, source_positions, receiver_positions)
"""

import numpy as np
import os
from scipy.optimize import minimize
from scipy.signal import butter, sosfiltfilt


# ===================================================================
# Precompute per-surface modal coupling
# ===================================================================

def precompute_surface_gamma_weights(mesh, ops, eigenvectors, dimensions):
    """
    Precompute per-surface, per-mode damping weights.

    For each surface s and mode i:
      w_s_i = 0.5 * rho*c^2 * phi_i^T * B_s * phi_i

    Then gamma_i = sum_s( w_s_i / Z_s )

    This allows fast recalculation of gamma when Z changes.

    Parameters
    ----------
    mesh : BoxMesh3D or similar
    ops : operator dict with B_total
    eigenvectors : (N, n_modes) from compute_room_modes
    dimensions : (Lx, Ly, Lz)

    Returns
    -------
    weights : dict of surface_label -> array (n_modes,)
        w_s_i for each surface and mode.
    """
    rho = 1.2
    c = 343.0
    rc2 = rho * c ** 2
    Lx, Ly, Lz = dimensions
    N = mesh.N_dof
    n_modes = eigenvectors.shape[1]
    tol = 1e-6

    # Identify boundary nodes per surface (box rooms)
    face_masks = {
        'floor':   mesh.z < tol,
        'ceiling': mesh.z > Lz - tol,
        'left':    mesh.x < tol,
        'right':   mesh.x > Lx - tol,
        'front':   mesh.y < tol,
        'back':    mesh.y > Ly - tol,
    }

    B_diag = np.array(ops['B_total'].diagonal())

    weights = {}
    for label, mask in face_masks.items():
        B_surface = np.zeros(N)
        B_surface[mask] = B_diag[mask]

        # w_s_i = 0.5 * rc2 * phi_i^T * B_s * phi_i
        w = np.zeros(n_modes)
        for i in range(n_modes):
            phi = eigenvectors[:, i]
            w[i] = 0.5 * rc2 * np.dot(phi, B_surface * phi)
        weights[label] = w

    return weights


def synthesize_ir_fast(eigenvalues, eigenvectors, omega, modal_amplitudes,
                       gamma, rec_idx, dt, T):
    """Fast IR synthesis from precomputed modal parameters."""
    n_modes = len(eigenvalues)
    Nt = int(round(T / dt))
    t = np.arange(Nt + 1) * dt
    ir = np.zeros(Nt + 1)

    phi_rec = eigenvectors[rec_idx, :]

    for i in range(n_modes):
        if abs(modal_amplitudes[i]) < 1e-30:
            continue
        if eigenvalues[i] < 1e-10 and gamma[i] < 1e-10:
            continue

        A = modal_amplitudes[i] * phi_rec[i]
        g = gamma[i]
        omega_d = np.sqrt(max(omega[i] ** 2 - g ** 2, 0))
        if omega_d > 0:
            ir += A * np.exp(-g * t) * np.cos(omega_d * t)
        elif omega[i] > 0:
            ir += A * np.exp(-g * t)

    return ir


class AxialModeCache:
    """
    Precomputed axial mode geometry for fast re-synthesis with varying alpha.

    Stores mode frequencies, source/receiver coupling, and pair geometry.
    The decay rate can be recomputed analytically from alpha without
    re-running the full axial_mode_ir().
    """

    def __init__(self, parallel_pairs, source, receiver,
                 f_min=0, f_max=4000, c=343.0, sr=44100):
        self.c = c
        self.sr = sr
        self.modes = []  # list of {freq, omega, S_n, R_n, weight, pair_labels, L, A_pair}

        src = np.asarray(source, dtype=float)
        rec = np.asarray(receiver, dtype=float)
        total_pair_area = sum(p.overlap_area for p in parallel_pairs)
        if total_pair_area < 1e-10:
            return

        for pair in parallel_pairs:
            L = pair.distance
            n_dir = pair.normal

            x_src = np.dot(src - pair.centroid_1, n_dir)
            x_rec = np.dot(rec - pair.centroid_1, n_dir)

            margin = 0.01 * L
            if x_src < -margin or x_src > L + margin:
                continue
            if x_rec < -margin or x_rec > L + margin:
                continue
            x_src = np.clip(x_src, 0.0, L)
            x_rec = np.clip(x_rec, 0.0, L)

            weight_pair = pair.overlap_area / total_pair_area
            n_max = int(np.floor(2.0 * L * f_max / c))

            for n in range(1, n_max + 1):
                f = n * c / (2.0 * L)
                if f < f_min:
                    continue
                S_n = np.cos(n * np.pi * x_src / L)
                R_n = np.cos(n * np.pi * x_rec / L)
                A = S_n * R_n * (2.0 / L) * weight_pair
                if abs(A) < 1e-15:
                    continue

                self.modes.append({
                    'freq': f,
                    'omega': 2.0 * np.pi * f,
                    'amplitude': A,
                    'pair_labels': (pair.label_1, pair.label_2),
                    'L': L,
                    'A_pair': pair.overlap_area * 2,
                })

    def synthesize(self, alpha_per_surface, T, room_volume=None,
                   room_surface_area=None):
        """Synthesize axial IR with given per-surface absorption."""
        c = self.c
        sr = self.sr
        rho_c = 1.2 * c
        n_samples = int(T * sr)
        t = np.arange(n_samples, dtype=np.float64) / sr
        ir = np.zeros(n_samples, dtype=np.float64)

        # Room-average gamma for coupling model
        gamma_room = 0.0
        if room_volume and room_surface_area:
            mean_alpha = np.mean([np.clip(a, 0.01, 0.99)
                                  for a in alpha_per_surface.values()])
            rt60 = 0.161 * room_volume / (
                -room_surface_area * np.log(1 - min(mean_alpha, 0.99)))
            gamma_room = 6.91 / max(rt60, 0.05)

        for m in self.modes:
            l1, l2 = m['pair_labels']
            a1 = np.clip(alpha_per_surface.get(l1, 0.05), 0.001, 0.999)
            a2 = np.clip(alpha_per_surface.get(l2, 0.05), 0.001, 0.999)
            R_prod = (1 - a1) * (1 - a2)
            if R_prod <= 0:
                continue
            gamma_pair = (c / (2 * m['L'])) * (-np.log(R_prod))

            # Coupling correction
            if room_surface_area and room_surface_area > 0:
                coupling = 1.0 - min(m['A_pair'] / room_surface_area, 1.0)
                gamma = (1.0 - coupling) * gamma_pair + coupling * gamma_room
            else:
                gamma = gamma_pair

            # Early termination
            if gamma > 0:
                t_80dB = 80.0 * np.log(10) / (20.0 * gamma)
                n_cut = min(int(t_80dB * sr) + 1, n_samples)
            else:
                n_cut = n_samples
            if n_cut < 10:
                continue

            ir[:n_cut] += (m['amplitude'] * np.exp(-gamma * t[:n_cut])
                           * np.cos(m['omega'] * t[:n_cut]))

        return ir


def compute_gamma_from_alpha(weights, alpha_per_surface, c=343.0):
    """
    Compute per-mode decay rates from per-surface absorption.

    Parameters
    ----------
    weights : dict of label -> array (n_modes,)
        From precompute_surface_gamma_weights().
    alpha_per_surface : dict of label -> float
        Absorption coefficient per surface.

    Returns
    -------
    gamma : array (n_modes,)
    """
    rho_c = 1.2 * c
    labels = list(weights.keys())
    n_modes = len(weights[labels[0]])
    gamma = np.zeros(n_modes)

    for label in labels:
        alpha = alpha_per_surface.get(label, 0.05)
        alpha = np.clip(alpha, 0.001, 0.999)
        R = np.sqrt(1.0 - alpha)
        Z = rho_c * (1 + R) / (1 - R)
        gamma += weights[label] / Z

    return gamma


def compute_band_t30(ir, sr, fc):
    """Compute T30 in a single octave band. Returns T30 or None."""
    try:
        from .acoustics_metrics import compute_t30
    except ImportError:
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


# ===================================================================
# Load measured RIRs
# ===================================================================

def load_measured_rirs(rir_dir):
    """Load all dodecahedron RIRs from BRAS directory.

    Returns list of (filename, ir, sr).
    """
    from scipy.io import wavfile

    rirs = []
    for fn in sorted(os.listdir(rir_dir)):
        if not fn.endswith('.wav') or 'Dodecahedron' not in fn:
            continue
        sr, data = wavfile.read(os.path.join(rir_dir, fn))
        ir = data.astype(np.float64)
        if ir.ndim > 1:
            ir = ir[:, 0]
        ir /= max(np.abs(ir).max(), 1e-10)
        rirs.append((fn, ir, sr))
    return rirs


def compute_measured_band_t30s(rirs, bands):
    """Compute per-band T30 for each measured RIR.

    Returns dict: band_fc -> list of T30 values (one per RIR).
    """
    measured = {fc: [] for fc in bands}
    for fn, ir, sr in rirs:
        for fc in bands:
            t30 = compute_band_t30(ir, sr, fc)
            if t30 is not None:
                measured[fc].append(t30)
    return measured


# ===================================================================
# Calibration
# ===================================================================

def calibrate_absorption(room, rir_dir, bands=(250, 500, 1000, 2000),
                         surface_labels=None, verbose=True):
    """
    Calibrate per-surface, per-band absorption from measured RIRs.

    Uses the modal ROM to simulate IRs at each receiver position,
    then optimizes per-surface absorption to minimize T30 residuals
    between simulation and measurement.

    Parameters
    ----------
    room : Room (already built)
        Must have eigenvalues, eigenvectors, mesh, ops.
    rir_dir : str
        Path to directory containing measured dodecahedron WAVs.
    bands : tuple of int
        Octave-band center frequencies to calibrate.
    surface_labels : list of str or None
        Which surfaces to calibrate. Default: all 6 box faces.
    verbose : bool
        Print progress.

    Returns
    -------
    calibrated_alpha : dict of band_fc -> dict of label -> alpha
        Calibrated absorption per surface per band.
    residuals : dict of band_fc -> float
        Final RMS T30 error per band.
    """
    if not room._built:
        raise RuntimeError("Room must be built first")
    if room._geometry_type != 'box':
        raise NotImplementedError("Calibration currently supports box rooms only")

    if surface_labels is None:
        surface_labels = ['floor', 'ceiling', 'front', 'back', 'left', 'right']

    Lx, Ly, Lz = room._dimensions
    sr = room.sr
    dt = 1.0 / sr
    c = 343.0

    # Load measured RIRs
    rirs = load_measured_rirs(rir_dir)
    if verbose:
        print(f"  Loaded {len(rirs)} measured RIRs")

    # Compute measured per-band T30
    measured_t30 = compute_measured_band_t30s(rirs, bands)
    if verbose:
        for fc in bands:
            vals = measured_t30[fc]
            if vals:
                print(f"  {fc:5d} Hz: measured T30 = {np.mean(vals):.3f}s "
                      f"(std={np.std(vals):.3f}, n={len(vals)})")

    # Precompute per-surface modal coupling weights
    weights = precompute_surface_gamma_weights(
        room.mesh, room.ops, room._eigenvectors, room._dimensions)

    # Precompute source coupling for each measured RIR
    # BRAS CR2: LS1 and LS2 are the two source positions
    # We need to know source/receiver positions from the filenames
    # For now, use the mesh nearest_node to find receivers
    # Source positions (approximate for BRAS CR2 seminar room)
    source_positions = {
        'LS1': (2.0, 3.35, 1.5),
        'LS2': (4.5, 4.0, 1.5),
    }

    # Compute modal amplitudes for each source position
    M = room.ops['M_diag']
    modal_amps = {}
    for ls_name, src_pos in source_positions.items():
        r2 = ((room.mesh.x - src_pos[0]) ** 2 +
              (room.mesh.y - src_pos[1]) ** 2 +
              (room.mesh.z - src_pos[2]) ** 2)
        p0 = np.exp(-r2 / 0.3 ** 2)
        modal_amps[ls_name] = room._eigenvectors.T @ (M * p0)

    omega = np.sqrt(np.maximum(room._eigenvalues, 0)) * c

    # Receiver positions (approximate for BRAS CR2)
    # These are rough — the exact positions are in the SketchUp file
    receiver_map = {
        'MP1': (6.0, 1.5, 1.2),
        'MP2': (6.0, 3.0, 1.2),
        'MP3': (6.0, 4.5, 1.2),
        'MP4': (3.0, 1.5, 1.2),
        'MP5': (3.0, 5.0, 1.2),
    }

    # For each RIR, identify source and find nearest receiver node
    rir_configs = []
    for fn, ir_meas, sr_meas in rirs:
        parts = fn.replace('.wav', '').split('_')
        ls_name = parts[2]  # LS1 or LS2
        mp_name = parts[3]  # MP1-MP5

        rec_pos = receiver_map.get(mp_name, (4.0, 3.0, 1.2))
        rec_idx = room.mesh.nearest_node(*rec_pos)

        rir_configs.append({
            'fn': fn,
            'ir_meas': ir_meas,
            'sr': sr_meas,
            'ls': ls_name,
            'mp': mp_name,
            'rec_idx': rec_idx,
            'modal_amps': modal_amps[ls_name],
        })

    # Calibrate per band
    calibrated_alpha = {}
    residuals = {}

    for fc in bands:
        if verbose:
            print(f"\n  Calibrating {fc} Hz band...")

        # Measured T30 per RIR at this band
        meas_per_rir = []
        valid_configs = []
        for cfg in rir_configs:
            t30 = compute_band_t30(cfg['ir_meas'], cfg['sr'], fc)
            if t30 is not None:
                meas_per_rir.append(t30)
                valid_configs.append(cfg)

        if len(meas_per_rir) < 3:
            if verbose:
                print(f"    Skipping — too few valid measurements ({len(meas_per_rir)})")
            continue

        meas_arr = np.array(meas_per_rir)

        # Precompute axial mode caches for each RIR config
        f_max_modal = np.max(np.sqrt(np.maximum(room._eigenvalues, 0))
                             * c / (2 * np.pi))
        V = Lx * Ly * Lz
        S = 2 * (Lx * Ly + Lx * Lz + Ly * Lz)

        axial_caches = []
        for cfg in valid_configs:
            cache = AxialModeCache(
                room._parallel_pairs,
                source_positions[cfg['ls']],
                receiver_map[cfg['mp']],
                f_min=0, f_max=4000, c=c, sr=sr)
            axial_caches.append(cache)

        # Precompute eigenvector frequencies for band filtering
        eigen_freqs = np.sqrt(np.maximum(room._eigenvalues, 0)) * c / (2 * np.pi)

        # Band limits for current octave
        fl_band = fc / np.sqrt(2)
        fh_band = fc * np.sqrt(2)

        # Eigenmodes in this band
        eigen_in_band = (eigen_freqs >= fl_band) & (eigen_freqs <= fh_band)

        # Objective: estimate T30 from energy-weighted decay rates
        # Much faster than full IR synthesis + Schroeder analysis.
        #
        # For modes in the octave band, T30 is determined by the
        # energy-weighted average decay rate:
        #   gamma_eff = sum(A_i^2 * gamma_i) / sum(A_i^2)
        #   T30 ≈ 6.91 / gamma_eff  (for -60 dB decay)
        #
        # This is exact for a single mode and a good approximation
        # when multiple modes have similar decay rates (which they do
        # within an octave band for a room with uniform-ish absorption).
        def objective(alpha_vec):
            alpha_dict = {label: alpha_vec[i]
                          for i, label in enumerate(surface_labels)}

            # Eigenmodes: gamma depends on alpha via precomputed weights
            gamma = compute_gamma_from_alpha(weights, alpha_dict, c)

            # Axial modes: compute decay rates for this alpha
            rho_c = 1.2 * c
            mean_alpha = np.mean(list(alpha_dict.values()))
            rt60_room = 0.161 * V / (-S * np.log(1 - min(max(mean_alpha, 0.01), 0.99)))
            gamma_room_val = 6.91 / max(rt60_room, 0.05)

            total_err = 0.0
            for k, cfg in enumerate(valid_configs):
                # Collect all modes (eigen + axial) in this octave band
                # with their amplitudes and decay rates
                energies = []
                gammas = []

                # Eigenmodes in band
                phi_rec = room._eigenvectors[cfg['rec_idx'], :]
                for i in np.where(eigen_in_band)[0]:
                    A = cfg['modal_amps'][i] * phi_rec[i]
                    if abs(A) > 1e-30:
                        energies.append(A ** 2)
                        gammas.append(gamma[i])

                # Axial modes in band
                for m in axial_caches[k].modes:
                    if fl_band <= m['freq'] <= fh_band:
                        l1, l2 = m['pair_labels']
                        a1 = np.clip(alpha_dict.get(l1, 0.05), 0.001, 0.999)
                        a2 = np.clip(alpha_dict.get(l2, 0.05), 0.001, 0.999)
                        R_prod = (1 - a1) * (1 - a2)
                        if R_prod <= 0:
                            continue
                        gp = (c / (2 * m['L'])) * (-np.log(R_prod))
                        # Coupling
                        if S > 0:
                            coup = 1.0 - min(m['A_pair'] / S, 1.0)
                            g = (1 - coup) * gp + coup * gamma_room_val
                        else:
                            g = gp
                        energies.append(m['amplitude'] ** 2)
                        gammas.append(g)

                if not energies:
                    total_err += meas_arr[k] ** 2
                    continue

                energies = np.array(energies)
                gammas = np.array(gammas)

                # Energy-weighted average decay rate
                gamma_eff = np.sum(energies * gammas) / np.sum(energies)
                t30_sim = 6.91 / max(gamma_eff, 0.01)

                total_err += (t30_sim - meas_arr[k]) ** 2
            return total_err

        # Initial guess: Eyring-inverted room-mean alpha
        mean_t30 = np.mean(meas_arr)
        V = Lx * Ly * Lz
        S = 2 * (Lx * Ly + Lx * Lz + Ly * Lz)
        alpha_init = 1.0 - np.exp(-0.161 * V / (S * max(mean_t30, 0.1)))
        x0 = np.full(len(surface_labels), alpha_init)

        # Bounds: absorption between 0.01 and 0.99
        bounds = [(0.01, 0.99)] * len(surface_labels)

        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 50, 'ftol': 1e-6})

        alpha_opt = {label: result.x[i]
                     for i, label in enumerate(surface_labels)}
        calibrated_alpha[fc] = alpha_opt

        # Final residual
        rms_err = np.sqrt(result.fun / len(valid_configs))
        residuals[fc] = rms_err

        if verbose:
            print(f"    Measured mean T30: {mean_t30:.3f}s")
            print(f"    Initial alpha (Eyring): {alpha_init:.4f}")
            print(f"    Optimized alpha per surface:")
            for label in surface_labels:
                print(f"      {label:10s}: {alpha_opt[label]:.4f}")
            print(f"    RMS T30 error: {rms_err:.4f}s "
                  f"({rms_err / mean_t30 * 100:.1f}%)")

    return calibrated_alpha, residuals


def print_calibration_summary(calibrated_alpha, surface_labels=None):
    """Print calibrated absorption as a formatted table."""
    if surface_labels is None:
        surface_labels = ['floor', 'ceiling', 'front', 'back', 'left', 'right']

    bands = sorted(calibrated_alpha.keys())
    header = f"  {'Surface':<10s}" + "".join(f"  {fc:>6d}Hz" for fc in bands)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label in surface_labels:
        row = f"  {label:<10s}"
        for fc in bands:
            alpha = calibrated_alpha.get(fc, {}).get(label, 0)
            row += f"  {alpha:>8.4f}"
        print(row)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from room_acoustics.room import Room

    print("=" * 70)
    print("  Absorption Calibration from BRAS CR2 Measured RIRs")
    print("=" * 70)

    # Build room
    room = Room.from_box(8.4, 6.7, 3.0, P=4, ppw=4, f_target=250)
    room.set_material_default('plaster')
    room.build(n_modes=100)

    rir_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'bras_data', '1 Scene descriptions',
        'CR2 small room (seminar room)', 'RIRs', 'wav')

    calibrated, residuals = calibrate_absorption(
        room, rir_dir, bands=(250, 500, 1000, 2000))

    print("\n" + "=" * 70)
    print("  CALIBRATED ABSORPTION COEFFICIENTS")
    print("=" * 70)
    print_calibration_summary(calibrated)

    print("\n  Residuals:")
    for fc, rms in sorted(residuals.items()):
        print(f"    {fc:5d} Hz: {rms:.4f}s RMS error")
