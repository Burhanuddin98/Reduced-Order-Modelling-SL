"""
Modal ROM for room acoustics — dispersion-free reverberation.

Instead of building a ROM basis from time-domain FOM snapshots (which
contain numerical dispersion), this approach uses the eigenmodes of the
spatial discretization as the basis. Each mode evolves at its exact
eigenfrequency with no accumulated phase error.

For PR (rigid walls): each mode oscillates as cos(omega_i * t)
For FI (absorbing):   each mode decays as exp(-gamma_i * t) * cos(...)

The result: accurate T30 prediction without mesh-resolution limitations
on reverberation time, because there is no time-stepping in the
traditional sense — each mode is evaluated analytically.

This combines:
  - Wave-based spatial discretization (SEM) for accurate eigenfrequencies
  - Modal superposition for dispersion-free time evolution
  - Model order reduction (keep only first r modes) for speed
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def compute_room_modes(ops, n_modes=200):
    """Compute the first n_modes eigenmodes of the room.

    Solves the generalized eigenvalue problem:
        S * phi = lambda * M * phi

    where lambda = omega^2 / c^2 (for PR boundaries).

    Returns
    -------
    eigenvalues : (n_modes,) — sorted, including lambda=0 (DC mode)
    eigenvectors : (N, n_modes) — M-orthonormal eigenmodes
    frequencies : (n_modes,) — eigenfrequencies in Hz
    """
    M_sp = diags(ops['M_diag'])
    S = ops['S']

    # Shift-invert to find smallest eigenvalues
    eigenvalues, eigenvectors = eigsh(S, k=n_modes, M=M_sp,
                                      sigma=0, which='LM')

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clip tiny negative eigenvalues to zero
    eigenvalues = np.maximum(eigenvalues, 0)

    # Frequencies
    c = 343.0
    frequencies = np.sqrt(eigenvalues) * c / (2 * np.pi)

    return eigenvalues, eigenvectors, frequencies


def modal_ir(mesh, ops, eigenvalues, eigenvectors, bc_type, bc_params,
             x0, y0, z0, sigma, dt, T, rec_idx):
    """Compute impulse response using modal superposition.

    Each mode evolves analytically — no time-stepping, no dispersion.

    For PR: p(x,t) = sum_i A_i * phi_i(x) * cos(omega_i * t)
    For FI: p(x,t) = sum_i A_i * phi_i(x) * exp(-gamma_i * t) * cos(omega_d_i * t)

    where gamma_i is the modal decay rate from the boundary impedance.

    Parameters
    ----------
    mesh : mesh object with .x, .y, .z coordinates
    ops : operator dict with M_diag, S, B_total
    eigenvalues : from compute_room_modes
    eigenvectors : from compute_room_modes
    bc_type : 'PR' or 'FI'
    bc_params : {} for PR, {'Z': scalar} or {'Z_per_node': array} for FI
    x0, y0, z0 : source position
    sigma : Gaussian pulse width
    dt : output time step
    T : total time
    rec_idx : receiver node index

    Returns
    -------
    dict with 'ir' (impulse response) and 'modal_info'
    """
    rho = 1.2
    c = 343.0
    rc2 = rho * c**2
    N = mesh.N_dof
    n_modes = len(eigenvalues)
    Nt = int(round(T / dt))

    # Initial condition: Gaussian pulse
    if hasattr(mesh, '_ensure_coords'):
        mesh._ensure_coords()
    r2 = (mesh.x - x0)**2 + (mesh.y - y0)**2 + (mesh.z - z0)**2
    p0 = np.exp(-r2 / sigma**2)

    # Project initial pressure onto eigenmodes
    # A_i = phi_i^T * M * p0  (since eigenvectors are M-orthonormal)
    M = ops['M_diag']
    modal_amplitudes = eigenvectors.T @ (M * p0)

    # Modal frequencies
    omega = np.sqrt(np.maximum(eigenvalues, 0)) * c  # rad/s

    # Modal decay rates for FI boundaries
    # The damping term in the FOM is: dp/dt += -(rc2/Z) * M^{-1} * B * p
    # Modal projection gives: 2*gamma_i = (rc2) * phi_i^T * (B/Z) * phi_i
    # where the division by Z is per-node and M-orthonormality absorbs M.
    gamma = np.zeros(n_modes)
    if bc_type == 'FI':
        B_diag = np.array(ops['B_total'].diagonal())

        if 'Z_per_node' in bc_params:
            Z_vec = np.asarray(bc_params['Z_per_node'], dtype=float)
            damping_vec = B_diag / Z_vec  # per-node B/Z
        else:
            Z = bc_params['Z']
            damping_vec = B_diag / Z

        # gamma_i = (rc2/2) * phi_i^T * (B/Z) * phi_i
        for i in range(n_modes):
            phi = eigenvectors[:, i]
            gamma[i] = 0.5 * rc2 * np.dot(phi, damping_vec * phi)

    # Receiver mode shapes
    phi_rec = eigenvectors[rec_idx, :]  # (n_modes,)

    # Synthesize IR analytically — no time-stepping loop
    t = np.arange(Nt + 1) * dt
    ir = np.zeros(Nt + 1)

    for i in range(n_modes):
        if abs(modal_amplitudes[i]) < 1e-30:
            continue
        # Skip DC mode (eigenvalue=0, constant pressure offset)
        if eigenvalues[i] < 1e-10 and gamma[i] < 1e-10:
            continue

        A = modal_amplitudes[i] * phi_rec[i]

        if bc_type == 'PR':
            if omega[i] > 0:
                ir += A * np.cos(omega[i] * t)
        else:
            g = gamma[i]
            omega_d = np.sqrt(max(omega[i]**2 - g**2, 0))
            if omega_d > 0:
                ir += A * np.exp(-g * t) * np.cos(omega_d * t)
            elif omega[i] > 0:
                ir += A * np.exp(-g * t)

    # Modal info for diagnostics
    modal_info = {
        'n_modes': n_modes,
        'frequencies_Hz': (omega / (2*np.pi)).tolist(),
        'decay_rates': gamma.tolist(),
        'modal_T60': [60.0 / (20 * g / np.log(10)) if g > 1e-10 else np.inf
                      for g in gamma],
        'amplitudes': modal_amplitudes.tolist(),
    }

    return dict(ir=ir, modal_info=modal_info)
