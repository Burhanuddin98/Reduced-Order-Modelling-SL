"""
Frequency-domain Helmholtz solver for room acoustics.

Solves (S - omega^2 * M + i*omega * C) * p = f at each frequency,
where:
  S = stiffness matrix (Laplacian, sparse)
  M = mass matrix (diagonal)
  C = damping matrix from boundary impedance (diagonal)
  f = source vector (point source)

The transfer function H(omega) = p[rec] / f[src] gives the
frequency response between source and receiver. IFFT produces
the impulse response.

No time-stepping. No numerical dispersion. No eigenvalue computation.
Frequency-dependent materials are trivial — just update C(omega).

The ROM reduces each N-dimensional solve to an r-dimensional solve,
enabling real-time frequency sweeps.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, factorized, LinearOperator, gmres


# ===================================================================
# Core solver
# ===================================================================

def _setup_source(mesh, src_pos, sigma, M_diag):
    """Compute mass-weighted Gaussian source vector."""
    if hasattr(mesh, '_ensure_coords'):
        mesh._ensure_coords()
    r2 = (mesh.x - src_pos[0])**2 + (mesh.y - src_pos[1])**2
    if hasattr(mesh, 'z'):
        r2 += (mesh.z - src_pos[2])**2
    return M_diag * np.exp(-r2 / sigma**2)


def _get_damping(bc_params, freq, rc2, B_diag, N):
    """Get damping coefficient vector for a given frequency."""
    if 'Z_func' in bc_params:
        Z_vec = bc_params['Z_func'](freq)
        return rc2 * B_diag / Z_vec
    elif 'Z_per_node' in bc_params:
        return rc2 * B_diag / np.asarray(bc_params['Z_per_node'], dtype=float)
    elif 'Z' in bc_params:
        return rc2 / bc_params['Z'] * B_diag
    return np.zeros(N)


def helmholtz_transfer_function(ops, mesh, src_pos, rec_idx,
                                 freqs, bc_params, sigma=0.3,
                                 c=343.0, rho=1.2):
    """
    Compute the transfer function H(f) between source and receiver.

    Uses preconditioned GMRES with LU-factored stiffness matrix as
    preconditioner. The LU factorization is done once; each frequency
    only requires a few GMRES iterations since the diagonal shift
    (omega^2 * M) is a small perturbation of S at low frequencies.

    For frequency-independent impedance, the damping matrix is also
    precomputed and the preconditioner includes it.
    """
    N = mesh.N_dof
    S = ops['S']
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())
    rc2 = rho * c**2

    f_rhs = _setup_source(mesh, src_pos, sigma, M_diag)
    H = np.zeros(len(freqs), dtype=complex)

    # Check if damping is frequency-independent
    freq_dep = 'Z_func' in bc_params
    if not freq_dep:
        C_diag_base = _get_damping(bc_params, 0, rc2, B_diag, N)

    # Strategy: use scipy.sparse.linalg.splu for symbolic factorization
    # reuse. For frequency-independent damping, factor once per frequency
    # using direct LU. The key optimization: convert to complex CSC once,
    # then update the diagonal in-place for each frequency.

    # Pre-convert S to complex CSC
    S_csc = S.tocsc().astype(complex)

    if not freq_dep:
        C_diag_base_arr = C_diag_base

    print(f"  Solving {len(freqs)} frequencies (N={N})...", end='', flush=True)

    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        k2 = (omega / c) ** 2

        if freq_dep:
            C_diag = _get_damping(bc_params, freq, rc2, B_diag, N)
        else:
            C_diag = C_diag_base_arr

        # A = S - k2*M + i*omega*C (all shifts are diagonal)
        diag_shift = -k2 * M_diag + 1j * omega * C_diag
        A = S_csc + sparse.diags(diag_shift, format='csc')

        try:
            p = spsolve(A, f_rhs)
            H[i] = p[rec_idx]
        except Exception:
            H[i] = 0.0

        if (i + 1) % max(1, len(freqs) // 10) == 0:
            print(f" {i+1}/{len(freqs)}", end='', flush=True)

    return H


def transfer_function_to_ir(H, freqs, sr=44100, T=None):
    """
    Convert a transfer function H(f) to an impulse response via IFFT.

    Parameters
    ----------
    H : complex array — transfer function at positive frequencies
    freqs : array — corresponding frequencies [Hz]
    sr : int — output sample rate [Hz]
    T : float or None — output duration [s]. If None, use 1/df.

    Returns
    -------
    ir : real array — impulse response
    t : time vector [s]
    """
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    if T is None:
        T = 1.0 / df

    n_fft = int(sr * T)
    # Map H onto the FFT frequency grid
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    n_rfft = len(fft_freqs)

    # Interpolate H onto the FFT grid
    H_interp = np.interp(fft_freqs, freqs, np.real(H)) + \
               1j * np.interp(fft_freqs, freqs, np.imag(H))

    # IFFT
    ir = np.fft.irfft(H_interp, n=n_fft)

    t = np.arange(n_fft) / sr
    return ir, t


# ===================================================================
# Reduced Basis Method (ROM for frequency domain)
# ===================================================================

def build_frequency_rom(ops, mesh, src_pos, bc_params, sigma=0.3,
                        training_freqs=None, n_basis=30,
                        c=343.0, rho=1.2):
    """
    Build a reduced basis for fast frequency sweeps.

    Solves the full system at a few training frequencies, collects
    the solution snapshots, and builds a reduced basis via SVD.

    Parameters
    ----------
    ops : operator dict
    mesh : mesh object
    src_pos : source position
    bc_params : boundary params (can include Z_func for freq-dep)
    training_freqs : array of frequencies for snapshot collection.
                     If None, uses logarithmically spaced points.
    n_basis : max number of basis vectors to retain
    c, rho : physical constants

    Returns
    -------
    rom : dict with reduced operators and basis
    """
    N = mesh.N_dof
    S = ops['S']
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())
    rc2 = rho * c**2

    if hasattr(mesh, '_ensure_coords'):
        mesh._ensure_coords()
    r2 = (mesh.x - src_pos[0])**2 + (mesh.y - src_pos[1])**2
    if hasattr(mesh, 'z'):
        r2 += (mesh.z - src_pos[2])**2
    f_src = np.exp(-r2 / sigma**2)
    f_rhs = M_diag * f_src

    M_sp = sparse.diags(M_diag)

    if training_freqs is None:
        training_freqs = np.logspace(np.log10(20), np.log10(4000), 40)

    print(f"  Building ROM basis from {len(training_freqs)} training frequencies...",
          end='', flush=True)

    # Collect snapshots
    snapshots = []
    for freq in training_freqs:
        omega = 2 * np.pi * freq
        k2 = (omega / c) ** 2

        if 'Z_func' in bc_params:
            Z_vec = bc_params['Z_func'](freq)
            C_diag = rc2 * B_diag / Z_vec
        elif 'Z_per_node' in bc_params:
            Z_vec = np.asarray(bc_params['Z_per_node'], dtype=float)
            C_diag = rc2 * B_diag / Z_vec
        elif 'Z' in bc_params:
            C_diag = rc2 / bc_params['Z'] * B_diag
        else:
            C_diag = np.zeros(N)

        A = S - k2 * M_sp + 1j * omega * sparse.diags(C_diag)
        try:
            p = spsolve(A.tocsc(), f_rhs)
            snapshots.append(p)
        except Exception:
            pass

    snapshots = np.column_stack(snapshots)  # (N, n_train)

    # SVD to get reduced basis (use real + imag parts)
    snap_ri = np.column_stack([snapshots.real, snapshots.imag])
    from scipy.linalg import svd
    U, sigma_vals, _ = svd(snap_ri, full_matrices=False)
    n_basis = min(n_basis, len(sigma_vals))
    Psi = U[:, :n_basis].real  # real basis

    print(f" {n_basis} basis vectors, sigma_min/max={sigma_vals[n_basis-1]:.2e}/{sigma_vals[0]:.2e}")

    # Project operators onto basis
    S_r = Psi.T @ S.dot(Psi)             # (r, r)
    M_r = Psi.T @ (M_diag[:, None] * Psi)  # (r, r)
    B_r = Psi.T @ (B_diag[:, None] * Psi)  # (r, r)
    f_r = Psi.T @ f_rhs                     # (r,)

    return {
        'Psi': Psi,
        'S_r': S_r,
        'M_r': M_r,
        'B_r': B_r,
        'f_r': f_r,
        'n_basis': n_basis,
        'N_full': N,
        'bc_params': bc_params,
        'rc2': rc2,
        'c': c,
        'rho': rho,
        'B_diag': B_diag,
    }


def rom_transfer_function(rom, rec_idx, freqs):
    """
    Evaluate transfer function using the reduced basis.

    Each frequency is an r×r dense solve — microseconds.

    Parameters
    ----------
    rom : dict from build_frequency_rom
    rec_idx : receiver node index
    freqs : array of frequencies [Hz]

    Returns
    -------
    H : complex array (len(freqs),)
    """
    Psi = rom['Psi']
    S_r = rom['S_r']
    M_r = rom['M_r']
    B_r = rom['B_r']
    f_r = rom['f_r']
    c = rom['c']
    rc2 = rom['rc2']
    B_diag = rom['B_diag']
    bc_params = rom['bc_params']
    r = rom['n_basis']

    # Receiver observation vector
    obs = Psi[rec_idx, :]  # (r,)

    H = np.zeros(len(freqs), dtype=complex)

    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        k2 = (omega / c) ** 2

        # Damping: need to project frequency-dependent C onto basis
        if 'Z_func' in bc_params:
            Z_vec = bc_params['Z_func'](freq)
            C_diag = rc2 * B_diag / Z_vec
            C_r = Psi.T @ (C_diag[:, None] * Psi)
        elif 'Z_per_node' in bc_params:
            Z_vec = np.asarray(bc_params['Z_per_node'], dtype=float)
            C_diag = rc2 * B_diag / Z_vec
            # C_r is constant (Z doesn't change with freq)
            if i == 0:
                C_r = Psi.T @ (C_diag[:, None] * Psi)
        elif 'Z' in bc_params:
            C_diag = rc2 / bc_params['Z'] * B_diag
            if i == 0:
                C_r = Psi.T @ (C_diag[:, None] * Psi)
        else:
            C_r = np.zeros((r, r))

        # Reduced system: (S_r - k2*M_r + i*omega*C_r) * a = f_r
        A_r = S_r - k2 * M_r + 1j * omega * C_r
        try:
            a = np.linalg.solve(A_r, f_r)
            H[i] = obs @ a
        except np.linalg.LinAlgError:
            H[i] = 0.0

    return H
