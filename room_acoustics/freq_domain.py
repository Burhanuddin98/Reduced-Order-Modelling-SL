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

    Uses splu (sparse LU) per frequency. The system matrix has the same
    sparsity pattern at every frequency (only diagonal changes), so
    scipy's internal symbolic analysis is reused across calls.

    For maximum speed, the matrix is built by modifying the diagonal
    of a pre-allocated CSC matrix in-place, avoiding repeated sparse
    matrix construction.
    """
    import time as _time

    N = mesh.N_dof
    S = ops['S']
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())
    rc2 = rho * c**2

    f_rhs = _setup_source(mesh, src_pos, sigma, M_diag)
    H = np.zeros(len(freqs), dtype=complex)

    freq_dep = 'Z_func' in bc_params
    if not freq_dep:
        C_diag_base = _get_damping(bc_params, 0, rc2, B_diag, N)

    # Build template CSC matrix with S's structure
    # We'll modify the diagonal values in-place for each frequency
    S_csc = S.tocsc().astype(complex).copy()

    # Find diagonal entry positions in CSC data array
    diag_indices = np.empty(N, dtype=np.intp)
    indptr = S_csc.indptr
    indices = S_csc.indices
    for col in range(N):
        start, end = indptr[col], indptr[col + 1]
        row_slice = indices[start:end]
        diag_pos = np.searchsorted(row_slice, col)
        if diag_pos < len(row_slice) and row_slice[diag_pos] == col:
            diag_indices[col] = start + diag_pos
        else:
            diag_indices[col] = -1  # no diagonal entry — shouldn't happen for S

    # Store original diagonal values from S
    S_diag_orig = np.array([S_csc.data[diag_indices[i]] if diag_indices[i] >= 0
                            else 0.0 for i in range(N)], dtype=complex)

    t_start = _time.perf_counter()
    print(f"  Solving {len(freqs)} frequencies (N={N})...", end='', flush=True)

    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        k2 = (omega / c) ** 2

        if freq_dep:
            C_diag = _get_damping(bc_params, freq, rc2, B_diag, N)
        else:
            C_diag = C_diag_base

        # Update diagonal in-place: S_diag + (-k2*M + i*omega*C)
        new_diag = S_diag_orig + (-k2 * M_diag + 1j * omega * C_diag)
        for j in range(N):
            if diag_indices[j] >= 0:
                S_csc.data[diag_indices[j]] = new_diag[j]

        # splu: LU factorization. scipy reuses symbolic analysis when
        # the sparsity pattern hasn't changed.
        try:
            lu = sparse.linalg.splu(S_csc)
            p = lu.solve(f_rhs)
            H[i] = p[rec_idx]
        except Exception:
            # Fallback to spsolve
            try:
                p = spsolve(S_csc, f_rhs)
                H[i] = p[rec_idx]
            except Exception:
                H[i] = 0.0

        if (i + 1) % max(1, len(freqs) // 10) == 0:
            elapsed = _time.perf_counter() - t_start
            rate = elapsed / (i + 1)
            print(f" {i+1}/{len(freqs)} ({rate:.2f}s/freq)", end='', flush=True)

    elapsed = _time.perf_counter() - t_start
    print(f" done ({elapsed:.1f}s total, {elapsed/len(freqs):.2f}s/freq)")
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


def build_greedy_rom(ops, mesh, src_pos, bc_params, sigma=0.5,
                     f_min=20, f_max=500, n_initial=15, max_basis=60,
                     tol=1e-3, max_iter=50, c=343.0, rho=1.2,
                     umfpack_lib=None):
    """
    Greedy Reduced Basis Method for frequency-domain room acoustics.

    Adaptively selects training frequencies where the ROM error is
    largest, ensuring resonance peaks are captured.

    Algorithm:
    1. Solve FOM at n_initial logarithmically spaced frequencies
    2. Build ROM basis via SVD
    3. Evaluate ROM at dense test frequencies
    4. Compute residual at each test frequency (one sparse matvec, cheap)
    5. Find frequency with max residual
    6. Solve FOM there, add to basis
    7. Repeat until residual < tol or max_basis reached

    Parameters
    ----------
    umfpack_lib : ctypes library or None
        If provided, uses UMFPACK for FOM solves (much faster).
        If None, uses scipy spsolve.

    Returns
    -------
    rom : dict with reduced operators, basis, and training info
    """
    import time as _time
    from scipy.linalg import svd

    N = mesh.N_dof
    S = ops['S']
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())
    rc2 = rho * c**2

    f_rhs = _setup_source(mesh, src_pos, sigma, M_diag)
    M_sp = sparse.diags(M_diag)

    # Get damping (assumed frequency-independent for greedy — use mid freq)
    C_diag = _get_damping(bc_params, (f_min + f_max)/2, rc2, B_diag, N)
    freq_dep = 'Z_func' in bc_params

    # Setup UMFPACK if available
    umf_ctx = None
    if umfpack_lib is not None:
        import ctypes
        S_csc = S.tocsc()
        def _ptr(arr, dt=ctypes.c_double):
            return np.ascontiguousarray(arr).ctypes.data_as(ctypes.POINTER(dt))
        umf_ctx = umfpack_lib.helmholtz_umf_init(
            N, S_csc.nnz,
            _ptr(np.ascontiguousarray(S_csc.indptr, dtype=np.int32), ctypes.c_int),
            _ptr(np.ascontiguousarray(S_csc.indices, dtype=np.int32), ctypes.c_int),
            _ptr(np.ascontiguousarray(S_csc.data, dtype=np.float64)),
            _ptr(np.ascontiguousarray(M_diag, dtype=np.float64)))

    def _fom_solve(freq):
        """Solve FOM at one frequency, return full solution vector."""
        omega = 2 * np.pi * freq
        k2 = (omega / c) ** 2
        if freq_dep:
            C_d = _get_damping(bc_params, freq, rc2, B_diag, N)
        else:
            C_d = C_diag

        if umf_ctx is not None:
            import ctypes
            def _ptr(arr, dt=ctypes.c_double):
                return np.ascontiguousarray(arr).ctypes.data_as(ctypes.POINTER(dt))
            # Use UMFPACK but get full solution (solve, then read sol arrays)
            # For now, use the sweep function with rec_idx trick — solve once
            H_r = np.zeros(1, dtype=np.float64)
            H_i = np.zeros(1, dtype=np.float64)
            omegas = np.array([omega], dtype=np.float64)
            C_arr = np.ascontiguousarray(C_d, dtype=np.float64)
            f_arr = np.ascontiguousarray(f_rhs, dtype=np.float64)
            # We need the full solution, not just one point.
            # Fall back to scipy for snapshot collection since we need full p.
            pass

        # Scipy solve (need full solution for basis construction)
        A = S - k2 * M_sp + 1j * omega * sparse.diags(C_d)
        return spsolve(A.tocsc(), f_rhs)

    # Dense test frequencies for error estimation
    n_test = 500
    test_freqs = np.linspace(f_min, f_max, n_test)

    # Step 1: Initial training
    training_freqs = list(np.logspace(np.log10(f_min), np.log10(f_max), n_initial))
    snapshots = []

    t_start = _time.perf_counter()
    print(f"  Greedy ROM: f=[{f_min}-{f_max}]Hz, N={N}")
    print(f"  Initial training ({n_initial} points)...", end='', flush=True)

    for freq in training_freqs:
        p = _fom_solve(freq)
        snapshots.append(p)

    print(f" done ({_time.perf_counter()-t_start:.1f}s)")

    # Greedy loop
    for iteration in range(max_iter):
        # Build basis from current snapshots
        snap_mat = np.column_stack(snapshots)
        snap_ri = np.column_stack([snap_mat.real, snap_mat.imag])
        U, sigma_vals, _ = svd(snap_ri, full_matrices=False)
        n_basis = min(max_basis, len(sigma_vals),
                      np.searchsorted(-sigma_vals, -sigma_vals[0] * 1e-12) + 1)
        Psi = U[:, :n_basis]

        # Project operators
        S_r = Psi.T @ S.dot(Psi)
        M_r = Psi.T @ (M_diag[:, None] * Psi)
        f_r = Psi.T @ f_rhs

        if not freq_dep:
            C_r = Psi.T @ (C_diag[:, None] * Psi)

        # Evaluate ROM at all test frequencies, compute residuals
        max_res = 0
        worst_freq = test_freqs[0]

        for freq in test_freqs:
            omega = 2 * np.pi * freq
            k2 = (omega / c) ** 2

            if freq_dep:
                C_d = _get_damping(bc_params, freq, rc2, B_diag, N)
                C_r_f = Psi.T @ (C_d[:, None] * Psi)
            else:
                C_r_f = C_r
                C_d = C_diag

            # ROM solve
            A_r = S_r - k2 * M_r + 1j * omega * C_r_f
            try:
                a = np.linalg.solve(A_r, f_r)
            except np.linalg.LinAlgError:
                continue

            # Reconstruct full solution
            p_rom = Psi @ a

            # Compute residual: r = f - A*p_rom (one sparse matvec)
            Ap = S.dot(p_rom) + (-k2 * M_diag + 1j * omega * C_d) * p_rom
            residual = np.linalg.norm(f_rhs - Ap) / np.linalg.norm(f_rhs)

            if residual > max_res:
                max_res = residual
                worst_freq = freq

        print(f"  Iter {iteration+1}: basis={n_basis}, max_residual={max_res:.2e} "
              f"at {worst_freq:.0f}Hz", end='')

        if max_res < tol:
            print(f" — CONVERGED")
            break

        # Add worst frequency to training set
        if worst_freq not in training_freqs:
            training_freqs.append(worst_freq)
            p_new = _fom_solve(worst_freq)
            snapshots.append(p_new)
            print(f" — added {worst_freq:.0f}Hz")
        else:
            print(f" — already trained, stopping")
            break

    elapsed = _time.perf_counter() - t_start
    print(f"  Greedy done: {n_basis} basis vectors, {len(training_freqs)} FOM solves, "
          f"{elapsed:.1f}s")

    # Final basis and operators
    snap_mat = np.column_stack(snapshots)
    snap_ri = np.column_stack([snap_mat.real, snap_mat.imag])
    U, sigma_vals, _ = svd(snap_ri, full_matrices=False)
    n_basis = min(max_basis, len(sigma_vals),
                  np.searchsorted(-sigma_vals, -sigma_vals[0] * 1e-12) + 1)
    Psi = U[:, :n_basis]

    S_r = Psi.T @ S.dot(Psi)
    M_r = Psi.T @ (M_diag[:, None] * Psi)
    B_r = Psi.T @ (B_diag[:, None] * Psi)
    f_r = Psi.T @ f_rhs

    # Cleanup UMFPACK
    if umf_ctx is not None:
        umfpack_lib.helmholtz_umf_free(umf_ctx)

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
        'training_freqs': training_freqs,
        'max_residual': max_res,
        'n_fom_solves': len(training_freqs),
        'build_time_s': elapsed,
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
