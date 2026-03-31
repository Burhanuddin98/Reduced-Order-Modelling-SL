"""
Laplace-domain solver for room acoustics (Sampedro Llopis et al. 2022).

Solves (s²M + c²S + s·c²·ρ/Z_s · M_C) p = s·p0·M

at complex frequencies s = σ + iω, where σ > 0 provides regularization
that avoids resonance singularities. This makes the ROM basis construction
well-conditioned — unlike the Helmholtz approach at real ω.

The system is split into 2N real equations for numerical stability,
and the time-domain IR is recovered via numerical inverse Laplace transform.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import svd


# ===================================================================
# Core Laplace-domain solver
# ===================================================================

def laplace_solve(ops, mesh, src_pos, sigma_src, s_values, bc_params,
                  c=343.0, rho=1.2):
    """
    Solve the Laplace-domain system at complex frequencies s = σ + iω.

    Follows Sampedro Llopis eq. (9)-(12): splits into 2N real system.

    Parameters
    ----------
    ops : operator dict (M_diag, S, B_total)
    mesh : mesh object
    src_pos : source position (x, y, z)
    sigma_src : Gaussian pulse width [m]
    s_values : array of complex frequencies s = σ + iω
    bc_params : dict with 'Z' or 'Z_per_node'
    c, rho : physical constants

    Returns
    -------
    P : complex array (N, len(s_values)) — full pressure field at each s
    """
    N = mesh.N_dof
    S = ops['S']
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())

    # Source: Gaussian pulse initial condition
    if hasattr(mesh, '_ensure_coords'):
        mesh._ensure_coords()
    r2 = (mesh.x - src_pos[0])**2 + (mesh.y - src_pos[1])**2
    if hasattr(mesh, 'z'):
        r2 += (mesh.z - src_pos[2])**2
    p0 = np.exp(-r2 / sigma_src**2)

    # Boundary impedance
    if 'Z_per_node' in bc_params:
        Z_vec = np.asarray(bc_params['Z_per_node'], dtype=float)
    elif 'Z' in bc_params:
        Z_vec = np.full(N, bc_params['Z'], dtype=float)
    else:
        Z_vec = np.full(N, 1e15, dtype=float)

    # Br = Bc = c²·ρ/Z_s · M_C (boundary damping, diagonal)
    Br_diag = c**2 * rho * B_diag / Z_vec

    # Sparse matrices
    M_sp = sparse.diags(M_diag, format='csc')
    S_sp = S.tocsc()
    c2S = c**2 * S_sp

    P = np.zeros((N, len(s_values)), dtype=complex)

    for i, s in enumerate(s_values):
        sigma = s.real
        omega = s.imag

        # Kr = (σ² - ω²)M + c²S + σ·Br
        Kr_diag = (sigma**2 - omega**2) * M_diag + sigma * Br_diag
        Kr = c2S + sparse.diags(Kr_diag, format='csc')

        # Kc = 2σωM + ω·Bc
        Kc_diag = 2 * sigma * omega * M_diag + omega * Br_diag
        Kc = sparse.diags(Kc_diag, format='csc')

        # RHS: Qr = σ·p0·M, Qc = ω·p0·M
        Qr = sigma * p0 * M_diag
        Qc = omega * p0 * M_diag

        # Build 2N×2N real system:
        # [Kr  -Kc] [pr]   [Qr]
        # [Kc   Kr] [pc] = [Qc]
        A_top = sparse.hstack([Kr, -Kc], format='csc')
        A_bot = sparse.hstack([Kc, Kr], format='csc')
        A = sparse.vstack([A_top, A_bot], format='csc')
        rhs = np.concatenate([Qr, Qc])

        # Solve
        x = spsolve(A, rhs)
        pr = x[:N]
        pc = x[N:]
        P[:, i] = pr + 1j * pc

    return P


def laplace_transfer_function(ops, mesh, src_pos, rec_idx, sigma_src,
                               s_values, bc_params, c=343.0, rho=1.2):
    """
    Compute transfer function H(s) = p[rec_idx] at complex frequencies.

    Parameters
    ----------
    s_values : array of complex s = σ + iω
    rec_idx : receiver node index

    Returns
    -------
    H : complex array (len(s_values),)
    """
    import time as _time

    N = mesh.N_dof
    S = ops['S']
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())

    if hasattr(mesh, '_ensure_coords'):
        mesh._ensure_coords()
    r2 = (mesh.x - src_pos[0])**2 + (mesh.y - src_pos[1])**2
    if hasattr(mesh, 'z'):
        r2 += (mesh.z - src_pos[2])**2
    p0 = np.exp(-r2 / sigma_src**2)

    if 'Z_per_node' in bc_params:
        Z_vec = np.asarray(bc_params['Z_per_node'], dtype=float)
    elif 'Z' in bc_params:
        Z_vec = np.full(N, bc_params['Z'], dtype=float)
    else:
        Z_vec = np.full(N, 1e15, dtype=float)

    Br_diag = c**2 * rho * B_diag / Z_vec
    c2S = c**2 * S.tocsc()

    H = np.zeros(len(s_values), dtype=complex)
    t0 = _time.perf_counter()
    print(f"  Laplace solve: {len(s_values)} points, N={N}...", end='', flush=True)

    for i, s in enumerate(s_values):
        sigma = s.real
        omega = s.imag

        Kr_diag = (sigma**2 - omega**2) * M_diag + sigma * Br_diag
        Kr = c2S + sparse.diags(Kr_diag, format='csc')

        Kc_diag = 2 * sigma * omega * M_diag + omega * Br_diag
        Kc = sparse.diags(Kc_diag, format='csc')

        Qr = sigma * p0 * M_diag
        Qc = omega * p0 * M_diag

        A = sparse.vstack([
            sparse.hstack([Kr, -Kc], format='csc'),
            sparse.hstack([Kc, Kr], format='csc'),
        ], format='csc')
        rhs = np.concatenate([Qr, Qc])

        x = spsolve(A, rhs)
        H[i] = x[rec_idx] + 1j * x[N + rec_idx]

        if (i + 1) % max(1, len(s_values) // 10) == 0:
            elapsed = _time.perf_counter() - t0
            print(f" {i+1}/{len(s_values)} ({elapsed/(i+1):.2f}s/pt)", end='', flush=True)

    elapsed = _time.perf_counter() - t0
    print(f" done ({elapsed:.1f}s)")
    return H


# ===================================================================
# Inverse Laplace transform → time-domain IR
# ===================================================================

def laplace_to_ir(H, s_values, T=3.5, sr=44100):
    """
    Convert Laplace-domain transfer function to time-domain IR.

    Uses the numerical inverse Laplace transform:
    For s = σ + iω evaluated at equally spaced ω values,
    the time-domain signal is recovered via:

        h(t) = (e^{σt} / π) Re[∫ H(σ+iω) e^{iωt} dω]

    which reduces to an IFFT when ω is uniformly spaced.

    Parameters
    ----------
    H : complex array — H(s) at the s_values
    s_values : complex array — s = σ + iω (must share same σ)
    T : float — output duration [s]
    sr : int — sample rate [Hz]

    Returns
    -------
    ir : real array — impulse response
    t : time array [s]
    """
    sigma = s_values[0].real
    omegas = np.imag(s_values)
    d_omega = omegas[1] - omegas[0] if len(omegas) > 1 else 1.0
    df = d_omega / (2 * np.pi)

    n_fft = int(sr * T)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Interpolate H onto FFT frequency grid
    H_interp = np.interp(2*np.pi*fft_freqs, omegas, np.real(H)) + \
               1j * np.interp(2*np.pi*fft_freqs, omegas, np.imag(H))

    # IFFT
    ir_raw = np.fft.irfft(H_interp, n=n_fft)

    # Undo the Laplace damping: multiply by e^{σt}
    t = np.arange(n_fft) / sr
    ir = ir_raw * np.exp(sigma * t)

    return ir, t


# ===================================================================
# Laplace-domain ROM (Reduced Basis Method)
# ===================================================================

def build_laplace_rom(ops, mesh, src_pos, sigma_src, bc_params,
                      sigma_lap=10.0, omega_max=2000*2*np.pi,
                      n_initial=20, max_basis=60, tol=1e-3,
                      max_iter=40, c=343.0, rho=1.2):
    """
    Greedy Reduced Basis Method in the Laplace domain.

    The key difference from Helmholtz-domain ROM: σ > 0 regularizes
    the system, preventing resonance singularities. This makes the
    basis construction well-conditioned.

    Parameters
    ----------
    sigma_lap : float — real part of s (Laplace damping). Typical: 5-20.
    omega_max : float — max angular frequency to cover.
    """
    import time as _time

    N = mesh.N_dof
    S = ops['S']
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())

    if hasattr(mesh, '_ensure_coords'):
        mesh._ensure_coords()
    r2 = (mesh.x - src_pos[0])**2 + (mesh.y - src_pos[1])**2
    if hasattr(mesh, 'z'):
        r2 += (mesh.z - src_pos[2])**2
    p0 = np.exp(-r2 / sigma_src**2)

    if 'Z_per_node' in bc_params:
        Z_vec = np.asarray(bc_params['Z_per_node'], dtype=float)
    elif 'Z' in bc_params:
        Z_vec = np.full(N, bc_params['Z'], dtype=float)
    else:
        Z_vec = np.full(N, 1e15, dtype=float)

    Br_diag = c**2 * rho * B_diag / Z_vec
    c2S = c**2 * S.tocsc()

    def _solve_fom(s):
        """Solve at one complex frequency s, return full solution."""
        sig = s.real; omg = s.imag
        Kr_diag = (sig**2 - omg**2) * M_diag + sig * Br_diag
        Kr = c2S + sparse.diags(Kr_diag, format='csc')
        Kc_diag = 2*sig*omg * M_diag + omg * Br_diag
        Kc = sparse.diags(Kc_diag, format='csc')
        Qr = sig * p0 * M_diag
        Qc = omg * p0 * M_diag

        A = sparse.vstack([
            sparse.hstack([Kr, -Kc], format='csc'),
            sparse.hstack([Kc, Kr], format='csc'),
        ], format='csc')
        rhs = np.concatenate([Qr, Qc])
        x = spsolve(A, rhs)
        return x[:N] + 1j * x[N:]

    def _compute_residual(Psi, s, a):
        """Compute relative residual norm at s for ROM solution a."""
        sig = s.real; omg = s.imag
        p_rom = Psi @ a

        # A*p = (s²M + c²S + s·Br)p
        s_val = sig + 1j * omg
        s2 = s_val**2
        Ap = s2 * M_diag * p_rom + c**2 * S.dot(p_rom) + s_val * Br_diag * p_rom

        # RHS = s·p0·M
        rhs = s_val * p0 * M_diag

        res = np.linalg.norm(rhs - Ap)
        rhs_norm = np.linalg.norm(rhs)
        return res / max(rhs_norm, 1e-30)

    # Initial training points: uniform in ω at fixed σ
    omega_train = np.linspace(0, omega_max, n_initial)
    s_train = [sigma_lap + 1j * w for w in omega_train]

    t_start = _time.perf_counter()
    print(f"  Laplace ROM: sigma={sigma_lap}, f_max={omega_max/(2*np.pi):.0f}Hz, N={N}")
    print(f"  Initial training ({n_initial} points)...", end='', flush=True)

    snapshots = []
    for s in s_train:
        p = _solve_fom(s)
        snapshots.append(p)

    t_init = _time.perf_counter() - t_start
    print(f" done ({t_init:.1f}s)")

    # Test points for error estimation
    n_test = 200
    omega_test = np.linspace(0, omega_max, n_test)
    s_test = [sigma_lap + 1j * w for w in omega_test]

    # Greedy loop
    training_s = list(s_train)

    for iteration in range(max_iter):
        # Build basis from snapshots (real + imaginary parts)
        snap_mat = np.column_stack(snapshots)
        snap_ri = np.column_stack([snap_mat.real, snap_mat.imag])
        U, sigma_vals, _ = svd(snap_ri, full_matrices=False)
        n_basis = min(max_basis, len(sigma_vals),
                      np.searchsorted(-sigma_vals, -sigma_vals[0] * 1e-14) + 1)
        Psi = U[:, :n_basis]

        # Project operators onto basis
        S_r = Psi.T @ (c**2 * S.dot(Psi))  # c²·Ψᵀ·S·Ψ
        M_r = Psi.T @ (M_diag[:, None] * Psi)  # Ψᵀ·M·Ψ
        Br_r = Psi.T @ (Br_diag[:, None] * Psi)  # Ψᵀ·Br·Ψ
        f_r_p0 = Psi.T @ (p0 * M_diag)  # Ψᵀ·M·p0

        # Find worst test point
        max_res = 0
        worst_s = s_test[0]

        for s in s_test:
            sig = s.real; omg = s.imag
            s_val = sig + 1j * omg
            s2 = s_val**2

            # ROM system: (s²M_r + S_r + s·Br_r) a = s·f_r
            A_r = s2 * M_r + S_r + s_val * Br_r
            rhs_r = s_val * f_r_p0

            try:
                a = np.linalg.solve(A_r, rhs_r)
            except np.linalg.LinAlgError:
                continue

            res = _compute_residual(Psi, s, a)
            if res > max_res:
                max_res = res
                worst_s = s

        print(f"  Iter {iteration+1}: basis={n_basis}, max_res={max_res:.2e} "
              f"at f={worst_s.imag/(2*np.pi):.0f}Hz", end='')

        if max_res < tol:
            print(f" — CONVERGED")
            break

        # Add worst point
        training_s.append(worst_s)
        p_new = _solve_fom(worst_s)
        snapshots.append(p_new)
        print(f" — added")

    elapsed = _time.perf_counter() - t_start
    print(f"  Done: {n_basis} basis, {len(training_s)} solves, {elapsed:.0f}s")

    # Final basis
    snap_mat = np.column_stack(snapshots)
    snap_ri = np.column_stack([snap_mat.real, snap_mat.imag])
    U, sigma_vals, _ = svd(snap_ri, full_matrices=False)
    n_basis = min(max_basis, len(sigma_vals),
                  np.searchsorted(-sigma_vals, -sigma_vals[0] * 1e-14) + 1)
    Psi = U[:, :n_basis]

    S_r = Psi.T @ (c**2 * S.dot(Psi))
    M_r = Psi.T @ (M_diag[:, None] * Psi)
    Br_r = Psi.T @ (Br_diag[:, None] * Psi)
    f_r_p0 = Psi.T @ (p0 * M_diag)

    return {
        'Psi': Psi, 'S_r': S_r, 'M_r': M_r, 'Br_r': Br_r,
        'f_r_p0': f_r_p0, 'n_basis': n_basis,
        'sigma_lap': sigma_lap, 'c': c,
        'training_s': training_s,
        'max_residual': max_res,
        'build_time_s': elapsed,
    }


def rom_laplace_sweep(rom, s_values, rec_idx):
    """
    Evaluate transfer function H(s) using the Laplace ROM.

    Each s is an r×r dense solve — microseconds.
    """
    Psi = rom['Psi']
    S_r = rom['S_r']
    M_r = rom['M_r']
    Br_r = rom['Br_r']
    f_r = rom['f_r_p0']
    obs = Psi[rec_idx, :]

    H = np.zeros(len(s_values), dtype=complex)

    for i, s in enumerate(s_values):
        s_val = complex(s)
        A_r = s_val**2 * M_r + S_r + s_val * Br_r
        rhs_r = s_val * f_r
        try:
            a = np.linalg.solve(A_r, rhs_r)
            H[i] = obs @ a
        except np.linalg.LinAlgError:
            H[i] = 0.0

    return H
