"""Laplace-domain FOM solver, Weeks ILT, Miki model, ROM basis + projection."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import svd
import time as _time

C_AIR = 343.0
RHO_AIR = 1.2


# ── Miki model ───────────────────────────────────────────────

def miki_impedance(f, sigma_flow, d_mat):
    f = np.asarray(f, dtype=complex)
    f_safe = np.where(np.abs(f) < 1.0, 1.0, f)
    X = np.abs(f_safe) / sigma_flow
    Zc = RHO_AIR*C_AIR*(1 + 0.0699*X**(-0.632) - 1j*0.107*X**(-0.632))
    kc = (2*np.pi*f_safe/C_AIR)*(1 + 0.109*X**(-0.618) - 1j*0.160*X**(-0.618))
    return -1j * Zc * np.cos(kc*d_mat) / np.sin(kc*d_mat)

def miki_admittance_scalar(f, sigma_flow, d_mat):
    return 1.0 / complex(miki_impedance(max(abs(f), 1.0), sigma_flow, d_mat))

def miki_absorption(f, sigma_flow, d_mat):
    Zs = miki_impedance(f, sigma_flow, d_mat)
    return 1 - np.abs((Zs - RHO_AIR*C_AIR) / (Zs + RHO_AIR*C_AIR))**2


# ── Weeks method ─────────────────────────────────────────────

def weeks_s_values(sigma, b, N):
    k = np.arange(N); theta = 2*np.pi*k/N; z = np.exp(1j*theta)
    z_safe = np.where(np.abs(1-z) < 1e-10, 1-1e-10, z)
    return sigma + b*(1+z_safe)/(1-z_safe), z_safe

def weeks_coefficients(H, b, z_safe):
    return np.fft.fft(H*(2*b/(1-z_safe))) / len(H)

def laguerre_eval(n, x):
    L = np.zeros((n, len(x))); L[0] = 1.0
    if n > 1: L[1] = 1.0 - x
    for k in range(1, n-1):
        L[k+1] = ((2*k+1-x)*L[k] - k*L[k-1]) / (k+1)
    return L

def weeks_reconstruct(a, sigma, b, t):
    return np.exp((sigma-b)*t) * np.real(a @ laguerre_eval(len(a), 2*b*t))

def laplace_to_ir(H, sigma, b, t):
    _, z_safe = weeks_s_values(sigma, b, len(H))
    return weeks_reconstruct(weeks_coefficients(H, b, z_safe), sigma, b, t)


# ── IFFT time reconstruction ─────────────────────────────────

def ifft_frequencies(f_max, N_freq):
    """Real frequencies for IFFT reconstruction.

    Returns s_values = sigma + i*2*pi*f for uniformly spaced f in [0, f_max].
    sigma is a small damping shift to ensure convergence.
    """
    sigma = 5.0  # small positive shift (abscissa of convergence)
    freqs = np.linspace(0, f_max, N_freq)
    omega = 2 * np.pi * freqs
    return sigma + 1j * omega, freqs, sigma


def ifft_to_ir(H, freqs, sigma, t_max, fs=44100):
    """Reconstruct time-domain IR from H(s) via inverse FFT.

    H(s) was evaluated at s = sigma + i*omega for uniformly spaced omega.
    The time signal is:
        p(t) = e^{sigma*t} * IFFT[ H(sigma + i*omega) ]
    scaled by df.

    Parameters
    ----------
    H : complex array (N_freq,) — transfer function at real frequencies
    freqs : array (N_freq,) — frequency values [Hz]
    sigma : float — Laplace shift used during evaluation
    t_max : float — desired IR duration [s]
    fs : int — output sample rate

    Returns
    -------
    ir : real array — impulse response at fs sample rate
    t : array — time values
    """
    N_freq = len(H)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # Build two-sided spectrum for IFFT
    # H is one-sided: f = [0, df, 2df, ..., f_max]
    # Mirror to get negative frequencies (conjugate symmetry)
    N_fft = 2 * (N_freq - 1)
    H_full = np.zeros(N_fft, dtype=complex)
    H_full[:N_freq] = H
    H_full[N_freq:] = np.conj(H[-2:0:-1])  # mirror without DC and Nyquist

    # IFFT
    ir_raw = np.fft.ifft(H_full).real * N_fft * df

    # The IFFT gives signal at dt = 1/(N_fft * df)
    dt_ifft = 1.0 / (N_fft * df)
    t_ifft = np.arange(N_fft) * dt_ifft

    # Apply Laplace shift correction: p(t) = e^{sigma*t} * ifft_result
    ir_raw *= np.exp(sigma * t_ifft)

    # Resample to desired fs and t_max
    t_out = np.arange(0, t_max, 1.0/fs)
    ir_out = np.interp(t_out, t_ifft, ir_raw, left=0, right=0)

    return ir_out, t_out


def sweep_fi_ifft(c2S, M, B, p0, N, f_max, N_freq, Zs, rec_idx, verbose=True):
    """FOM sweep at real frequencies for IFFT reconstruction. Returns H, freqs, sigma."""
    s_vals, freqs, sigma = ifft_frequencies(f_max, N_freq)
    Br = C_AIR**2 * RHO_AIR * B / Zs
    Ns = len(s_vals)
    H = np.zeros(Ns, dtype=complex)
    t0 = _time.perf_counter()
    for i, s in enumerate(s_vals):
        H[i] = _solve_laplace(c2S, M, B, p0, N, s, Br)[rec_idx]
        if verbose and (i+1) % max(1, Ns//10) == 0:
            el = _time.perf_counter()-t0; eta = el/(i+1)*(Ns-i-1)
            print(f'  {i+1}/{Ns} ({el:.0f}s, ETA {eta:.0f}s)', end='', flush=True)
    if verbose:
        print(f' done ({_time.perf_counter()-t0:.0f}s)')
    return H, freqs, sigma


def sweep_fd_ifft(c2S, M, B, p0, N, f_max, N_freq, sigma_flow, d_mat,
                  rec_idx, verbose=True):
    """FOM sweep at real frequencies for freq-dep BC + IFFT."""
    s_vals, freqs, sigma = ifft_frequencies(f_max, N_freq)
    Ns = len(s_vals)
    H = np.zeros(Ns, dtype=complex)
    t0 = _time.perf_counter()
    for i, s in enumerate(s_vals):
        f = max(abs(s.imag)/(2*np.pi), 1.0)
        Ys = miki_admittance_scalar(f, sigma_flow, d_mat)
        Br = C_AIR**2 * RHO_AIR * Ys * B
        H[i] = _solve_laplace(c2S, M, B, p0, N, s, Br)[rec_idx]
        if verbose and (i+1) % max(1, Ns//10) == 0:
            el = _time.perf_counter()-t0; eta = el/(i+1)*(Ns-i-1)
            print(f'  {i+1}/{Ns} ({el:.0f}s, ETA {eta:.0f}s)', end='', flush=True)
    if verbose:
        print(f' done ({_time.perf_counter()-t0:.0f}s)')
    return H, freqs, sigma


def sweep_persurface_ifft(c2S, M, B_labels, p0, N, f_max, N_freq,
                           mat_data, rec_idx, verbose=True):
    """FOM sweep with per-surface freq-dep absorption for IFFT.

    mat_data: dict {face_label: (freqs_array, alpha_array)}
    """
    s_vals, freqs, sigma = ifft_frequencies(f_max, N_freq)
    Ns = len(s_vals)
    H = np.zeros(Ns, dtype=complex)
    t0 = _time.perf_counter()
    for i, s in enumerate(s_vals):
        f = max(abs(s.imag)/(2*np.pi), 1.0)
        Br_diag = np.zeros(N)
        for face, (f_mat, alpha_mat) in mat_data.items():
            a = np.clip(np.interp(min(f, f_mat[-1]), f_mat, alpha_mat), 0.001, 0.999)
            Z = _alpha_to_Z_internal(a)
            if face in B_labels:
                Br_diag += C_AIR**2 * RHO_AIR * B_labels[face] / Z
        sig, omg = s.real, s.imag
        Kr = c2S + sparse.diags((sig**2-omg**2)*M + sig*Br_diag, format='csc')
        Kc = sparse.diags(2*sig*omg*M + omg*Br_diag, format='csc')
        A = sparse.bmat([[Kr,-Kc],[Kc,Kr]], format='csc')
        rhs = np.concatenate([sig*p0*M, omg*p0*M])
        x = spsolve(A, rhs)
        H[i] = x[rec_idx] + 1j*x[N+rec_idx]
        if verbose and (i+1) % max(1, Ns//10) == 0:
            el = _time.perf_counter()-t0; eta = el/(i+1)*(Ns-i-1)
            print(f'  {i+1}/{Ns} ({el:.0f}s, ETA {eta:.0f}s)', end='', flush=True)
    if verbose:
        print(f' done ({_time.perf_counter()-t0:.0f}s)')
    return H, freqs, sigma


def _alpha_to_Z_internal(alpha):
    """Quick alpha -> Z conversion (same as materials.absorption_to_impedance)."""
    rho_c = RHO_AIR * C_AIR
    return rho_c * (1 + np.sqrt(1 - alpha)) / np.sqrt(alpha)


# ── Laplace FOM ──────────────────────────────────────────────

def _solve_laplace(c2S, M, B, p0, N, s, Br_diag):
    sig, omg = s.real, s.imag
    Kr = c2S + sparse.diags((sig**2-omg**2)*M + sig*Br_diag.real - omg*Br_diag.imag, format='csc')
    Kc = sparse.diags(2*sig*omg*M + omg*Br_diag.real + sig*Br_diag.imag, format='csc')
    A = sparse.bmat([[Kr,-Kc],[Kc,Kr]], format='csc')
    rhs = np.concatenate([sig*p0*M, omg*p0*M])
    x = spsolve(A, rhs)
    return x[:N] + 1j*x[N:]


def sweep_fi(c2S, M, B, p0, N, s_vals, Zs, rec_idx, verbose=True):
    """FOM sweep for frequency-independent BC. Returns H(rec)."""
    Br = C_AIR**2 * RHO_AIR * B / Zs
    Ns = len(s_vals)
    H = np.zeros(Ns, dtype=complex)
    t0 = _time.perf_counter()
    for i, s in enumerate(s_vals):
        H[i] = _solve_laplace(c2S, M, B, p0, N, s, Br)[rec_idx]
        if verbose and (i+1) % max(1, Ns//10) == 0:
            el = _time.perf_counter()-t0
            print(f'  {i+1}/{Ns} ({el:.0f}s)', end='', flush=True)
    if verbose:
        print(f' done ({_time.perf_counter()-t0:.0f}s)')
    return H


def sweep_fd(c2S, M, B, p0, N, s_vals, sigma_flow, d_mat, rec_idx, verbose=True):
    """FOM sweep for frequency-dependent BC."""
    Ns = len(s_vals)
    H = np.zeros(Ns, dtype=complex)
    t0 = _time.perf_counter()
    for i, s in enumerate(s_vals):
        f = max(abs(s.imag)/(2*np.pi), 1.0)
        Ys = miki_admittance_scalar(f, sigma_flow, d_mat)
        Br = C_AIR**2 * RHO_AIR * Ys * B
        H[i] = _solve_laplace(c2S, M, B, p0, N, s, Br)[rec_idx]
        if verbose and (i+1) % max(1, Ns//10) == 0:
            el = _time.perf_counter()-t0
            print(f'  {i+1}/{Ns} ({el:.0f}s)', end='', flush=True)
    if verbose:
        print(f' done ({_time.perf_counter()-t0:.0f}s)')
    return H


def sweep_fi_fullfield(c2S, M, B, p0, N, s_vals, Zs, verbose=True):
    """Full-field sweep for snapshot collection."""
    Br = C_AIR**2 * RHO_AIR * B / Zs
    snaps = []
    t0 = _time.perf_counter()
    for i, s in enumerate(s_vals):
        snaps.append(_solve_laplace(c2S, M, B, p0, N, s, Br))
        if verbose and (i+1) % max(1, len(s_vals)//10) == 0:
            el = _time.perf_counter()-t0
            print(f'  {i+1}/{len(s_vals)} ({el:.0f}s)', end='', flush=True)
    if verbose:
        print(f' done ({_time.perf_counter()-t0:.0f}s)')
    return snaps


def sweep_fd_fullfield(c2S, M, B, p0, N, s_vals, sigma_flow, d_mat, verbose=True):
    """Full-field sweep for freq-dep snapshots."""
    snaps = []
    t0 = _time.perf_counter()
    for i, s in enumerate(s_vals):
        f = max(abs(s.imag)/(2*np.pi), 1.0)
        Ys = miki_admittance_scalar(f, sigma_flow, d_mat)
        Br = C_AIR**2 * RHO_AIR * Ys * B
        snaps.append(_solve_laplace(c2S, M, B, p0, N, s, Br))
        if verbose and (i+1) % max(1, len(s_vals)//10) == 0:
            el = _time.perf_counter()-t0
            print(f'  {i+1}/{len(s_vals)} ({el:.0f}s)', end='', flush=True)
    if verbose:
        print(f' done ({_time.perf_counter()-t0:.0f}s)')
    return snaps


# ── ROM basis ────────────────────────────────────────────────

def build_basis(snapshots, eps_pod=1e-6):
    """Cotangent-lift SVD. Returns (Psi, Nrb, singular_values)."""
    S_cl = np.column_stack([
        np.column_stack([p.real for p in snapshots]),
        np.column_stack([p.imag for p in snapshots]),
    ])
    U, sv, _ = svd(S_cl, full_matrices=False)
    energy = np.cumsum(sv**2) / np.sum(sv**2)
    Nrb = min(int(np.searchsorted(energy, 1-eps_pod)+1), len(sv))
    return U[:, :Nrb], Nrb, sv


def project_operators(ops, Psi, p0, rec_idx):
    """Project FOM operators onto ROM basis."""
    M = ops['M_diag']; B = np.array(ops['B_total'].diagonal())
    return dict(
        M_r=Psi.T @ (M[:,None]*Psi),
        S_r=Psi.T @ ops['S'].dot(Psi),
        MC_r=Psi.T @ (B[:,None]*Psi),
        f_r=Psi.T @ (p0*M),
        obs=Psi[rec_idx,:].copy(),
    )


def rom_solve_fi(rom_ops, s, Zs):
    Br_r = C_AIR**2*RHO_AIR/Zs * rom_ops['MC_r']
    A_r = s**2*rom_ops['M_r'] + C_AIR**2*rom_ops['S_r'] + s*Br_r
    return rom_ops['obs'] @ np.linalg.solve(A_r, s*rom_ops['f_r'])

def rom_solve_fd(rom_ops, s, sigma_flow, d_mat):
    f = max(abs(s.imag)/(2*np.pi), 1.0)
    Ys = miki_admittance_scalar(f, sigma_flow, d_mat)
    Br = s*C_AIR**2*RHO_AIR*Ys
    A_r = s**2*rom_ops['M_r'] + C_AIR**2*rom_ops['S_r'] + Br*rom_ops['MC_r']
    return rom_ops['obs'] @ np.linalg.solve(A_r, s*rom_ops['f_r'])

def rom_sweep_fi(rom_ops, s_vals, Zs):
    return np.array([rom_solve_fi(rom_ops, s, Zs) for s in s_vals])

def rom_sweep_fd(rom_ops, s_vals, sigma_flow, d_mat):
    return np.array([rom_solve_fd(rom_ops, s, sigma_flow, d_mat) for s in s_vals])
