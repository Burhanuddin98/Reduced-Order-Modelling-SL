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
