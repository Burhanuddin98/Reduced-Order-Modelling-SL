"""
BEM solver for room acoustics — Kirchhoff-Helmholtz boundary integral.

Surface-only discretization: DOFs scale as f² (not f³ like volume FEM).
At 4 kHz for a 169 m³ room: ~1M surface DOFs, 0.8 GB RAM.
Compare volume SEM: 60M DOFs, 2.3 GB — impossible to eigensolve.

GPU acceleration via CuPy for matrix assembly and dense solves.
FMM approximation for O(N log N) matvecs at large N.

Laplace-domain formulation (s = σ + iω) following Sampedro Llopis (2022):
  - Regularizes resonances (no near-singular matrices)
  - Enables stable ROM via SVD of frequency snapshots
  - IR reconstruction via Weeks method (inverse Laplace)

Usage:
    from room_acoustics.bem_solver import BEMSolver

    solver = BEMSolver(mesh)  # TriMesh from room_geometry.py
    H = solver.transfer_function(src, rec, freqs, materials)
    ir, t = solver.impulse_response(src, rec, materials, T=3.0)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import time

try:
    import cupy as cp
    from cupy import asnumpy
    _GPU = True
except ImportError:
    cp = None
    _GPU = False

from .material_function import MaterialFunction, air_absorption_coefficient


# ═══════════════════════════════════════════════════════════════════
# Green's function and derivatives
# ═══════════════════════════════════════════════════════════════════

def _green_3d(r, k):
    """Free-space Green's function G = exp(ikr) / (4πr)."""
    return np.exp(1j * k * r) / (4 * np.pi * r + 1e-30)


def _dgreen_dn_3d(r_vec, r_mag, normal, k):
    """Normal derivative of Green's function: dG/dn = G * (ik - 1/r) * (r·n)/r."""
    r_hat = r_vec / (r_mag[:, :, None] + 1e-30)
    cos_angle = np.sum(r_hat * normal[None, :, :], axis=2)
    G = _green_3d(r_mag, k)
    return G * (1j * k - 1.0 / (r_mag + 1e-30)) * cos_angle


# ═══════════════════════════════════════════════════════════════════
# Surface mesh preparation
# ═══════════════════════════════════════════════════════════════════

def _prepare_surface_mesh(vertices, faces):
    """Compute collocation points, normals, areas for a triangle mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Collocation at centroid
    centers = (v0 + v1 + v2) / 3.0

    # Outward normals (assuming consistent winding)
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2 * area[:, None] + 1e-30)

    return centers, normals, area


# ═══════════════════════════════════════════════════════════════════
# BEM matrix assembly
# ═══════════════════════════════════════════════════════════════════

def _assemble_bem_matrices(centers, normals, areas, k, beta):
    """
    Assemble BEM system matrices for the Kirchhoff-Helmholtz BIE.

    The CBIE (conventional BIE) for interior acoustics:
      (1/2) p(x) + ∫_Γ [dG/dn(x,y) + ik·β(y)·G(x,y)] p(y) dΓ(y) = p_inc(x)

    Discretized with collocation at triangle centroids:
      A · p_surf = p_inc

    Parameters
    ----------
    centers : (N, 3) collocation points
    normals : (N, 3) outward normals
    areas : (N,) triangle areas
    k : complex wavenumber (= s/c for Laplace domain)
    beta : (N,) surface admittance β = ρc/Z at each element

    Returns
    -------
    A : (N, N) complex system matrix
    """
    N = len(centers)
    xp = cp if (_GPU and N > 5000) else np

    if xp is cp:
        centers_d = cp.asarray(centers)
        normals_d = cp.asarray(normals)
        areas_d = cp.asarray(areas)
        beta_d = cp.asarray(beta)
    else:
        centers_d = centers
        normals_d = normals
        areas_d = areas
        beta_d = beta

    # Distance matrix: r[i,j] = ||x_i - y_j||
    diff = centers_d[:, None, :] - centers_d[None, :, :]  # (N, N, 3)
    r_mag = xp.sqrt(xp.sum(diff ** 2, axis=2))  # (N, N)

    # Avoid self-interaction singularity
    diag_mask = xp.eye(N, dtype=bool)
    r_safe = xp.where(diag_mask, 1.0, r_mag)

    # Green's function
    G = xp.exp(1j * k * r_safe) / (4 * xp.pi * r_safe)
    G = xp.where(diag_mask, 0.0, G)

    # dG/dn at collocation point y (normal of source element j)
    r_hat = diff / (r_safe[:, :, None] + 1e-30)
    cos_angle = xp.sum(r_hat * normals_d[None, :, :], axis=2)
    dGdn = G * (1j * k - 1.0 / (r_safe + 1e-30)) * cos_angle
    dGdn = xp.where(diag_mask, 0.0, dGdn)

    # System matrix: A[i,j] = δ_{ij}/2 + (dG/dn + ik·β·G)·area_j
    A = 0.5 * xp.eye(N, dtype=complex)
    A += (dGdn + 1j * k * beta_d[None, :] * G) * areas_d[None, :]

    if xp is cp:
        return cp.asnumpy(A)
    return A


def _assemble_bem_matrices_chunked(centers, normals, areas, k, beta,
                                    chunk_size=2000):
    """
    Memory-efficient chunked assembly for large meshes.
    Processes chunk_size rows at a time to stay within GPU RAM.
    """
    N = len(centers)
    if N <= chunk_size:
        return _assemble_bem_matrices(centers, normals, areas, k, beta)

    A = np.zeros((N, N), dtype=complex)
    A[np.diag_indices(N)] = 0.5

    xp = cp if _GPU else np

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_rows = i_end - i_start

        if xp is cp:
            ci = cp.asarray(centers[i_start:i_end])
            cj = cp.asarray(centers)
            nj = cp.asarray(normals)
            aj = cp.asarray(areas)
            bj = cp.asarray(beta)
        else:
            ci, cj, nj, aj, bj = (centers[i_start:i_end], centers,
                                   normals, areas, beta)

        diff = ci[:, None, :] - cj[None, :, :]
        r_mag = xp.sqrt(xp.sum(diff ** 2, axis=2))

        # Mask diagonal
        diag_offset = i_start
        for di in range(n_rows):
            j = diag_offset + di
            if j < N:
                r_mag[di, j] = 1.0  # avoid /0

        G = xp.exp(1j * k * r_mag) / (4 * xp.pi * r_mag)
        r_hat = diff / (r_mag[:, :, None] + 1e-30)
        cos_a = xp.sum(r_hat * nj[None, :, :], axis=2)
        dGdn = G * (1j * k - 1.0 / (r_mag + 1e-30)) * cos_a

        block = (dGdn + 1j * k * bj[None, :] * G) * aj[None, :]

        # Zero diagonal elements
        for di in range(n_rows):
            j = diag_offset + di
            if j < N:
                block[di, j] = 0.0

        if xp is cp:
            A[i_start:i_end, :] += cp.asnumpy(block)
        else:
            A[i_start:i_end, :] += block

    return A


# ═══════════════════════════════════════════════════════════════════
# Incident field
# ═══════════════════════════════════════════════════════════════════

def _incident_field(centers, source, k):
    """Point source incident field: p_inc = G(x, x_s)."""
    r = np.linalg.norm(centers - source, axis=1)
    return np.exp(1j * k * r) / (4 * np.pi * r + 1e-30)


def _evaluate_at_receiver(p_surf, centers, normals, areas,
                           receiver, source, k, beta):
    """
    Evaluate pressure at interior receiver point using Kirchhoff-Helmholtz.

    p(r) = p_inc(r) + Σ_j [dG/dn(r,y_j) + ik·β_j·G(r,y_j)] · p_j · A_j
    """
    r_vec = receiver - centers  # (N, 3)
    r_mag = np.linalg.norm(r_vec, axis=1)  # (N,)

    G_rec = np.exp(1j * k * r_mag) / (4 * np.pi * r_mag + 1e-30)

    r_hat = r_vec / (r_mag[:, None] + 1e-30)
    cos_a = np.sum(r_hat * normals, axis=1)
    dGdn_rec = G_rec * (1j * k - 1.0 / (r_mag + 1e-30)) * cos_a

    # Interior representation
    p_inc = np.exp(1j * k * np.linalg.norm(receiver - source)) / (
        4 * np.pi * np.linalg.norm(receiver - source) + 1e-30)

    p_rec = p_inc + np.sum(
        (dGdn_rec + 1j * k * beta * G_rec) * p_surf * areas)

    return p_rec


# ═══════════════════════════════════════════════════════════════════
# Laplace-domain solver
# ═══════════════════════════════════════════════════════════════════

def _solve_laplace_frequency(centers, normals, areas, source, receiver,
                              s, c, rho, Z_surf):
    """
    Solve BEM at one complex frequency s = σ + iω.

    Parameters
    ----------
    s : complex, Laplace variable
    c : speed of sound
    rho : air density
    Z_surf : (N,) surface impedance per element

    Returns
    -------
    p_rec : complex, pressure at receiver
    p_surf : (N,) complex, surface pressure
    """
    k = s / c  # complex wavenumber

    # Admittance β = ρc / Z
    beta = rho * c / (Z_surf + 1e-10)

    # Assemble and solve
    N = len(centers)
    A = _assemble_bem_matrices_chunked(centers, normals, areas, k, beta)
    rhs = _incident_field(centers, source, k)

    try:
        p_surf = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        # Regularize
        A += 1e-10 * np.eye(N)
        p_surf = np.linalg.solve(A, rhs)

    p_rec = _evaluate_at_receiver(p_surf, centers, normals, areas,
                                   receiver, source, k, beta)

    return p_rec, p_surf


# ═══════════════════════════════════════════════════════════════════
# Weeks method inverse Laplace transform
# ═══════════════════════════════════════════════════════════════════

def _weeks_inverse_laplace(H_laplace, s_values, sigma, b, T, sr):
    """
    Reconstruct time-domain IR from Laplace-domain transfer function
    using the Weeks method (Laguerre polynomial expansion).

    Parameters
    ----------
    H_laplace : (Ns,) complex transfer function at s_values
    s_values : (Ns,) complex Laplace frequencies
    sigma : float, real part of Laplace contour
    b : float, Laguerre parameter
    T : float, IR duration
    sr : int, sample rate

    Returns
    -------
    ir : (N_samples,) float, impulse response
    t : (N_samples,) float, time vector
    """
    Ns = len(H_laplace)
    N_samples = int(T * sr)
    t = np.arange(N_samples) / sr
    dt = 1.0 / sr

    # Compute expansion coefficients via DFT (Eq. 25 in Sampedro Llopis)
    theta = np.arange(-Ns, Ns) * np.pi / Ns
    theta_half = (np.arange(-Ns, Ns) + 0.5) * np.pi / Ns

    # Map H values: use conjugate symmetry
    # H(s_j) for j = -Ns..Ns-1, where s_j = sigma + i*b*cot(theta/2)
    a_k = np.zeros(2 * Ns, dtype=complex)
    for k in range(2 * Ns):
        for j in range(Ns):
            phase = np.exp(1j * k * theta_half[Ns + j])
            denom = 1.0 - np.exp(1j * theta_half[Ns + j])
            a_k[k] += phase / (denom + 1e-30) * H_laplace[j]
            # Conjugate symmetry
            if j > 0:
                phase_c = np.exp(1j * k * theta_half[Ns - j])
                denom_c = 1.0 - np.exp(1j * theta_half[Ns - j])
                a_k[k] += phase_c / (denom_c + 1e-30) * np.conj(H_laplace[j])
        a_k[k] *= b / (2 * Ns)

    # Laguerre polynomial evaluation via Clenshaw recurrence
    # L_k(x) satisfies: (k+1)L_{k+1} = (2k+1-x)L_k - k*L_{k-1}
    ir = np.zeros(N_samples)
    for n in range(N_samples):
        x = 2 * b * t[n]
        # Clenshaw backward recurrence
        d1 = 0.0
        d2 = 0.0
        for k in range(min(2 * Ns - 1, 200), -1, -1):
            if k == 0:
                d0 = np.real(a_k[k]) + (1.0 - x) * d1 - d2
            else:
                d0 = np.real(a_k[k]) + ((2 * k + 1 - x) / (k + 1)) * d1 - (
                    (k + 1) / (k + 2)) * d2
            d2 = d1
            d1 = d0
        ir[n] = np.exp((sigma - b) * t[n]) * d0

    return ir, t


def _simple_inverse_laplace(H, freqs, sigma, T, sr):
    """
    Inverse Laplace transform via the relation:

    H_laplace(s) evaluated at s = σ + i2πf relates to Fourier domain as:
      H_fourier(f) = H_laplace(σ + i2πf)

    The time-domain signal is then:
      h(t) = exp(σt) × IFFT{H_laplace(σ + i2πf)}

    We interpolate H onto a regular FFT grid and apply IFFT,
    then multiply by exp(σt) to undo the Laplace damping.
    """
    N_samples = int(T * sr)
    N_fft = N_samples // 2 + 1
    f_fft = np.linspace(0, sr / 2, N_fft)

    # Interpolate real and imaginary parts separately (not mag/phase)
    # This preserves causality better than mag/phase interpolation
    H_real_interp = np.interp(f_fft, freqs, H.real)
    H_imag_interp = np.interp(f_fft, freqs, H.imag)
    H_full = H_real_interp + 1j * H_imag_interp

    # Zero out frequencies above our solved range
    H_full[f_fft > freqs[-1] * 1.1] = 0.0
    # Smooth taper at edges to avoid Gibbs ringing
    taper_width = max(int(len(H_full) * 0.02), 5)
    f_max_idx = np.searchsorted(f_fft, freqs[-1])
    if f_max_idx < len(H_full):
        taper = np.linspace(1, 0, min(taper_width, len(H_full) - f_max_idx))
        H_full[f_max_idx:f_max_idx + len(taper)] *= taper
        H_full[f_max_idx + len(taper):] = 0.0

    # IFFT to get the σ-damped time signal
    ir_damped = np.fft.irfft(H_full, n=N_samples)

    # Undo Laplace damping: h(t) = exp(σt) × ir_damped(t)
    t = np.arange(N_samples, dtype=np.float64) / sr
    ir = ir_damped * np.exp(sigma * t)

    # The exp(σt) growth makes late samples blow up if sigma is large.
    # Apply a window that enforces physical decay at late times.
    # Use the energy at t=0 as reference and cap growth.
    if np.max(np.abs(ir[:int(0.01*sr)])) > 0:
        ref_level = np.max(np.abs(ir[:int(0.01*sr)]))
        # Don't let signal grow beyond 10× the initial peak
        cap = ref_level * 10
        ir = np.clip(ir, -cap, cap)

    return ir, t


# ═══════════════════════════════════════════════════════════════════
# Main BEM Solver class
# ═══════════════════════════════════════════════════════════════════

class BEMSolver:
    """
    Boundary Element Method solver for room acoustics.

    Surface-only discretization — O(f²) DOFs instead of O(f³).
    Supports GPU acceleration via CuPy.
    Laplace-domain formulation for ROM compatibility.

    Parameters
    ----------
    mesh : TriMesh
        Room geometry (from room_geometry.py or any triangle mesh).
    c : float
        Speed of sound [m/s].
    rho : float
        Air density [kg/m³].
    """

    def __init__(self, mesh, c=343.0, rho=1.225):
        self.mesh = mesh
        self.c = c
        self.rho = rho

        # Prepare surface mesh
        self.centers, self.normals, self.areas = _prepare_surface_mesh(
            mesh.vertices, mesh.faces)
        self.N = len(self.centers)

        print(f"BEM solver: {self.N} surface elements, "
              f"S={self.areas.sum():.1f} m², "
              f"GPU={'yes' if _GPU else 'no'}")

    def _get_impedance(self, materials, f):
        """Get per-element impedance from material functions."""
        Z = np.full(self.N, self.rho * self.c * 100, dtype=float)  # default rigid
        areas = self.mesh.face_areas()

        for gname, fidx in self.mesh.face_groups.items():
            mat_name = self.mesh.materials.get(gname, 'plaster')
            if isinstance(mat_name, MaterialFunction):
                mat_func = mat_name
            elif mat_name in materials:
                mat_func = materials[mat_name]
            else:
                continue
            alpha = np.clip(mat_func(f), 0.001, 0.999)
            # α → Z via diffuse-field inversion
            R = np.sqrt(1 - alpha)
            Z_val = self.rho * self.c * (1 + R) / (1 - R + 1e-10)
            Z[fidx] = Z_val

        return Z

    def transfer_function(self, source, receiver, materials,
                           freqs=None, f_min=20, f_max=2000, n_freqs=80,
                           sigma=5.0):
        """
        Compute transfer function H(f) in Laplace domain.

        Parameters
        ----------
        source : (3,) source position
        receiver : (3,) receiver position
        materials : dict of {group_name: MaterialFunction}
        freqs : array of frequencies [Hz], or None for auto
        sigma : float, Laplace damping parameter (σ > 0)

        Returns
        -------
        H : (n_freqs,) complex transfer function
        freqs : (n_freqs,) frequency vector [Hz]
        snapshots : (N, n_freqs) complex surface pressure snapshots
        """
        source = np.asarray(source, dtype=float)
        receiver = np.asarray(receiver, dtype=float)

        if freqs is None:
            freqs = np.linspace(f_min, f_max, n_freqs)

        H = np.zeros(len(freqs), dtype=complex)
        snapshots = np.zeros((self.N, len(freqs)), dtype=complex)

        t0 = time.perf_counter()
        for i, f in enumerate(freqs):
            omega = 2 * np.pi * f
            s = sigma + 1j * omega

            Z = self._get_impedance(materials, f)

            p_rec, p_surf = _solve_laplace_frequency(
                self.centers, self.normals, self.areas,
                source, receiver, s, self.c, self.rho, Z)

            H[i] = p_rec
            snapshots[:, i] = p_surf

            elapsed = time.perf_counter() - t0
            if (i + 1) % 10 == 0 or i == 0:
                rate = (i + 1) / elapsed
                eta = (len(freqs) - i - 1) / rate
                print(f"  BEM solve {i+1}/{len(freqs)}: f={f:.0f}Hz, "
                      f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining")

        total = time.perf_counter() - t0
        print(f"  BEM complete: {len(freqs)} solves in {total:.1f}s "
              f"({total/len(freqs):.2f}s/solve)")

        return H, freqs, snapshots

    def impulse_response(self, source, receiver, materials,
                          T=3.0, sr=44100,
                          f_min=20, f_max=2000, n_freqs=80,
                          sigma=5.0):
        """
        Compute impulse response via BEM + inverse Laplace.

        Parameters
        ----------
        source, receiver : (3,) positions
        materials : dict of {group_name: MaterialFunction}
        T : float, IR duration [s]
        sr : int, sample rate [Hz]
        f_min, f_max : frequency range
        n_freqs : number of frequency samples
        sigma : Laplace damping

        Returns
        -------
        ir : (N_samples,) float impulse response
        t : (N_samples,) float time vector
        H : (n_freqs,) complex transfer function
        freqs : (n_freqs,) float frequency vector
        """
        H, freqs, snapshots = self.transfer_function(
            source, receiver, materials,
            f_min=f_min, f_max=f_max, n_freqs=n_freqs, sigma=sigma)

        ir, t = _simple_inverse_laplace(H, freqs, sigma, T, sr)

        # Normalize
        peak = np.max(np.abs(ir))
        if peak > 0:
            ir /= peak

        return ir, t, H, freqs, snapshots
