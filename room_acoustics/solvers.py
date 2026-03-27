"""
FOM and ROM solvers for 2-D and 3-D room acoustics.

Formulations
------------
  p-v   : pressure–velocity  (Linearized Euler Equations)
  p-Φ   : pressure–potential (Hamiltonian / structure-preserving)
  mod-pΦ: modified p-Φ with boundary-energy enrichment (stable ROM)

Boundary conditions
-------------------
  PR  : perfectly reflecting (rigid walls)
  FI  : frequency-independent impedance
  LR  : locally reacting, frequency-dependent (ADE method)

Time integration: explicit RK4.
ROM: POD (p-v) and PSD cotangent-lift (p-Φ).
"""

import numpy as np
from scipy import sparse
from scipy.linalg import svd as cpu_svd, schur

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    GPU = True
except ImportError:
    GPU = False

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_AIR   = 343.0     # speed of sound  [m/s]
RHO_AIR = 1.2       # density of air  [kg/m³]


def _fi_coefficient(bc_params, rc2, M_inv, B_diag):
    """Compute the FI boundary damping coefficient vector.

    Supports both uniform Z and per-node Z_per_node.
    """
    if 'Z_per_node' in bc_params:
        Z_vec = np.asarray(bc_params['Z_per_node'], dtype=float)
        return -rc2 * M_inv * B_diag / Z_vec
    else:
        Z = bc_params['Z']
        return -rc2 / Z * M_inv * B_diag

# ---------------------------------------------------------------------------
# Miki model  (for LR boundaries)
# ---------------------------------------------------------------------------

def miki_impedance(f, sigma_mat, d_mat):
    """
    Surface impedance of porous material on rigid backing (Miki 1990).

    Parameters
    ----------
    f         : array of frequencies [Hz]  (>0)
    sigma_mat : flow resistivity  [N·s/m⁴]
    d_mat     : material thickness [m]

    Returns
    -------
    Zs : complex array — surface impedance
    """
    f = np.asarray(f, dtype=float)
    X = f / sigma_mat
    Zc = RHO_AIR * C_AIR * (1.0 + 0.0699 * X**(-0.632)
                             - 1j * 0.1071 * X**(-0.618))
    kc = (2.0 * np.pi * f / C_AIR) * (1.0 + 0.1093 * X**(-0.618)
                                        - 1j * 0.1597 * X**(-0.683))
    Zs = -1j * Zc / np.tan(kc * d_mat)
    return Zs


def fit_admittance_poles(f, Ys, n_poles=4, n_iter=8):
    """
    Vector fitting of admittance Y_s(ω) with complex conjugate pole pairs.
    Implements the Gustavsen–Semlyen (1999) algorithm (vectfit3-style).

    Initial poles are placed as complex conjugate pairs for better capture
    of resonant boundary behaviour.  After iteration the poles are relocated
    and residues recomputed.  Complex poles always appear in conjugate pairs
    to keep the time-domain ADE real-valued.

    Returns (Y_inf, A_k, lam_k) where lam_k are *positive* decay rates:
        Y_s(jω) ≈ Y_inf + Σ_k A_k / (lam_k + jω)       [real poles]
                         + Σ_k (C_k/(p_k+jω) + C_k*/(p_k*+jω))  [complex pairs]

    For the ADE solver only real poles are currently used, so complex poles
    are split into equivalent real-pole pairs via partial-fraction expansion.
    """
    omega = 2.0 * np.pi * f
    s = 1j * omega
    Nf = len(f)

    # --- initial poles: complex conjugate pairs spread across bandwidth ---
    n_real = n_poles % 2
    n_cpairs = n_poles // 2
    poles = []
    freqs = np.logspace(np.log10(omega[1]), np.log10(omega[-1]), n_cpairs + n_real)
    idx = 0
    for _ in range(n_cpairs):
        alpha = freqs[idx] * 0.05          # small real part (damping)
        beta  = freqs[idx]                  # imaginary part (frequency)
        poles.append(-alpha + 1j * beta)
        poles.append(-alpha - 1j * beta)
        idx += 1
    for _ in range(n_real):
        poles.append(-freqs[idx])
        idx += 1
    poles = np.array(poles, dtype=complex)
    Np = len(poles)

    # --- iterative pole relocation ----------------------------------------
    for _it in range(n_iter):
        # Build real-valued linear system
        # f(s) * sigma(s) = numerator(s)
        # where sigma(s) = 1 + Σ c̃_k/(s-p_k),  numerator = d + Σ c_k/(s-p_k)
        A_mat = np.zeros((2 * Nf, 2 * Np + 1))
        b_vec = np.zeros(2 * Nf)

        for i in range(Nf):
            phi = 1.0 / (s[i] - poles)          # (Np,) complex
            row = np.concatenate([phi, [1.0 + 0j], -Ys[i] * phi])
            A_mat[2*i,   :] = row.real
            A_mat[2*i+1, :] = row.imag
            b_vec[2*i]   = Ys[i].real
            b_vec[2*i+1] = Ys[i].imag

        x, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        c_den = x[Np + 1:]                      # denominator residues

        # New poles = eigenvalues of  diag(poles) - ones @ c_den^T
        H = np.diag(poles) - np.ones((Np, 1)) @ c_den[None, :]
        new_poles = np.linalg.eigvals(H)

        # Force stability: flip unstable poles
        new_poles = np.where(np.real(new_poles) > 0,
                             -np.real(new_poles) + 1j * np.imag(new_poles),
                             new_poles)

        # Enforce conjugate pairing
        sorted_poles = []
        used = np.zeros(Np, dtype=bool)
        for k in range(Np):
            if used[k]:
                continue
            p = new_poles[k]
            if abs(p.imag) < 1e-10 * abs(p):
                sorted_poles.append(p.real + 0j)
                used[k] = True
            else:
                # find conjugate partner
                dists = np.abs(new_poles - p.conj())
                dists[used] = np.inf
                dists[k] = np.inf
                partner = np.argmin(dists)
                pm = 0.5 * (p + new_poles[partner].conj())
                sorted_poles.append(pm)
                sorted_poles.append(pm.conj())
                used[k] = True
                used[partner] = True
        poles = np.array(sorted_poles, dtype=complex)

    # --- final residue computation with fixed poles -----------------------
    A2 = np.zeros((2 * Nf, Np + 1))
    b2 = np.zeros(2 * Nf)
    for i in range(Nf):
        phi = 1.0 / (s[i] - poles)
        A2[2*i,   :Np] = phi.real
        A2[2*i+1, :Np] = phi.imag
        A2[2*i,   Np]  = 1.0
        b2[2*i]   = Ys[i].real
        b2[2*i+1] = Ys[i].imag
    x2, *_ = np.linalg.lstsq(A2, b2, rcond=None)
    residues = x2[:Np]
    Y_inf = x2[Np]

    # --- convert to ADE-compatible real-pole format -----------------------
    # Each complex conjugate pair  C/(s-p) + C*/(s-p*)  is equivalent to
    # two coupled real ODEs (the ψ^(1), ψ^(2) accumulators in the paper).
    # For simplicity, approximate each pair as two real poles via
    # partial-fraction splitting that preserves the low-frequency response.
    real_lam = []
    real_A   = []
    for k in range(Np):
        p = poles[k]
        r = residues[k]
        if abs(p.imag) < 1e-10 * abs(p):
            real_lam.append(-p.real)
            real_A.append(r.real)
        else:
            if p.imag > 0:      # only process each pair once
                alpha = -p.real
                beta  = abs(p.imag)
                # 2 Re[C/(s-p)] = 2(Re(C)(s+α) + Im(C)β) / ((s+α)²+β²)
                # approximate as two real poles at α ± δ
                delta = beta * 0.5
                lam1 = alpha + delta
                lam2 = alpha - delta
                if lam2 < 1e-6:
                    lam2 = 1e-6
                A1 = r.real + r.imag * beta / (2 * delta) if delta > 0 else r.real
                A2_val = r.real - r.imag * beta / (2 * delta) if delta > 0 else r.real
                real_lam.extend([lam1, lam2])
                real_A.extend([A1, A2_val])

    return float(Y_inf), np.array(real_A), np.array(real_lam)


# ---------------------------------------------------------------------------
# Initial condition (Gaussian pulse)
# ---------------------------------------------------------------------------

def gaussian_pulse(mesh, x0, y0, sigma):
    """Gaussian pressure pulse centred at (x0, y0)."""
    return np.exp(-((mesh.x - x0)**2 + (mesh.y - y0)**2) / sigma**2)


# ---------------------------------------------------------------------------
# FOM: p-Φ  formulation  (Hamiltonian, structure-preserving)
# ---------------------------------------------------------------------------

def fom_pphi(mesh, ops, bc_type, bc_params, x0, y0, sigma,
             dt, T, rec_idx=None, store_snapshots=False,
             store_boundary_pressure=False, snap_stride=1):
    """
    Time-domain FOM using p-Φ formulation with RK4.

    bc_type   : 'PR', 'FI', or 'LR'
    bc_params : dict — {'Z': ...} for FI,
                       {'sigma_mat':…, 'd_mat':…} for LR
    rec_idx   : node index for impulse-response recording
    snap_stride : store every N-th snapshot (saves memory for long runs)
    """
    rho, c = RHO_AIR, C_AIR
    N = mesh.N_dof
    Nt = int(round(T / dt))

    # unpack operators  (CPU arrays)
    M_inv = ops['M_inv']
    S     = ops['S']
    B_tot = ops['B_total']

    # precompute: K1 Φ = ρc² M⁻¹ S Φ ,  K2 p = -(1/ρ) p
    rc2 = rho * c**2

    # --- boundary setup ---------------------------------------------------
    bnd_nodes = mesh.all_boundary_nodes()
    B_diag = np.array(B_tot.diagonal())

    # FI
    if bc_type == 'FI':
        Z = bc_params['Z']
        # coefficient for boundary term in p equation
        # p_t += -ρc² M⁻¹ B (p/Z)  →  coeff = -ρc²/Z * M⁻¹ * B_diag
        fi_coeff = -rc2 / Z * M_inv * B_diag

    # LR  (ADE)
    ade_state = None
    if bc_type == 'LR':
        freq = np.linspace(10, 2000, 500)
        Zs = miki_impedance(freq, bc_params['sigma_mat'], bc_params['d_mat'])
        Ys = 1.0 / Zs
        Y_inf, A_k, lam_k = fit_admittance_poles(freq, Ys, n_poles=4)
        n_acc = len(lam_k)
        # accumulators: phi_k[k, boundary_node_local]
        Nb = len(bnd_nodes)
        ade_phi = np.zeros((n_acc, Nb))

    # --- state vectors ----------------------------------------------------
    p   = gaussian_pulse(mesh, x0, y0, sigma)
    Phi = np.zeros(N)

    # storage
    ir = np.zeros(Nt + 1) if rec_idx is not None else None
    if rec_idx is not None:
        ir[0] = p[rec_idx]

    snaps_p   = [p.copy()] if store_snapshots else None
    snaps_Phi = [Phi.copy()] if store_snapshots else None
    snaps_pb  = []
    energy    = np.zeros(Nt + 1)
    energy[0] = _energy_pphi(p, Phi, ops, mesh)

    # --- RHS function -----------------------------------------------------
    def rhs(p_loc, Phi_loc, ade_loc=None):
        """Return (dp, dPhi, d_ade) for one RK stage."""
        # p_t = ρc² M⁻¹ S Φ  +  boundary
        dp = rc2 * M_inv * S.dot(Phi_loc)
        dPhi = -(1.0 / rho) * p_loc

        if bc_type == 'FI':
            dp += fi_coeff * p_loc
        elif bc_type == 'LR':
            p_bnd = p_loc[bnd_nodes]
            # v_n = Y_inf * p + Σ A_k * phi_k
            v_n = Y_inf * p_bnd
            for k in range(n_acc):
                v_n += A_k[k] * ade_loc[k]
            # boundary term:  -ρc² M⁻¹ B v_n
            bnd_term = np.zeros(N)
            bnd_term[bnd_nodes] = B_diag[bnd_nodes] * v_n
            dp += -rc2 * M_inv * bnd_term
            # ADE:  dφ_k/dt = -λ_k φ_k + p_bnd
            d_ade = np.zeros_like(ade_loc)
            for k in range(n_acc):
                d_ade[k] = -lam_k[k] * ade_loc[k] + p_bnd
            return dp, dPhi, d_ade

        return dp, dPhi, None

    # --- RK4 time loop ----------------------------------------------------
    for n in range(Nt):
        ade = ade_phi if bc_type == 'LR' else None

        k1p, k1P, k1a = rhs(p, Phi, ade)
        a1 = ade + 0.5 * dt * k1a if k1a is not None else None
        k2p, k2P, k2a = rhs(p + 0.5*dt*k1p, Phi + 0.5*dt*k1P, a1)
        a2 = ade + 0.5 * dt * k2a if k2a is not None else None
        k3p, k3P, k3a = rhs(p + 0.5*dt*k2p, Phi + 0.5*dt*k2P, a2)
        a3 = ade + dt * k3a if k3a is not None else None
        k4p, k4P, k4a = rhs(p + dt*k3p, Phi + dt*k3P, a3)

        p   += dt / 6.0 * (k1p + 2*k2p + 2*k3p + k4p)
        Phi += dt / 6.0 * (k1P + 2*k2P + 2*k3P + k4P)
        if bc_type == 'LR':
            ade_phi += dt / 6.0 * (k1a + 2*k2a + 2*k3a + k4a)

        if rec_idx is not None:
            ir[n + 1] = p[rec_idx]
        if store_snapshots and (n + 1) % snap_stride == 0:
            snaps_p.append(p.copy())
            snaps_Phi.append(Phi.copy())
        if store_boundary_pressure and (n + 1) % snap_stride == 0:
            pb = np.zeros(N)
            pb[bnd_nodes] = p[bnd_nodes]
            snaps_pb.append(pb.copy())
        energy[n + 1] = _energy_pphi(p, Phi, ops, mesh)

    result = dict(ir=ir, energy=energy, p_final=p, Phi_final=Phi)
    if store_snapshots:
        result['snaps_p'] = np.array(snaps_p)
        result['snaps_Phi'] = np.array(snaps_Phi)
    if store_boundary_pressure and snaps_pb:
        result['snaps_pb'] = np.array(snaps_pb)
    return result


# ---------------------------------------------------------------------------
# FOM: p-v  formulation  (Linearized Euler)
# ---------------------------------------------------------------------------

def fom_pv(mesh, ops, bc_type, bc_params, x0, y0, sigma,
           dt, T, rec_idx=None, store_snapshots=False):
    """Time-domain FOM using p-v formulation with RK4."""
    rho, c = RHO_AIR, C_AIR
    N = mesh.N_dof
    Nt = int(round(T / dt))
    rc2 = rho * c**2

    M_inv = ops['M_inv']
    Sx    = ops['Sx']
    Sy    = ops['Sy']
    B_tot = ops['B_total']
    B_diag = np.array(B_tot.diagonal())

    # precompute sparse transposes
    SxT = Sx.T.tocsr()
    SyT = Sy.T.tocsr()

    # FI boundary
    if bc_type == 'FI':
        Z = bc_params['Z']
        fi_coeff = -rc2 / Z * M_inv * B_diag

    bnd_nodes = mesh.all_boundary_nodes()

    # LR (ADE)
    if bc_type == 'LR':
        freq = np.linspace(10, 2000, 500)
        Zs = miki_impedance(freq, bc_params['sigma_mat'], bc_params['d_mat'])
        Ys = 1.0 / Zs
        Y_inf, A_k, lam_k = fit_admittance_poles(freq, Ys, n_poles=4)
        n_acc = len(lam_k)
        Nb = len(bnd_nodes)
        ade_phi = np.zeros((n_acc, Nb))

    # state
    p_vec = gaussian_pulse(mesh, x0, y0, sigma)
    u_vec = np.zeros(N)
    v_vec = np.zeros(N)

    ir = np.zeros(Nt + 1) if rec_idx is not None else None
    if rec_idx is not None:
        ir[0] = p_vec[rec_idx]
    snaps_p = [p_vec.copy()] if store_snapshots else None
    snaps_u = [u_vec.copy()] if store_snapshots else None
    snaps_v = [v_vec.copy()] if store_snapshots else None
    energy = np.zeros(Nt + 1)
    energy[0] = _energy_pv(p_vec, u_vec, v_vec, ops, mesh)

    def rhs(p_l, u_l, v_l, ade_l=None):
        du = -(1.0 / rho) * M_inv * Sx.dot(p_l)
        dv = -(1.0 / rho) * M_inv * Sy.dot(p_l)
        dp = rc2 * M_inv * (SxT.dot(u_l) + SyT.dot(v_l))

        if bc_type == 'FI':
            dp += fi_coeff * p_l
        elif bc_type == 'LR':
            p_bnd = p_l[bnd_nodes]
            v_n = Y_inf * p_bnd
            for k in range(n_acc):
                v_n += A_k[k] * ade_l[k]
            bnd_t = np.zeros(N)
            bnd_t[bnd_nodes] = B_diag[bnd_nodes] * v_n
            dp += -rc2 * M_inv * bnd_t
            d_ade = np.zeros_like(ade_l)
            for k in range(n_acc):
                d_ade[k] = -lam_k[k] * ade_l[k] + p_bnd
            return du, dv, dp, d_ade

        return du, dv, dp, None

    for n in range(Nt):
        ade = ade_phi if bc_type == 'LR' else None

        k1u, k1v, k1p, k1a = rhs(p_vec, u_vec, v_vec, ade)
        a1 = ade + 0.5*dt*k1a if k1a is not None else None
        k2u, k2v, k2p, k2a = rhs(p_vec+0.5*dt*k1p, u_vec+0.5*dt*k1u,
                                   v_vec+0.5*dt*k1v, a1)
        a2 = ade + 0.5*dt*k2a if k2a is not None else None
        k3u, k3v, k3p, k3a = rhs(p_vec+0.5*dt*k2p, u_vec+0.5*dt*k2u,
                                   v_vec+0.5*dt*k2v, a2)
        a3 = ade + dt*k3a if k3a is not None else None
        k4u, k4v, k4p, k4a = rhs(p_vec+dt*k3p, u_vec+dt*k3u,
                                   v_vec+dt*k3v, a3)

        p_vec += dt/6*(k1p + 2*k2p + 2*k3p + k4p)
        u_vec += dt/6*(k1u + 2*k2u + 2*k3u + k4u)
        v_vec += dt/6*(k1v + 2*k2v + 2*k3v + k4v)
        if bc_type == 'LR':
            ade_phi += dt/6*(k1a + 2*k2a + 2*k3a + k4a)

        if rec_idx is not None:
            ir[n + 1] = p_vec[rec_idx]
        if store_snapshots:
            snaps_p.append(p_vec.copy())
            snaps_u.append(u_vec.copy())
            snaps_v.append(v_vec.copy())
        energy[n + 1] = _energy_pv(p_vec, u_vec, v_vec, ops, mesh)

    result = dict(ir=ir, energy=energy, p_final=p_vec)
    if store_snapshots:
        result['snaps_p'] = np.array(snaps_p)
        result['snaps_u'] = np.array(snaps_u)
        result['snaps_v'] = np.array(snaps_v)
    return result


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def _energy_pphi(p, Phi, ops, mesh):
    """Total acoustic energy (interior) for p-Φ formulation."""
    rho, c = RHO_AIR, C_AIR
    S = ops['S']
    M = ops['M_diag']
    # H = (ρ/2)|∇Φ|²  +  p²/(2ρc²)
    # Discrete:  (ρ/2) Φ^T S Φ  +  (1/(2ρc²)) p^T M p
    kinetic  = 0.5 * rho * Phi @ S.dot(Phi)
    pressure = 0.5 / (rho * c**2) * (p * M) @ p
    return kinetic + pressure


def _energy_pv(p, u, v, ops, mesh):
    """Total acoustic energy for p-v formulation."""
    rho, c = RHO_AIR, C_AIR
    M = ops['M_diag']
    kinetic  = 0.5 * rho * ((u * M) @ u + (v * M) @ v)
    pressure = 0.5 / (rho * c**2) * (p * M) @ p
    return kinetic + pressure


# ---------------------------------------------------------------------------
# Analytical solution (rigid rectangular room)
# ---------------------------------------------------------------------------

def analytical_rigid_rect(mesh, x0, y0, sigma, rx, ry, dt, T, n_modes=120):
    """
    Analytical impulse response for a rigid rectangular room.

    Uses modal expansion with Gaussian pulse IC.
    """
    Lx, Ly = mesh.Lx, mesh.Ly
    c = C_AIR
    Nt = int(round(T / dt))
    t = np.arange(Nt + 1) * dt

    ir = np.zeros(Nt + 1)
    for m in range(n_modes):
        for n in range(n_modes):
            if m == 0 and n == 0:
                eps = 1.0
            elif m == 0 or n == 0:
                eps = 2.0
            else:
                eps = 4.0

            omega_mn = c * np.pi * np.sqrt((m / Lx)**2 + (n / Ly)**2)

            # Coefficient from Gaussian IC  (approximate: integral extended to ∞)
            A_mn = (eps / (Lx * Ly)) * (sigma**2 * np.pi) * \
                   np.cos(m * np.pi * x0 / Lx) * np.cos(n * np.pi * y0 / Ly) * \
                   np.exp(-((m * np.pi * sigma / (2 * Lx))**2 +
                            (n * np.pi * sigma / (2 * Ly))**2))

            mode_val = np.cos(m * np.pi * rx / Lx) * np.cos(n * np.pi * ry / Ly)

            if omega_mn == 0:
                ir += A_mn * mode_val
            else:
                ir += A_mn * mode_val * np.cos(omega_mn * t)

    return t, ir


# ---------------------------------------------------------------------------
# ROM — Proper Orthogonal / Symplectic Decomposition
# ---------------------------------------------------------------------------

def _enrich_with_dc(U, N):
    """Ensure the constant (DC / zero-frequency) mode is in basis U (N, Nrb).

    If the uniform-pressure vector is not well-represented by U,
    prepend it and re-orthogonalise via QR.  Returns (U_new, added).
    """
    e0 = np.ones(N) / np.sqrt(N)
    proj = U.T @ e0
    residual = np.linalg.norm(e0 - U @ proj)
    if residual > 1e-6:
        U_aug = np.column_stack([e0, U])
        U_aug, _ = np.linalg.qr(U_aug, mode='reduced')
        return U_aug, True
    return U, False


def build_pod_basis(snapshots, eps_pod=1e-6):
    """
    POD basis from snapshot matrix.

    Parameters
    ----------
    snapshots : (Nt, N) array
    eps_pod   : energy tolerance for truncation

    Returns
    -------
    Phi : (N, Nrb) basis
    sigma : singular values
    Nrb : number of retained modes
    """
    S = snapshots.T          # (N, Nt)
    U, sigma, _ = cpu_svd(S, full_matrices=False)
    # determine Nrb
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    Nrb = int(np.searchsorted(energy, 1.0 - eps_pod) + 1)
    Nrb = max(Nrb, 1)
    return U[:, :Nrb], sigma, Nrb


def build_psd_basis(snaps_p, snaps_Phi, eps_pod=1e-6, enrich_dc=True):
    """
    PSD cotangent-lift basis for p-Φ formulation.

    Combined snapshot: [p(t0)…p(tN), Φ(t0)…Φ(tN)]  → single SVD.
    Symplectic basis Ψ_H used for both p and Φ.

    If *enrich_dc* is True, the constant-pressure (DC) mode is guaranteed
    to be in the basis — critical for absorbing-boundary ROMs where the
    spatial mean must be representable to decay correctly.
    """
    S_combined = np.hstack([snaps_p.T, snaps_Phi.T])   # (N, 2*Nt)
    U, sigma, _ = cpu_svd(S_combined, full_matrices=False)
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    Nrb = int(np.searchsorted(energy, 1.0 - eps_pod) + 1)
    Nrb = max(Nrb, 1)
    basis = U[:, :Nrb]
    if enrich_dc:
        basis, added = _enrich_with_dc(basis, basis.shape[0])
        if added:
            Nrb = basis.shape[1]
    return basis, sigma, Nrb


def build_modified_psd_basis(snaps_p, snaps_Phi, snaps_pb, eps_pod=1e-6,
                             enrich_dc=True):
    """
    Modified PSD basis enriched with boundary pressure (stable ROM).
    """
    S_combined = np.hstack([snaps_p.T, snaps_Phi.T, snaps_pb.T])
    U, sigma, _ = cpu_svd(S_combined, full_matrices=False)
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    Nrb = int(np.searchsorted(energy, 1.0 - eps_pod) + 1)
    Nrb = max(Nrb, 1)
    basis = U[:, :Nrb]
    if enrich_dc:
        basis, added = _enrich_with_dc(basis, basis.shape[0])
        if added:
            Nrb = basis.shape[1]
    return basis, sigma, Nrb


# ---------------------------------------------------------------------------
# ROM online solver  (p-Φ formulation)
# ---------------------------------------------------------------------------

def _stabilize_propagator(P, bc_type):
    """Schur-based eigenvalue stabilization for an RK4 propagator matrix.

    PR  — force all |λ| = 1  (energy conservation)
    FI  — clip  |λ| > 1 to 1 (dissipative, no growth)
    LR  — same as FI
    """
    T_s, Q = schur(P, output='complex')
    for i in range(T_s.shape[0]):
        mag = abs(T_s[i, i])
        if bc_type == 'PR':
            if mag > 1e-15:
                T_s[i, i] /= mag
        else:
            if mag > 1.0:
                T_s[i, i] /= mag
    return np.real(Q @ T_s @ Q.conj().T)


def rom_pphi(mesh, ops, Psi_H, bc_type, bc_params, x0, y0, sigma,
             dt, T, rec_idx=None, Nrb_override=None):
    """
    ROM time-domain solver using p-Φ with symplectic (PSD) basis.

    For PR and FI boundaries the RK4 propagator matrix is precomputed
    and eigenvalue-stabilised via Schur decomposition — each time step
    is then a single dense matvec (very fast, guaranteed stable).

    For LR boundaries (state-dependent ADE accumulators in full boundary
    space) explicit RK4 is retained.

    Psi_H : (N, Nrb) — symplectic basis (same for p and Φ)
    """
    rho, c = RHO_AIR, C_AIR
    rc2 = rho * c**2
    N = mesh.N_dof
    Nt = int(round(T / dt))

    Psi = Psi_H
    if Nrb_override is not None and Nrb_override < Psi.shape[1]:
        Psi = Psi[:, :Nrb_override]
    Nrb = Psi.shape[1]

    # reduced operators
    S_full = ops['S']
    B_diag = np.array(ops['B_total'].diagonal())
    M_inv = ops['M_inv']

    K1Psi = rc2 * (M_inv[:, None] * S_full.dot(Psi))   # (N, Nrb)
    K1_r  = Psi.T @ K1Psi                               # (Nrb, Nrb)
    K2    = -1.0 / rho                                   # scalar

    bnd_nodes = mesh.all_boundary_nodes()

    # initial conditions projected
    p0 = gaussian_pulse(mesh, x0, y0, sigma)
    a_p   = Psi.T @ p0
    a_Phi = np.zeros(Nrb)

    ir = np.zeros(Nt + 1) if rec_idx is not None else None
    if rec_idx is not None:
        ir[0] = Psi[rec_idx, :] @ a_p

    # ------------------------------------------------------------------
    # PR / FI: propagator matrix approach (fast + stable)
    # ------------------------------------------------------------------
    if bc_type in ('PR', 'FI'):
        A_r = np.zeros((2 * Nrb, 2 * Nrb))
        A_r[:Nrb, Nrb:] = K1_r
        A_r[Nrb:, :Nrb] = K2 * np.eye(Nrb)

        if bc_type == 'FI':
            fi_vec = _fi_coefficient(bc_params, rc2, M_inv, B_diag)
            K3_r = Psi.T @ (fi_vec[:, None] * Psi)
            A_r[:Nrb, :Nrb] += K3_r

        # RK4 propagator
        dtA = dt * A_r
        dtA2 = dtA @ dtA
        P = (np.eye(2 * Nrb) + dtA + dtA2 / 2.0
             + dtA2 @ dtA / 6.0 + dtA2 @ dtA2 / 24.0)

        P = _stabilize_propagator(P, bc_type)

        # observation vector
        obs = np.zeros(2 * Nrb)
        if rec_idx is not None:
            obs[:Nrb] = Psi[rec_idx, :]

        state = np.concatenate([a_p, a_Phi])
        for n in range(Nt):
            state = P @ state
            if rec_idx is not None:
                ir[n + 1] = obs @ state

        return dict(ir=ir)

    # ------------------------------------------------------------------
    # LR: explicit RK4 (ADE accumulators live in full boundary space)
    # ------------------------------------------------------------------
    freq = np.linspace(10, 2000, 500)
    Zs = miki_impedance(freq, bc_params['sigma_mat'], bc_params['d_mat'])
    Ys = 1.0 / Zs
    Y_inf, A_k, lam_k = fit_admittance_poles(freq, Ys, n_poles=4)
    n_acc = len(lam_k)
    Nb = len(bnd_nodes)
    ade_phi = np.zeros((n_acc, Nb))
    Psi_bnd = Psi[bnd_nodes, :]
    bnd_B = B_diag[bnd_nodes]

    if bc_type == 'FI':
        fi_vec = _fi_coefficient(bc_params, rc2, M_inv, B_diag)
        K3_r = Psi.T @ (fi_vec[:, None] * Psi)

    def rhs(ap, aPhi, ade_l=None):
        dap = K1_r @ aPhi
        daPhi = K2 * ap

        if bc_type == 'FI':
            dap += K3_r @ ap
        elif bc_type == 'LR':
            p_bnd = Psi_bnd @ ap
            v_n = Y_inf * p_bnd
            for k in range(n_acc):
                v_n += A_k[k] * ade_l[k]
            bnd_full = np.zeros(N)
            bnd_full[bnd_nodes] = bnd_B * v_n
            dap += Psi.T @ (-rc2 * M_inv * bnd_full)
            d_ade = np.zeros_like(ade_l)
            for k in range(n_acc):
                d_ade[k] = -lam_k[k] * ade_l[k] + p_bnd
            return dap, daPhi, d_ade
        return dap, daPhi, None

    for n in range(Nt):
        ade = ade_phi if bc_type == 'LR' else None

        k1p, k1P, k1a = rhs(a_p, a_Phi, ade)
        a1 = ade + 0.5*dt*k1a if k1a is not None else None
        k2p, k2P, k2a = rhs(a_p+0.5*dt*k1p, a_Phi+0.5*dt*k1P, a1)
        a2 = ade + 0.5*dt*k2a if k2a is not None else None
        k3p, k3P, k3a = rhs(a_p+0.5*dt*k2p, a_Phi+0.5*dt*k2P, a2)
        a3 = ade + dt*k3a if k3a is not None else None
        k4p, k4P, k4a = rhs(a_p+dt*k3p, a_Phi+dt*k3P, a3)

        a_p   += dt/6*(k1p + 2*k2p + 2*k3p + k4p)
        a_Phi += dt/6*(k1P + 2*k2P + 2*k3P + k4P)
        if bc_type == 'LR':
            ade_phi += dt/6*(k1a + 2*k2a + 2*k3a + k4a)

        if rec_idx is not None:
            ir[n + 1] = Psi[rec_idx, :] @ a_p

    return dict(ir=ir)


# ---------------------------------------------------------------------------
# ROM online solver  (p-v formulation, POD)
# ---------------------------------------------------------------------------

def rom_pv(mesh, ops, Psi_p, Psi_u, Psi_v, bc_type, bc_params,
           x0, y0, sigma, dt, T, rec_idx=None):
    """ROM using p-v formulation with separate POD bases."""
    rho, c = RHO_AIR, C_AIR
    rc2 = rho * c**2
    N = mesh.N_dof
    Nt = int(round(T / dt))
    M_inv = ops['M_inv']
    Sx = ops['Sx']; Sy = ops['Sy']
    SxT = Sx.T.tocsr(); SyT = Sy.T.tocsr()
    B_diag = np.array(ops['B_total'].diagonal())
    bnd_nodes = mesh.all_boundary_nodes()

    Nrb_p = Psi_p.shape[1]
    Nrb_u = Psi_u.shape[1]
    Nrb_v = Psi_v.shape[1]

    # reduced operators
    L1_r = Psi_u.T @ (-(1.0/rho) * (M_inv[:, None] * Sx.dot(Psi_p)))
    L2_r = Psi_v.T @ (-(1.0/rho) * (M_inv[:, None] * Sy.dot(Psi_p)))
    L3_r = Psi_p.T @ (rc2 * (M_inv[:, None] * SxT.dot(Psi_u)))
    L4_r = Psi_p.T @ (rc2 * (M_inv[:, None] * SyT.dot(Psi_v)))

    if bc_type == 'FI':
        fi_vec = _fi_coefficient(bc_params, rc2, M_inv, B_diag)
        L5_r = Psi_p.T @ (fi_vec[:, None] * Psi_p)

    # IC
    p0 = gaussian_pulse(mesh, x0, y0, sigma)
    ap = Psi_p.T @ p0
    au = np.zeros(Nrb_u)
    av = np.zeros(Nrb_v)

    ir = np.zeros(Nt + 1) if rec_idx is not None else None
    if rec_idx is not None:
        ir[0] = Psi_p[rec_idx, :] @ ap

    def rhs_r(ap_l, au_l, av_l):
        du = L1_r @ ap_l
        dv = L2_r @ ap_l
        dp = L3_r @ au_l + L4_r @ av_l
        if bc_type == 'FI':
            dp += L5_r @ ap_l
        return du, dv, dp

    for n in range(Nt):
        k1u, k1v, k1p = rhs_r(ap, au, av)
        k2u, k2v, k2p = rhs_r(ap+0.5*dt*k1p, au+0.5*dt*k1u, av+0.5*dt*k1v)
        k3u, k3v, k3p = rhs_r(ap+0.5*dt*k2p, au+0.5*dt*k2u, av+0.5*dt*k2v)
        k4u, k4v, k4p = rhs_r(ap+dt*k3p, au+dt*k3u, av+dt*k3v)

        ap += dt/6*(k1p + 2*k2p + 2*k3p + k4p)
        au += dt/6*(k1u + 2*k2u + 2*k3u + k4u)
        av += dt/6*(k1v + 2*k2v + 2*k3v + k4v)

        if rec_idx is not None:
            ir[n + 1] = Psi_p[rec_idx, :] @ ap

    return dict(ir=ir)


# ---------------------------------------------------------------------------
# Stability analysis (eigenvalues of reduced operators)
# ---------------------------------------------------------------------------

def eigenvalue_analysis(ops, Psi_H=None, formulation='pphi'):
    """
    Compute eigenvalues of assembled operator matrix.

    For p-Φ:  K = [[K1, 0], [0, K2*I]]  rearranged as block system
    """
    rho, c = RHO_AIR, C_AIR
    rc2 = rho * c**2
    M_inv = ops['M_inv']
    S = ops['S']
    N = S.shape[0]

    if Psi_H is not None:
        Nrb = Psi_H.shape[1]
        K1Psi = rc2 * (M_inv[:, None] * S.dot(Psi_H))
        K1_r = Psi_H.T @ K1Psi
        K2 = -1.0 / rho

        # assembled operator:  [0, K1_r; K2*I, 0]
        A_rom = np.zeros((2*Nrb, 2*Nrb))
        A_rom[:Nrb, Nrb:] = K1_r
        A_rom[Nrb:, :Nrb] = K2 * np.eye(Nrb)
        return np.linalg.eigvals(A_rom)
    else:
        # FOM eigenvalues (only feasible for small systems)
        if N > 3000:
            print(f"  [warn] FOM eigenvalue analysis skipped (N={N} too large)")
            return None
        K1 = rc2 * (M_inv[:, None] * S.toarray())
        K2 = -1.0 / rho
        A_fom = np.zeros((2*N, 2*N))
        A_fom[:N, N:] = K1
        A_fom[N:, :N] = K2 * np.eye(N)
        return np.linalg.eigvals(A_fom)


# ---------------------------------------------------------------------------
# GPU-accelerated 3-D FOM  (p-Φ formulation)
# ---------------------------------------------------------------------------

def gaussian_pulse_3d(mesh, x0, y0, z0, sigma):
    """Gaussian pressure pulse in 3D."""
    mesh._ensure_coords()
    return np.exp(-((mesh.x-x0)**2 + (mesh.y-y0)**2 + (mesh.z-z0)**2) / sigma**2)


def fom_pphi_3d_gpu(mesh, ops, bc_type, bc_params, x0, y0, z0, sigma,
                    dt, T, rec_idx=None, store_snapshots=False, snap_stride=1,
                    store_boundary_pressure=False):
    """
    GPU-accelerated 3-D FOM using p-Φ with RK4.

    Sparse matvec runs on GPU (CuPy); snapshots stay on CPU.
    Falls back to CPU if CuPy unavailable.
    """
    rho, c = RHO_AIR, C_AIR
    rc2 = rho * c**2
    N = mesh.N_dof
    Nt = int(round(T / dt))

    # Detect backend
    try:
        import cupy as _cp
        import cupyx.scipy.sparse as _cps
        use_gpu = True
        xp = _cp
        print(f"    [GPU] Using CuPy on {_cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except Exception:
        use_gpu = False
        xp = np

    # Move operators to device
    if use_gpu:
        d_M_inv = xp.asarray(ops['M_inv'])
        d_S     = _cps.csr_matrix(ops['S'])
        d_B_diag = xp.asarray(np.array(ops['B_total'].diagonal()))
    else:
        d_M_inv = ops['M_inv']
        d_S     = ops['S']
        d_B_diag = np.array(ops['B_total'].diagonal())

    # Boundary setup
    bnd_nodes = mesh.all_boundary_nodes()
    fi_coeff_d = None
    if bc_type == 'FI':
        B_diag = np.array(ops['B_total'].diagonal())
        fi_coeff = _fi_coefficient(bc_params, rc2, ops['M_inv'], B_diag)
        fi_coeff_d = xp.asarray(fi_coeff) if use_gpu else fi_coeff

    # Initial state
    p0_cpu = gaussian_pulse_3d(mesh, x0, y0, z0, sigma)
    p   = xp.asarray(p0_cpu) if use_gpu else p0_cpu.copy()
    Phi = xp.zeros(N)

    ir = np.zeros(Nt + 1) if rec_idx is not None else None
    if rec_idx is not None:
        ir[0] = float(p[rec_idx])

    snaps_p = [p0_cpu.copy()] if store_snapshots else None
    snaps_Phi = [np.zeros(N)] if store_snapshots else None
    snaps_pb = []
    bnd_nodes_cpu = mesh.all_boundary_nodes()

    def rhs(p_l, Phi_l):
        dp = rc2 * d_M_inv * d_S.dot(Phi_l)
        dPhi = -(1.0 / rho) * p_l
        if bc_type == 'FI':
            dp += fi_coeff_d * p_l
        return dp, dPhi

    for n in range(Nt):
        k1p, k1P = rhs(p, Phi)
        k2p, k2P = rhs(p + 0.5*dt*k1p, Phi + 0.5*dt*k1P)
        k3p, k3P = rhs(p + 0.5*dt*k2p, Phi + 0.5*dt*k2P)
        k4p, k4P = rhs(p + dt*k3p, Phi + dt*k3P)

        p   += dt/6 * (k1p + 2*k2p + 2*k3p + k4p)
        Phi += dt/6 * (k1P + 2*k2P + 2*k3P + k4P)

        if rec_idx is not None:
            ir[n+1] = float(p[rec_idx])
        if (store_snapshots or store_boundary_pressure) and (n+1) % snap_stride == 0:
            p_cpu = p.get() if use_gpu else p.copy()
            if store_snapshots:
                snaps_p.append(p_cpu)
                snaps_Phi.append(Phi.get() if use_gpu else Phi.copy())
            if store_boundary_pressure:
                pb = np.zeros(N)
                pb[bnd_nodes_cpu] = p_cpu[bnd_nodes_cpu]
                snaps_pb.append(pb)

    result = dict(ir=ir)
    if store_snapshots:
        result['snaps_p'] = np.array(snaps_p)
        result['snaps_Phi'] = np.array(snaps_Phi)
    if store_boundary_pressure and snaps_pb:
        result['snaps_pb'] = np.array(snaps_pb)
    return result


def rom_pphi_3d(mesh, ops, Psi_H, bc_type, bc_params,
                x0, y0, z0, sigma, dt, T, rec_idx=None, Nrb_override=None):
    """
    ROM for 3-D p-Φ formulation using the **RK4 propagator matrix**
    with eigenvalue stabilization.

    For a linear ODE  x' = A x,  the RK4 scheme reduces to  x_{n+1} = P x_n.
    After computing P, its eigenvalues are checked:
      - PR: spectral radius is forced to exactly 1.0 (energy conservation)
      - FI: eigenvalues with |λ|>1 are projected inside the unit circle
    This guarantees long-term stability while preserving the mode structure.
    """
    rho, c = RHO_AIR, C_AIR
    rc2 = rho * c**2
    N = mesh.N_dof
    Nt = int(round(T / dt))

    Psi = Psi_H
    if Nrb_override is not None and Nrb_override < Psi.shape[1]:
        Psi = Psi[:, :Nrb_override]
    Nrb = Psi.shape[1]

    M_inv = ops['M_inv']
    S = ops['S']
    B_diag = np.array(ops['B_total'].diagonal())

    # Build reduced system matrix A_r (2Nrb x 2Nrb)
    print(f"    ROM: building {Nrb}x{Nrb} reduced operators + propagator...",
          end='', flush=True)
    K1Psi = rc2 * (M_inv[:, None] * S.dot(Psi))
    K1_r = Psi.T @ K1Psi
    del K1Psi
    K2 = -1.0 / rho

    A_r = np.zeros((2 * Nrb, 2 * Nrb))
    A_r[:Nrb, Nrb:] = K1_r
    A_r[Nrb:, :Nrb] = K2 * np.eye(Nrb)

    if bc_type == 'FI':
        fi_vec = _fi_coefficient(bc_params, rc2, M_inv, B_diag)
        K3_r = Psi.T @ (fi_vec[:, None] * Psi)
        A_r[:Nrb, :Nrb] = K3_r

    # RK4 propagator
    dtA = dt * A_r
    dtA2 = dtA @ dtA
    P = np.eye(2*Nrb) + dtA + dtA2/2.0 + dtA2@dtA/6.0 + dtA2@dtA2/24.0
    del dtA, dtA2, A_r

    # --- Eigenvalue stabilization via Schur decomposition ---
    # Schur is numerically robust: P = Q T Q^H with unitary Q (cond=1).
    # Eigenvalues sit on the diagonal of T.  Modifying them in-place
    # avoids the ill-conditioned inv(V) that plagues eig-based methods.
    T_s, Q = schur(P, output='complex')
    eigs = np.diag(T_s)
    spectral_radius = np.max(np.abs(eigs))
    n_unstable = np.sum(np.abs(eigs) > 1.0 + 1e-14)

    if bc_type == 'PR':
        # Conservative system: force ALL |λ| to exactly 1
        for i in range(T_s.shape[0]):
            mag = abs(T_s[i, i])
            if mag > 1e-15:
                T_s[i, i] /= mag
    else:
        # Dissipative system: clip any |λ| > 1 to the unit circle
        for i in range(T_s.shape[0]):
            mag = abs(T_s[i, i])
            if mag > 1.0:
                T_s[i, i] /= mag

    P = np.real(Q @ T_s @ Q.conj().T)
    sr_new = np.max(np.abs(np.diag(T_s)))
    if n_unstable > 0:
        print(f" stabilized ({n_unstable} modes, "
              f"rho {spectral_radius:.8f}->{sr_new:.8f})", end='')

    print(" done.")

    # IC
    p0 = gaussian_pulse_3d(mesh, x0, y0, z0, sigma)
    state = np.zeros(2 * Nrb)
    state[:Nrb] = Psi.T @ p0
    del p0

    obs = np.zeros(2 * Nrb)
    if rec_idx is not None:
        obs[:Nrb] = Psi[rec_idx, :]

    ir = np.zeros(Nt + 1) if rec_idx is not None else None
    if rec_idx is not None:
        ir[0] = obs @ state

    for n in range(Nt):
        state = P @ state
        if rec_idx is not None:
            ir[n + 1] = obs @ state

    return dict(ir=ir)
