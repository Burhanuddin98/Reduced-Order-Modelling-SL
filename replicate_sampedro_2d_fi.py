"""
replicate_sampedro_2d_fi.py
============================
Standalone replication of:

  Sampedro Llopis et al. (2022)
  "Reduced basis methods for numerical room acoustic simulations
   with parametrized boundaries"
  J. Acoust. Soc. Am. 152(2), pp. 851-865

Case: 2D square domain, frequency-independent (FI) boundaries.

  Domain : 2 m x 2 m
  SEM    : Ne=20 elements/dir, P=4 polynomial order -> N=6561 DOFs
  Source : Gaussian pulse, sigma=0.3 m^2
  Receiver: (0.2, 0.2) m
  BC     : Frequency-independent impedance Zs (parametrized)

Pipeline:
  1. SEM mesh + Kronecker operator assembly
  2. Laplace-domain FOM solver  (Paper Eq. 9-12)
  3. Parametric snapshot collection (Paper Sec. II.C)
  4. SVD + PSD cotangent lift   (Paper Eq. 37-38)
  5. Operator projection         (Paper Eq. 44)
  6. Online ROM evaluation       (Paper Eq. 40-43)
  7. Weeks method ILT            (Paper Eq. 24-29)
  8. FOM vs ROM comparison + speedup study

Usage:
  cd ROM
  python replicate_sampedro_2d_fi.py
"""

import os, sys, time
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.linalg import svd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Imports from existing SEM assembly ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from room_acoustics.sem import RectMesh2D, assemble_2d_operators

# ── Physical constants ───────────────────────────────────────
C_AIR = 343.0     # speed of sound [m/s]
RHO_AIR = 1.2     # air density [kg/m^3]

# ═════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════
# Domain
Lx, Ly = 2.0, 2.0          # room dimensions [m]
Ne = 20                     # elements per direction
P = 4                       # polynomial order

# Source / receiver
SRC_POS = (1.0, 1.0)       # source position [m]
SIGMA_SRC = 0.3             # Gaussian spatial variance [m^2]
REC_POS = (0.2, 0.2)       # receiver position [m]

# Training impedances (Paper: Nsnap=3 is sufficient, Fig. 9)
Z_TRAIN = [500.0, 8000.0, 15500.0]   # [Pa s/m^3]

# Test impedances (NOT in training set — Paper Fig. 3)
Z_TEST = [5000.0, 15000.0]

# Weeks method (Paper Sec. II.B, 3D case: sigma=20, b=800)
SIGMA_W = 20.0              # Laplace shift parameter
B_W = 800.0                 # Laguerre scaling parameter
NS = 1000                   # complex frequencies (Laguerre terms)

# Time reconstruction
T_MAX = 0.1                 # impulse response length [s]
NT = 4000                   # time samples

# ROM
TOL_POD = 1e-6              # POD energy tolerance (Paper Eq. 36)

# Output
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'sampedro_replication')


# ═════════════════════════════════════════════════════════════
# STEP 1: MESH + OPERATOR ASSEMBLY
# ═════════════════════════════════════════════════════════════
def step1_mesh_and_operators():
    """Build 2D SEM mesh and assemble M, S, B operators."""
    print("=" * 64)
    print("  Sampedro Llopis et al. (2022) — 2D FI Replication")
    print("=" * 64)

    mesh = RectMesh2D(Lx, Ly, Ne, Ne, P)
    ops = assemble_2d_operators(mesh)
    N = mesh.N_dof
    print(f"\n[1] Mesh: {Lx}x{Ly} m, Ne={Ne}, P={P}, N={N} DOFs")

    # Source: Gaussian pulse  p0 = exp(-|x-x0|^2 / sigma^2)
    r2 = (mesh.x - SRC_POS[0])**2 + (mesh.y - SRC_POS[1])**2
    p0 = np.exp(-r2 / SIGMA_SRC**2)

    # Receiver
    rec_idx = mesh.nearest_node(*REC_POS)
    print(f"    Source: {SRC_POS}, Receiver: {REC_POS} -> node {rec_idx}")
    print(f"    Receiver coords: ({mesh.x[rec_idx]:.4f}, {mesh.y[rec_idx]:.4f})")

    return mesh, ops, p0, rec_idx


# ═════════════════════════════════════════════════════════════
# STEP 2: FOM SOLVER  (Paper Eq. 9-12)
#
#   (s^2 M + c^2 S + s c^2 rho/Zs MC) p = s p0 M
#
# Split into 2N x 2N real system:
#   [Kr -Kc] [pr]   [Qr]
#   [Kc  Kr] [pc] = [Qc]
#
# Kr = (sig^2 - omg^2)M + c^2 S + sig * c^2 rho/Zs * MC
# Kc = 2 sig omg M + omg * c^2 rho/Zs * MC
# Qr = sig * p0 * M,  Qc = omg * p0 * M
# ═════════════════════════════════════════════════════════════
class FOMSolver:
    """Laplace-domain full-order model solver."""

    def __init__(self, ops, p0, N):
        self.M_diag = ops['M_diag']
        self.S_csc = (C_AIR**2 * ops['S']).tocsc()
        self.B_diag = np.array(ops['B_total'].diagonal())
        self.p0 = p0
        self.N = N

    def solve(self, s, Zs):
        """Solve FOM at complex frequency s with impedance Zs.

        Returns full pressure field (complex N-vector).
        """
        sig, omg = s.real, s.imag
        Br = C_AIR**2 * RHO_AIR * self.B_diag / Zs

        Kr_diag = (sig**2 - omg**2) * self.M_diag + sig * Br
        Kr = self.S_csc + sparse.diags(Kr_diag, format='csc')

        Kc_diag = 2 * sig * omg * self.M_diag + omg * Br
        Kc = sparse.diags(Kc_diag, format='csc')

        Qr = sig * self.p0 * self.M_diag
        Qc = omg * self.p0 * self.M_diag

        A = sparse.bmat([[Kr, -Kc], [Kc, Kr]], format='csc')
        rhs = np.concatenate([Qr, Qc])
        x = spsolve(A, rhs)
        return x[:self.N] + 1j * x[self.N:]

    def transfer_function(self, s, Zs, rec_idx):
        """H(s) = p(receiver, s) for a given impedance."""
        return self.solve(s, Zs)[rec_idx]


# ═════════════════════════════════════════════════════════════
# STEP 3: WEEKS METHOD  (Paper Eq. 24-29)
#
# Mobius mapping:  s = sigma + b*(1+z)/(1-z),  z = e^{i theta}
# Laguerre expansion:
#   p(t) = e^{(sigma-b)t} sum_k a_k L_k(2 b t)
# Coefficients via FFT:
#   a_k = (1/N) FFT[ H(s_k) * 2b/(1-z_k) ]
# ═════════════════════════════════════════════════════════════
def weeks_s_values(sigma, b, N_terms):
    """Complex frequencies on the Mobius-mapped unit circle."""
    k = np.arange(N_terms)
    theta = 2 * np.pi * k / N_terms
    z = np.exp(1j * theta)
    z_safe = np.where(np.abs(1 - z) < 1e-10, 1 - 1e-10, z)
    return sigma + b * (1 + z_safe) / (1 - z_safe), z_safe


def weeks_coefficients(H_values, b, z_safe):
    """Laguerre expansion coefficients from H(s) values (Eq. 25)."""
    weight = 2 * b / (1 - z_safe)
    G = H_values * weight
    return np.fft.fft(G) / len(G)


def laguerre_eval(n_max, x):
    """Evaluate L_0(x) .. L_{n_max-1}(x) via 3-term recurrence."""
    L = np.zeros((n_max, len(x)))
    L[0] = 1.0
    if n_max > 1:
        L[1] = 1.0 - x
    for k in range(1, n_max - 1):
        L[k + 1] = ((2 * k + 1 - x) * L[k] - k * L[k - 1]) / (k + 1)
    return L


def weeks_reconstruct(a_coeffs, sigma, b, t):
    """Time-domain signal from Laguerre coefficients (Eq. 24)."""
    N_terms = len(a_coeffs)
    L = laguerre_eval(N_terms, 2 * b * t)          # (N_terms, len(t))
    envelope = np.exp((sigma - b) * t)              # decaying exponential
    return envelope * np.real(a_coeffs @ L)


# ═════════════════════════════════════════════════════════════
# STEP 4: OFFLINE — SNAPSHOT COLLECTION  (Paper Sec. II.C)
# ═════════════════════════════════════════════════════════════
def step4_collect_snapshots(fom, s_vals):
    """Solve FOM at all (s, Z_train) pairs. Return list of N-vectors."""
    n_train = len(Z_TRAIN)
    n_freq = len(s_vals)
    print(f"\n[4] Offline: {n_train} impedances x {n_freq} frequencies "
          f"= {n_train * n_freq} FOM solves")

    all_snapshots = []
    t_total = time.perf_counter()

    for iz, Zs in enumerate(Z_TRAIN):
        print(f"    Z={Zs:>8.0f}: ", end='', flush=True)
        t0 = time.perf_counter()
        batch = []
        for i_s, s in enumerate(s_vals):
            p = fom.solve(s, Zs)
            batch.append(p)
            if (i_s + 1) % max(1, n_freq // 5) == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i_s + 1) * (n_freq - i_s - 1)
                print(f"{i_s+1}/{n_freq} ({elapsed:.0f}s, "
                      f"ETA {eta:.0f}s) ", end='', flush=True)
        all_snapshots.extend(batch)
        print(f"done ({time.perf_counter()-t0:.0f}s)")

    t_offline = time.perf_counter() - t_total
    print(f"    Total offline solves: {t_offline:.0f}s")
    return all_snapshots, t_offline


# ═════════════════════════════════════════════════════════════
# STEP 5: SVD + PSD COTANGENT LIFT  (Paper Eq. 37-38)
#
# Cotangent lift snapshot matrix:
#   S_cl = [Re(snapshots), Im(snapshots)]  in R^{N x 2*n_snap}
#
# SVD -> U, select Nrb columns -> Psi (the symplectic basis)
# ═════════════════════════════════════════════════════════════
def step5_build_basis(all_snapshots, N):
    """SVD of cotangent-lift snapshot matrix -> ROM basis Psi."""
    print(f"\n[5] SVD + cotangent lift...")
    t0 = time.perf_counter()

    # Cotangent lift: separate real and imaginary parts
    S_real = np.column_stack([p.real for p in all_snapshots])
    S_imag = np.column_stack([p.imag for p in all_snapshots])
    S_cl = np.column_stack([S_real, S_imag])
    del S_real, S_imag
    print(f"    Snapshot matrix: {S_cl.shape[0]} x {S_cl.shape[1]} "
          f"({S_cl.nbytes / 1e6:.0f} MB)")

    # Thin SVD
    U, sigma_sv, _ = svd(S_cl, full_matrices=False)
    del S_cl
    t_svd = time.perf_counter() - t0
    print(f"    SVD time: {t_svd:.1f}s")

    # Energy-based truncation (Paper Eq. 36)
    energy = np.cumsum(sigma_sv**2) / np.sum(sigma_sv**2)
    Nrb = int(np.searchsorted(energy, 1.0 - TOL_POD) + 1)
    Nrb = min(Nrb, len(sigma_sv))

    Psi = U[:, :Nrb].copy()
    del U

    print(f"    POD tolerance: {TOL_POD}")
    print(f"    Basis size: Nrb = {Nrb}")
    print(f"    sigma_1 = {sigma_sv[0]:.3e}, "
          f"sigma_{Nrb} = {sigma_sv[Nrb-1]:.3e}, "
          f"sigma_last = {sigma_sv[-1]:.3e}")

    return Psi, Nrb, sigma_sv, t_svd


# ═════════════════════════════════════════════════════════════
# STEP 6: OPERATOR PROJECTION  (Paper Eq. 44)
#
# M_r  = Psi^T M Psi           (Nrb x Nrb)
# S_r  = Psi^T S Psi           (Nrb x Nrb, pure stiffness)
# MC_r = Psi^T MC Psi          (Nrb x Nrb, boundary mass)
# f_r  = Psi^T (M * p0)        (Nrb,)
# obs  = Psi[rec, :]           (Nrb,)
# ═════════════════════════════════════════════════════════════
def step6_project_operators(ops, Psi, p0, rec_idx):
    """Project FOM operators onto the ROM basis."""
    print(f"\n[6] Projecting operators onto {Psi.shape[1]}-dim subspace...")
    t0 = time.perf_counter()

    M_diag = ops['M_diag']
    S = ops['S']
    B_diag = np.array(ops['B_total'].diagonal())

    M_r = Psi.T @ (M_diag[:, None] * Psi)
    S_r = Psi.T @ S.dot(Psi)                   # pure stiffness (no c^2)
    MC_r = Psi.T @ (B_diag[:, None] * Psi)     # boundary mass
    f_r = Psi.T @ (p0 * M_diag)
    obs = Psi[rec_idx, :].copy()

    dt = time.perf_counter() - t0
    print(f"    Done in {dt:.2f}s")

    return dict(M_r=M_r, S_r=S_r, MC_r=MC_r, f_r=f_r, obs=obs)


# ═════════════════════════════════════════════════════════════
# STEP 7: ONLINE ROM SOLVER  (Paper Eq. 40-43)
#
# A_r = s^2 M_r + c^2 S_r + s * c^2 rho / Zs * MC_r
# rhs = s * f_r
# a   = solve(A_r, rhs)         (Nrb-vector, complex)
# H_ROM(s) = obs . a
# ═════════════════════════════════════════════════════════════
class ROMSolver:
    """Parametric Laplace-domain reduced-order model solver."""

    def __init__(self, rom_ops):
        self.M_r = rom_ops['M_r']
        self.S_r = rom_ops['S_r']
        self.MC_r = rom_ops['MC_r']
        self.f_r = rom_ops['f_r']
        self.obs = rom_ops['obs']

    def transfer_function(self, s, Zs):
        """H_ROM(s) at receiver for given impedance Zs."""
        Br_r = C_AIR**2 * RHO_AIR / Zs * self.MC_r
        A_r = s**2 * self.M_r + C_AIR**2 * self.S_r + s * Br_r
        a = np.linalg.solve(A_r, s * self.f_r)
        return self.obs @ a


# ═════════════════════════════════════════════════════════════
# STEP 8: FULL PIPELINE — EVALUATE + COMPARE
# ═════════════════════════════════════════════════════════════
def step8_evaluate(fom, rom, rec_idx, s_vals, z_safe, t_eval):
    """Compare FOM and ROM impulse responses at test impedances."""
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for Z_test in Z_TEST:
        print(f"\n{'=' * 64}")
        print(f"  TEST: Zs = {Z_test:.0f} Pa s/m^3")
        print(f"{'=' * 64}")

        # ── FOM: evaluate H(s) at all Weeks frequencies ──────
        print(f"  FOM ({NS} solves)...", end='', flush=True)
        t0 = time.perf_counter()
        H_fom = np.array([fom.transfer_function(s, Z_test, rec_idx)
                          for s in s_vals])
        t_fom_eval = time.perf_counter() - t0
        print(f" {t_fom_eval:.1f}s")

        # ── ROM: evaluate H(s) at all Weeks frequencies ──────
        print(f"  ROM ({NS} solves, Nrb={rom.M_r.shape[0]})...", end='',
              flush=True)
        t0 = time.perf_counter()
        H_rom = np.array([rom.transfer_function(s, Z_test)
                          for s in s_vals])
        t_rom_eval = time.perf_counter() - t0
        print(f" {t_rom_eval:.3f}s")

        # ── Weeks ILT: coefficients + time reconstruction ────
        a_fom = weeks_coefficients(H_fom, B_W, z_safe)
        a_rom = weeks_coefficients(H_rom, B_W, z_safe)

        ir_fom = weeks_reconstruct(a_fom, SIGMA_W, B_W, t_eval)
        ir_rom = weeks_reconstruct(a_rom, SIGMA_W, B_W, t_eval)

        # ── Error metrics ────────────────────────────────────
        err_abs = np.abs(ir_fom - ir_rom)
        err_max = np.max(err_abs)
        p_max = np.max(np.abs(ir_fom))
        err_rel = err_max / max(p_max, 1e-30)
        speedup = t_fom_eval / max(t_rom_eval, 1e-9)

        print(f"\n  Results:")
        print(f"    |p_FOM|_max       = {p_max:.4e} Pa")
        print(f"    |p_FOM - p_ROM|   = {err_max:.4e} Pa")
        print(f"    Relative error    = {err_rel:.4e}")
        print(f"    FOM eval time     = {t_fom_eval:.2f}s")
        print(f"    ROM eval time     = {t_rom_eval:.4f}s")
        print(f"    Online speedup    = {speedup:.0f}x")

        results.append(dict(
            Z=Z_test, ir_fom=ir_fom, ir_rom=ir_rom,
            err_max=err_max, err_rel=err_rel, speedup=speedup,
            t_fom=t_fom_eval, t_rom=t_rom_eval,
            H_fom=H_fom, H_rom=H_rom,
        ))

        # ── Plot: IR comparison + error ──────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        t_ms = t_eval * 1000
        axes[0].plot(t_ms, ir_fom, 'b-', lw=0.7, label='FOM')
        axes[0].plot(t_ms, ir_rom, 'r--', lw=0.7,
                     label=f'ROM (Nrb={rom.M_r.shape[0]})')
        axes[0].set_ylabel('Pressure [Pa]')
        axes[0].legend(loc='upper right')
        axes[0].set_title(f'Sampedro Llopis (2022) — 2D FI,  '
                          f'Zs = {Z_test:.0f} Pa·s/m³')
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(t_ms, err_abs + 1e-30, 'k-', lw=0.5)
        axes[1].set_xlabel('Time [ms]')
        axes[1].set_ylabel('|FOM − ROM| [Pa]')
        axes[1].set_title(f'Error: max={err_max:.2e},  '
                          f'rel={err_rel:.2e},  '
                          f'speedup={speedup:.0f}×')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(OUT_DIR, f'ir_Z{int(Z_test)}.png')
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"    Saved: {fname}")

    return results


def plot_svd_decay(sigma_sv, Nrb):
    """Singular value decay plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(sigma_sv / sigma_sv[0], 'b-', lw=1)
    ax.axvline(Nrb, color='r', ls='--', label=f'Nrb = {Nrb}')
    ax.set_xlabel('Basis index')
    ax.set_ylabel('Normalized singular value')
    ax.set_title('Singular value decay (cotangent lift, Paper Eq. 35-36)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, 'svd_decay.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"    SVD decay plot: {fname}")


# ═════════════════════════════════════════════════════════════
# STEP 9: SPEEDUP STUDY  (Paper Fig. 5)
#
# Vary Nrb, measure error and speedup at one test impedance.
# ═════════════════════════════════════════════════════════════
def step9_speedup_study(ops, Psi_full, p0, rec_idx, fom,
                        s_vals, z_safe, t_eval, sigma_sv):
    """Speedup vs error for varying ROM sizes (Paper Fig. 5)."""
    Z_test = Z_TEST[0]
    Nrb_max = Psi_full.shape[1]

    # FOM reference (reuse if already computed)
    print(f"\n[9] Speedup study: Zs={Z_test:.0f}, Nrb = 7..{Nrb_max}")
    print(f"    Computing FOM reference...", end='', flush=True)
    t0 = time.perf_counter()
    H_fom = np.array([fom.transfer_function(s, Z_test, rec_idx)
                      for s in s_vals])
    t_fom = time.perf_counter() - t0
    a_fom = weeks_coefficients(H_fom, B_W, z_safe)
    ir_fom = weeks_reconstruct(a_fom, SIGMA_W, B_W, t_eval)
    print(f" {t_fom:.1f}s")

    # Test several ROM sizes
    nrb_list = sorted(set([7, 18, 30, 44, 82, 150, 300] +
                          [Nrb_max]) & set(range(1, Nrb_max + 1)))
    errors = []
    speedups = []

    for nrb in nrb_list:
        Psi_sub = Psi_full[:, :nrb]
        rom_ops_sub = step6_project_operators(ops, Psi_sub, p0, rec_idx)
        rom_sub = ROMSolver(rom_ops_sub)

        t0 = time.perf_counter()
        H_rom = np.array([rom_sub.transfer_function(s, Z_test)
                          for s in s_vals])
        t_rom = time.perf_counter() - t0

        a_rom = weeks_coefficients(H_rom, B_W, z_safe)
        ir_rom = weeks_reconstruct(a_rom, SIGMA_W, B_W, t_eval)

        err = np.max(np.abs(ir_fom - ir_rom))
        spd = t_fom / max(t_rom, 1e-9)
        errors.append(err)
        speedups.append(spd)
        print(f"    Nrb={nrb:>4d}: error={err:.2e}, speedup={spd:.0f}x, "
              f"ROM={t_rom:.3f}s")

    # ── Plot: speedup vs error (Paper Fig. 5a) ──────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(errors, speedups, 'bo-', markersize=6)
    for i, nrb in enumerate(nrb_list):
        ax1.annotate(f'  {nrb}', (errors[i], speedups[i]), fontsize=8)
    ax1.set_xlabel('Max absolute error [Pa]')
    ax1.set_ylabel('Speedup (FOM / ROM)')
    ax1.set_title(f'Speedup vs Error (Zs={Z_test:.0f})')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.invert_xaxis()

    ax2.semilogy(nrb_list, errors, 'ro-', markersize=6, label='Error')
    ax2b = ax2.twinx()
    ax2b.plot(nrb_list, speedups, 'bs-', markersize=6, label='Speedup')
    ax2.set_xlabel('ROM size (Nrb)')
    ax2.set_ylabel('Max error [Pa]', color='r')
    ax2b.set_ylabel('Speedup', color='b')
    ax2.set_title('Error & Speedup vs ROM size')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, 'speedup_study.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"    Speedup study plot: {fname}")

    return nrb_list, errors, speedups


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_global = time.perf_counter()

    # ── Step 1: Mesh ─────────────────────────────────────────
    mesh, ops, p0, rec_idx = step1_mesh_and_operators()
    N = mesh.N_dof

    # ── Step 2: FOM solver ───────────────────────────────────
    fom = FOMSolver(ops, p0, N)

    # Quick timing calibration
    print(f"\n[2] FOM solver timing...", end='', flush=True)
    s_test = SIGMA_W + 1j * 2000.0
    t0 = time.perf_counter()
    n_cal = 5
    for _ in range(n_cal):
        fom.solve(s_test, 5000.0)
    ms_per = (time.perf_counter() - t0) / n_cal * 1000
    print(f" {ms_per:.1f} ms/solve")
    est_offline = len(Z_TRAIN) * NS * ms_per / 1000
    print(f"    Estimated offline: {est_offline:.0f}s "
          f"({len(Z_TRAIN)} x {NS} solves)")

    # ── Step 3: Weeks frequencies ────────────────────────────
    print(f"\n[3] Weeks method: sigma={SIGMA_W}, b={B_W}, Ns={NS}")
    s_vals, z_safe = weeks_s_values(SIGMA_W, B_W, NS)
    omega_sorted = np.sort(s_vals.imag)
    print(f"    Im(s) range: [{omega_sorted[1]:.0f}, "
          f"{omega_sorted[-1]:.0f}] rad/s")
    print(f"    ~freq range: [{omega_sorted[1]/(2*np.pi):.0f}, "
          f"{omega_sorted[-1]/(2*np.pi):.0f}] Hz")

    # ── Step 4: Offline snapshots ────────────────────────────
    all_snapshots, t_offline = step4_collect_snapshots(fom, s_vals)

    # ── Step 5: SVD + basis ──────────────────────────────────
    Psi, Nrb, sigma_sv, t_svd = step5_build_basis(all_snapshots, N)
    del all_snapshots
    plot_svd_decay(sigma_sv, Nrb)

    # ── Step 6: Project operators ────────────────────────────
    rom_ops = step6_project_operators(ops, Psi, p0, rec_idx)

    # ── Step 7: ROM solver ───────────────────────────────────
    rom = ROMSolver(rom_ops)

    # ── Step 8: Evaluate + compare ───────────────────────────
    t_eval = np.linspace(1e-6, T_MAX, NT)
    results = step8_evaluate(fom, rom, rec_idx, s_vals, z_safe, t_eval)

    # ── Step 9: Speedup study ────────────────────────────────
    nrb_list, errors, speedups = step9_speedup_study(
        ops, Psi, p0, rec_idx, fom, s_vals, z_safe, t_eval, sigma_sv)

    # ── Summary ──────────────────────────────────────────────
    t_total = time.perf_counter() - t_global
    print(f"\n{'=' * 64}")
    print(f"  SUMMARY")
    print(f"{'=' * 64}")
    print(f"  N (FOM DOFs)     : {N}")
    print(f"  Nrb (ROM DOFs)   : {Nrb}")
    print(f"  Ns (frequencies) : {NS}")
    print(f"  Offline time     : {t_offline + t_svd:.0f}s "
          f"(solves: {t_offline:.0f}s, SVD: {t_svd:.1f}s)")
    for r in results:
        print(f"  Z={r['Z']:>6.0f}: err={r['err_rel']:.2e}, "
              f"speedup={r['speedup']:.0f}x")
    print(f"  Total wall time  : {t_total:.0f}s")
    print(f"  Output directory  : {OUT_DIR}")
    print(f"{'=' * 64}")


if __name__ == '__main__':
    main()
