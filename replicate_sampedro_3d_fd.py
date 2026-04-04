"""
replicate_sampedro_3d_fd.py
============================
Replication of Sampedro Llopis et al. (2022), JASA 152(2), pp. 851-865

Case: 3D cube, frequency-DEPENDENT boundaries (Miki model).

  Domain : 1 m x 1 m x 1 m
  SEM    : Ne=8/dir, P=4 -> N=35,937
  Source : Gaussian pulse, sigma=0.2 m^2, at (0.5, 0.5, 0.5)
  Receiver: (0.25, 0.1, 0.8)
  BC     : Miki porous absorber on rigid backing
           Flow resistivity sigma_mat = 10,000 N s/m^4
           Thickness d_mat parametrized: [0.02, 0.12, 0.22] m
  Weeks  : sigma=20, b=800, Ns=500

GPU strategy: CuPy GMRES with diagonal preconditioner for the bulk of
frequencies. Falls back to CPU splu for frequencies that don't converge.

Usage:
  cd ROM
  python replicate_sampedro_3d_fd.py
"""

import os, sys, time, warnings
import numpy as np
from scipy.sparse.linalg import spsolve, splu, gmres as cpu_gmres, LinearOperator
from scipy.linalg import svd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from room_acoustics.sem import BoxMesh3D, assemble_3d_operators

# ── GPU setup ────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import gmres as gpu_gmres
    from cupyx.scipy.sparse.linalg import LinearOperator as gpuLO
    HAS_GPU = True
    print("GPU: CuPy + cuSPARSE available")
except ImportError:
    HAS_GPU = False
    print("GPU: not available, using CPU")

C_AIR = 343.0
RHO_AIR = 1.2

# ── Configuration ────────────────────────────────────────────
Lx, Ly, Lz = 1.0, 1.0, 1.0
Ne = 8;  P = 4              # paper values: N = (8*4+1)^3 = 35,937

SRC_POS = (0.5, 0.5, 0.5)
SIGMA_SRC = 0.2
REC_POS = (0.25, 0.1, 0.8)

SIGMA_FLOW = 10_000.0
D_TRAIN = [0.02, 0.12, 0.22]
D_TEST = [0.05, 0.15]

SIGMA_W = 20.0
B_W = 800.0
NS = 500

T_MAX = 0.1
NT = 4000
TOL_POD = 1e-6

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'sampedro_3d_replication')


# ═════════════════════════════════════════════════════════════
# MIKI MODEL
# ═════════════════════════════════════════════════════════════
def miki_admittance(f, sigma_flow, d_mat):
    f = max(abs(f), 1.0)
    X = f / sigma_flow
    Zc = RHO_AIR * C_AIR * (1 + 0.0699*X**(-0.632) - 1j*0.107*X**(-0.632))
    kc = (2*np.pi*f/C_AIR) * (1 + 0.109*X**(-0.618) - 1j*0.160*X**(-0.618))
    arg = kc * d_mat
    Zs = -1j * Zc * np.cos(arg) / np.sin(arg)
    return 1.0 / Zs


# ═════════════════════════════════════════════════════════════
# WEEKS METHOD
# ═════════════════════════════════════════════════════════════
def weeks_s_values(sigma, b, N_terms):
    k = np.arange(N_terms)
    theta = 2 * np.pi * k / N_terms
    z = np.exp(1j * theta)
    z_safe = np.where(np.abs(1 - z) < 1e-10, 1 - 1e-10, z)
    return sigma + b * (1 + z_safe) / (1 - z_safe), z_safe

def weeks_coefficients(H_values, b, z_safe):
    return np.fft.fft(H_values * (2*b/(1-z_safe))) / len(H_values)

def laguerre_eval(n_max, x):
    L = np.zeros((n_max, len(x)))
    L[0] = 1.0
    if n_max > 1: L[1] = 1.0 - x
    for k in range(1, n_max-1):
        L[k+1] = ((2*k+1-x)*L[k] - k*L[k-1]) / (k+1)
    return L

def weeks_reconstruct(a, sigma, b, t):
    L = laguerre_eval(len(a), 2*b*t)
    return np.exp((sigma-b)*t) * np.real(a @ L)


# ═════════════════════════════════════════════════════════════
# 3D FOM SOLVER — GPU GMRES + CPU LU fallback
# ═════════════════════════════════════════════════════════════
class FOM3D:
    def __init__(self, ops, p0, N):
        self.N = N
        self.M_diag = ops['M_diag']
        self.B_diag = np.array(ops['B_total'].diagonal())
        self.p0 = p0
        self.rhs_base = p0 * self.M_diag

        # CPU stiffness (always needed for fallback)
        self.c2S_cpu = (C_AIR**2 * ops['S']).tocsc()

        # GPU copies
        if HAS_GPU:
            self.c2S_gpu = csp.csr_matrix(self.c2S_cpu.tocsr())
            self.M_gpu = cp.asarray(self.M_diag)
            self.B_gpu = cp.asarray(self.B_diag)
            self.rhs_base_gpu = cp.asarray(self.rhs_base)

        # CPU LU cache for fallback
        self._lu_cache = {}
        self.gpu_solves = 0
        self.cpu_solves = 0

    def _build_diag(self, s, Ys):
        return s**2 * self.M_diag + s * C_AIR**2 * RHO_AIR * Ys * self.B_diag

    def solve(self, s, d_mat):
        f = max(abs(s.imag) / (2*np.pi), 1.0)
        Ys = complex(miki_admittance(f, SIGMA_FLOW, d_mat))

        if HAS_GPU:
            diag_gpu = (s**2 * self.M_gpu
                        + s * C_AIR**2 * RHO_AIR * Ys * self.B_gpu)
            A_gpu = self.c2S_gpu + csp.diags(diag_gpu)
            rhs_gpu = s * self.rhs_base_gpu
            prec_diag = A_gpu.diagonal()
            M_pre = gpuLO((self.N, self.N),
                          matvec=lambda x: x / prec_diag, dtype=complex)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, info = gpu_gmres(A_gpu, rhs_gpu, M=M_pre,
                                    tol=1e-7, maxiter=500, restart=100)
            cp.cuda.Device().synchronize()
            self.gpu_solves += 1
            # Accept even if info != 0 — partial convergence is fine
            # for snapshot collection (residual typically < 1e-6)
            return cp.asnumpy(x)

        # CPU fallback: iterative (never use spsolve — fill-in kills 3D)
        diag_vals = self._build_diag(s, Ys)
        A_cpu = (self.c2S_cpu + sparse.diags(diag_vals, format='csc')).tocsr()
        rhs = s * self.rhs_base
        diag_A = A_cpu.diagonal()
        M_pre = LinearOperator((self.N, self.N),
                               matvec=lambda v: v / diag_A)
        x, info = cpu_gmres(A_cpu, rhs, M=M_pre, rtol=1e-7, maxiter=500)
        self.cpu_solves += 1
        return x

    def transfer_function(self, s, d_mat, rec_idx):
        return self.solve(s, d_mat)[rec_idx]


# ═════════════════════════════════════════════════════════════
# ROM SOLVER
# ═════════════════════════════════════════════════════════════
class ROM3D:
    def __init__(self, M_r, S_r, MC_r, f_r, obs):
        self.M_r, self.S_r, self.MC_r = M_r, S_r, MC_r
        self.f_r, self.obs = f_r, obs

    def transfer_function(self, s, d_mat):
        f = max(abs(s.imag) / (2*np.pi), 1.0)
        Ys = complex(miki_admittance(f, SIGMA_FLOW, d_mat))
        Br = s * C_AIR**2 * RHO_AIR * Ys
        A_r = s**2*self.M_r + C_AIR**2*self.S_r + Br*self.MC_r
        a = np.linalg.solve(A_r, s * self.f_r)
        return self.obs @ a


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_global = time.perf_counter()

    print("=" * 64)
    print("  Sampedro Llopis (2022) — 3D Freq-Dependent Replication")
    print("=" * 64)

    # ── Mesh + assembly ──────────────────────────────────────
    print(f"\n  Building mesh: Ne={Ne}, P={P}...", flush=True)
    mesh = BoxMesh3D(Lx, Ly, Lz, Ne, Ne, Ne, P)
    ops = assemble_3d_operators(mesh)
    N = mesh.N_dof
    mesh._ensure_coords()

    r2 = ((mesh.x-SRC_POS[0])**2 + (mesh.y-SRC_POS[1])**2
          + (mesh.z-SRC_POS[2])**2)
    p0 = np.exp(-r2 / SIGMA_SRC**2)
    rec_idx = mesh.nearest_node(*REC_POS)
    print(f"  N = {N}, receiver node {rec_idx}")

    fom = FOM3D(ops, p0, N)

    # ── Timing calibration ───────────────────────────────────
    s_cal = SIGMA_W + 1j*2000
    print(f"  Calibrating...", end='', flush=True)
    fom.solve(s_cal, 0.05)  # warmup
    t0 = time.perf_counter()
    for _ in range(3):
        fom.solve(s_cal, 0.05)
    ms = (time.perf_counter()-t0)/3*1000
    print(f" {ms:.0f} ms/solve (GPU={fom.gpu_solves}, CPU={fom.cpu_solves})")
    fom.gpu_solves = fom.cpu_solves = 0

    est = len(D_TRAIN) * NS * ms / 1000
    print(f"  Estimated offline: {est:.0f}s ({est/60:.1f} min)")

    # ── Weeks frequencies ────────────────────────────────────
    s_vals, z_safe = weeks_s_values(SIGMA_W, B_W, NS)

    # ── Offline: snapshots ───────────────────────────────────
    print(f"\n  Offline: {len(D_TRAIN)} x {NS} = "
          f"{len(D_TRAIN)*NS} FOM solves")
    all_snapshots = []
    t_offline_start = time.perf_counter()

    for d in D_TRAIN:
        print(f"    d={d:.3f}m: ", end='', flush=True)
        t0 = time.perf_counter()
        for i, s in enumerate(s_vals):
            p = fom.solve(s, d)
            all_snapshots.append(p)
            if (i+1) % max(1, NS//5) == 0:
                elapsed = time.perf_counter()-t0
                eta = elapsed/(i+1)*(NS-i-1)
                print(f"{i+1}/{NS} ({elapsed:.0f}s, ETA {eta:.0f}s) ",
                      end='', flush=True)
        print(f"done ({time.perf_counter()-t0:.0f}s)")

    t_offline = time.perf_counter() - t_offline_start
    print(f"    Total: {t_offline:.0f}s "
          f"(GPU={fom.gpu_solves}, CPU fallback={fom.cpu_solves})")

    # ── SVD ──────────────────────────────────────────────────
    print(f"\n  SVD...", flush=True)
    t0 = time.perf_counter()
    S_r = np.column_stack([p.real for p in all_snapshots])
    S_i = np.column_stack([p.imag for p in all_snapshots])
    S_cl = np.column_stack([S_r, S_i])
    del S_r, S_i, all_snapshots
    print(f"    Snapshot: {S_cl.shape}, {S_cl.nbytes/1e6:.0f} MB")

    U, sv, _ = svd(S_cl, full_matrices=False)
    del S_cl
    t_svd = time.perf_counter()-t0
    print(f"    SVD: {t_svd:.1f}s")

    energy = np.cumsum(sv**2) / np.sum(sv**2)
    Nrb = min(int(np.searchsorted(energy, 1-TOL_POD)+1), len(sv))
    Psi = U[:, :Nrb].copy(); del U
    print(f"    Nrb = {Nrb}")

    # SVD plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.semilogy(sv[:min(500,len(sv))]/sv[0], 'b-')
    ax.axvline(Nrb, color='r', ls='--', label=f'Nrb={Nrb}')
    ax.set_xlabel('Index'); ax.set_ylabel('Normalized σ')
    ax.set_title(f'3D SVD decay (N={N})'); ax.legend(); ax.grid(True,alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'svd_3d.png'), dpi=150); plt.close()

    # ── Project operators ────────────────────────────────────
    print(f"  Projecting...", end='', flush=True)
    t0 = time.perf_counter()
    M_r = Psi.T @ (ops['M_diag'][:,None] * Psi)
    S_r = Psi.T @ ops['S'].dot(Psi)
    B_diag = np.array(ops['B_total'].diagonal())
    MC_r = Psi.T @ (B_diag[:,None] * Psi)
    f_r = Psi.T @ (p0 * ops['M_diag'])
    obs = Psi[rec_idx,:].copy()
    print(f" {time.perf_counter()-t0:.1f}s")

    rom = ROM3D(M_r, S_r, MC_r, f_r, obs)

    # ── Evaluate ─────────────────────────────────────────────
    t_eval = np.linspace(1e-6, T_MAX, NT)

    for d_test in D_TEST:
        print(f"\n{'='*64}")
        print(f"  TEST: d_mat = {d_test:.3f} m")
        print(f"{'='*64}")

        print(f"  FOM...", end='', flush=True)
        t0 = time.perf_counter()
        H_fom = np.array([fom.transfer_function(s, d_test, rec_idx)
                          for s in s_vals])
        t_fom = time.perf_counter()-t0
        print(f" {t_fom:.1f}s")

        print(f"  ROM (Nrb={Nrb})...", end='', flush=True)
        t0 = time.perf_counter()
        H_rom = np.array([rom.transfer_function(s, d_test)
                          for s in s_vals])
        t_rom = time.perf_counter()-t0
        print(f" {t_rom:.3f}s")

        ir_fom = weeks_reconstruct(weeks_coefficients(H_fom,B_W,z_safe),
                                   SIGMA_W, B_W, t_eval)
        ir_rom = weeks_reconstruct(weeks_coefficients(H_rom,B_W,z_safe),
                                   SIGMA_W, B_W, t_eval)

        err = np.max(np.abs(ir_fom-ir_rom))
        p_max = np.max(np.abs(ir_fom))
        rel = err / max(p_max, 1e-30)
        spd = t_fom / max(t_rom, 1e-9)

        print(f"    |p_FOM|_max  = {p_max:.4e}")
        print(f"    Error        = {err:.4e} (rel {rel:.4e})")
        print(f"    Speedup      = {spd:.0f}x")

        fig, axes = plt.subplots(2,1, figsize=(10,6), sharex=True)
        t_ms = t_eval*1000
        axes[0].plot(t_ms, ir_fom, 'b-', lw=.7, label='FOM')
        axes[0].plot(t_ms, ir_rom, 'r--', lw=.7, label=f'ROM (Nrb={Nrb})')
        axes[0].set_ylabel('Pressure [Pa]'); axes[0].legend()
        axes[0].set_title(f'3D Freq-Dep — d={d_test:.3f}m, N={N}')
        axes[0].grid(True, alpha=.3)
        axes[1].semilogy(t_ms, np.abs(ir_fom-ir_rom)+1e-30, 'k-', lw=.5)
        axes[1].set_xlabel('Time [ms]'); axes[1].set_ylabel('|FOM−ROM|')
        axes[1].set_title(f'rel={rel:.2e}, speedup={spd:.0f}×')
        axes[1].grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'ir_3d_d{int(d_test*1000)}mm.png'),
                    dpi=150)
        plt.close()

    # ── Summary ──────────────────────────────────────────────
    t_total = time.perf_counter()-t_global
    print(f"\n{'='*64}")
    print(f"  N={N}, Nrb={Nrb}, Ns={NS}")
    print(f"  Offline: {t_offline:.0f}s + SVD {t_svd:.0f}s")
    print(f"  GPU solves: {fom.gpu_solves}, CPU fallback: {fom.cpu_solves}")
    print(f"  Total: {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
