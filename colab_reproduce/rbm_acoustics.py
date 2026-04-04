"""
rbm_acoustics.py — Self-contained Reduced Basis Method for Room Acoustics
===========================================================================
Standalone implementation of:
  Sampedro Llopis et al. (2022)
  "Reduced basis methods for numerical room acoustic simulations
   with parametrized boundaries"
  J. Acoust. Soc. Am. 152(2), pp. 851-865

Everything needed to reproduce all paper figures in one file.
No external dependencies beyond numpy, scipy, matplotlib.

Modules:
  1. GLL quadrature + SEM mesh (2D rect, 3D box)
  2. Kronecker operator assembly (M, S, B)
  3. Laplace-domain FOM solver
  4. Weeks method inverse Laplace transform
  5. Miki impedance model
  6. SVD / PSD cotangent lift basis construction
  7. ROM projection + online solver
  8. Time-domain p-Phi RK4 solver (for cross-validation)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import svd
from numpy.polynomial import legendre
import time as _time
import warnings

# ── GPU backend ──────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import gmres as gpu_gmres
    from cupyx.scipy.sparse.linalg import LinearOperator as gpuLO
    HAS_GPU = True
    _gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f"GPU: {_gpu_name}")
except Exception:
    HAS_GPU = False
    print("GPU: not available, using CPU")


# ═════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═════════════════════════════════════════════════════════════
C_AIR = 343.0      # speed of sound [m/s]
RHO_AIR = 1.2      # air density [kg/m^3]


# ═════════════════════════════════════════════════════════════
# 1. GLL QUADRATURE
# ═════════════════════════════════════════════════════════════
def gll_points_weights(P):
    """GLL quadrature nodes and weights on [-1,1]."""
    if P == 0: return np.array([0.0]), np.array([2.0])
    if P == 1: return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    N = P
    c = np.zeros(N+1); c[N] = 1.0
    dc = legendre.legder(c)
    interior = np.sort(np.real(legendre.legroots(dc)))
    xi = np.concatenate([[-1.0], interior, [1.0]])
    PN = legendre.legval(xi, c)
    w = 2.0 / (N*(N+1)*PN**2)
    return xi, w


def gll_derivative_matrix(xi):
    """Differentiation matrix at GLL nodes."""
    N = len(xi)-1
    c = np.zeros(N+1); c[N] = 1.0
    LN = legendre.legval(xi, c)
    D = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i,j] = LN[i] / (LN[j] * (xi[i]-xi[j]))
    D[0,0] = -N*(N+1)/4.0
    if N > 0: D[N,N] = N*(N+1)/4.0
    return D


# ═════════════════════════════════════════════════════════════
# 2. SEM MESH + 1D ASSEMBLY
# ═════════════════════════════════════════════════════════════
def _lagrange_basis_at(xi_nodes, x_eval):
    n, m = len(xi_nodes), len(x_eval)
    L = np.ones((m, n))
    for j in range(n):
        for k in range(n):
            if k != j:
                L[:,j] *= (x_eval - xi_nodes[k]) / (xi_nodes[j] - xi_nodes[k])
    return L

def _lagrange_deriv_at(xi_nodes, x_eval):
    n, m = len(xi_nodes), len(x_eval)
    dL = np.zeros((m, n))
    for j in range(n):
        for k in range(n):
            if k == j: continue
            term = np.ones(m) / (xi_nodes[j] - xi_nodes[k])
            for l in range(n):
                if l != j and l != k:
                    term *= (x_eval - xi_nodes[l]) / (xi_nodes[j] - xi_nodes[l])
            dL[:,j] += term
    return dL


def assemble_1d(N_el, P, h, w, D):
    """1D SEM operators: mass, stiffness, gradient."""
    xi, _ = gll_points_weights(P)
    Ng = N_el*P + 1
    M = np.zeros(Ng)
    K = np.zeros((Ng, Ng))
    G = np.zeros((Ng, Ng))
    W = np.diag(w)
    K_ref = D.T @ W @ D
    n_gauss = P + 1
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    L_at_g = _lagrange_basis_at(xi, xg)
    dL_at_g = _lagrange_deriv_at(xi, xg)
    G_ref = np.zeros((P+1, P+1))
    for q in range(n_gauss):
        G_ref += wg[q] * np.outer(L_at_g[q,:], dL_at_g[q,:])
    for e in range(N_el):
        idx = np.arange(e*P, e*P+P+1)
        M[idx] += (h/2.0) * w
        K[np.ix_(idx, idx)] += (2.0/h) * K_ref
        G[np.ix_(idx, idx)] += G_ref
    return M, K, G


class RectMesh2D:
    """Structured 2D quad mesh on [0,Lx] x [0,Ly]."""
    def __init__(self, Lx, Ly, Nex, Ney, P):
        self.Lx, self.Ly = Lx, Ly
        self.Nex, self.Ney, self.P = Nex, Ney, P
        self.xi, self.w = gll_points_weights(P)
        self.D = gll_derivative_matrix(self.xi)
        self.hx, self.hy = Lx/Nex, Ly/Ney
        self.Ngx, self.Ngy = Nex*P+1, Ney*P+1
        self.N_dof = self.Ngx * self.Ngy
        self.Mx, self.Kx, self.Gx = assemble_1d(Nex, P, self.hx, self.w, self.D)
        self.My, self.Ky, self.Gy = assemble_1d(Ney, P, self.hy, self.w, self.D)
        x1d = self._c1d(Nex, self.hx)
        y1d = self._c1d(Ney, self.hy)
        xx, yy = np.meshgrid(x1d, y1d)
        self.x, self.y = xx.ravel(), yy.ravel()

    def _c1d(self, N_el, h):
        Ng = N_el*self.P+1; c = np.zeros(Ng)
        for e in range(N_el):
            for i in range(self.P+1):
                c[e*self.P+i] = e*h + (self.xi[i]+1)/2*h
        return c

    def nearest_node(self, rx, ry):
        return int(np.argmin((self.x-rx)**2 + (self.y-ry)**2))

    def boundary_nodes(self, edge):
        if edge == 'bottom': return np.arange(self.Ngx)
        if edge == 'top':    return np.arange(self.Ngx) + (self.Ngy-1)*self.Ngx
        if edge == 'left':   return np.arange(self.Ngy) * self.Ngx
        if edge == 'right':  return np.arange(self.Ngy) * self.Ngx + (self.Ngx-1)

    def all_boundary_nodes(self):
        s = set()
        for e in ('bottom','top','left','right'):
            s.update(self.boundary_nodes(e).tolist())
        return np.array(sorted(s))


class BoxMesh3D:
    """Structured 3D hex mesh on [0,Lx] x [0,Ly] x [0,Lz]."""
    def __init__(self, Lx, Ly, Lz, Nex, Ney, Nez, P):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.Nex, self.Ney, self.Nez, self.P = Nex, Ney, Nez, P
        self.xi, self.w = gll_points_weights(P)
        self.D = gll_derivative_matrix(self.xi)
        self.hx, self.hy, self.hz = Lx/Nex, Ly/Ney, Lz/Nez
        self.Ngx, self.Ngy, self.Ngz = Nex*P+1, Ney*P+1, Nez*P+1
        self.N_dof = self.Ngx * self.Ngy * self.Ngz
        self.Mx, self.Kx, self.Gx = assemble_1d(Nex, P, self.hx, self.w, self.D)
        self.My, self.Ky, self.Gy = assemble_1d(Ney, P, self.hy, self.w, self.D)
        self.Mz, self.Kz, self.Gz = assemble_1d(Nez, P, self.hz, self.w, self.D)
        self._coords_built = False

    def _c1d(self, N_el, h):
        Ng = N_el*self.P+1; c = np.zeros(Ng)
        for e in range(N_el):
            for i in range(self.P+1):
                c[e*self.P+i] = e*h + (self.xi[i]+1)/2*h
        return c

    def _ensure_coords(self):
        if self._coords_built: return
        x1d = self._c1d(self.Nex, self.hx)
        y1d = self._c1d(self.Ney, self.hy)
        z1d = self._c1d(self.Nez, self.hz)
        xx, yy, zz = np.meshgrid(x1d, y1d, z1d, indexing='ij')
        self.x = xx.ravel(order='F')
        self.y = yy.ravel(order='F')
        self.z = zz.ravel(order='F')
        self._coords_built = True

    def nearest_node(self, rx, ry, rz):
        self._ensure_coords()
        return int(np.argmin((self.x-rx)**2+(self.y-ry)**2+(self.z-rz)**2))

    def _face_nodes(self, axis, side):
        Nx, Ny, Nz = self.Ngx, self.Ngy, self.Ngz
        if axis == 0:
            gx = 0 if side == 0 else Nx-1
            return np.array([gx+gy*Nx+gz*Nx*Ny for gz in range(Nz) for gy in range(Ny)])
        elif axis == 1:
            gy = 0 if side == 0 else Ny-1
            return np.array([gx+gy*Nx+gz*Nx*Ny for gz in range(Nz) for gx in range(Nx)])
        else:
            gz = 0 if side == 0 else Nz-1
            return np.array([gx+gy*Nx+gz*Nx*Ny for gy in range(Ny) for gx in range(Nx)])

    def all_boundary_nodes(self):
        s = set()
        for axis in range(3):
            for side in (0,1):
                s.update(self._face_nodes(axis, side).tolist())
        return np.array(sorted(s))


# ═════════════════════════════════════════════════════════════
# 3. OPERATOR ASSEMBLY (Kronecker products)
# ═════════════════════════════════════════════════════════════
def assemble_2d(mesh):
    """2D SEM operators via Kronecker products."""
    Mx, Kx = mesh.Mx, mesh.Kx
    My, Ky = mesh.My, mesh.Ky
    N = mesh.N_dof
    M_diag = np.kron(My, Mx)
    M_inv = 1.0 / M_diag
    sMx, sMy = sparse.diags(Mx), sparse.diags(My)
    sKx, sKy = sparse.csr_matrix(Kx), sparse.csr_matrix(Ky)
    S = sparse.kron(sMy, sKx, format='csr') + sparse.kron(sKy, sMx, format='csr')
    # Boundary mass
    b_total = np.zeros(N)
    for edge in ('bottom','top'):
        nodes = mesh.boundary_nodes(edge)
        for gx in range(mesh.Ngx): b_total[nodes[gx]] += Mx[gx]
    for edge in ('left','right'):
        nodes = mesh.boundary_nodes(edge)
        for gy in range(mesh.Ngy): b_total[nodes[gy]] += My[gy]
    B_total = sparse.diags(b_total)
    return dict(M_diag=M_diag, M_inv=M_inv, S=S, B_total=B_total)


def assemble_3d(mesh):
    """3D SEM operators via triple Kronecker products."""
    Mx, Kx = mesh.Mx, mesh.Kx
    My, Ky = mesh.My, mesh.Ky
    Mz, Kz = mesh.Mz, mesh.Kz
    N = mesh.N_dof
    Mxy = np.kron(My, Mx)
    M_diag = np.kron(Mz, Mxy)
    M_inv = 1.0 / M_diag
    sMx, sMy, sMz = sparse.diags(Mx), sparse.diags(My), sparse.diags(Mz)
    sKx, sKy, sKz = sparse.csr_matrix(Kx), sparse.csr_matrix(Ky), sparse.csr_matrix(Kz)
    sMxy = sparse.kron(sMy, sMx, format='csr')
    S = (sparse.kron(sMz, sparse.kron(sMy, sKx, format='csr'), format='csr') +
         sparse.kron(sMz, sparse.kron(sKy, sMx, format='csr'), format='csr') +
         sparse.kron(sKz, sMxy, format='csr'))
    # Boundary mass
    b_total = np.zeros(N)
    Myz = np.kron(Mz, My); Mxz = np.kron(Mz, Mx); Mxy_1d = np.kron(My, Mx)
    for side in (0,1):
        b_total[mesh._face_nodes(0, side)] += Myz
        b_total[mesh._face_nodes(1, side)] += Mxz
        b_total[mesh._face_nodes(2, side)] += Mxy_1d
    B_total = sparse.diags(b_total)
    return dict(M_diag=M_diag, M_inv=M_inv, S=S, B_total=B_total)


# ═════════════════════════════════════════════════════════════
# 4. MIKI IMPEDANCE MODEL
# ═════════════════════════════════════════════════════════════
def miki_impedance(f, sigma_flow, d_mat):
    """Surface impedance of porous absorber on rigid backing (Miki 1990)."""
    f = np.asarray(f, dtype=complex)
    f_safe = np.where(np.abs(f) < 1.0, 1.0, f)
    X = np.abs(f_safe) / sigma_flow
    Zc = RHO_AIR*C_AIR*(1 + 0.0699*X**(-0.632) - 1j*0.107*X**(-0.632))
    kc = (2*np.pi*f_safe/C_AIR)*(1 + 0.109*X**(-0.618) - 1j*0.160*X**(-0.618))
    return -1j * Zc * np.cos(kc*d_mat) / np.sin(kc*d_mat)

def miki_admittance_scalar(f, sigma_flow, d_mat):
    """Scalar admittance at one frequency."""
    return 1.0 / complex(miki_impedance(max(abs(f), 1.0), sigma_flow, d_mat))

def miki_absorption(f, sigma_flow, d_mat):
    """Normal incidence absorption coefficient."""
    Zs = miki_impedance(f, sigma_flow, d_mat)
    return 1 - np.abs((Zs - RHO_AIR*C_AIR) / (Zs + RHO_AIR*C_AIR))**2


# ═════════════════════════════════════════════════════════════
# 5. LAPLACE-DOMAIN FOM SOLVER (Paper Eq. 9-12)
#
# GPU path: complex N×N system with GMRES + diagonal preconditioner
# CPU path: real 2N×2N system with scipy spsolve (exact)
# ═════════════════════════════════════════════════════════════

def _cpu_solve(c2S, M_diag, B_diag, p0, N, s, Br_diag):
    """CPU: 2N×2N real system, exact via spsolve."""
    sig, omg = s.real, s.imag
    Kr = c2S + sparse.diags((sig**2-omg**2)*M_diag + sig*Br_diag.real - omg*Br_diag.imag, format='csc')
    Kc = sparse.diags(2*sig*omg*M_diag + omg*Br_diag.real + sig*Br_diag.imag, format='csc')
    A = sparse.bmat([[Kr,-Kc],[Kc,Kr]], format='csc')
    rhs = np.concatenate([sig*p0*M_diag, omg*p0*M_diag])
    x = spsolve(A, rhs)
    return x[:N] + 1j*x[N:]


def _gpu_solve(c2S_gpu, M_gpu, B_gpu_br, rhs_base_gpu, N, s):
    """GPU: complex N×N GMRES with diagonal preconditioner."""
    diag = s**2 * M_gpu + s * B_gpu_br
    A = c2S_gpu + csp.diags(diag)
    rhs = s * rhs_base_gpu
    prec_d = A.diagonal()
    M_pre = gpuLO((N, N), matvec=lambda x: x / prec_d, dtype=complex)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            x, _ = gpu_gmres(A, rhs, M=M_pre, tol=1e-8, maxiter=500, restart=100)
        except TypeError:
            x, _ = gpu_gmres(A, rhs, M=M_pre, atol=1e-8, maxiter=500, restart=100)
    cp.cuda.Device().synchronize()
    return x


class GPUContext:
    """Pre-allocated GPU arrays for fast repeated solves."""
    def __init__(self, c2S_cpu, M_diag, B_diag, p0):
        self.N = len(M_diag)
        self.c2S = csp.csr_matrix(c2S_cpu.tocsr())
        self.M = cp.asarray(M_diag, dtype=np.float64)
        self.B = cp.asarray(B_diag, dtype=np.float64)
        self.rhs_base = cp.asarray(p0 * M_diag, dtype=complex)
        self._Br_cache = {}

    def set_fi(self, Zs):
        key = ('fi', Zs)
        if key not in self._Br_cache:
            self._Br_cache[key] = C_AIR**2 * RHO_AIR * self.B / Zs
        self._Br = self._Br_cache[key]

    def set_fd(self, f, sigma_flow, d_mat):
        Ys = miki_admittance_scalar(f, sigma_flow, d_mat)
        self._Br = C_AIR**2 * RHO_AIR * Ys * self.B

    def solve(self, s):
        return _gpu_solve(self.c2S, self.M, self._Br, self.rhs_base, self.N, s)

    def solve_to_cpu(self, s):
        return cp.asnumpy(self.solve(s))

    def solve_at_rec(self, s, rec_idx):
        return complex(cp.asnumpy(self.solve(s)[rec_idx]))


def make_gpu_ctx(c2S, M_diag, B_diag, p0):
    """Create GPU context if GPU available, else return None."""
    if HAS_GPU:
        return GPUContext(c2S, M_diag, B_diag, p0)
    return None


def laplace_fom_fi(c2S, M_diag, B_diag, p0, N, s, Zs):
    """Solve for FI boundaries (CPU)."""
    Br = C_AIR**2 * RHO_AIR * B_diag / Zs
    return _cpu_solve(c2S, M_diag, B_diag, p0, N, s, Br)

def laplace_fom_fd(c2S, M_diag, B_diag, p0, N, s, sigma_flow, d_mat):
    """Solve for freq-dependent boundaries (CPU)."""
    f = max(abs(s.imag)/(2*np.pi), 1.0)
    Ys = miki_admittance_scalar(f, sigma_flow, d_mat)
    Br = C_AIR**2 * RHO_AIR * Ys * B_diag
    return _cpu_solve(c2S, M_diag, B_diag, p0, N, s, Br)


def laplace_sweep_fi(c2S, M_diag, B_diag, p0, N, s_vals, Zs, rec_idx, verbose=True):
    """Sweep for FI case. Auto GPU if available."""
    Ns = len(s_vals)
    H = np.zeros(Ns, dtype=complex)
    t0 = _time.perf_counter()
    gpu = make_gpu_ctx(c2S, M_diag, B_diag, p0)
    if gpu:
        gpu.set_fi(Zs)
        for i, s in enumerate(s_vals):
            H[i] = gpu.solve_at_rec(s, rec_idx)
            if verbose and (i+1) % max(1, Ns//10) == 0:
                el = _time.perf_counter()-t0; eta = el/(i+1)*(Ns-i-1)
                print(f'  {i+1}/{Ns} ({el:.0f}s, ETA {eta:.0f}s)', end='', flush=True)
    else:
        for i, s in enumerate(s_vals):
            H[i] = laplace_fom_fi(c2S, M_diag, B_diag, p0, N, s, Zs)[rec_idx]
            if verbose and (i+1) % max(1, Ns//5) == 0:
                el = _time.perf_counter()-t0
                print(f'  {i+1}/{Ns} ({el:.0f}s)', end='', flush=True)
    if verbose:
        el = _time.perf_counter()-t0
        tag = 'GPU' if gpu else 'CPU'
        print(f' done ({el:.0f}s, {el/Ns*1000:.0f}ms/pt, {tag})')
    return H


def laplace_sweep_fd(c2S, M_diag, B_diag, p0, N, s_vals, sigma_flow, d_mat,
                     rec_idx, verbose=True):
    """Sweep for freq-dependent case. Auto GPU."""
    Ns = len(s_vals)
    H = np.zeros(Ns, dtype=complex)
    t0 = _time.perf_counter()
    gpu = make_gpu_ctx(c2S, M_diag, B_diag, p0)
    for i, s in enumerate(s_vals):
        if gpu:
            f = max(abs(s.imag)/(2*np.pi), 1.0)
            gpu.set_fd(f, sigma_flow, d_mat)
            H[i] = gpu.solve_at_rec(s, rec_idx)
        else:
            H[i] = laplace_fom_fd(c2S, M_diag, B_diag, p0, N, s, sigma_flow, d_mat)[rec_idx]
        if verbose and (i+1) % max(1, Ns//10) == 0:
            el = _time.perf_counter()-t0; eta = el/(i+1)*(Ns-i-1)
            print(f'  {i+1}/{Ns} ({el:.0f}s, ETA {eta:.0f}s)', end='', flush=True)
    if verbose:
        el = _time.perf_counter()-t0
        tag = 'GPU' if gpu else 'CPU'
        print(f' done ({el:.0f}s, {el/Ns*1000:.0f}ms/pt, {tag})')
    return H


def laplace_sweep_fi_fullfield(c2S, M_diag, B_diag, p0, N, s_vals, Zs, verbose=True):
    """Full-field sweep for FI (snapshots). Auto GPU."""
    Ns = len(s_vals)
    snaps = []
    t0 = _time.perf_counter()
    gpu = make_gpu_ctx(c2S, M_diag, B_diag, p0)
    if gpu:
        gpu.set_fi(Zs)
        for i, s in enumerate(s_vals):
            snaps.append(gpu.solve_to_cpu(s))
            if verbose and (i+1) % max(1, Ns//10) == 0:
                el = _time.perf_counter()-t0; eta = el/(i+1)*(Ns-i-1)
                print(f'  {i+1}/{Ns} ({el:.0f}s, ETA {eta:.0f}s)', end='', flush=True)
    else:
        for i, s in enumerate(s_vals):
            snaps.append(laplace_fom_fi(c2S, M_diag, B_diag, p0, N, s, Zs))
            if verbose and (i+1) % max(1, Ns//5) == 0:
                el = _time.perf_counter()-t0
                print(f'  {i+1}/{Ns} ({el:.0f}s)', end='', flush=True)
    if verbose:
        el = _time.perf_counter()-t0
        tag = 'GPU' if gpu else 'CPU'
        print(f' done ({el:.0f}s, {el/Ns*1000:.0f}ms/pt, {tag})')
    return snaps


def laplace_sweep_fd_fullfield(c2S, M_diag, B_diag, p0, N, s_vals,
                                sigma_flow, d_mat, verbose=True):
    """Full-field sweep for freq-dependent (snapshots). Auto GPU."""
    Ns = len(s_vals)
    snaps = []
    t0 = _time.perf_counter()
    gpu = make_gpu_ctx(c2S, M_diag, B_diag, p0)
    for i, s in enumerate(s_vals):
        if gpu:
            f = max(abs(s.imag)/(2*np.pi), 1.0)
            gpu.set_fd(f, sigma_flow, d_mat)
            snaps.append(gpu.solve_to_cpu(s))
        else:
            snaps.append(laplace_fom_fd(c2S, M_diag, B_diag, p0, N, s, sigma_flow, d_mat))
        if verbose and (i+1) % max(1, Ns//10) == 0:
            el = _time.perf_counter()-t0; eta = el/(i+1)*(Ns-i-1)
            print(f'  {i+1}/{Ns} ({el:.0f}s, ETA {eta:.0f}s)', end='', flush=True)
    if verbose:
        el = _time.perf_counter()-t0
        tag = 'GPU' if gpu else 'CPU'
        print(f' done ({el:.0f}s, {el/Ns*1000:.0f}ms/pt, {tag})')
    return snaps


# ═════════════════════════════════════════════════════════════
# 6. WEEKS METHOD (Paper Eq. 24-29)
# ═════════════════════════════════════════════════════════════
def weeks_s_values(sigma, b, N_terms):
    """Complex frequencies on the Weeks/Mobius contour."""
    k = np.arange(N_terms)
    theta = 2*np.pi*k/N_terms
    z = np.exp(1j*theta)
    z_safe = np.where(np.abs(1-z) < 1e-10, 1-1e-10, z)
    return sigma + b*(1+z_safe)/(1-z_safe), z_safe

def weeks_coefficients(H, b, z_safe):
    """Laguerre expansion coefficients via FFT."""
    return np.fft.fft(H * (2*b/(1-z_safe))) / len(H)

def laguerre_eval(n_max, x):
    """Evaluate L_0..L_{n-1} at x."""
    L = np.zeros((n_max, len(x))); L[0] = 1.0
    if n_max > 1: L[1] = 1.0 - x
    for k in range(1, n_max-1):
        L[k+1] = ((2*k+1-x)*L[k] - k*L[k-1]) / (k+1)
    return L

def weeks_reconstruct(a, sigma, b, t):
    """Time-domain signal from Laguerre coefficients."""
    return np.exp((sigma-b)*t) * np.real(a @ laguerre_eval(len(a), 2*b*t))

def laplace_to_ir(H, sigma, b, t):
    """H(s) values -> time-domain IR via Weeks method."""
    _, z_safe = weeks_s_values(sigma, b, len(H))
    a = weeks_coefficients(H, b, z_safe)
    return weeks_reconstruct(a, sigma, b, t)


# ═════════════════════════════════════════════════════════════
# 7. SVD BASIS CONSTRUCTION (Paper Eq. 33-38)
# ═════════════════════════════════════════════════════════════
def build_basis(snapshots, eps_pod=1e-6):
    """Cotangent-lift SVD basis from complex snapshots."""
    S_r = np.column_stack([p.real for p in snapshots])
    S_i = np.column_stack([p.imag for p in snapshots])
    S_cl = np.column_stack([S_r, S_i])
    U, sv, _ = svd(S_cl, full_matrices=False)
    energy = np.cumsum(sv**2) / np.sum(sv**2)
    Nrb = min(int(np.searchsorted(energy, 1.0-eps_pod)+1), len(sv))
    return U[:, :Nrb], Nrb, sv


# ═════════════════════════════════════════════════════════════
# 8. ROM PROJECTION + ONLINE SOLVER (Paper Eq. 40-44)
# ═════════════════════════════════════════════════════════════
def project_operators(ops, Psi, p0, rec_idx):
    """Project FOM operators onto ROM basis."""
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())
    return dict(
        M_r  = Psi.T @ (M_diag[:,None] * Psi),
        S_r  = Psi.T @ ops['S'].dot(Psi),
        MC_r = Psi.T @ (B_diag[:,None] * Psi),
        f_r  = Psi.T @ (p0 * M_diag),
        obs  = Psi[rec_idx,:].copy(),
    )

def rom_solve_fi(rom_ops, s, Zs):
    """ROM H(s) for frequency-independent BC."""
    Br_r = C_AIR**2 * RHO_AIR / Zs * rom_ops['MC_r']
    A_r = s**2*rom_ops['M_r'] + C_AIR**2*rom_ops['S_r'] + s*Br_r
    a = np.linalg.solve(A_r, s*rom_ops['f_r'])
    return rom_ops['obs'] @ a

def rom_solve_fd(rom_ops, s, sigma_flow, d_mat):
    """ROM H(s) for frequency-dependent BC."""
    f = max(abs(s.imag)/(2*np.pi), 1.0)
    Ys = miki_admittance_scalar(f, sigma_flow, d_mat)
    Br = s * C_AIR**2 * RHO_AIR * Ys
    A_r = s**2*rom_ops['M_r'] + C_AIR**2*rom_ops['S_r'] + Br*rom_ops['MC_r']
    a = np.linalg.solve(A_r, s*rom_ops['f_r'])
    return rom_ops['obs'] @ a

def rom_sweep_fi(rom_ops, s_vals, Zs):
    """ROM sweep for FI case."""
    return np.array([rom_solve_fi(rom_ops, s, Zs) for s in s_vals])

def rom_sweep_fd(rom_ops, s_vals, sigma_flow, d_mat):
    """ROM sweep for freq-dep case."""
    return np.array([rom_solve_fd(rom_ops, s, sigma_flow, d_mat) for s in s_vals])


# ═════════════════════════════════════════════════════════════
# 9. TIME-DOMAIN p-Phi RK4 SOLVER (for cross-validation)
# ═════════════════════════════════════════════════════════════
def td_solve_fi(mesh, ops, src_x, src_y, sigma, Zs, dt, T, rec_idx):
    """2D time-domain p-Phi FOM with FI boundaries."""
    N = mesh.N_dof
    rho, c = RHO_AIR, C_AIR
    rc2 = rho * c**2
    M_inv = ops['M_inv']
    S = ops['S']
    B_diag = np.array(ops['B_total'].diagonal())
    fi_coeff = -rc2 / Zs * M_inv * B_diag
    Nt = int(round(T/dt))

    r2 = (mesh.x-src_x)**2 + (mesh.y-src_y)**2
    p = np.exp(-r2/sigma**2)
    Phi = np.zeros(N)
    ir = np.zeros(Nt+1)
    ir[0] = p[rec_idx]

    def rhs(p_l, Phi_l):
        dp = rc2 * M_inv * S.dot(Phi_l) + fi_coeff * p_l
        dPhi = -(1.0/rho) * p_l
        return dp, dPhi

    for n in range(Nt):
        k1p, k1P = rhs(p, Phi)
        k2p, k2P = rhs(p+0.5*dt*k1p, Phi+0.5*dt*k1P)
        k3p, k3P = rhs(p+0.5*dt*k2p, Phi+0.5*dt*k2P)
        k4p, k4P = rhs(p+dt*k3p, Phi+dt*k3P)
        p   += dt/6*(k1p+2*k2p+2*k3p+k4p)
        Phi += dt/6*(k1P+2*k2P+2*k3P+k4P)
        ir[n+1] = p[rec_idx]

    return np.arange(Nt+1)*dt, ir


def td_solve_3d_fi(mesh, ops, src_pos, sigma, Zs, dt, T, rec_idx):
    """3D time-domain p-Phi FOM with FI boundaries (CPU)."""
    N = mesh.N_dof
    rho, c = RHO_AIR, C_AIR
    rc2 = rho * c**2
    M_inv = ops['M_inv']
    S = ops['S']
    B_diag = np.array(ops['B_total'].diagonal())
    fi_coeff = -rc2 / Zs * M_inv * B_diag
    Nt = int(round(T/dt))

    mesh._ensure_coords()
    r2 = (mesh.x-src_pos[0])**2 + (mesh.y-src_pos[1])**2 + (mesh.z-src_pos[2])**2
    p = np.exp(-r2/sigma**2)
    Phi = np.zeros(N)
    ir = np.zeros(Nt+1)
    ir[0] = p[rec_idx]

    def rhs(p_l, Phi_l):
        dp = rc2 * M_inv * S.dot(Phi_l) + fi_coeff * p_l
        dPhi = -(1.0/rho) * p_l
        return dp, dPhi

    for n in range(Nt):
        k1p, k1P = rhs(p, Phi)
        k2p, k2P = rhs(p+0.5*dt*k1p, Phi+0.5*dt*k1P)
        k3p, k3P = rhs(p+0.5*dt*k2p, Phi+0.5*dt*k2P)
        k4p, k4P = rhs(p+dt*k3p, Phi+dt*k3P)
        p   += dt/6*(k1p+2*k2p+2*k3p+k4p)
        Phi += dt/6*(k1P+2*k2P+2*k3P+k4P)
        ir[n+1] = p[rec_idx]

    return np.arange(Nt+1)*dt, ir


# ═════════════════════════════════════════════════════════════
# 10. UTILITY: Gaussian pulse, setup helpers
# ═════════════════════════════════════════════════════════════
def gaussian_pulse_2d(mesh, x0, y0, sigma):
    r2 = (mesh.x-x0)**2 + (mesh.y-y0)**2
    return np.exp(-r2/sigma**2)

def gaussian_pulse_3d(mesh, x0, y0, z0, sigma):
    mesh._ensure_coords()
    r2 = (mesh.x-x0)**2 + (mesh.y-y0)**2 + (mesh.z-z0)**2
    return np.exp(-r2/sigma**2)

def setup_2d(Lx, Ly, Ne, P, src, rec_pos, sigma_src):
    """Build mesh, operators, source, receiver for 2D case."""
    mesh = RectMesh2D(Lx, Ly, Ne, Ne, P)
    ops = assemble_2d(mesh)
    p0 = gaussian_pulse_2d(mesh, src[0], src[1], sigma_src)
    rec = mesh.nearest_node(rec_pos[0], rec_pos[1])
    c2S = (C_AIR**2 * ops['S']).tocsc()
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())
    return mesh, ops, p0, rec, c2S, M_diag, B_diag

def setup_3d(Lx, Ly, Lz, Ne, P, src, rec_pos, sigma_src):
    """Build mesh, operators, source, receiver for 3D case."""
    mesh = BoxMesh3D(Lx, Ly, Lz, Ne, Ne, Ne, P)
    ops = assemble_3d(mesh)
    p0 = gaussian_pulse_3d(mesh, src[0], src[1], src[2], sigma_src)
    rec = mesh.nearest_node(rec_pos[0], rec_pos[1], rec_pos[2])
    c2S = (C_AIR**2 * ops['S']).tocsc()
    M_diag = ops['M_diag']
    B_diag = np.array(ops['B_total'].diagonal())
    return mesh, ops, p0, rec, c2S, M_diag, B_diag
