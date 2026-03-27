"""
Spectral Element Method (SEM) for 2D rectangular domains.
GLL quadrature, mesh generation, and operator assembly using Kronecker products.

GPU acceleration via CuPy for sparse matrix operations when available.
"""

import numpy as np
from scipy import sparse
from numpy.polynomial import legendre

# ---------------------------------------------------------------------------
# GPU backend selection
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def use_gpu():
    """Return (xp, xp_sparse) — CuPy if available, else NumPy/SciPy."""
    if GPU_AVAILABLE:
        return cp, cp_sparse
    return np, sparse


# ---------------------------------------------------------------------------
# GLL Quadrature
# ---------------------------------------------------------------------------

def gll_points_weights(P):
    """
    Gauss-Lobatto-Legendre quadrature points and weights.

    Parameters
    ----------
    P : int  — polynomial order (P+1 points returned)

    Returns
    -------
    xi : ndarray (P+1,)  — nodes on [-1, 1]
    w  : ndarray (P+1,)  — quadrature weights
    """
    if P == 0:
        return np.array([0.0]), np.array([2.0])
    if P == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    N = P
    c = np.zeros(N + 1); c[N] = 1.0          # P_N in Legendre basis
    dc = legendre.legder(c)                    # P'_N
    interior = np.sort(np.real(legendre.legroots(dc)))

    xi = np.concatenate([[-1.0], interior, [1.0]])
    PN = legendre.legval(xi, c)
    w = 2.0 / (N * (N + 1) * PN ** 2)
    return xi, w


def gll_derivative_matrix(xi):
    """
    Differentiation matrix D[i,j] = l'_j(xi_i) for GLL nodes.
    """
    N = len(xi) - 1
    c = np.zeros(N + 1); c[N] = 1.0
    LN = legendre.legval(xi, c)

    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = LN[i] / (LN[j] * (xi[i] - xi[j]))
    D[0, 0] = -N * (N + 1) / 4.0
    if N > 0:
        D[N, N] = N * (N + 1) / 4.0
    return D


# ---------------------------------------------------------------------------
# 1-D SEM assembly (reference building block)
# ---------------------------------------------------------------------------

def _gauss_quadrature(n):
    """Gauss-Legendre quadrature with *n* points (exact for degree 2n-1)."""
    return np.polynomial.legendre.leggauss(n)


def _lagrange_basis_at(xi_nodes, x_eval):
    """
    Evaluate Lagrange basis polynomials defined on *xi_nodes* at *x_eval*.

    Returns
    -------
    L : (len(x_eval), len(xi_nodes))  — L[q, j] = l_j(x_eval[q])
    """
    n = len(xi_nodes)
    m = len(x_eval)
    L = np.ones((m, n))
    for j in range(n):
        for k in range(n):
            if k != j:
                L[:, j] *= (x_eval - xi_nodes[k]) / (xi_nodes[j] - xi_nodes[k])
    return L


def _lagrange_deriv_at(xi_nodes, x_eval):
    """
    Evaluate derivatives of Lagrange basis polynomials at *x_eval*.

    Returns
    -------
    dL : (len(x_eval), len(xi_nodes))  — dL[q, j] = l'_j(x_eval[q])
    """
    n = len(xi_nodes)
    m = len(x_eval)
    dL = np.zeros((m, n))
    for j in range(n):
        for k in range(n):
            if k == j:
                continue
            term = np.ones(m) / (xi_nodes[j] - xi_nodes[k])
            for l in range(n):
                if l != j and l != k:
                    term *= (x_eval - xi_nodes[l]) / (xi_nodes[j] - xi_nodes[l])
            dL[:, j] += term
    return dL


def assemble_1d(N_el, P, h, w, D):
    """
    Assemble 1-D global SEM operators for *N_el* uniform elements of width *h*.

    The gradient matrix G uses **exact Gauss quadrature** (not GLL) so that
    the discrete identity  Sx^T M^{-1} Sx ≈ S  holds to high precision,
    ensuring p-v and p-Φ formulations agree as in the reference papers.

    Returns
    -------
    M : ndarray (Ng,)          — diagonal (lumped) mass
    K : ndarray (Ng, Ng)       — stiffness  ∫ l'_i l'_j dx
    G : ndarray (Ng, Ng)       — gradient   ∫ l'_j l_i dx  (exact quadrature)
    """
    Ng = N_el * P + 1
    M = np.zeros(Ng)
    K = np.zeros((Ng, Ng))
    G = np.zeros((Ng, Ng))

    W = np.diag(w)
    K_ref = D.T @ W @ D          # 1-D reference stiffness (GLL is exact here)

    # Exact gradient via Gauss quadrature
    # The integrand l'_j(ξ) l_i(ξ) is degree (P-1)+P = 2P-1.
    # Gauss-Legendre with P+1 points is exact for degree 2(P+1)-1 = 2P+1 ≥ 2P-1. ✓
    xi_gll = w * 0  # placeholder — we use the GLL nodes stored externally
    # We need the GLL nodes; they come in via D's definition.  Recover from D:
    #   D[i,j] = l'_j(xi_i)   where xi are GLL points.
    # Instead, recompute GLL points from P directly.
    xi_gll_pts, _ = gll_points_weights(P)
    n_gauss = P + 1
    xg, wg = _gauss_quadrature(n_gauss)
    L_at_g  = _lagrange_basis_at(xi_gll_pts, xg)    # (n_gauss, P+1)
    dL_at_g = _lagrange_deriv_at(xi_gll_pts, xg)    # (n_gauss, P+1)
    # G_ref[i, j] = ∫ l'_j(ξ) l_i(ξ) dξ ≈ Σ_q wg[q] * dL[q,j] * L[q,i]
    G_ref = np.zeros((P + 1, P + 1))
    for q in range(n_gauss):
        G_ref += wg[q] * np.outer(L_at_g[q, :], dL_at_g[q, :])

    for e in range(N_el):
        idx = np.arange(e * P, e * P + P + 1)
        M[idx] += (h / 2.0) * w
        K[np.ix_(idx, idx)] += (2.0 / h) * K_ref
        G[np.ix_(idx, idx)] += G_ref           # Jacobians cancel for gradient

    return M, K, G


# ---------------------------------------------------------------------------
# 2-D Rectangular Mesh
# ---------------------------------------------------------------------------

class RectMesh2D:
    """
    Structured quad SEM mesh on [0, Lx] × [0, Ly].

    Node ordering: (gx, gy) → gx + gy * Ngx   (x-index varies fastest)
    """

    def __init__(self, Lx, Ly, Nex, Ney, P):
        self.Lx, self.Ly = Lx, Ly
        self.Nex, self.Ney = Nex, Ney
        self.P = P

        # GLL reference data
        self.xi, self.w = gll_points_weights(P)
        self.D = gll_derivative_matrix(self.xi)

        # Grid sizes
        self.hx = Lx / Nex
        self.hy = Ly / Ney
        self.Ngx = Nex * P + 1
        self.Ngy = Ney * P + 1
        self.N_dof = self.Ngx * self.Ngy

        # 1-D operators
        self.Mx, self.Kx, self.Gx = assemble_1d(Nex, P, self.hx, self.w, self.D)
        self.My, self.Ky, self.Gy = assemble_1d(Ney, P, self.hy, self.w, self.D)

        # Coordinates
        self.x1d = self._coord_1d(Nex, self.hx)
        self.y1d = self._coord_1d(Ney, self.hy)
        # node ordering: gx + gy*Ngx  →  x varies fastest
        # meshgrid default ('xy') gives shape (Ngy, Ngx) so ravel is row-major = gy*Ngx + gx ✓
        xx, yy = np.meshgrid(self.x1d, self.y1d)   # 'xy' default
        self.x = xx.ravel()     # [x1d[0..Ngx-1] repeated Ngy times]
        self.y = yy.ravel()     # [y1d[0] * Ngx, y1d[1] * Ngx, ...]

    # ---- helpers ----------------------------------------------------------
    def _coord_1d(self, N_el, h):
        Ng = N_el * self.P + 1
        c = np.zeros(Ng)
        for e in range(N_el):
            for i in range(self.P + 1):
                g = e * self.P + i
                c[g] = e * h + (self.xi[i] + 1.0) / 2.0 * h
        return c

    def idx(self, gx, gy):
        """Global DOF index from grid coordinates."""
        return gx + gy * self.Ngx

    def nearest_node(self, rx, ry):
        """Index of the node closest to (rx, ry)."""
        return int(np.argmin((self.x - rx)**2 + (self.y - ry)**2))

    def boundary_nodes(self, edge):
        """Global indices on *edge* ∈ {'bottom','top','left','right'}."""
        if edge == 'bottom':
            return np.arange(self.Ngx)                        # gy=0
        if edge == 'top':
            return np.arange(self.Ngx) + (self.Ngy - 1) * self.Ngx
        if edge == 'left':
            return np.arange(self.Ngy) * self.Ngx              # gx=0
        if edge == 'right':
            return np.arange(self.Ngy) * self.Ngx + (self.Ngx - 1)
        raise ValueError(edge)

    def all_boundary_nodes(self):
        """Unique set of boundary node indices."""
        s = set()
        for e in ('bottom', 'top', 'left', 'right'):
            s.update(self.boundary_nodes(e).tolist())
        return np.array(sorted(s))


# ---------------------------------------------------------------------------
# 2-D Operator Assembly (Kronecker products)
# ---------------------------------------------------------------------------

def assemble_2d_operators(mesh):
    """
    Build all 2-D SEM operators from 1-D Kronecker products.

    Returns dict with keys:
        M_diag, M_inv  — diagonal mass / inverse (1-D arrays)
        S               — Laplacian stiffness  (sparse, N×N)
        Sx, Sy          — gradient matrices     (sparse, N×N)
        B_edges         — per-edge boundary mass  {str: sparse}
        B_total         — combined boundary mass  (sparse)
    """
    Mx, Kx, Gx = mesh.Mx, mesh.Kx, mesh.Gx
    My, Ky, Gy = mesh.My, mesh.Ky, mesh.Gy
    N = mesh.N_dof

    # Mass (diagonal) — Kronecker of two diagonal vectors
    M_diag = np.kron(My, Mx)
    M_inv = 1.0 / M_diag

    # Convert 1-D dense to sparse for kron
    sMx = sparse.diags(Mx)
    sMy = sparse.diags(My)
    sKx = sparse.csr_matrix(Kx)
    sKy = sparse.csr_matrix(Ky)
    sGx = sparse.csr_matrix(Gx)
    sGy = sparse.csr_matrix(Gy)

    S_xx = sparse.kron(sMy, sKx, format='csr')
    S_yy = sparse.kron(sKy, sMx, format='csr')
    S    = S_xx + S_yy

    Sx = sparse.kron(sMy, sGx, format='csr')
    Sy = sparse.kron(sGy, sMx, format='csr')

    # Boundary mass matrices (diagonal — GLL quadrature)
    B_edges = {}
    for edge in ('bottom', 'top', 'left', 'right'):
        b = np.zeros(N)
        nodes = mesh.boundary_nodes(edge)
        if edge in ('bottom', 'top'):
            # 1-D mass along x at fixed gy
            for gx in range(mesh.Ngx):
                b[nodes[gx]] = Mx[gx]
        else:
            # 1-D mass along y at fixed gx
            for gy in range(mesh.Ngy):
                b[nodes[gy]] = My[gy]
        B_edges[edge] = sparse.diags(b)

    B_total = sum(B_edges.values())

    return dict(
        M_diag=M_diag, M_inv=M_inv,
        S=S, S_xx=S_xx, S_yy=S_yy,
        Sx=Sx, Sy=Sy,
        B_edges=B_edges, B_total=B_total,
    )


# ---------------------------------------------------------------------------
# 3-D Rectangular (Box) Mesh
# ---------------------------------------------------------------------------

class BoxMesh3D:
    """
    Structured hex SEM mesh on [0,Lx] x [0,Ly] x [0,Lz].

    Node ordering: gx + gy*Ngx + gz*Ngx*Ngy   (x fastest, then y, then z)
    """

    def __init__(self, Lx, Ly, Lz, Nex, Ney, Nez, P):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.Nex, self.Ney, self.Nez = Nex, Ney, Nez
        self.P = P
        self.ndim = 3

        self.xi, self.w = gll_points_weights(P)
        self.D = gll_derivative_matrix(self.xi)

        self.hx, self.hy, self.hz = Lx/Nex, Ly/Ney, Lz/Nez
        self.Ngx = Nex*P + 1
        self.Ngy = Ney*P + 1
        self.Ngz = Nez*P + 1
        self.N_dof = self.Ngx * self.Ngy * self.Ngz

        self.Mx, self.Kx, self.Gx = assemble_1d(Nex, P, self.hx, self.w, self.D)
        self.My, self.Ky, self.Gy = assemble_1d(Ney, P, self.hy, self.w, self.D)
        self.Mz, self.Kz, self.Gz = assemble_1d(Nez, P, self.hz, self.w, self.D)

        self.x1d = self._c1d(Nex, self.hx)
        self.y1d = self._c1d(Ney, self.hy)
        self.z1d = self._c1d(Nez, self.hz)

        # Build coordinates lazily (can be large)
        self._coords_built = False

    def _c1d(self, N_el, h):
        Ng = N_el * self.P + 1
        c = np.zeros(Ng)
        for e in range(N_el):
            for i in range(self.P + 1):
                c[e * self.P + i] = e * h + (self.xi[i] + 1.0) / 2.0 * h
        return c

    def _ensure_coords(self):
        if self._coords_built:
            return
        # x fastest, y middle, z slowest
        xx, yy, zz = np.meshgrid(self.x1d, self.y1d, self.z1d, indexing='ij')
        # ravel with Fortran order so x varies fastest
        self.x = xx.ravel(order='F')
        self.y = yy.ravel(order='F')
        self.z = zz.ravel(order='F')
        self._coords_built = True

    def nearest_node(self, rx, ry, rz):
        self._ensure_coords()
        return int(np.argmin((self.x-rx)**2 + (self.y-ry)**2 + (self.z-rz)**2))

    def _face_nodes(self, axis, side):
        """Node indices on a face.  axis=0,1,2 for x,y,z; side=0 or 1."""
        Nx, Ny, Nz = self.Ngx, self.Ngy, self.Ngz
        if axis == 0:   # x-face
            gx = 0 if side == 0 else Nx - 1
            return np.array([gx + gy*Nx + gz*Nx*Ny
                             for gz in range(Nz) for gy in range(Ny)])
        elif axis == 1: # y-face
            gy = 0 if side == 0 else Ny - 1
            return np.array([gx + gy*Nx + gz*Nx*Ny
                             for gz in range(Nz) for gx in range(Nx)])
        else:           # z-face
            gz = 0 if side == 0 else Nz - 1
            return np.array([gx + gy*Nx + gz*Nx*Ny
                             for gy in range(Ny) for gx in range(Nx)])

    def all_boundary_nodes(self):
        s = set()
        for axis in range(3):
            for side in (0, 1):
                s.update(self._face_nodes(axis, side).tolist())
        return np.array(sorted(s))


def assemble_3d_operators(mesh):
    """
    Build 3-D SEM operators from triple Kronecker products.

    Memory-conscious: builds sparse matrices one at a time, never
    holds more than 2 large sparse matrices simultaneously.

    Returns dict with same keys as 2D version plus Sz.
    """
    Mx, Kx, Gx = mesh.Mx, mesh.Kx, mesh.Gx
    My, Ky, Gy = mesh.My, mesh.Ky, mesh.Gy
    Mz, Kz, Gz = mesh.Mz, mesh.Kz, mesh.Gz
    N = mesh.N_dof

    print(f"    3D assembly: N={N}, building mass...", end='', flush=True)
    # Mass diagonal: kron(Mz, kron(My, Mx))
    Mxy = np.kron(My, Mx)
    M_diag = np.kron(Mz, Mxy)
    M_inv = 1.0 / M_diag

    # Sparse 1D
    sMx = sparse.diags(Mx); sMy = sparse.diags(My); sMz = sparse.diags(Mz)
    sKx = sparse.csr_matrix(Kx); sKy = sparse.csr_matrix(Ky); sKz = sparse.csr_matrix(Kz)
    sGx = sparse.csr_matrix(Gx); sGy = sparse.csr_matrix(Gy); sGz = sparse.csr_matrix(Gz)

    # Stiffness: S = S_xx + S_yy + S_zz
    # S_xx = kron(Mz, kron(My, Kx))
    # S_yy = kron(Mz, kron(Ky, Mx))
    # S_zz = kron(Kz, kron(My, Mx))
    print(" stiffness...", end='', flush=True)
    sMxy = sparse.kron(sMy, sMx, format='csr')
    sKxMy = sparse.kron(sMy, sKx, format='csr')
    sKyMx = sparse.kron(sKy, sMx, format='csr')

    S_xx = sparse.kron(sMz, sKxMy, format='csr')
    S_yy = sparse.kron(sMz, sKyMx, format='csr')
    S_zz = sparse.kron(sKz, sMxy, format='csr')
    S = S_xx + S_yy + S_zz
    del S_xx, S_yy, S_zz, sKxMy, sKyMx

    # Gradient: Sx = kron(Mz, kron(My, Gx)), etc.
    print(" gradient...", end='', flush=True)
    Sx = sparse.kron(sMz, sparse.kron(sMy, sGx, format='csr'), format='csr')
    Sy = sparse.kron(sMz, sparse.kron(sGy, sMx, format='csr'), format='csr')
    Sz = sparse.kron(sGz, sMxy, format='csr')

    # Boundary mass: sum over 6 faces
    print(" boundary...", end='', flush=True)
    b_total = np.zeros(N)
    # Each face's boundary mass is the 2D mass of the remaining directions
    # x-faces (axis=0): boundary mass = kron(Mz, My) at fixed gx
    Myz = np.kron(Mz, My)
    for side in (0, 1):
        nodes = mesh._face_nodes(0, side)
        b_total[nodes] += Myz
    # y-faces (axis=1): boundary mass = kron(Mz, Mx) at fixed gy
    Mxz = np.kron(Mz, Mx)
    for side in (0, 1):
        nodes = mesh._face_nodes(1, side)
        b_total[nodes] += Mxz
    # z-faces (axis=2): boundary mass = kron(My, Mx) at fixed gz
    Mxy_1d = np.kron(My, Mx)
    for side in (0, 1):
        nodes = mesh._face_nodes(2, side)
        b_total[nodes] += Mxy_1d

    B_total = sparse.diags(b_total)
    print(" done.")

    return dict(
        M_diag=M_diag, M_inv=M_inv,
        S=S, Sx=Sx, Sy=Sy, Sz=Sz,
        B_total=B_total,
    )


def to_gpu(ops):
    """Move operator dict to GPU (CuPy).  No-op if CuPy unavailable."""
    if not GPU_AVAILABLE:
        return ops
    g = {}
    for k, v in ops.items():
        if isinstance(v, np.ndarray):
            g[k] = cp.asarray(v)
        elif sparse.issparse(v):
            g[k] = cp_sparse.csr_matrix(v)
        elif isinstance(v, dict):
            g[k] = {ek: (cp_sparse.csr_matrix(ev) if sparse.issparse(ev)
                         else cp.asarray(ev) if isinstance(ev, np.ndarray)
                         else ev)
                     for ek, ev in v.items()}
        else:
            g[k] = v
    return g
