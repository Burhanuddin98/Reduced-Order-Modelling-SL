"""Spectral Element Method: GLL quadrature, mesh, Kronecker assembly."""

import numpy as np
from scipy import sparse
from numpy.polynomial import legendre


def gll_points_weights(P):
    if P == 0: return np.array([0.0]), np.array([2.0])
    if P == 1: return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    c = np.zeros(P+1); c[P] = 1.0
    interior = np.sort(np.real(legendre.legroots(legendre.legder(c))))
    xi = np.concatenate([[-1.0], interior, [1.0]])
    PN = legendre.legval(xi, c)
    return xi, 2.0 / (P*(P+1)*PN**2)


def gll_derivative_matrix(xi):
    N = len(xi)-1
    c = np.zeros(N+1); c[N] = 1.0
    LN = legendre.legval(xi, c)
    D = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j: D[i,j] = LN[i]/(LN[j]*(xi[i]-xi[j]))
    D[0,0] = -N*(N+1)/4.0
    if N > 0: D[N,N] = N*(N+1)/4.0
    return D


def _lagrange_basis_at(xi, x):
    n, m = len(xi), len(x)
    L = np.ones((m, n))
    for j in range(n):
        for k in range(n):
            if k != j: L[:,j] *= (x-xi[k])/(xi[j]-xi[k])
    return L

def _lagrange_deriv_at(xi, x):
    n, m = len(xi), len(x)
    dL = np.zeros((m, n))
    for j in range(n):
        for k in range(n):
            if k == j: continue
            t = np.ones(m)/(xi[j]-xi[k])
            for l in range(n):
                if l != j and l != k: t *= (x-xi[l])/(xi[j]-xi[l])
            dL[:,j] += t
    return dL


def assemble_1d(N_el, P, h, w, D):
    xi, _ = gll_points_weights(P)
    Ng = N_el*P+1
    M = np.zeros(Ng); K = np.zeros((Ng,Ng)); G = np.zeros((Ng,Ng))
    K_ref = D.T @ np.diag(w) @ D
    xg, wg = np.polynomial.legendre.leggauss(P+1)
    L_g = _lagrange_basis_at(xi, xg)
    dL_g = _lagrange_deriv_at(xi, xg)
    G_ref = sum(wg[q]*np.outer(L_g[q,:], dL_g[q,:]) for q in range(P+1))
    for e in range(N_el):
        idx = np.arange(e*P, e*P+P+1)
        M[idx] += (h/2)*w
        K[np.ix_(idx,idx)] += (2/h)*K_ref
        G[np.ix_(idx,idx)] += G_ref
    return M, K, G


class RectMesh2D:
    def __init__(self, Lx, Ly, Nex, Ney, P):
        self.Lx, self.Ly, self.P = Lx, Ly, P
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
        self.ndim = 2

    def _c1d(self, N_el, h):
        c = np.zeros(N_el*self.P+1)
        for e in range(N_el):
            for i in range(self.P+1):
                c[e*self.P+i] = e*h + (self.xi[i]+1)/2*h
        return c

    def nearest_node(self, rx, ry):
        return int(np.argmin((self.x-rx)**2+(self.y-ry)**2))

    def boundary_nodes(self, edge):
        if edge == 'bottom': return np.arange(self.Ngx)
        if edge == 'top': return np.arange(self.Ngx)+(self.Ngy-1)*self.Ngx
        if edge == 'left': return np.arange(self.Ngy)*self.Ngx
        if edge == 'right': return np.arange(self.Ngy)*self.Ngx+(self.Ngx-1)

    def all_boundary_nodes(self):
        s = set()
        for e in ('bottom','top','left','right'):
            s.update(self.boundary_nodes(e).tolist())
        return np.array(sorted(s))


class BoxMesh3D:
    def __init__(self, Lx, Ly, Lz, Nex, Ney, Nez, P):
        self.Lx, self.Ly, self.Lz, self.P = Lx, Ly, Lz, P
        self.xi, self.w = gll_points_weights(P)
        self.D = gll_derivative_matrix(self.xi)
        self.hx, self.hy, self.hz = Lx/Nex, Ly/Ney, Lz/Nez
        self.Ngx, self.Ngy, self.Ngz = Nex*P+1, Ney*P+1, Nez*P+1
        self.N_dof = self.Ngx*self.Ngy*self.Ngz
        self.Mx, self.Kx, self.Gx = assemble_1d(Nex, P, self.hx, self.w, self.D)
        self.My, self.Ky, self.Gy = assemble_1d(Ney, P, self.hy, self.w, self.D)
        self.Mz, self.Kz, self.Gz = assemble_1d(Nez, P, self.hz, self.w, self.D)
        self._coords_built = False
        self.ndim = 3

    def _c1d(self, N_el, h):
        c = np.zeros(N_el*self.P+1)
        for e in range(N_el):
            for i in range(self.P+1):
                c[e*self.P+i] = e*h + (self.xi[i]+1)/2*h
        return c

    def _ensure_coords(self):
        if self._coords_built: return
        x1d = self._c1d(int(round(self.Lx/self.hx)), self.hx)
        y1d = self._c1d(int(round(self.Ly/self.hy)), self.hy)
        z1d = self._c1d(int(round(self.Lz/self.hz)), self.hz)
        xx, yy, zz = np.meshgrid(x1d, y1d, z1d, indexing='ij')
        self.x, self.y, self.z = xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')
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


def assemble_2d(mesh):
    N = mesh.N_dof
    M_diag = np.kron(mesh.My, mesh.Mx)
    sMx, sMy = sparse.diags(mesh.Mx), sparse.diags(mesh.My)
    S = (sparse.kron(sMy, sparse.csr_matrix(mesh.Kx), format='csr') +
         sparse.kron(sparse.csr_matrix(mesh.Ky), sMx, format='csr'))
    B_labels = {}
    for edge in ('bottom','top','left','right'):
        b = np.zeros(N)
        nodes = mesh.boundary_nodes(edge)
        if edge in ('bottom','top'):
            for gx in range(mesh.Ngx): b[nodes[gx]] += mesh.Mx[gx]
        else:
            for gy in range(mesh.Ngy): b[nodes[gy]] += mesh.My[gy]
        B_labels[edge] = b
    b_total = sum(B_labels.values())
    return dict(M_diag=M_diag, M_inv=1.0/M_diag, S=S,
                B_total=sparse.diags(b_total), B_labels=B_labels)


def assemble_3d(mesh):
    N = mesh.N_dof
    M_diag = np.kron(mesh.Mz, np.kron(mesh.My, mesh.Mx))
    sMx, sMy, sMz = sparse.diags(mesh.Mx), sparse.diags(mesh.My), sparse.diags(mesh.Mz)
    sKx, sKy, sKz = sparse.csr_matrix(mesh.Kx), sparse.csr_matrix(mesh.Ky), sparse.csr_matrix(mesh.Kz)
    sMxy = sparse.kron(sMy, sMx, format='csr')
    S = (sparse.kron(sMz, sparse.kron(sMy, sKx, format='csr'), format='csr') +
         sparse.kron(sMz, sparse.kron(sKy, sMx, format='csr'), format='csr') +
         sparse.kron(sKz, sMxy, format='csr'))
    Myz = np.kron(mesh.Mz, mesh.My)
    Mxz = np.kron(mesh.Mz, mesh.Mx)
    Mxy = np.kron(mesh.My, mesh.Mx)
    face_names = {(0,0):'x_min',(0,1):'x_max',(1,0):'y_min',(1,1):'y_max',(2,0):'z_min',(2,1):'z_max'}
    face_mass = {0: Myz, 1: Mxz, 2: Mxy}
    B_labels = {}
    for (axis, side), label in face_names.items():
        b = np.zeros(N)
        b[mesh._face_nodes(axis, side)] += face_mass[axis]
        B_labels[label] = b
    b_total = sum(B_labels.values())
    return dict(M_diag=M_diag, M_inv=1.0/M_diag, S=S,
                B_total=sparse.diags(b_total), B_labels=B_labels)
