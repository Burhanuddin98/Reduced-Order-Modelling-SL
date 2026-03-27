"""
Unstructured quad SEM assembly for arbitrary 2D geometries.

Takes a quad mesh (from Gmsh) and produces the same operator dict as
``assemble_2d_operators`` so all existing FOM/ROM solvers work unchanged.

Element-by-element assembly with isoparametric mapping:
  - GLL nodes placed on each element via tensor product on [-1,1]^2
  - Jacobian computed per quadrature point for non-rectangular elements
  - Diagonal (lumped) mass matrix via GLL quadrature
  - Stiffness and gradient assembled per element using reference matrices
"""

import numpy as np
from scipy import sparse

from .sem import gll_points_weights, gll_derivative_matrix


# ---------------------------------------------------------------------------
# Bilinear shape functions for the 4-node reference quad
# ---------------------------------------------------------------------------

def _bilinear_shape(xi, eta):
    """Shape functions N_i(xi, eta) for 4-node quad.
    Node ordering: 0=(-1,-1), 1=(+1,-1), 2=(+1,+1), 3=(-1,+1)
    """
    return np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    ])


def _bilinear_grad(xi, eta):
    """Gradient of shape functions: [dN/dxi; dN/deta], shape (2, 4)."""
    return np.array([
        [-.25*(1-eta),  .25*(1-eta),  .25*(1+eta), -.25*(1+eta)],
        [-.25*(1-xi),  -.25*(1+xi),   .25*(1+xi),   .25*(1-xi)],
    ])


# ---------------------------------------------------------------------------
# Unstructured quad mesh with high-order (GLL) nodes
# ---------------------------------------------------------------------------

class UnstructuredQuadMesh2D:
    """
    Unstructured quad SEM mesh for arbitrary 2D room geometries.

    Takes corner-node connectivity from Gmsh and creates high-order GLL
    nodes within each element via isoparametric mapping.

    Parameters
    ----------
    nodes : (N_corner, 2) array -- vertex coordinates from Gmsh
    quads : (N_el, 4) int array -- quad connectivity (corner node indices)
    boundary_data : dict mapping label -> list of (n1, n2) corner edge pairs
    P : int -- polynomial order
    """

    def __init__(self, nodes, quads, boundary_data, P):
        self.P = P
        self.ndim = 2
        self.corner_nodes = np.asarray(nodes, dtype=float)
        self.corner_quads = np.asarray(quads, dtype=int)
        self.N_el = len(self.corner_quads)

        self.xi_gll, self.w_gll = gll_points_weights(P)
        self.D_gll = gll_derivative_matrix(self.xi_gll)
        self.n_loc = (P + 1) ** 2

        self._build_global_numbering()
        self._build_coordinates()
        self._identify_boundaries(boundary_data)

    def _build_global_numbering(self):
        """Assign global DOF indices, merging shared nodes at element interfaces."""
        P = self.P
        n1d = P + 1

        self.elem_dof = np.full((self.N_el, self.n_loc), -1, dtype=int)

        vertex_map = {}
        edge_map = {}
        next_dof = [0]

        def alloc(n=1):
            start = next_dof[0]
            next_dof[0] += n
            return np.arange(start, start + n)

        corner_local = [(0, 0), (P, 0), (P, P), (0, P)]
        edge_corners = [(0, 1), (1, 2), (2, 3), (3, 0)]

        def edge_local_ij(edge_idx):
            if edge_idx == 0: return [(i, 0) for i in range(n1d)]
            if edge_idx == 1: return [(P, j) for j in range(n1d)]
            if edge_idx == 2: return [(i, P) for i in range(n1d-1, -1, -1)]
            return [(0, j) for j in range(n1d-1, -1, -1)]

        for e in range(self.N_el):
            q = self.corner_quads[e]

            for lc, (li, lj) in enumerate(corner_local):
                cn = int(q[lc])
                if cn not in vertex_map:
                    vertex_map[cn] = alloc(1)[0]
                self.elem_dof[e, li + lj * n1d] = vertex_map[cn]

            for edge_idx in range(4):
                c0l, c1l = edge_corners[edge_idx]
                cn0, cn1 = int(q[c0l]), int(q[c1l])
                ekey = (min(cn0, cn1), max(cn0, cn1))
                interior_pairs = edge_local_ij(edge_idx)[1:-1]

                if ekey not in edge_map:
                    dofs = alloc(P - 1)
                    if cn0 == ekey[0]:
                        edge_map[ekey] = dofs
                    else:
                        edge_map[ekey] = dofs[::-1].copy()

                stored = edge_map[ekey]
                interior_dofs = stored if cn0 == ekey[0] else stored[::-1]

                for k, (li, lj) in enumerate(interior_pairs):
                    self.elem_dof[e, li + lj * n1d] = interior_dofs[k]

            for j in range(1, P):
                for i in range(1, P):
                    self.elem_dof[e, i + j * n1d] = alloc(1)[0]

        self.N_dof = next_dof[0]

    def _build_coordinates(self):
        """Compute physical (x,y) for every global DOF via isoparametric map."""
        P = self.P
        n1d = P + 1
        xi = self.xi_gll
        coords = np.zeros((self.N_dof, 2))
        assigned = np.zeros(self.N_dof, dtype=bool)

        for e in range(self.N_el):
            corners = self.corner_nodes[self.corner_quads[e]]
            for j in range(n1d):
                for i in range(n1d):
                    g = self.elem_dof[e, i + j * n1d]
                    if not assigned[g]:
                        N = _bilinear_shape(xi[i], xi[j])
                        coords[g] = N @ corners
                        assigned[g] = True

        self.x = coords[:, 0]
        self.y = coords[:, 1]

    def _identify_boundaries(self, boundary_data):
        """Build boundary node sets from Gmsh edge data."""
        P = self.P
        n1d = P + 1
        self._boundary_nodes_per_label = {}
        all_bnd = set()

        # Build lookup: corner edge key -> (element, edge_idx)
        edge_to_elem = {}
        for e in range(self.N_el):
            q = self.corner_quads[e]
            for eidx, (c0l, c1l) in enumerate([(0,1),(1,2),(2,3),(3,0)]):
                cn0, cn1 = int(q[c0l]), int(q[c1l])
                ekey = (min(cn0, cn1), max(cn0, cn1))
                edge_to_elem[ekey] = (e, eidx)

        for label, edges in boundary_data.items():
            label_nodes = set()
            for cn0, cn1 in edges:
                ekey = (min(cn0, cn1), max(cn0, cn1))
                if ekey not in edge_to_elem:
                    continue
                e, eidx = edge_to_elem[ekey]
                dof = self.elem_dof[e]

                if eidx == 0:
                    gdofs = [dof[i] for i in range(n1d)]
                elif eidx == 1:
                    gdofs = [dof[P + j*n1d] for j in range(n1d)]
                elif eidx == 2:
                    gdofs = [dof[i + P*n1d] for i in range(n1d)]
                else:
                    gdofs = [dof[j*n1d] for j in range(n1d)]
                label_nodes.update(gdofs)

            self._boundary_nodes_per_label[label] = np.array(
                sorted(label_nodes), dtype=int)
            all_bnd.update(label_nodes)

        self._all_boundary = np.array(sorted(all_bnd), dtype=int)

    def all_boundary_nodes(self):
        return self._all_boundary

    def boundary_nodes(self, label):
        return self._boundary_nodes_per_label[label]

    def nearest_node(self, rx, ry):
        return int(np.argmin((self.x - rx)**2 + (self.y - ry)**2))


# ---------------------------------------------------------------------------
# Operator assembly
# ---------------------------------------------------------------------------

def _element_geometry(corners, xi, n1d):
    """Precompute Jacobian data at all GLL points for one element.

    Returns
    -------
    wdetJ : (n1d, n1d) -- quadrature weight * |det(J)|
    Jinv  : (n1d, n1d, 2, 2) -- inverse Jacobian at each GLL point
    """
    w_xi, _ = gll_points_weights(n1d - 1)  # reuse
    wdetJ = np.zeros((n1d, n1d))
    Jinv = np.zeros((n1d, n1d, 2, 2))

    for j in range(n1d):
        for i in range(n1d):
            dN = _bilinear_grad(xi[i], xi[j])
            J_raw = dN @ corners  # (2,2): row0=[dx/dxi, dy/dxi], row1=[dx/deta, dy/deta]

            detJ = J_raw[0, 0] * J_raw[1, 1] - J_raw[0, 1] * J_raw[1, 0]
            inv_d = 1.0 / detJ

            Jinv[i, j, 0, 0] =  J_raw[1, 1] * inv_d   # dxi/dx
            Jinv[i, j, 0, 1] = -J_raw[1, 0] * inv_d   # dxi/dy
            Jinv[i, j, 1, 0] = -J_raw[0, 1] * inv_d   # deta/dx
            Jinv[i, j, 1, 1] =  J_raw[0, 0] * inv_d   # deta/dy

            wdetJ[i, j] = w_xi[i] * w_xi[j] * abs(detJ)

    return wdetJ, Jinv


def assemble_unstructured_2d_operators(mesh):
    """
    Element-by-element SEM assembly for unstructured quad meshes.

    Returns dict compatible with all FOM/ROM solvers:
        M_diag, M_inv, S, Sx, Sy, B_total
    """
    P = mesh.P
    n1d = P + 1
    n_loc = mesh.n_loc
    N = mesh.N_dof
    N_el = mesh.N_el
    xi = mesh.xi_gll
    w = mesh.w_gll
    D = mesh.D_gll

    M_diag = np.zeros(N)

    # COO triplets for sparse matrices
    s_r, s_c, s_v = [], [], []
    sx_r, sx_c, sx_v = [], [], []
    sy_r, sy_c, sy_v = [], [], []

    for e in range(N_el):
        corners = mesh.corner_nodes[mesh.corner_quads[e]]
        dof = mesh.elem_dof[e]

        # Element stiffness, gradient, mass via reference-space operations
        # Using vectorized approach with D matrix

        # Precompute Jacobian data
        wdetJ = np.zeros((n1d, n1d))
        # Inverse Jacobian components at each GLL point
        dxi_dx  = np.zeros((n1d, n1d))
        dxi_dy  = np.zeros((n1d, n1d))
        deta_dx = np.zeros((n1d, n1d))
        deta_dy = np.zeros((n1d, n1d))

        for j in range(n1d):
            for i in range(n1d):
                dN = _bilinear_grad(xi[i], xi[j])
                J_raw = dN @ corners

                detJ = J_raw[0, 0] * J_raw[1, 1] - J_raw[0, 1] * J_raw[1, 0]
                inv_d = 1.0 / detJ

                dxi_dx[i, j]  =  J_raw[1, 1] * inv_d
                dxi_dy[i, j]  = -J_raw[1, 0] * inv_d
                deta_dx[i, j] = -J_raw[0, 1] * inv_d
                deta_dy[i, j] =  J_raw[0, 0] * inv_d
                wdetJ[i, j]   = w[i] * w[j] * abs(detJ)

                # Mass (diagonal)
                g = dof[i + j * n1d]
                M_diag[g] += wdetJ[i, j]

        # Build element stiffness and gradient matrices (n_loc x n_loc)
        # Using the fact that basis functions are tensor products of 1D GLL:
        #   phi_{(a,b)}(xi,eta) = l_a(xi) * l_b(eta)
        #
        # Physical gradient at GLL point (i,j):
        #   d(phi_{a,b})/dx = dxi_dx * D[i,a]*delta(j,b) + deta_dx * delta(i,a)*D[j,b]
        #   d(phi_{a,b})/dy = dxi_dy * D[i,a]*delta(j,b) + deta_dy * delta(i,a)*D[j,b]

        # Stiffness: S_e[(a,b), (c,d)] = sum_{i,j} wdetJ[i,j] *
        #   (grad phi_{a,b} . grad phi_{c,d}) at (i,j)
        #
        # This has 3 types of nonzero blocks:
        # 1) b==d: sum_i wdetJ[i,b] * G11[i,b] * D[i,a]*D[i,c]  (xi-xi)
        # 2) a==c: sum_j wdetJ[a,j] * G22[a,j] * D[j,b]*D[j,d]  (eta-eta)
        # 3) mixed: wdetJ[c,b] * G12[c,b] * D[c,a]*D[b,d]        (xi-eta)
        #         + wdetJ[a,d] * G12[a,d] * D[a,c]*D[d,b]        (eta-xi)

        G11 = dxi_dx**2 + dxi_dy**2
        G12 = dxi_dx * deta_dx + dxi_dy * deta_dy
        G22 = deta_dx**2 + deta_dy**2

        Se = np.zeros((n_loc, n_loc))

        # Block 1: b == d (xi-xi coupling)
        for b in range(n1d):
            # W_diag[i] = wdetJ[i,b] * G11[i,b]
            W_diag = wdetJ[:, b] * G11[:, b]
            # Contribution: D^T diag(W) D — a (n1d x n1d) matrix
            block = D.T @ np.diag(W_diag) @ D
            for a in range(n1d):
                for c in range(n1d):
                    Se[a + b*n1d, c + b*n1d] += block[a, c]

        # Block 2: a == c (eta-eta coupling)
        for a in range(n1d):
            W_diag = wdetJ[a, :] * G22[a, :]
            block = D.T @ np.diag(W_diag) @ D
            for b_idx in range(n1d):
                for d in range(n1d):
                    Se[a + b_idx*n1d, a + d*n1d] += block[b_idx, d]

        # Block 3: mixed (xi-eta and eta-xi cross terms)
        for a in range(n1d):
            for b in range(n1d):
                for c in range(n1d):
                    for d in range(n1d):
                        # xi-eta: q=(c,b)
                        v = wdetJ[c, b] * G12[c, b] * D[c, a] * D[b, d]
                        # eta-xi: q=(a,d)
                        v += wdetJ[a, d] * G12[a, d] * D[a, c] * D[d, b]
                        Se[a + b*n1d, c + d*n1d] += v

        # Gradient matrices
        # Sx_e[(a,b), (c,d)] = sum_{i,j} wdetJ[i,j] * phi_{a,b}(i,j) * d(phi_{c,d})/dx(i,j)
        # phi_{a,b}(i,j) = delta(i,a)*delta(j,b) [GLL collocated]
        # So Sx_e[(a,b),(c,d)] = wdetJ[a,b] * d(phi_{c,d})/dx at (a,b)
        #   = wdetJ[a,b] * [dxi_dx[a,b]*D[a,c]*delta(b,d) + deta_dx[a,b]*delta(a,c)*D[b,d]]

        Sx_e = np.zeros((n_loc, n_loc))
        Sy_e = np.zeros((n_loc, n_loc))

        for b in range(n1d):
            for a in range(n1d):
                ab = a + b * n1d
                w_ab = wdetJ[a, b]
                # xi-derivative contributions: d=b fixed, c varies
                for c in range(n1d):
                    cd = c + b * n1d
                    Sx_e[ab, cd] += w_ab * dxi_dx[a, b] * D[a, c]
                    Sy_e[ab, cd] += w_ab * dxi_dy[a, b] * D[a, c]
                # eta-derivative contributions: c=a fixed, d varies
                for d in range(n1d):
                    cd = a + d * n1d
                    Sx_e[ab, cd] += w_ab * deta_dx[a, b] * D[b, d]
                    Sy_e[ab, cd] += w_ab * deta_dy[a, b] * D[b, d]

        # Scatter element matrices into global COO
        for loc_a in range(n_loc):
            g_a = dof[loc_a]
            for loc_b in range(n_loc):
                g_b = dof[loc_b]
                if abs(Se[loc_a, loc_b]) > 1e-30:
                    s_r.append(g_a); s_c.append(g_b)
                    s_v.append(Se[loc_a, loc_b])
                if abs(Sx_e[loc_a, loc_b]) > 1e-30:
                    sx_r.append(g_a); sx_c.append(g_b)
                    sx_v.append(Sx_e[loc_a, loc_b])
                if abs(Sy_e[loc_a, loc_b]) > 1e-30:
                    sy_r.append(g_a); sy_c.append(g_b)
                    sy_v.append(Sy_e[loc_a, loc_b])

    S  = sparse.coo_matrix((s_v,  (s_r,  s_c)),  shape=(N, N)).tocsr()
    Sx = sparse.coo_matrix((sx_v, (sx_r, sx_c)), shape=(N, N)).tocsr()
    Sy = sparse.coo_matrix((sy_v, (sy_r, sy_c)), shape=(N, N)).tocsr()

    # Boundary mass
    B_diag = _assemble_boundary_mass(mesh)
    B_total = sparse.diags(B_diag)

    M_inv = 1.0 / M_diag

    return dict(
        M_diag=M_diag, M_inv=M_inv,
        S=S, Sx=Sx, Sy=Sy,
        B_total=B_total,
    )


def _assemble_boundary_mass(mesh):
    """Diagonal boundary mass via 1D GLL quadrature along boundary edges."""
    P = mesh.P
    n1d = P + 1
    w1d = mesh.w_gll
    N = mesh.N_dof

    B_diag = np.zeros(N)
    bnd_set = set(mesh._all_boundary.tolist())

    for e in range(mesh.N_el):
        corners = mesh.corner_nodes[mesh.corner_quads[e]]
        dof = mesh.elem_dof[e]

        edge_defs = [
            ([(i, 0) for i in range(n1d)], corners[0], corners[1]),       # edge 0: bottom
            ([(P, j) for j in range(n1d)], corners[1], corners[2]),       # edge 1: right
            ([(i, P) for i in range(n1d)], corners[3], corners[2]),       # edge 2: top
            ([(0, j) for j in range(n1d)], corners[0], corners[3]),       # edge 3: left
        ]

        for local_ij, c0, c1 in edge_defs:
            global_dofs = [dof[i + j * n1d] for i, j in local_ij]
            if not all(g in bnd_set for g in global_dofs):
                continue

            edge_len = np.linalg.norm(c1 - c0)
            jac_1d = edge_len / 2.0

            for k, g in enumerate(global_dofs):
                B_diag[g] += w1d[k] * jac_1d

    return B_diag
