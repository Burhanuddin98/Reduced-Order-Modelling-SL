"""
Unstructured SEM assembly for arbitrary 2D and extruded 3D geometries.

2D: Takes a quad mesh (from Gmsh) and produces the same operator dict as
``assemble_2d_operators`` so all existing FOM/ROM solvers work unchanged.

3D: Takes an extruded hex mesh (2D quads stacked in z) and produces the
same operator dict as ``assemble_3d_operators``.

Element-by-element assembly with isoparametric mapping:
  - GLL nodes placed on each element via tensor product on [-1,1]^d
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


# ===================================================================
# 3D Extruded Hex Mesh
# ===================================================================

def _trilinear_grad(xi, eta, zeta):
    """Gradient of 8-node hex shape functions: [dN/dxi, dN/deta, dN/dzeta].

    Node ordering (VTK hex):
        0=(-1,-1,-1), 1=(+1,-1,-1), 2=(+1,+1,-1), 3=(-1,+1,-1),
        4=(-1,-1,+1), 5=(+1,-1,+1), 6=(+1,+1,+1), 7=(-1,+1,+1)

    Returns (3, 8) array.
    """
    xm, xp = 1 - xi, 1 + xi
    em, ep = 1 - eta, 1 + eta
    zm, zp = 1 - zeta, 1 + zeta
    return 0.125 * np.array([
        # dN/dxi
        [-em*zm,  em*zm,  ep*zm, -ep*zm,
         -em*zp,  em*zp,  ep*zp, -ep*zp],
        # dN/deta
        [-xm*zm, -xp*zm,  xp*zm,  xm*zm,
         -xm*zp, -xp*zp,  xp*zp,  xm*zp],
        # dN/dzeta
        [-xm*em, -xp*em, -xp*ep, -xm*ep,
          xm*em,  xp*em,  xp*ep,  xm*ep],
    ])


class UnstructuredHexMesh3D:
    """
    Extruded hex SEM mesh for arbitrary 3D room geometries.

    Built from a 2D quad mesh extruded vertically. GLL nodes are placed
    as tensor products on [-1,1]^3 within each hex element.

    Parameters
    ----------
    mesh_3d_data : dict
        Output from ``extrude_quad_mesh``.
    nodes_2d : (N_corner_2d, 2) array
        Original 2D corner coordinates (from Gmsh).
    quads_2d : (N_el_2d, 4) int array
        Original 2D quad connectivity.
    P : int
        Polynomial order.
    """

    def __init__(self, mesh_3d_data, nodes_2d, quads_2d, P):
        self.P = P
        self.ndim = 3

        self.corner_nodes = np.asarray(mesh_3d_data['nodes_3d'], dtype=float)
        self.corner_hexes = np.asarray(mesh_3d_data['hexes'], dtype=int)
        self.N_el = len(self.corner_hexes)
        self.Lz = mesh_3d_data['Lz']
        self.n_layers = mesh_3d_data['n_layers']

        self.xi_gll, self.w_gll = gll_points_weights(P)
        self.D_gll = gll_derivative_matrix(self.xi_gll)
        n1d = P + 1
        self.n_loc = n1d ** 3  # local DOFs per hex element

        self._nodes_2d = np.asarray(nodes_2d, dtype=float)
        self._quads_2d = np.asarray(quads_2d, dtype=int)
        self._N2 = len(nodes_2d)

        self._build_global_numbering()
        self._build_coordinates()
        self._identify_boundaries(mesh_3d_data['boundary'])

    def _build_global_numbering(self):
        """Assign global DOF indices for extruded hex mesh.

        Exploits extrusion structure: the 2D quad mesh has a well-defined
        node numbering, and each z-layer replicates it. Shared faces between
        vertically adjacent hex elements share all (P+1)^2 DOFs.
        """
        P = self.P
        n1d = P + 1
        N_el = self.N_el
        N_el_2d = len(self._quads_2d)
        n_layers = self.n_layers
        N2 = self._N2

        # First build 2D high-order DOF numbering using a temporary 2D mesh
        # to reuse the existing logic
        temp_2d = UnstructuredQuadMesh2D.__new__(UnstructuredQuadMesh2D)
        temp_2d.P = P
        temp_2d.corner_nodes = self._nodes_2d
        temp_2d.corner_quads = self._quads_2d
        temp_2d.N_el = N_el_2d
        temp_2d.xi_gll = self.xi_gll
        temp_2d.w_gll = self.w_gll
        temp_2d.D_gll = self.D_gll
        temp_2d.n_loc = n1d ** 2
        temp_2d._build_global_numbering()

        N_dof_2d = temp_2d.N_dof
        elem_dof_2d = temp_2d.elem_dof  # (N_el_2d, n1d^2)

        # 3D numbering: for each z-layer k, the 2D DOFs are offset by
        # k * N_dof_2d_layer. Between z-GLL nodes within a layer,
        # new DOFs are interior to that layer.

        # z-direction: within each hex layer k (between z-levels k and k+1),
        # there are n1d GLL nodes in z. The bottom (kz=0) and top (kz=P)
        # are shared with adjacent layers.

        # Total z-levels of GLL nodes: n_layers * P + 1
        N_z_gll = n_layers * P + 1

        # Global DOF: (2d_dof, z_gll_level) -> 2d_dof + z_gll * N_dof_2d
        self.N_dof = N_dof_2d * N_z_gll

        # Build elem_dof for each 3D element
        self.elem_dof = np.zeros((N_el, self.n_loc), dtype=int)

        for layer in range(n_layers):
            for e2d in range(N_el_2d):
                e3d = layer * N_el_2d + e2d
                dof_2d = elem_dof_2d[e2d]  # (n1d^2,)

                for kz in range(n1d):
                    gz = layer * P + kz  # global z-GLL index
                    z_offset = gz * N_dof_2d
                    for loc_2d in range(n1d ** 2):
                        loc_3d = loc_2d + kz * n1d * n1d
                        self.elem_dof[e3d, loc_3d] = dof_2d[loc_2d] + z_offset

        self._N_dof_2d = N_dof_2d
        self._elem_dof_2d = elem_dof_2d

    def _build_coordinates(self):
        """Compute (x,y,z) for every global DOF."""
        P = self.P
        n1d = P + 1
        xi = self.xi_gll
        N_el_2d = len(self._quads_2d)
        N_dof_2d = self._N_dof_2d
        n_layers = self.n_layers
        hz = self.Lz / n_layers

        # Build 2D coordinates
        coords_2d = np.zeros((N_dof_2d, 2))
        assigned = np.zeros(N_dof_2d, dtype=bool)
        for e in range(N_el_2d):
            corners = self._nodes_2d[self._quads_2d[e]]
            dof = self._elem_dof_2d[e]
            for j in range(n1d):
                for i in range(n1d):
                    g = dof[i + j * n1d]
                    if not assigned[g]:
                        N = _bilinear_shape(xi[i], xi[j])
                        coords_2d[g] = N @ corners
                        assigned[g] = True

        # Replicate to 3D
        N_z_gll = n_layers * P + 1
        self.x = np.zeros(self.N_dof)
        self.y = np.zeros(self.N_dof)
        self.z = np.zeros(self.N_dof)

        for gz in range(N_z_gll):
            layer = min(gz // P, n_layers - 1)
            kz_local = gz - layer * P
            z_phys = layer * hz + (xi[kz_local] + 1.0) / 2.0 * hz

            offset = gz * N_dof_2d
            self.x[offset:offset + N_dof_2d] = coords_2d[:, 0]
            self.y[offset:offset + N_dof_2d] = coords_2d[:, 1]
            self.z[offset:offset + N_dof_2d] = z_phys

        self._coords_built = True

    def _ensure_coords(self):
        """Compatibility with BoxMesh3D interface."""
        pass  # coords already built in __init__

    def _identify_boundaries(self, boundary_data):
        """Build boundary node sets from extruded face data."""
        P = self.P
        n1d = P + 1
        N_el_2d = len(self._quads_2d)
        N_dof_2d = self._N_dof_2d
        n_layers = self.n_layers
        N2 = self._N2

        self._boundary_nodes_per_label = {}
        all_bnd = set()

        # Floor (z=0): all 2D DOFs at gz=0
        floor_dofs = set(range(N_dof_2d))
        self._boundary_nodes_per_label['floor'] = np.array(
            sorted(floor_dofs), dtype=int)
        all_bnd.update(floor_dofs)

        # Ceiling (z=Lz): all 2D DOFs at gz = N_z_gll - 1
        gz_top = n_layers * P
        ceil_dofs = set(range(gz_top * N_dof_2d, (gz_top + 1) * N_dof_2d))
        self._boundary_nodes_per_label['ceiling'] = np.array(
            sorted(ceil_dofs), dtype=int)
        all_bnd.update(ceil_dofs)

        # Vertical walls: need to identify 2D boundary nodes, then
        # replicate across all z-levels
        # Build a temporary 2D mesh to get boundary info
        for label, faces in boundary_data.items():
            if label in ('floor', 'ceiling'):
                continue

            # Extract the 2D corner edge pairs from the face data
            # Each face is (n0_bot, n1_bot, n1_top, n0_top) for one z-layer
            # The 2D edge is (n0 % N2, n1 % N2) — take from layer 0
            edges_2d = set()
            for face in faces:
                n0, n1 = face[0], face[1]
                edges_2d.add((n0 % N2, n1 % N2))

            # Find which 2D GLL DOFs lie on these edges
            wall_2d_dofs = set()
            for cn0, cn1 in edges_2d:
                ekey = (min(cn0, cn1), max(cn0, cn1))
                for e in range(N_el_2d):
                    q = self._quads_2d[e]
                    for eidx, (c0l, c1l) in enumerate(
                            [(0,1),(1,2),(2,3),(3,0)]):
                        ec0, ec1 = int(q[c0l]), int(q[c1l])
                        if (min(ec0, ec1), max(ec0, ec1)) == ekey:
                            dof = self._elem_dof_2d[e]
                            if eidx == 0:
                                gdofs = [dof[i] for i in range(n1d)]
                            elif eidx == 1:
                                gdofs = [dof[P + j*n1d] for j in range(n1d)]
                            elif eidx == 2:
                                gdofs = [dof[i + P*n1d] for i in range(n1d)]
                            else:
                                gdofs = [dof[j*n1d] for j in range(n1d)]
                            wall_2d_dofs.update(gdofs)
                            break

            # Replicate across all z-levels
            N_z_gll = n_layers * P + 1
            wall_3d_dofs = set()
            for gz in range(N_z_gll):
                for d2 in wall_2d_dofs:
                    wall_3d_dofs.add(d2 + gz * N_dof_2d)

            self._boundary_nodes_per_label[label] = np.array(
                sorted(wall_3d_dofs), dtype=int)
            all_bnd.update(wall_3d_dofs)

        self._all_boundary = np.array(sorted(all_bnd), dtype=int)

    def all_boundary_nodes(self):
        return self._all_boundary

    def boundary_nodes(self, label):
        return self._boundary_nodes_per_label[label]

    def nearest_node(self, rx, ry, rz):
        return int(np.argmin((self.x - rx)**2 + (self.y - ry)**2
                             + (self.z - rz)**2))


# ===================================================================
# 3D hex operator assembly
# ===================================================================

def assemble_unstructured_3d_operators(mesh):
    """
    Element-by-element SEM assembly for extruded 3D hex meshes.

    Returns dict compatible with 3D FOM/ROM solvers:
        M_diag, M_inv, S, Sx, Sy, Sz, B_total
    """
    P = mesh.P
    n1d = P + 1
    n_loc = mesh.n_loc  # n1d^3
    N = mesh.N_dof
    N_el = mesh.N_el
    xi = mesh.xi_gll
    w = mesh.w_gll
    D = mesh.D_gll

    print(f"    3D unstructured assembly: N={N}, {N_el} elements...",
          end='', flush=True)

    M_diag = np.zeros(N)
    s_r, s_c, s_v = [], [], []

    for e in range(N_el):
        corners = mesh.corner_nodes[mesh.corner_hexes[e]]  # (8, 3)
        dof = mesh.elem_dof[e]  # (n_loc,)

        # Local index: a = i + j*n1d + k*n1d^2
        # where i=xi, j=eta, k=zeta direction

        # Precompute Jacobian-derived quantities at all GLL points
        n1d2 = n1d * n1d
        wdetJ_all = np.zeros(n_loc)
        # Metric tensor G = J^{-T} J^{-1}, stored as g[alpha,beta] at each point
        # We need 6 unique components: g11, g12, g13, g22, g23, g33
        g11 = np.zeros(n_loc)
        g12 = np.zeros(n_loc)
        g13 = np.zeros(n_loc)
        g22 = np.zeros(n_loc)
        g23 = np.zeros(n_loc)
        g33 = np.zeros(n_loc)

        # Also need Jinv for gradient matrices
        Jinv_all = np.zeros((n_loc, 3, 3))

        for k in range(n1d):
            for j in range(n1d):
                for i in range(n1d):
                    loc = i + j * n1d + k * n1d2
                    dN = _trilinear_grad(xi[i], xi[j], xi[k])  # (3, 8)
                    J = dN @ corners  # (3, 3)

                    detJ = np.linalg.det(J)
                    Ji = np.linalg.inv(J)  # (3, 3): Ji[alpha, r]
                    w3 = w[i] * w[j] * w[k] * abs(detJ)

                    wdetJ_all[loc] = w3
                    M_diag[dof[loc]] += w3
                    Jinv_all[loc] = Ji

                    # G = Ji^T @ Ji -> g[alpha, beta] = sum_r Ji[r,alpha]*Ji[r,beta]
                    # But Ji here is dxi_alpha/dx_r  (ref-to-phys inverse)
                    # g_ab = sum_r (dxi_a/dx_r)(dxi_b/dx_r)
                    for a_idx in range(3):
                        for b_idx in range(a_idx, 3):
                            val = np.dot(Ji[a_idx, :], Ji[b_idx, :])
                            if a_idx == 0 and b_idx == 0: g11[loc] = val
                            elif a_idx == 0 and b_idx == 1: g12[loc] = val
                            elif a_idx == 0 and b_idx == 2: g13[loc] = val
                            elif a_idx == 1 and b_idx == 1: g22[loc] = val
                            elif a_idx == 1 and b_idx == 2: g23[loc] = val
                            elif a_idx == 2 and b_idx == 2: g33[loc] = val

        # Build element stiffness matrix Se (n_loc x n_loc)
        # Basis: phi_{(a,b,c)}(xi,eta,zeta) = l_a(xi)*l_b(eta)*l_c(zeta)
        #
        # Stiffness integral at GLL point q=(i,j,k):
        # grad phi_{(a,b,c)} has ref components:
        #   dxi:   D[i,a]*d(j,b)*d(k,c)
        #   deta:  d(i,a)*D[j,b]*d(k,c)
        #   dzeta: d(i,a)*d(j,b)*D[k,c]
        #
        # Se[(a,b,c), (a2,b2,c2)] = sum_q wdetJ * (grad1 . G . grad2)
        #
        # Non-zero blocks:
        # (1) b==b2, c==c2 (xi-xi): sum_i wdetJ[i,b,c]*g11*D[i,a]*D[i,a2]
        # (2) a==a2, c==c2 (eta-eta): sum_j wdetJ[a,j,c]*g22*D[j,b]*D[j,b2]
        # (3) a==a2, b==b2 (zeta-zeta): sum_k wdetJ[a,b,k]*g33*D[k,c]*D[k,c2]
        # (4-6) mixed terms via g12, g13, g23

        Se = np.zeros((n_loc, n_loc))

        def loc3(i, j, k):
            return i + j * n1d + k * n1d2

        # Diagonal blocks: xi-xi
        for c in range(n1d):
            for b in range(n1d):
                W = np.array([wdetJ_all[loc3(i, b, c)] * g11[loc3(i, b, c)]
                              for i in range(n1d)])
                block = D.T @ np.diag(W) @ D
                for a in range(n1d):
                    for a2 in range(n1d):
                        Se[loc3(a, b, c), loc3(a2, b, c)] += block[a, a2]

        # eta-eta
        for c in range(n1d):
            for a in range(n1d):
                W = np.array([wdetJ_all[loc3(a, j, c)] * g22[loc3(a, j, c)]
                              for j in range(n1d)])
                block = D.T @ np.diag(W) @ D
                for b in range(n1d):
                    for b2 in range(n1d):
                        Se[loc3(a, b, c), loc3(a, b2, c)] += block[b, b2]

        # zeta-zeta
        for b in range(n1d):
            for a in range(n1d):
                W = np.array([wdetJ_all[loc3(a, b, k)] * g33[loc3(a, b, k)]
                              for k in range(n1d)])
                block = D.T @ np.diag(W) @ D
                for c in range(n1d):
                    for c2 in range(n1d):
                        Se[loc3(a, b, c), loc3(a, b, c2)] += block[c, c2]

        # Mixed xi-eta (g12): q=(a2,b,c) and q=(a,b2,c)
        for c in range(n1d):
            for a in range(n1d):
                for b in range(n1d):
                    for a2 in range(n1d):
                        for b2 in range(n1d):
                            v = (wdetJ_all[loc3(a2, b, c)]
                                 * g12[loc3(a2, b, c)]
                                 * D[a2, a] * D[b, b2])
                            v += (wdetJ_all[loc3(a, b2, c)]
                                  * g12[loc3(a, b2, c)]
                                  * D[a, a2] * D[b2, b])
                            Se[loc3(a, b, c), loc3(a2, b2, c)] += v

        # Mixed xi-zeta (g13): q=(a2,b,c) and q=(a,b,c2)
        for b in range(n1d):
            for a in range(n1d):
                for c in range(n1d):
                    for a2 in range(n1d):
                        for c2 in range(n1d):
                            v = (wdetJ_all[loc3(a2, b, c)]
                                 * g13[loc3(a2, b, c)]
                                 * D[a2, a] * D[c, c2])
                            v += (wdetJ_all[loc3(a, b, c2)]
                                  * g13[loc3(a, b, c2)]
                                  * D[a, a2] * D[c2, c])
                            Se[loc3(a, b, c), loc3(a2, b, c2)] += v

        # Mixed eta-zeta (g23): q=(a,b2,c) and q=(a,b,c2)
        for a in range(n1d):
            for b in range(n1d):
                for c in range(n1d):
                    for b2 in range(n1d):
                        for c2 in range(n1d):
                            v = (wdetJ_all[loc3(a, b2, c)]
                                 * g23[loc3(a, b2, c)]
                                 * D[b2, b] * D[c, c2])
                            v += (wdetJ_all[loc3(a, b, c2)]
                                  * g23[loc3(a, b, c2)]
                                  * D[b, b2] * D[c2, c])
                            Se[loc3(a, b, c), loc3(a, b2, c2)] += v

        # Scatter to global COO
        for la in range(n_loc):
            ga = dof[la]
            for lb in range(n_loc):
                v = Se[la, lb]
                if abs(v) > 1e-30:
                    s_r.append(ga)
                    s_c.append(dof[lb])
                    s_v.append(v)

        if (e + 1) % max(1, N_el // 5) == 0:
            print(f" {e+1}/{N_el}", end='', flush=True)

    print(" assembling...", end='', flush=True)
    S = sparse.coo_matrix((s_v, (s_r, s_c)), shape=(N, N)).tocsr()

    # Boundary mass (2D quadrature on boundary faces)
    print(" boundary...", end='', flush=True)
    B_diag = _assemble_3d_boundary_mass(mesh)
    B_total = sparse.diags(B_diag)

    M_inv = 1.0 / M_diag
    print(" done.")

    return dict(
        M_diag=M_diag, M_inv=M_inv,
        S=S, B_total=B_total,
    )


def _assemble_3d_boundary_mass(mesh):
    """Diagonal boundary mass for 3D extruded mesh.

    Floor/ceiling: 2D GLL quadrature on each quad face.
    Vertical walls: 2D GLL quadrature on each extruded edge face.
    """
    P = mesh.P
    n1d = P + 1
    w = mesh.w_gll
    xi = mesh.xi_gll
    N = mesh.N_dof
    N_dof_2d = mesh._N_dof_2d
    N_el_2d = len(mesh._quads_2d)
    n_layers = mesh.n_layers
    hz = mesh.Lz / n_layers

    B_diag = np.zeros(N)

    # Floor (gz=0) and ceiling (gz=n_layers*P): 2D quadrature on each quad
    for e in range(N_el_2d):
        corners = mesh._nodes_2d[mesh._quads_2d[e]]
        dof_2d = mesh._elem_dof_2d[e]

        for j in range(n1d):
            for i in range(n1d):
                dN = _bilinear_grad(xi[i], xi[j])
                J_raw = dN @ corners
                detJ2 = abs(J_raw[0, 0] * J_raw[1, 1]
                            - J_raw[0, 1] * J_raw[1, 0])
                w2 = w[i] * w[j] * detJ2
                d2 = dof_2d[i + j * n1d]

                # Floor
                B_diag[d2] += w2
                # Ceiling
                gz_top = n_layers * P
                B_diag[d2 + gz_top * N_dof_2d] += w2

    # Vertical walls: for each boundary edge, quadrature on the
    # (edge_param, z) face. Edge Jacobian * hz/2 for z direction.
    bnd_2d_set = set()
    for label in mesh._boundary_nodes_per_label:
        if label in ('floor', 'ceiling'):
            continue
        # Get 2D DOFs for this wall (at gz=0)
        dofs_3d = mesh._boundary_nodes_per_label[label]
        for d in dofs_3d:
            if d < N_dof_2d:
                bnd_2d_set.add(d)

    # Iterate over elements and their edges
    bnd_2d_set_frozen = frozenset(bnd_2d_set)
    for e in range(N_el_2d):
        corners = mesh._nodes_2d[mesh._quads_2d[e]]
        dof_2d = mesh._elem_dof_2d[e]

        edge_defs = [
            ([(i, 0) for i in range(n1d)], corners[0], corners[1]),
            ([(P, j) for j in range(n1d)], corners[1], corners[2]),
            ([(i, P) for i in range(n1d)], corners[3], corners[2]),
            ([(0, j) for j in range(n1d)], corners[0], corners[3]),
        ]

        for local_ij, c0, c1 in edge_defs:
            gdofs_2d = [dof_2d[i + j * n1d] for i, j in local_ij]
            if not all(g in bnd_2d_set_frozen for g in gdofs_2d):
                continue

            edge_len = np.linalg.norm(c1 - c0)
            jac_edge = edge_len / 2.0
            jac_z = hz / 2.0

            # Quadrature on (edge_param, z) face
            for layer in range(n_layers):
                for kz in range(n1d):
                    gz = layer * P + kz
                    z_offset = gz * N_dof_2d
                    for m, g2d in enumerate(gdofs_2d):
                        B_diag[g2d + z_offset] += (w[m] * jac_edge
                                                   * w[kz] * jac_z)

    return B_diag
