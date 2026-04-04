"""Unstructured tetrahedral mesh with per-surface boundary mass."""

import numpy as np
from scipy import sparse


class TetMesh:
    """P1 tetrahedral mesh with named boundary surfaces.

    Parameters
    ----------
    nodes : (N_nodes, 3) array — vertex coordinates
    tets : (N_tets, 4) array — tet connectivity (node indices)
    boundary : dict — {label: (N_faces, 3) array of triangle faces}
    """

    def __init__(self, nodes, tets, boundary):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.tets = np.asarray(tets, dtype=int)
        self.boundary = boundary  # {label: (n_faces, 3)}
        self.N_dof = len(self.nodes)
        self.x = self.nodes[:, 0]
        self.y = self.nodes[:, 1]
        self.z = self.nodes[:, 2]
        self.ndim = 3

    def nearest_node(self, rx, ry, rz):
        return int(np.argmin(
            (self.x - rx)**2 + (self.y - ry)**2 + (self.z - rz)**2))

    def all_boundary_nodes(self):
        s = set()
        for faces in self.boundary.values():
            s.update(faces.ravel().tolist())
        return np.array(sorted(s))


def _tet_volume(v0, v1, v2, v3):
    """Signed volume of tetrahedron."""
    return np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0


def _tri_area(v0, v1, v2):
    """Area of triangle."""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def assemble_tet(mesh):
    """Assemble P1 tet FEM operators.

    Returns dict with:
        M_diag : lumped mass (N,)
        M_inv : inverse lumped mass (N,)
        S : stiffness matrix (N, N) sparse
        B_total : total boundary mass diagonal (N, N) sparse
        B_labels : per-surface boundary mass {label: ndarray (N,)}
    """
    N = mesh.N_dof
    nodes = mesh.nodes
    tets = mesh.tets

    # ── Stiffness + Mass ─────────────────────────────────────
    # P1 tet: gradients are constant per element
    # grad phi_i = (1/6V) * n_i  where n_i is the inward face normal
    rows, cols, vals_S = [], [], []
    M_diag = np.zeros(N)

    for tet in tets:
        v = nodes[tet]  # (4, 3)
        vol = abs(_tet_volume(v[0], v[1], v[2], v[3]))
        if vol < 1e-30:
            continue

        # Mass: lumped = vol/4 per node
        for i in range(4):
            M_diag[tet[i]] += vol / 4.0

        # Stiffness: K_ij = vol * grad_phi_i . grad_phi_j
        # For P1 tet, grad_phi is computed from the shape function gradients
        # Using the Jacobian approach
        J = np.column_stack([v[1]-v[0], v[2]-v[0], v[3]-v[0]])  # (3,3)
        detJ = abs(np.linalg.det(J))
        if detJ < 1e-30:
            continue
        Jinv = np.linalg.inv(J)

        # Reference gradients of P1 basis:
        # phi0 = 1-xi-eta-zeta, phi1=xi, phi2=eta, phi3=zeta
        grad_ref = np.array([
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float64)  # (4, 3)

        # Physical gradients: grad_phys = grad_ref @ Jinv
        grad_phys = grad_ref @ Jinv  # (4, 3)

        # K_local[i,j] = (detJ/6) * grad_phys[i] . grad_phys[j]
        K_local = (detJ / 6.0) * (grad_phys @ grad_phys.T)

        for i in range(4):
            for j in range(4):
                rows.append(tet[i])
                cols.append(tet[j])
                vals_S.append(K_local[i, j])

    S = sparse.csr_matrix((vals_S, (rows, cols)), shape=(N, N))
    S = (S + S.T) / 2  # symmetrize

    # ── Boundary mass (per surface) ──────────────────────────
    B_labels = {}
    B_total_diag = np.zeros(N)

    for label, faces in mesh.boundary.items():
        B_diag = np.zeros(N)
        for face in faces:
            v = nodes[face]  # (3, 3)
            area = _tri_area(v[0], v[1], v[2])
            # Lumped: area/3 per node
            for i in range(3):
                B_diag[face[i]] += area / 3.0
        B_labels[label] = B_diag
        B_total_diag += B_diag

    M_inv = np.zeros(N)
    nonzero = M_diag > 1e-30
    M_inv[nonzero] = 1.0 / M_diag[nonzero]

    return dict(
        M_diag=M_diag,
        M_inv=M_inv,
        S=S,
        B_total=sparse.diags(B_total_diag),
        B_labels=B_labels,
    )
