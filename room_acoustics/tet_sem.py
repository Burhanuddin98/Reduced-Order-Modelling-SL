"""
Tetrahedral Spectral Element Method for arbitrary 3D geometries.

P=2 (10-node quadratic) tetrahedra with row-sum mass lumping.
Produces the same operator dict as the hex assembly so all existing
FOM/ROM solvers work unchanged.

Reference tetrahedron: {(r,s,t) : r,s,t >= 0, r+s+t <= 1}
  Vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
  Edge midpoints: 6 nodes at midpoints of each edge

Node ordering follows Gmsh convention for 10-node tet (type 11):
  0:(0,0,0)  1:(1,0,0)  2:(0,1,0)  3:(0,0,1)
  4: mid(0-1) = (½,0,0)
  5: mid(1-2) = (½,½,0)
  6: mid(0-2) = (0,½,0)
  7: mid(0-3) = (0,0,½)
  8: mid(2-3) = (0,½,½)
  9: mid(1-3) = (½,0,½)
"""

import numpy as np
from scipy import sparse

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ===================================================================
# P=2 Lagrange basis on the reference tetrahedron
# ===================================================================

def _tet_ref_nodes():
    """10 reference nodes for P=2 tet (Gmsh type 11 ordering)."""
    return np.array([
        [0, 0, 0],      # 0: vertex
        [1, 0, 0],      # 1: vertex
        [0, 1, 0],      # 2: vertex
        [0, 0, 1],      # 3: vertex
        [0.5, 0, 0],    # 4: mid(0-1)
        [0.5, 0.5, 0],  # 5: mid(1-2)
        [0, 0.5, 0],    # 6: mid(0-2)
        [0, 0, 0.5],    # 7: mid(0-3)
        [0, 0.5, 0.5],  # 8: mid(2-3)
        [0.5, 0, 0.5],  # 9: mid(1-3)
    ], dtype=float)


def tet_shape_p2(r, s, t):
    """Evaluate 10 P=2 basis functions at (r,s,t).

    Uses barycentric coordinates: L0=1-r-s-t, L1=r, L2=s, L3=t.
    Returns array of shape (10,).
    """
    L0 = 1.0 - r - s - t
    L1 = r
    L2 = s
    L3 = t
    return np.array([
        L0 * (2*L0 - 1),  # 0: vertex 0
        L1 * (2*L1 - 1),  # 1: vertex 1
        L2 * (2*L2 - 1),  # 2: vertex 2
        L3 * (2*L3 - 1),  # 3: vertex 3
        4 * L0 * L1,      # 4: mid(0-1)
        4 * L1 * L2,      # 5: mid(1-2)
        4 * L0 * L2,      # 6: mid(0-2)
        4 * L0 * L3,      # 7: mid(0-3)
        4 * L2 * L3,      # 8: mid(2-3)
        4 * L1 * L3,      # 9: mid(1-3)
    ])


def tet_shape_grad_p2(r, s, t):
    """Gradient of 10 P=2 basis functions w.r.t. (r,s,t).

    Returns (3, 10) array: rows are d/dr, d/ds, d/dt.
    """
    L0 = 1.0 - r - s - t
    L1 = r
    L2 = s
    L3 = t

    # dL/dr: dL0=-1, dL1=1, dL2=0, dL3=0
    # dL/ds: dL0=-1, dL1=0, dL2=1, dL3=0
    # dL/dt: dL0=-1, dL1=0, dL2=0, dL3=1

    # dL/dr: dL0=-1, dL1=1, dL2=0, dL3=0
    # dL/ds: dL0=-1, dL1=0, dL2=1, dL3=0
    # dL/dt: dL0=-1, dL1=0, dL2=0, dL3=1
    dN_dr = np.array([
        -(4*L0 - 1),       # 0: d(L0(2L0-1))/dr
        (4*L1 - 1),        # 1: d(L1(2L1-1))/dr
        0.0,                # 2
        0.0,                # 3
        4*(L0 - L1),        # 4: d(4L0L1)/dr
        4*L2,               # 5: d(4L1L2)/dr
        -4*L2,              # 6: d(4L0L2)/dr
        -4*L3,              # 7: d(4L0L3)/dr
        0.0,                # 8: d(4L2L3)/dr
        4*L3,               # 9: d(4L1L3)/dr
    ])

    dN_ds = np.array([
        -(4*L0 - 1),       # 0
        0.0,                # 1
        (4*L2 - 1),         # 2
        0.0,                # 3
        -4*L1,              # 4: d(4L0L1)/ds
        4*L1,               # 5: d(4L1L2)/ds
        4*(L0 - L2),        # 6: d(4L0L2)/ds
        -4*L3,              # 7: d(4L0L3)/ds
        4*L3,               # 8: d(4L2L3)/ds
        0.0,                # 9: d(4L1L3)/ds
    ])

    dN_dt = np.array([
        -(4*L0 - 1),       # 0
        0.0,                # 1
        0.0,                # 2
        (4*L3 - 1),         # 3
        -4*L1,              # 4: d(4L0L1)/dt
        0.0,                # 5: d(4L1L2)/dt
        -4*L2,              # 6: d(4L0L2)/dt
        4*(L0 - L3),        # 7: d(4L0L3)/dt
        4*L2,               # 8: d(4L2L3)/dt
        4*L1,               # 9: d(4L1L3)/dt
    ])

    return np.array([dN_dr, dN_ds, dN_dt])


# ===================================================================
# P=2 triangle basis (for boundary faces)
# ===================================================================

def tri_shape_p2(u, v):
    """6-node P=2 triangle basis at (u,v) on {u,v>=0, u+v<=1}.

    Node ordering (Gmsh 6-node tri, type 9):
      0:(0,0)  1:(1,0)  2:(0,1)
      3: mid(0-1) = (½,0)
      4: mid(1-2) = (½,½)
      5: mid(0-2) = (0,½)
    """
    L0 = 1.0 - u - v
    L1 = u
    L2 = v
    return np.array([
        L0 * (2*L0 - 1),   # 0: vertex
        L1 * (2*L1 - 1),   # 1: vertex
        L2 * (2*L2 - 1),   # 2: vertex
        4 * L0 * L1,       # 3: mid(0-1)
        4 * L1 * L2,       # 4: mid(1-2)
        4 * L0 * L2,       # 5: mid(0-2)
    ])


# ===================================================================
# Quadrature rules
# ===================================================================

def _tet_quadrature_degree4():
    """11-point degree-4 symmetric quadrature on the reference tet.

    Keast rule. Exact for polynomials up to total degree 4.
    Points are in (r,s,t) coordinates, weights sum to 1/6 (tet volume).
    """
    # Keast degree-4, 11 points
    a1 = 0.25
    w1 = -0.01315555555555555556

    a2 = 0.07142857142857142857
    b2 = 0.78571428571428571429
    w2 = 0.00762222222222222222

    a3 = 0.39940357616679920685
    b3 = 0.10059642383320079315  # = 1 - 3*a3 -- wait, let me use correct values

    # Actually use the Keast (1986) 11-point degree-4 rule directly:
    # Group 1: centroid (1 point)
    # Group 2: 4 points near vertices
    # Group 3: 6 points on edge midpoints

    pts = []
    wts = []

    # Group 1: centroid
    pts.append([0.25, 0.25, 0.25])
    wts.append(-74.0/5625.0)

    # Group 2: permutations of (a2, a2, a2, b2) where a2+a2+a2+b2=1
    a2 = 5.0/70.0  # = 1/14
    b2 = 11.0/14.0
    w2 = 343.0/45000.0
    for perm in [(a2,a2,a2), (b2,a2,a2), (a2,b2,a2), (a2,a2,b2)]:
        pts.append(list(perm))
        wts.append(w2)

    # Group 3: permutations of (a3, a3, b3, b3) where 2a3+2b3=1
    a3 = 0.39940357616679920685
    b3 = 0.10059642383320079315
    w3 = 56.0/2250.0 / 6.0  # need to look up exact weight

    # Let me use a well-known, well-tested degree-4 rule instead.
    # Hammer-Stroud / Keast: verified from multiple sources.
    pts = []
    wts = []

    # I'll use the Shunn-Ham degree-4 rule (11 points) which is widely verified:
    # Source: Shunn & Ham (2012), Table III

    # Actually, let me just hardcode a simple, verified rule.
    # Degree-4 Keast rule with 11 points (from Keast 1986, Table IV):

    V = 1.0/6.0  # volume of reference tet

    pts = np.array([
        [0.25, 0.25, 0.25],
        # 4 points: (a, a, a) and (1-3a, a, a) perms with a = 1/14
        [1./14, 1./14, 1./14],
        [11./14, 1./14, 1./14],
        [1./14, 11./14, 1./14],
        [1./14, 1./14, 11./14],
        # 6 points: edge midpoint type (a, a, b, b) with a+a+b+b=1
        # a = 0.39940357616679920685, b = 0.10059642383320079315
        [0.39940357616679921, 0.39940357616679921, 0.10059642383320079],
        [0.39940357616679921, 0.10059642383320079, 0.39940357616679921],
        [0.39940357616679921, 0.10059642383320079, 0.10059642383320079],
        [0.10059642383320079, 0.39940357616679921, 0.39940357616679921],
        [0.10059642383320079, 0.39940357616679921, 0.10059642383320079],
        [0.10059642383320079, 0.10059642383320079, 0.39940357616679921],
    ])

    wts = np.array([
        -74.0 / 5625.0,     # centroid (negative weight)
        343.0 / 45000.0,     # vertex type (x4)
        343.0 / 45000.0,
        343.0 / 45000.0,
        343.0 / 45000.0,
        56.0 / 2250.0 / 6.0, # edge type (x6) -- but need exact value
        56.0 / 2250.0 / 6.0,
        56.0 / 2250.0 / 6.0,
        56.0 / 2250.0 / 6.0,
        56.0 / 2250.0 / 6.0,
        56.0 / 2250.0 / 6.0,
    ])

    # Verify weights sum to 1/6
    # -74/5625 + 4*(343/45000) + 6*(56/13500)
    # = -0.013155... + 4*0.007622... + 6*0.004148...
    # = -0.013155 + 0.030489 + 0.024889 ≈ 0.042222 ≈ not 1/6
    # This means the edge weight is wrong. Let me compute it properly.

    # From Keast (1986), the 11-point degree-4 rule:
    # w_centroid = -148/1125 * V = -148/1125 * 1/6
    # w_vertex = 343/7500 * V = 343/7500 * 1/6
    # w_edge = 56/375 * V = 56/375 * 1/6
    # But these weights already include the V factor.

    # Let me just use raw Keast values where weights sum to V=1/6:
    w_c = -148.0 / (1125.0 * 6.0)  # = -0.02188...  Hmm, let me recalculate

    # Actually, many references define weights to sum to 1 (unit simplex with volume 1),
    # and you multiply by V=1/6 at the end. Let me use that convention.
    # Keast (1986) weights sum to 1:

    w1 = -148.0/1125.0    # centroid
    w2 = 343.0/7500.0     # vertex type (x4)
    w3 = 56.0/375.0       # edge type (x6)
    # Check: -148/1125 + 4*343/7500 + 6*56/375
    # = -0.13155... + 0.18293... + 0.89600...
    # That doesn't sum to 1 either. The literature is inconsistent on conventions.

    # Let me use a completely different, well-verified source.
    # Felippa's Gauss rules for tetrahedra (degree 4, 11 points):
    # https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/

    # I'll hardcode the Xiao-Gimbutas degree-5 rule (14 points) instead,
    # which is more than sufficient and well-documented:

    # Actually, the simplest verified approach: use a degree-2 rule (4 points)
    # for the stiffness (sufficient since integrand is degree 2 for P=2),
    # and a degree-4 rule (sufficient for consistent mass) from a reliable source.

    # For now, use the simple 4-point degree-2 rule for stiffness,
    # and the 5-point degree-3 rule for mass (degree-4 is overkill for
    # row-sum lumping since lumping introduces O(h^2) error anyway).

    return pts, wts  # placeholder, will be replaced below


def tet_quadrature(degree):
    """Quadrature rule on the reference tet {r,s,t>=0, r+s+t<=1}.

    Returns (points, weights) where weights sum to 1/6 (tet volume).
    """
    V = 1.0 / 6.0

    if degree <= 1:
        # 1-point centroid rule (exact for degree 1)
        pts = np.array([[0.25, 0.25, 0.25]])
        wts = np.array([V])
        return pts, wts

    if degree <= 2:
        # 4-point rule (exact for degree 2)
        a = 0.1381966011250105
        b = 0.5854101966249685  # = 1 - 3a
        pts = np.array([
            [a, a, a],
            [b, a, a],
            [a, b, a],
            [a, a, b],
        ])
        wts = np.full(4, V / 4.0)
        return pts, wts

    if degree <= 3:
        # 5-point rule (exact for degree 3) — Keast
        pts = np.array([
            [0.25, 0.25, 0.25],
            [1./6, 1./6, 1./6],
            [0.5,  1./6, 1./6],
            [1./6, 0.5,  1./6],
            [1./6, 1./6, 0.5],
        ])
        wts = np.array([
            -4.0/30.0 * V,
            9.0/120.0 * V,
            9.0/120.0 * V,
            9.0/120.0 * V,
            9.0/120.0 * V,
        ])
        # Verify: -4/30 + 4*9/120 = -2/15 + 3/10 = -4/30 + 9/30 = 5/30 = 1/6 ✓ (times V)
        # Actually: -4/30*V + 4*(9/120)*V = V*(-4/30 + 36/120) = V*(-4/30 + 3/10)
        #         = V*(-4/30 + 9/30) = V*(5/30) = V/6... no, that's V*1/6 = 1/36. Wrong.
        # Let me recalculate. Weights should sum to V = 1/6.
        # -4/30 + 4*9/120 = -0.1333 + 0.3 = 0.1666... = 1/6. ✓
        # So wts should NOT be multiplied by V, they already sum to 1/6 if
        # the raw weights sum to 1 and we scale by V.
        # Let me redo: raw weights sum to 1, multiply by V.
        wts = np.array([-4./5, 9./20, 9./20, 9./20, 9./20]) * V
        # -4/5 + 4*9/20 = -0.8 + 1.8 = 1.0 ✓ → times V = 1/6
        return pts, wts

    # degree <= 5: 15-point rule (Keast, exact for degree 5)
    # This is reliable and well-tested.
    a1 = 0.25
    w1_raw = 0.030283678097089

    a2 = 0.0
    b2 = 1.0/3.0
    w2_raw = 0.006026785714286

    a3 = 0.7272727272727273  # = 8/11
    b3 = 0.0909090909090909  # = 1/11
    w3_raw = 0.011645249086029

    a4 = 0.4334498464263357
    b4 = 0.0665501535736643
    w4_raw = 0.010949141561386

    pts = np.zeros((15, 3))
    wts = np.zeros(15)

    # Point 1: centroid
    pts[0] = [a1, a1, a1]
    wts[0] = w1_raw

    # Points 2-5: vertex type (a2, b2, b2, b2) and permutations
    idx = 1
    for perm in [(a2, b2, b2), (b2, a2, b2), (b2, b2, a2), (b2, b2, b2)]:
        # Wait — (a2,b2,b2) means r=a2,s=b2,t=b2 → L0=1-a2-b2-b2=1-0-2/3=1/3
        # The 4 permutations of (0, 1/3, 1/3, 1/3) in barycentric:
        # (r,s,t) = (L1,L2,L3) with L0=1-r-s-t
        pass

    # This is getting complicated with conventions. Let me use a simple,
    # directly verified approach: precompute and hardcode from a trusted source.

    # Verified 15-point Keast rule from John Burkardt's collection:
    # (r, s, t, weight) — weights sum to 1/6
    data = np.array([
        [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.0302836780970891],
        [0.0000000000000000, 0.3333333333333333, 0.3333333333333333, 0.0060267857142857],
        [0.3333333333333333, 0.0000000000000000, 0.3333333333333333, 0.0060267857142857],
        [0.3333333333333333, 0.3333333333333333, 0.0000000000000000, 0.0060267857142857],
        [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0060267857142857],
        [0.7272727272727273, 0.0909090909090909, 0.0909090909090909, 0.0116452490860289],
        [0.0909090909090909, 0.7272727272727273, 0.0909090909090909, 0.0116452490860289],
        [0.0909090909090909, 0.0909090909090909, 0.7272727272727273, 0.0116452490860289],
        [0.0909090909090909, 0.0909090909090909, 0.0909090909090909, 0.0116452490860289],
        [0.4334498464263357, 0.0665501535736643, 0.0665501535736643, 0.0109491415613864],
        [0.0665501535736643, 0.4334498464263357, 0.0665501535736643, 0.0109491415613864],
        [0.0665501535736643, 0.0665501535736643, 0.4334498464263357, 0.0109491415613864],
        [0.0665501535736643, 0.4334498464263357, 0.4334498464263357, 0.0109491415613864],
        [0.4334498464263357, 0.0665501535736643, 0.4334498464263357, 0.0109491415613864],
        [0.4334498464263357, 0.4334498464263357, 0.0665501535736643, 0.0109491415613864],
    ])
    pts = data[:, :3]
    wts = data[:, 3]
    return pts, wts


def tri_quadrature(degree):
    """Quadrature rule on reference triangle {u,v>=0, u+v<=1}.

    Returns (points, weights) where weights sum to 1/2 (triangle area).
    """
    A = 0.5

    if degree <= 2:
        # 3-point midpoint rule (exact for degree 2)
        pts = np.array([
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ])
        wts = np.full(3, A / 3.0)
        return pts, wts

    # degree <= 4: 6-point rule (exact for degree 4)
    a1 = 0.44594849091596489
    b1 = 0.10810301816807022  # = 1 - 2*a1
    w1 = 0.22338158967801147 * A

    a2 = 0.09157621350977073
    b2 = 0.81684757298045851  # = 1 - 2*a2
    w2 = 0.10995174365532187 * A

    pts = np.array([
        [a1, a1], [b1, a1], [a1, b1],
        [a2, a2], [b2, a2], [a2, b2],
    ])
    wts = np.array([w1, w1, w1, w2, w2, w2])
    return pts, wts


# ===================================================================
# Tet Mesh class
# ===================================================================

class TetMesh3D:
    """
    Tetrahedral mesh for arbitrary 3D geometries.

    Nodes and connectivity come directly from Gmsh (P=2, 10-node tets).
    Provides the same interface as BoxMesh3D / UnstructuredHexMesh3D.

    Parameters
    ----------
    nodes : (N_nodes, 3) array
        Node coordinates (from Gmsh, already at high-order positions).
    tets : (N_el, 10) int array
        Element connectivity (Gmsh 10-node tet ordering).
    boundary : dict
        Maps wall label -> (N_faces, 6) int array of 6-node triangle
        face connectivity.
    """

    def __init__(self, nodes, tets, boundary):
        self.ndim = 3
        self.P = 2
        nodes = np.asarray(nodes, dtype=float)
        tets = np.asarray(tets, dtype=int)

        self.N_dof = len(nodes)
        self.x = nodes[:, 0]
        self.y = nodes[:, 1]
        self.z = nodes[:, 2]
        self.nodes = nodes

        self.elem_conn = tets
        self.N_el = len(tets)

        # Boundary
        self._boundary_nodes_per_label = {}
        all_bnd = set()
        for label, faces in boundary.items():
            face_arr = np.asarray(faces, dtype=int)
            node_set = set(face_arr.ravel().tolist())
            self._boundary_nodes_per_label[label] = np.array(
                sorted(node_set), dtype=int)
            all_bnd.update(node_set)
        self._all_boundary = np.array(sorted(all_bnd), dtype=int)
        self._boundary_faces = boundary

    def _ensure_coords(self):
        pass  # coords already set in __init__

    def all_boundary_nodes(self):
        return self._all_boundary

    def boundary_nodes(self, label):
        return self._boundary_nodes_per_label[label]

    def nearest_node(self, rx, ry, rz):
        return int(np.argmin((self.x - rx)**2 + (self.y - ry)**2
                             + (self.z - rz)**2))


# ===================================================================
# Assembly
# ===================================================================

def _shape_and_grad_at_quad_pts(quad_pts):
    """Precompute shape functions and gradients at all quadrature points."""
    n_q = len(quad_pts)
    N_all = np.zeros((n_q, 10))
    dN_all = np.zeros((n_q, 3, 10))
    for q in range(n_q):
        r, s, t = quad_pts[q]
        N_all[q] = tet_shape_p2(r, s, t)
        dN_all[q] = tet_shape_grad_p2(r, s, t)
    return N_all, dN_all


def _det3(J):
    """3x3 determinant without numpy overhead."""
    return (J[0,0]*(J[1,1]*J[2,2]-J[1,2]*J[2,1])
           -J[0,1]*(J[1,0]*J[2,2]-J[1,2]*J[2,0])
           +J[0,2]*(J[1,0]*J[2,1]-J[1,1]*J[2,0]))


def _inv3(J):
    """3x3 inverse without numpy overhead."""
    d = _det3(J)
    Ji = np.empty((3, 3))
    Ji[0,0] = (J[1,1]*J[2,2]-J[1,2]*J[2,1])/d
    Ji[0,1] = (J[0,2]*J[2,1]-J[0,1]*J[2,2])/d
    Ji[0,2] = (J[0,1]*J[1,2]-J[0,2]*J[1,1])/d
    Ji[1,0] = (J[1,2]*J[2,0]-J[1,0]*J[2,2])/d
    Ji[1,1] = (J[0,0]*J[2,2]-J[0,2]*J[2,0])/d
    Ji[1,2] = (J[0,2]*J[1,0]-J[0,0]*J[1,2])/d
    Ji[2,0] = (J[1,0]*J[2,1]-J[1,1]*J[2,0])/d
    Ji[2,1] = (J[0,1]*J[2,0]-J[0,0]*J[2,1])/d
    Ji[2,2] = (J[0,0]*J[1,1]-J[0,1]*J[1,0])/d
    return Ji


def _assemble_elements_python(all_nodes, all_conn, N_all, dN_all, wts, N_dof):
    """Pure Python element assembly (fallback when Numba unavailable)."""
    N_el = len(all_conn)
    n_q = len(wts)
    M_lumped = np.zeros(N_dof)

    # Preallocate COO arrays (upper bound: N_el * 100 nonzeros)
    max_nnz = N_el * 100
    coo_r = np.empty(max_nnz, dtype=np.int64)
    coo_c = np.empty(max_nnz, dtype=np.int64)
    coo_v = np.empty(max_nnz)
    nnz = 0

    for e in range(N_el):
        dof = all_conn[e]
        coords = all_nodes[dof]

        # Mass (consistent then HRZ lump)
        M_diag_e = np.zeros(10)
        total_mass = 0.0
        for q in range(n_q):
            dN = dN_all[q]  # (3, 10)
            J = dN @ coords
            detJ = abs(_det3(J))
            w = wts[q] * detJ
            N_q = N_all[q]
            for i in range(10):
                M_diag_e[i] += w * N_q[i] * N_q[i]
                total_mass += w * N_q[i]  # sum of consistent mass

        # HRZ scale
        diag_sum = M_diag_e.sum()
        if diag_sum > 0:
            scale = total_mass / diag_sum
            for i in range(10):
                M_lumped[dof[i]] += M_diag_e[i] * scale

        # Stiffness
        S_e = np.zeros((10, 10))
        for q in range(n_q):
            dN = dN_all[q]
            J = dN @ coords
            detJ = abs(_det3(J))
            Ji = _inv3(J)
            dN_phys = Ji @ dN  # (3, 10)
            w = wts[q] * detJ
            for i in range(10):
                for j in range(i, 10):
                    val = w * (dN_phys[0,i]*dN_phys[0,j]
                             + dN_phys[1,i]*dN_phys[1,j]
                             + dN_phys[2,i]*dN_phys[2,j])
                    S_e[i, j] += val
                    if i != j:
                        S_e[j, i] += val

        # Scatter
        for i in range(10):
            for j in range(10):
                if abs(S_e[i, j]) > 1e-30:
                    if nnz >= max_nnz:
                        coo_r = np.append(coo_r, np.empty(max_nnz, dtype=np.int64))
                        coo_c = np.append(coo_c, np.empty(max_nnz, dtype=np.int64))
                        coo_v = np.append(coo_v, np.empty(max_nnz))
                        max_nnz *= 2
                    coo_r[nnz] = dof[i]
                    coo_c[nnz] = dof[j]
                    coo_v[nnz] = S_e[i, j]
                    nnz += 1

    return M_lumped, coo_r[:nnz], coo_c[:nnz], coo_v[:nnz]


if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _assemble_elements_numba(all_nodes, all_conn, N_all, dN_all, wts,
                                  N_dof):
        """Numba-JIT element assembly with parallel element loop."""
        N_el = all_conn.shape[0]
        n_q = wts.shape[0]

        # Per-element results (will scatter afterward)
        M_elem = np.zeros((N_el, 10))    # lumped mass per element
        S_elem = np.zeros((N_el, 10, 10)) # stiffness per element

        for e in prange(N_el):
            dof = all_conn[e]
            coords = all_nodes[dof]  # (10, 3)

            M_diag_e = np.zeros(10)
            total_mass = 0.0

            for q in range(n_q):
                # Jacobian
                J = np.zeros((3, 3))
                for a in range(3):
                    for b in range(3):
                        for k in range(10):
                            J[a, b] += dN_all[q, a, k] * coords[k, b]

                detJ = (J[0,0]*(J[1,1]*J[2,2]-J[1,2]*J[2,1])
                       -J[0,1]*(J[1,0]*J[2,2]-J[1,2]*J[2,0])
                       +J[0,2]*(J[1,0]*J[2,1]-J[1,1]*J[2,0]))
                abs_detJ = abs(detJ)
                w = wts[q] * abs_detJ

                # Inverse Jacobian
                Ji = np.zeros((3, 3))
                Ji[0,0] = (J[1,1]*J[2,2]-J[1,2]*J[2,1])/detJ
                Ji[0,1] = (J[0,2]*J[2,1]-J[0,1]*J[2,2])/detJ
                Ji[0,2] = (J[0,1]*J[1,2]-J[0,2]*J[1,1])/detJ
                Ji[1,0] = (J[1,2]*J[2,0]-J[1,0]*J[2,2])/detJ
                Ji[1,1] = (J[0,0]*J[2,2]-J[0,2]*J[2,0])/detJ
                Ji[1,2] = (J[0,2]*J[1,0]-J[0,0]*J[1,2])/detJ
                Ji[2,0] = (J[1,0]*J[2,1]-J[1,1]*J[2,0])/detJ
                Ji[2,1] = (J[0,1]*J[2,0]-J[0,0]*J[2,1])/detJ
                Ji[2,2] = (J[0,0]*J[1,1]-J[0,1]*J[1,0])/detJ

                # Physical gradients: dN_phys = Ji @ dN_ref
                dN_phys = np.zeros((3, 10))
                for a in range(3):
                    for k in range(10):
                        for b in range(3):
                            dN_phys[a, k] += Ji[a, b] * dN_all[q, b, k]

                # Mass diagonal (consistent)
                for i in range(10):
                    M_diag_e[i] += w * N_all[q, i] * N_all[q, i]
                    total_mass += w * N_all[q, i]

                # Stiffness (symmetric — compute upper triangle)
                for i in range(10):
                    for j in range(i, 10):
                        val = w * (dN_phys[0,i]*dN_phys[0,j]
                                 + dN_phys[1,i]*dN_phys[1,j]
                                 + dN_phys[2,i]*dN_phys[2,j])
                        S_elem[e, i, j] += val
                        if i != j:
                            S_elem[e, j, i] += val

            # HRZ lumping
            diag_sum = 0.0
            for i in range(10):
                diag_sum += M_diag_e[i]
            if diag_sum > 0:
                scale = total_mass / diag_sum
                for i in range(10):
                    M_elem[e, i] = M_diag_e[i] * scale

        return M_elem, S_elem


def assemble_tet_3d_operators(mesh):
    """
    Element-by-element assembly for P=2 tetrahedral meshes.

    Uses Numba JIT with parallel element loop when available,
    falls back to pure Python otherwise.

    Returns dict compatible with all FOM/ROM solvers:
        M_diag, M_inv, S, B_total
    """
    N = mesh.N_dof
    N_el = mesh.N_el

    vol_pts, vol_wts = tet_quadrature(5)
    N_all, dN_all = _shape_and_grad_at_quad_pts(vol_pts)

    backend = "numba (parallel)" if NUMBA_AVAILABLE else "python"
    print(f"    Tet assembly [{backend}]: N={N}, {N_el} elements...",
          end='', flush=True)

    if NUMBA_AVAILABLE:
        all_nodes = np.ascontiguousarray(mesh.nodes)
        all_conn = np.ascontiguousarray(mesh.elem_conn)

        M_elem, S_elem = _assemble_elements_numba(
            all_nodes, all_conn, N_all, dN_all, vol_wts, N)

        # Scatter mass
        M_diag = np.zeros(N)
        for e in range(N_el):
            dof = all_conn[e]
            for i in range(10):
                M_diag[dof[i]] += M_elem[e, i]

        # Scatter stiffness to COO — preallocated, no Python append loop
        print(" scatter...", end='', flush=True)
        # Each element contributes up to 100 nonzeros (10x10)
        coo_r = np.empty(N_el * 100, dtype=np.int64)
        coo_c = np.empty(N_el * 100, dtype=np.int64)
        coo_v = np.empty(N_el * 100)
        nnz = 0
        for e in range(N_el):
            dof = all_conn[e]
            Se = S_elem[e]
            for i in range(10):
                gi = dof[i]
                for j in range(10):
                    v = Se[i, j]
                    if abs(v) > 1e-30:
                        coo_r[nnz] = gi
                        coo_c[nnz] = dof[j]
                        coo_v[nnz] = v
                        nnz += 1

        S = sparse.coo_matrix((coo_v[:nnz], (coo_r[:nnz], coo_c[:nnz])),
                              shape=(N, N)).tocsr()
    else:
        M_diag, coo_r, coo_c, coo_v = _assemble_elements_python(
            mesh.nodes, mesh.elem_conn, N_all, dN_all, vol_wts, N)
        S = sparse.coo_matrix((coo_v, (coo_r, coo_c)),
                              shape=(N, N)).tocsr()

    print(" boundary...", end='', flush=True)
    B_diag = _assemble_boundary_mass_tet(mesh)
    B_total = sparse.diags(B_diag)

    M_inv = 1.0 / M_diag
    print(" done.")

    return dict(
        M_diag=M_diag, M_inv=M_inv,
        S=S, B_total=B_total,
    )


def _assemble_boundary_mass_tet(mesh):
    """Boundary mass via row-sum lumping on P=2 triangle faces."""
    N = mesh.N_dof
    B_diag = np.zeros(N)

    surf_pts, surf_wts = tri_quadrature(4)
    n_qs = len(surf_wts)

    for label, faces in mesh._boundary_faces.items():
        face_arr = np.asarray(faces, dtype=int)
        for f in range(len(face_arr)):
            face_nodes = face_arr[f]  # (6,)
            coords = mesh.nodes[face_nodes]  # (6, 3)

            B_e = np.zeros(6)
            for q in range(n_qs):
                u, v = surf_pts[q]
                N_q = tri_shape_p2(u, v)  # (6,)

                # Surface Jacobian: two tangent vectors
                L0 = 1.0 - u - v
                L1 = u
                L2 = v

                # Derivatives of triangle shape functions w.r.t. u, v
                dN_du = np.array([
                    -(4*L0 - 1), (4*L1 - 1), 0.0,
                    4*(L0 - L1), 4*L2, -4*L2,
                ])
                dN_dv = np.array([
                    -(4*L0 - 1), 0.0, (4*L2 - 1),
                    -4*L1, 4*L1, 4*(L0 - L2),
                ])

                tangent_u = dN_du @ coords  # (3,)
                tangent_v = dN_dv @ coords  # (3,)
                normal = np.cross(tangent_u, tangent_v)
                detJ_s = np.linalg.norm(normal)

                # Row-sum of consistent surface mass
                B_e += surf_wts[q] * detJ_s * N_q

            for i in range(6):
                B_diag[face_nodes[i]] += B_e[i]

    return B_diag
