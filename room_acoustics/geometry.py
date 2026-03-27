"""
Room geometry definition and Gmsh quad mesh generation.

Defines room shapes as 2D polygons (with optional holes for columns),
then calls Gmsh to produce all-quad meshes suitable for spectral element
methods.
"""

import numpy as np

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


class RoomGeometry:
    """
    2D room geometry defined by an outer polygon and optional inner holes.

    Parameters
    ----------
    vertices : list of (x, y) tuples
        Outer boundary vertices in counter-clockwise order.
    wall_labels : list of str, optional
        Label for each wall segment (len = len(vertices)).
        Defaults to 'wall' for all segments.
    holes : list of list of (x, y), optional
        Each hole is a list of vertices (clockwise) defining a column
        or obstacle to subtract from the room.
    """

    def __init__(self, vertices, wall_labels=None, holes=None):
        self.vertices = [tuple(v) for v in vertices]
        n = len(self.vertices)
        self.wall_labels = wall_labels or ['wall'] * n
        self.holes = holes or []
        assert len(self.wall_labels) == n

    @property
    def n_walls(self):
        return len(self.vertices)

    def bounding_box(self):
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return min(xs), max(xs), min(ys), max(ys)


# ---------------------------------------------------------------------------
# Factory functions for common room shapes
# ---------------------------------------------------------------------------

def rectangular_room(Lx, Ly):
    """Simple rectangular room [0,Lx] x [0,Ly]."""
    verts = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
    labels = ['bottom', 'right', 'top', 'left']
    return RoomGeometry(verts, labels)


def l_shaped_room(Lx, Ly, notch_x, notch_y):
    """
    L-shaped room: rectangle [0,Lx]x[0,Ly] with upper-right corner
    [notch_x, Lx] x [notch_y, Ly] removed.

    Vertices (counter-clockwise):
        (0,0) -> (Lx,0) -> (Lx,notch_y) -> (notch_x,notch_y)
        -> (notch_x,Ly) -> (0,Ly) -> back to start
    """
    assert 0 < notch_x < Lx and 0 < notch_y < Ly
    verts = [
        (0, 0), (Lx, 0), (Lx, notch_y), (notch_x, notch_y),
        (notch_x, Ly), (0, Ly),
    ]
    labels = ['bottom', 'right_lower', 'notch_bottom', 'notch_right',
              'top', 'left']
    return RoomGeometry(verts, labels)


def t_shaped_room(Lx, Ly, stem_w, stem_h):
    """
    T-shaped room: horizontal bar [0,Lx]x[Ly-bar_h, Ly] with vertical
    stem of width stem_w centered at Lx/2, extending down by stem_h.
    """
    bar_h = Ly - stem_h
    x_left = (Lx - stem_w) / 2
    x_right = (Lx + stem_w) / 2
    verts = [
        (x_left, 0), (x_right, 0),           # stem bottom
        (x_right, bar_h), (Lx, bar_h),       # stem-to-bar right
        (Lx, Ly), (0, Ly),                    # bar top
        (0, bar_h), (x_left, bar_h),          # bar-to-stem left
    ]
    labels = ['stem_bottom', 'stem_right', 'bar_bottom_right', 'bar_right',
              'bar_top', 'bar_left', 'bar_bottom_left', 'stem_left']
    return RoomGeometry(verts, labels)


def room_with_column(Lx, Ly, col_center, col_radius, n_col_sides=8):
    """
    Rectangular room with a polygonal column (hole) approximating a circle.
    """
    outer = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
    labels = ['bottom', 'right', 'top', 'left']

    cx, cy = col_center
    angles = np.linspace(0, 2 * np.pi, n_col_sides, endpoint=False)
    # Clockwise for hole
    hole = [(cx + col_radius * np.cos(a), cy + col_radius * np.sin(a))
            for a in reversed(angles)]

    return RoomGeometry(outer, labels, holes=[hole])


# ---------------------------------------------------------------------------
# Gmsh mesh generation
# ---------------------------------------------------------------------------

def generate_quad_mesh(geometry, h_target, P=4, verbose=False):
    """
    Generate an all-quad mesh for a RoomGeometry using Gmsh.

    Parameters
    ----------
    geometry : RoomGeometry
    h_target : float
        Target element edge length [m].
    P : int
        Polynomial order (used only for info, not meshing).
    verbose : bool
        Print Gmsh output.

    Returns
    -------
    dict with keys:
        nodes     : (N_nodes, 2) float array — vertex coordinates
        quads     : (N_el, 4) int array — quad connectivity (corner node indices)
        boundary  : dict mapping wall_label -> list of (n1, n2) edge node pairs
        wall_labels : list of str — unique wall labels
    """
    if not GMSH_AVAILABLE:
        raise ImportError("Gmsh Python API required: pip install gmsh")

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("room")

    verts = geometry.vertices
    n = len(verts)

    # Create outer boundary points and lines
    outer_points = []
    for i, (x, y) in enumerate(verts):
        outer_points.append(gmsh.model.occ.addPoint(x, y, 0, h_target))

    outer_lines = []
    for i in range(n):
        p1 = outer_points[i]
        p2 = outer_points[(i + 1) % n]
        outer_lines.append(gmsh.model.occ.addLine(p1, p2))

    outer_loop = gmsh.model.occ.addCurveLoop(outer_lines)
    loops = [outer_loop]

    # Holes (columns)
    hole_lines_all = []
    for hole in geometry.holes:
        nh = len(hole)
        hole_pts = []
        for (x, y) in hole:
            hole_pts.append(gmsh.model.occ.addPoint(x, y, 0, h_target))
        h_lines = []
        for i in range(nh):
            h_lines.append(gmsh.model.occ.addLine(hole_pts[i],
                                                    hole_pts[(i + 1) % nh]))
        h_loop = gmsh.model.occ.addCurveLoop(h_lines)
        loops.append(h_loop)
        hole_lines_all.extend(h_lines)

    surf = gmsh.model.occ.addPlaneSurface(loops)
    gmsh.model.occ.synchronize()

    # Meshing options: force all-quad via recombination
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # blossom-quad
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_target * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_target * 1.5)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)  # linear elements

    # Physical groups for boundary identification
    for i, line_tag in enumerate(outer_lines):
        label = geometry.wall_labels[i]
        pg = gmsh.model.addPhysicalGroup(1, [line_tag])
        gmsh.model.setPhysicalName(1, pg, label)

    if hole_lines_all:
        pg = gmsh.model.addPhysicalGroup(1, hole_lines_all)
        gmsh.model.setPhysicalName(1, pg, "column")

    pg_surf = gmsh.model.addPhysicalGroup(2, [surf])
    gmsh.model.setPhysicalName(2, pg_surf, "room")

    gmsh.model.mesh.generate(2)

    # Extract mesh data
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # node_tags are 1-based; build a map to 0-based
    tag_to_idx = {}
    coords_2d = np.zeros((len(node_tags), 2))
    for i, tag in enumerate(node_tags):
        tag_to_idx[int(tag)] = i
        coords_2d[i, 0] = node_coords[3 * i]
        coords_2d[i, 1] = node_coords[3 * i + 1]

    # Extract quad elements (type 3 = 4-node quad)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    quads = None
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 3:  # 4-node quad
            n_el = len(etags)
            raw = np.array(enodes, dtype=int).reshape(n_el, 4)
            quads = np.array([[tag_to_idx[t] for t in row] for row in raw])

    if quads is None:
        # Try triangles (type 2) and warn
        gmsh.finalize()
        raise RuntimeError(
            "Gmsh did not produce quad elements. Try adjusting h_target "
            "or simplifying the geometry.")

    # Check for leftover triangles
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 2 and len(etags) > 0:
            gmsh.finalize()
            raise RuntimeError(
                f"Gmsh produced {len(etags)} triangles alongside quads. "
                f"Try adjusting h_target or mesh options.")

    # Extract boundary edges per physical group
    boundary = {}
    phys_groups = gmsh.model.getPhysicalGroups(1)
    for dim, tag in phys_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        edges = []
        for ent in entities:
            _, e_node_tags = gmsh.model.mesh.getElements(1, ent)[1:]
            # 2-node line elements
            for enodes in e_node_tags:
                raw_edges = np.array(enodes, dtype=int).reshape(-1, 2)
                for n1, n2 in raw_edges:
                    edges.append((tag_to_idx[n1], tag_to_idx[n2]))
        boundary[name] = edges

    wall_labels = list(boundary.keys())

    gmsh.finalize()

    return dict(
        nodes=coords_2d,
        quads=quads,
        boundary=boundary,
        wall_labels=wall_labels,
    )
