"""
Gmsh tetrahedral mesh generation and import for arbitrary 3D geometries.

Two entry points:
  - generate_tet_mesh(): create mesh from a 2D floor plan + height
  - import_msh_file(): load an existing .msh file
"""

import numpy as np

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


def generate_tet_mesh(geometry_2d, Lz, h_target, P=2, verbose=False):
    """
    Generate a P=2 tetrahedral mesh by extruding a 2D room geometry.

    Parameters
    ----------
    geometry_2d : RoomGeometry
        Floor plan (from geometry.py).
    Lz : float
        Room height [m].
    h_target : float
        Target element edge length.
    P : int
        Polynomial order (default 2 — 10-node tets).
    verbose : bool
        Print Gmsh output.

    Returns
    -------
    dict with keys:
        nodes    : (N_nodes, 3) float array
        tets     : (N_el, n_nodes_per_tet) int array
        boundary : dict mapping label -> (N_faces, n_nodes_per_tri) int array
        wall_labels : list of str
    """
    if not GMSH_AVAILABLE:
        raise ImportError("Gmsh Python API required: pip install gmsh")

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("room3d")

    verts = geometry_2d.vertices
    n = len(verts)

    # Build 2D floor plan
    floor_points = []
    for x, y in verts:
        floor_points.append(gmsh.model.occ.addPoint(x, y, 0, h_target))

    floor_lines = []
    for i in range(n):
        p1 = floor_points[i]
        p2 = floor_points[(i + 1) % n]
        floor_lines.append(gmsh.model.occ.addLine(p1, p2))

    floor_loop = gmsh.model.occ.addCurveLoop(floor_lines)
    floor_surf = gmsh.model.occ.addPlaneSurface([floor_loop])

    # Extrude to create 3D volume
    n_extrude_layers = max(1, int(round(Lz / h_target)))
    extruded = gmsh.model.occ.extrude(
        [(2, floor_surf)], 0, 0, Lz,
        numElements=[n_extrude_layers], recombine=False)

    gmsh.model.occ.synchronize()

    # Identify surfaces for physical groups
    # The extrude returns: [(3, vol_tag), (2, top_surf), (2, side1), ...]
    vol_tag = None
    side_surfs = []
    top_surf_tag = None

    for dim, tag in extruded:
        if dim == 3:
            vol_tag = tag
        elif dim == 2:
            if top_surf_tag is None:
                top_surf_tag = tag  # first 2D entity is the top surface
            else:
                side_surfs.append(tag)

    # Physical groups
    pg_vol = gmsh.model.addPhysicalGroup(3, [vol_tag])
    gmsh.model.setPhysicalName(3, pg_vol, "room")

    pg_floor = gmsh.model.addPhysicalGroup(2, [floor_surf])
    gmsh.model.setPhysicalName(2, pg_floor, "floor")

    pg_ceil = gmsh.model.addPhysicalGroup(2, [top_surf_tag])
    gmsh.model.setPhysicalName(2, pg_ceil, "ceiling")

    # Assign wall labels to side surfaces
    # Side surfaces correspond to the extruded floor edges
    for i, surf_tag in enumerate(side_surfs):
        label = geometry_2d.wall_labels[i] if i < len(geometry_2d.wall_labels) else f'wall_{i}'
        pg = gmsh.model.addPhysicalGroup(2, [surf_tag])
        gmsh.model.setPhysicalName(2, pg, label)

    # Mesh options
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder", P)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_target * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_target * 1.5)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    gmsh.model.mesh.generate(3)

    result = _extract_mesh_data(P)
    gmsh.finalize()
    return result


def import_msh_file(filepath, P=2):
    """
    Import a Gmsh .msh file and extract tet mesh data.

    The .msh file must contain:
    - 3D tet elements (type 11 for P=2)
    - Physical surface groups for boundary identification
    """
    if not GMSH_AVAILABLE:
        raise ImportError("Gmsh Python API required: pip install gmsh")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(filepath)

    result = _extract_mesh_data(P)
    gmsh.finalize()
    return result


def import_surface_mesh(filepath, h_target, P=2, verbose=False):
    """
    Import an STL/OBJ surface mesh and create a volume tet mesh.

    Works with any closed (watertight) surface mesh. Gmsh classifies
    the surface into patches, creates a volume, and tet-meshes it.

    Parameters
    ----------
    filepath : str
        Path to .stl or .obj file.
    h_target : float
        Target element edge length [m].
    P : int
        Polynomial order (default 2).
    verbose : bool
        Print Gmsh output.

    Returns
    -------
    Same dict as generate_tet_mesh / import_msh_file.
    """
    import math
    if not GMSH_AVAILABLE:
        raise ImportError("Gmsh Python API required: pip install gmsh")

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("imported")
    gmsh.merge(filepath)

    # Classify the imported triangulation into geometric surfaces
    angle = 40 * math.pi / 180
    gmsh.model.mesh.classifySurfaces(angle, True, True, angle)
    gmsh.model.mesh.createGeometry()
    gmsh.model.mesh.createTopology()

    ents = gmsh.model.getEntities()
    surfs = [t for d, t in ents if d == 2]
    vols = [t for d, t in ents if d == 3]

    if not vols:
        # Create volume manually from surface loop
        sl = gmsh.model.geo.addSurfaceLoop(surfs)
        vol = gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()
        vols = [vol]

    # Physical groups
    pg = gmsh.model.addPhysicalGroup(3, vols)
    gmsh.model.setPhysicalName(3, pg, "room")
    for i, s in enumerate(surfs):
        pg = gmsh.model.addPhysicalGroup(2, [s])
        gmsh.model.setPhysicalName(2, pg, f"wall_{i}")

    # Mesh options
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_target * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_target * 2.0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", P)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    gmsh.model.mesh.generate(3)

    result = _extract_mesh_data(P)
    gmsh.finalize()
    return result


def _extract_mesh_data(P):
    """Extract nodes, tets, and boundary faces from the current Gmsh model."""

    # Nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    tag_to_idx = {}
    nodes = np.zeros((len(node_tags), 3))
    for i, tag in enumerate(node_tags):
        tag_to_idx[int(tag)] = i
        nodes[i, 0] = node_coords[3 * i]
        nodes[i, 1] = node_coords[3 * i + 1]
        nodes[i, 2] = node_coords[3 * i + 2]

    # Tet elements from volume
    # P=2: type 11 (10-node tet)
    tet_type = 11 if P == 2 else (29 if P == 3 else 4)
    nodes_per_tet = {4: 4, 11: 10, 29: 20}[tet_type]

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
    tets = None
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == tet_type:
            n_el = len(etags)
            raw = np.array(enodes, dtype=int).reshape(n_el, nodes_per_tet)
            tets = np.array([[tag_to_idx[int(t)] for t in row] for row in raw])

    if tets is None:
        raise RuntimeError(
            f"No {nodes_per_tet}-node tet elements found in mesh. "
            f"Check Mesh.ElementOrder={P}.")

    # Boundary faces from physical surface groups
    # P=2: type 9 (6-node triangle)
    tri_type = 9 if P == 2 else (21 if P == 3 else 2)
    nodes_per_tri = {2: 3, 9: 6, 21: 10}[tri_type]

    boundary = {}
    phys_groups = gmsh.model.getPhysicalGroups(2)
    for dim, tag in phys_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        faces = []
        for ent in entities:
            et, _, en = gmsh.model.mesh.getElements(2, ent)
            for etype, enodes in zip(et, en):
                if etype == tri_type:
                    raw = np.array(enodes, dtype=int).reshape(-1, nodes_per_tri)
                    for row in raw:
                        faces.append([tag_to_idx[int(t)] for t in row])
        if faces:
            boundary[name] = np.array(faces, dtype=int)

    return dict(
        nodes=nodes,
        tets=tets,
        boundary=boundary,
        wall_labels=list(boundary.keys()),
    )
