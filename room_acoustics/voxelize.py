"""
Voxelization engine — convert mesh geometry to regular 3D grid.

Identifies interior air voxels, boundary voxels (air adjacent to solid),
and surface normals. Used for FDTD simulation and boundary detection.

Usage:
    from room_acoustics.voxelize import voxelize_box, voxelize_stl

    # Box room
    air, origin, vox_size = voxelize_box(8.4, 6.7, 3.0, dx=0.05)

    # From STL mesh
    air, origin, vox_size = voxelize_stl("room.stl", dx=0.05)

    # Find boundary voxels
    boundary_ijk, boundary_normals = find_boundary_voxels(air)
"""

import numpy as np


def voxelize_box(Lx, Ly, Lz, dx=0.05, padding=2):
    """
    Create a voxel grid for a rectangular box room.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Room dimensions [m].
    dx : float
        Voxel size [m] (isotropic).
    padding : int
        Extra voxels of solid around the room (for boundary conditions).

    Returns
    -------
    air : ndarray (nx, ny, nz), uint8
        1 = air, 0 = solid.
    origin : (float, float, float)
        World-space origin of voxel (0,0,0).
    dx : float
        Voxel size used.
    """
    nx = int(np.ceil(Lx / dx)) + 2 * padding
    ny = int(np.ceil(Ly / dx)) + 2 * padding
    nz = int(np.ceil(Lz / dx)) + 2 * padding

    air = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Interior air region (inside the box walls)
    ni = int(np.round(Lx / dx))
    nj = int(np.round(Ly / dx))
    nk = int(np.round(Lz / dx))

    air[padding:padding + ni,
        padding:padding + nj,
        padding:padding + nk] = 1

    origin = (-padding * dx, -padding * dx, -padding * dx)
    return air, origin, dx


def voxelize_stl(stl_path, dx=0.05, padding=4):
    """
    Voxelize an STL/OBJ mesh using flood-fill exterior detection.

    Parameters
    ----------
    stl_path : str
        Path to STL or OBJ mesh file.
    dx : float
        Voxel size [m].
    padding : int
        Extra voxels around bounding box.

    Returns
    -------
    air : ndarray (nx, ny, nz), uint8
        1 = air (interior), 0 = solid/exterior.
    origin : (float, float, float)
        World-space origin.
    dx : float
        Voxel size used.
    """
    # Load mesh vertices to get bounding box
    vertices = _load_mesh_vertices(stl_path)
    if vertices is None or len(vertices) == 0:
        raise ValueError(f"Could not load mesh from {stl_path}")

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin

    nx = int(np.ceil(extent[0] / dx)) + 2 * padding
    ny = int(np.ceil(extent[1] / dx)) + 2 * padding
    nz = int(np.ceil(extent[2] / dx)) + 2 * padding

    origin = (vmin[0] - padding * dx,
              vmin[1] - padding * dx,
              vmin[2] - padding * dx)

    # Start with all solid, then flood-fill exterior from corners
    grid = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Rasterize mesh triangles into the grid as solid
    # (simplified: mark voxels containing triangle vertices/edges)
    _rasterize_mesh(grid, vertices, origin, dx)

    # Flood-fill from corner to find exterior
    exterior = _flood_fill_exterior(grid)

    # Air = everything that's not exterior and not solid surface
    # Interior = not exterior
    air = np.zeros_like(grid)
    air[~exterior & (grid == 0)] = 1

    return air, origin, dx


def find_boundary_voxels(air):
    """
    Find air voxels adjacent to solid (boundary voxels).

    Parameters
    ----------
    air : ndarray (nx, ny, nz), uint8
        1 = air, 0 = solid.

    Returns
    -------
    boundary_ijk : ndarray (M, 3), int32
        (i, j, k) indices of boundary voxels.
    boundary_normals : ndarray (M, 3), float32
        Approximate outward normal at each boundary voxel
        (points from solid toward air).
    """
    air_bool = (air == 1)
    solid = ~air_bool

    # Boundary = air voxel adjacent to solid in any of 6 directions
    boundary = np.zeros_like(air_bool)
    normals_acc = np.zeros(air.shape + (3,), dtype=np.float32)

    for axis, direction in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        shifted_solid = np.roll(solid, direction, axis=axis)
        touch = air_bool & shifted_solid
        boundary |= touch

        # Normal contribution: points away from the solid neighbor
        normal_vec = np.zeros(3)
        normal_vec[axis] = -direction  # points from solid toward air
        normals_acc[touch, :] += normal_vec

    ijk = np.argwhere(boundary).astype(np.int32)

    # Normalize normals
    if len(ijk) > 0:
        normals = normals_acc[boundary]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normals = (normals / norms).astype(np.float32)
    else:
        normals = np.zeros((0, 3), dtype=np.float32)

    return ijk, normals


def boundary_to_world(ijk, origin, dx):
    """Convert boundary voxel indices to world-space coordinates."""
    pts = np.zeros((len(ijk), 3), dtype=np.float32)
    pts[:, 0] = origin[0] + ijk[:, 0] * dx
    pts[:, 1] = origin[1] + ijk[:, 1] * dx
    pts[:, 2] = origin[2] + ijk[:, 2] * dx
    return pts


def connected_components(boundary_mask, min_size=30, connectivity=6):
    """
    Split boundary voxels into connected components (surface regions).

    Parameters
    ----------
    boundary_mask : ndarray (nx, ny, nz), bool
        True at boundary voxels.
    min_size : int
        Minimum voxels per component (smaller are discarded).
    connectivity : int
        6 (faces only) or 26 (faces + edges + corners).

    Returns
    -------
    regions : dict of str -> ndarray (nx, ny, nz), bool
        Named regions ('region_0', 'region_1', ...).
    """
    boundary_mask = boundary_mask.astype(bool)
    nx, ny, nz = boundary_mask.shape
    visited = np.zeros_like(boundary_mask, dtype=bool)

    if connectivity == 6:
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                   (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    else:
        offsets = [(dx, dy, dz)
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if not (dx == dy == dz == 0)]

    regions = {}
    label = 0

    for (x, y, z) in np.argwhere(boundary_mask):
        if visited[x, y, z]:
            continue

        # BFS flood fill
        stack = [(x, y, z)]
        visited[x, y, z] = True
        voxels = [(x, y, z)]

        while stack:
            cx, cy, cz = stack.pop()
            for dx, dy, dz in offsets:
                nx_ = cx + dx
                ny_ = cy + dy
                nz_ = cz + dz
                if (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz
                        and not visited[nx_, ny_, nz_]
                        and boundary_mask[nx_, ny_, nz_]):
                    visited[nx_, ny_, nz_] = True
                    stack.append((nx_, ny_, nz_))
                    voxels.append((nx_, ny_, nz_))

        if len(voxels) >= min_size:
            mask = np.zeros_like(boundary_mask)
            for v in voxels:
                mask[v] = True
            regions[f'region_{label}'] = mask
            label += 1

    return regions


# ===================================================================
# Internal helpers
# ===================================================================

def _flood_fill_exterior(grid):
    """Flood-fill from domain corners to identify exterior voxels."""
    nx, ny, nz = grid.shape
    exterior = np.zeros_like(grid, dtype=bool)

    # Start from all corners/edges of the domain
    seeds = [(0, 0, 0)]
    queue = []
    for s in seeds:
        if grid[s] == 0 and not exterior[s]:
            queue.append(s)
            exterior[s] = True

    # BFS flood fill through non-solid voxels
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
               (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while queue:
        cx, cy, cz = queue.pop(0)
        for dx, dy, dz in offsets:
            nx_ = cx + dx
            ny_ = cy + dy
            nz_ = cz + dz
            if (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz
                    and not exterior[nx_, ny_, nz_]
                    and grid[nx_, ny_, nz_] == 0):
                exterior[nx_, ny_, nz_] = True
                queue.append((nx_, ny_, nz_))

    return exterior


def _load_mesh_vertices(path):
    """Load vertex positions from STL/OBJ/PLY file."""
    ext = path.lower().rsplit('.', 1)[-1]

    if ext == 'stl':
        return _load_stl(path)
    elif ext == 'obj':
        return _load_obj(path)
    else:
        raise ValueError(f"Unsupported mesh format: .{ext}")


def _load_stl(path):
    """Load vertices from binary or ASCII STL."""
    import struct
    with open(path, 'rb') as f:
        header = f.read(80)
        n_tris = struct.unpack('<I', f.read(4))[0]

        if n_tris > 0 and n_tris < 10_000_000:
            # Binary STL
            verts = []
            for _ in range(n_tris):
                f.read(12)  # normal
                for _ in range(3):
                    x, y, z = struct.unpack('<fff', f.read(12))
                    verts.append((x, y, z))
                f.read(2)  # attribute
            return np.array(verts, dtype=np.float64)

    # Try ASCII STL
    verts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('vertex'):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.array(verts, dtype=np.float64) if verts else None


def _load_obj(path):
    """Load vertices from OBJ file."""
    verts = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.array(verts, dtype=np.float64) if verts else None


def _rasterize_mesh(grid, vertices, origin, dx):
    """Mark voxels near mesh vertices as solid (simple vertex rasterization)."""
    ox, oy, oz = origin
    for v in vertices:
        i = int(np.round((v[0] - ox) / dx))
        j = int(np.round((v[1] - oy) / dx))
        k = int(np.round((v[2] - oz) / dx))
        nx, ny, nz = grid.shape
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            grid[i, j, k] = 2  # solid surface marker
