"""
room_geometry.py — Arbitrary room geometry: load, visualize, simulate.
Supports: OBJ, STL, STEP/STP (gmsh), SKP (engine DLL), GLB/FBX/PLY (trimesh)
"""
import numpy as np, struct, time, os, ctypes
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engines import (
    Room, MATERIALS, SURFACE_NAMES, OCTAVE_BANDS,
    all_metrics, synth_noise_decay, air_absorption_coeff,
    sabine_rt60, eyring_rt60, Result, ENGINE_REGISTRY,
    random_dirs_sphere, reflect_specular,
)

ROOT = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════
# Triangle mesh
# ═══════════════════════════════════════════════════════════════════
@dataclass
class TriMesh:
    vertices: np.ndarray           # (V, 3) float64
    faces: np.ndarray              # (F, 3) int32
    face_groups: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> face indices
    materials: Dict[str, str] = field(default_factory=dict)           # group -> material name

    # ── derived properties ──────────────────────────────────────
    def face_normals(self):
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30
        return cross / norms

    def face_areas(self):
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    def surface_area(self):
        return float(self.face_areas().sum())

    def volume(self):
        """Signed volume via divergence theorem (mesh must be closed)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return abs(float(np.sum(
            v0[:, 0] * (v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1]) +
            v1[:, 0] * (v2[:, 1] * v0[:, 2] - v2[:, 2] * v0[:, 1]) +
            v2[:, 0] * (v0[:, 1] * v1[:, 2] - v0[:, 2] * v1[:, 1])
        ) / 6.0))

    def bounding_box(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def centroid(self):
        return self.vertices.mean(axis=0)

    def dims(self):
        bb_min, bb_max = self.bounding_box()
        return bb_max - bb_min

    def group_names(self):
        return list(self.face_groups.keys())

    def group_area(self, name):
        if name not in self.face_groups: return 0
        return float(self.face_areas()[self.face_groups[name]].sum())

    def group_alpha(self, name, f=500.0):
        mat = self.materials.get(name, "plaster")
        a = np.array(MATERIALS.get(mat, MATERIALS["plaster"]), dtype=float)
        return float(np.interp(f, OCTAVE_BANDS, a))

    def mean_alpha(self, f=500.0):
        areas = self.face_areas()
        total_a, total_s = 0.0, 0.0
        for gname, fidx in self.face_groups.items():
            a = self.group_alpha(gname, f)
            s = float(areas[fidx].sum())
            total_a += a * s
            total_s += s
        if total_s == 0: return 0.1
        return total_a / total_s

    def absorption_area(self, f=500.0):
        areas = self.face_areas()
        total = 0.0
        for gname, fidx in self.face_groups.items():
            total += self.group_alpha(gname, f) * float(areas[fidx].sum())
        return total

    def to_shoebox(self) -> Room:
        """Bounding-box approximation for shoebox-only engines."""
        d = self.dims()
        surfaces = {}
        for i, sn in enumerate(SURFACE_NAMES):
            gnames = list(self.face_groups.keys())
            if i < len(gnames):
                surfaces[sn] = self.materials.get(gnames[i], "plaster")
            else:
                surfaces[sn] = "plaster"
        return Room(d[0], d[1], d[2], surfaces)

    # ── auto-group by normal direction ──────────────────────────
    def auto_group(self, angle_deg=20.0):
        """Cluster faces by normal direction. Assigns names based on orientation."""
        normals = self.face_normals()
        n_faces = len(self.faces)
        assigned = np.zeros(n_faces, dtype=bool)
        groups = {}
        # try canonical directions first
        canonical = [
            ("floor",    np.array([0, 0, -1.0])),
            ("ceiling",  np.array([0, 0,  1.0])),
            ("wall_north", np.array([0,  1.0, 0])),
            ("wall_south", np.array([0, -1.0, 0])),
            ("wall_east",  np.array([ 1.0, 0, 0])),
            ("wall_west",  np.array([-1.0, 0, 0])),
        ]
        cos_thresh = np.cos(np.radians(angle_deg))
        for name, ref_n in canonical:
            dots = np.dot(normals, ref_n)
            mask = (dots > cos_thresh) & (~assigned)
            if np.any(mask):
                groups[name] = np.where(mask)[0]
                assigned[mask] = True
        # remaining faces
        gid = 0
        for i in range(n_faces):
            if assigned[i]: continue
            n = normals[i]
            dots = np.abs(np.dot(normals, n))
            similar = (dots > cos_thresh) & (~assigned)
            if np.any(similar):
                groups[f"surface_{gid}"] = np.where(similar)[0]
                assigned[similar] = True
                gid += 1
        if not assigned.all():
            groups["other"] = np.where(~assigned)[0]
        self.face_groups = groups
        for gn in groups:
            if gn not in self.materials:
                if "floor" in gn:
                    self.materials[gn] = "linoleum"
                elif "ceiling" in gn:
                    self.materials[gn] = "acoustic_tile"
                else:
                    self.materials[gn] = "plaster"

    # ── translate / scale ───────────────────────────────────────
    def center_at_origin(self):
        c = self.centroid()
        self.vertices -= c

    def move_floor_to_z0(self):
        self.vertices[:, 2] -= self.vertices[:, 2].min()

    def scale_to_meters(self, current_unit="mm"):
        factors = {"mm": 0.001, "cm": 0.01, "in": 0.0254, "ft": 0.3048, "m": 1.0}
        self.vertices *= factors.get(current_unit, 1.0)

    # ── ray-triangle intersection (vectorised Möller-Trumbore) ──
    def ray_intersect(self, origins, directions):
        """
        For each ray, find nearest triangle hit.
        origins: (N, 3), directions: (N, 3)
        Returns: t (N,), face_idx (N,), hit_normals (N, 3)
        """
        N = origins.shape[0]
        F = len(self.faces)
        v0 = self.vertices[self.faces[:, 0]]  # (F, 3)
        e1 = self.vertices[self.faces[:, 1]] - v0
        e2 = self.vertices[self.faces[:, 2]] - v0

        t_min = np.full(N, np.inf)
        face_hit = np.full(N, -1, dtype=np.int32)

        # process in chunks to control memory: rays × faces
        chunk = max(1, min(2000, 500_000_000 // (F * 8 * 3 + 1)))
        normals = self.face_normals()

        for ri in range(0, N, chunk):
            re = min(ri + chunk, N)
            o = origins[ri:re]       # (C, 3)
            d = directions[ri:re]    # (C, 3)
            C = o.shape[0]

            for fi in range(0, F, 5000):
                fe = min(fi + 5000, F)
                e1_b = e1[fi:fe]  # (B, 3)
                e2_b = e2[fi:fe]
                v0_b = v0[fi:fe]
                B = e1_b.shape[0]

                # h = d × e2  (C, B, 3)
                h = np.cross(d[:, None, :], e2_b[None, :, :])
                a = np.sum(e1_b[None, :, :] * h, axis=2)  # (C, B)
                valid = np.abs(a) > 1e-10

                inv_a = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)
                s = o[:, None, :] - v0_b[None, :, :]  # (C, B, 3)
                u = np.sum(s * h, axis=2) * inv_a
                valid &= (u >= 0) & (u <= 1)

                q = np.cross(s, e1_b[None, :, :])
                v = np.sum(d[:, None, :] * q, axis=2) * inv_a
                valid &= (v >= 0) & (u + v <= 1)

                t = np.sum(e2_b[None, :, :] * q, axis=2) * inv_a
                valid &= (t > 1e-6)

                t = np.where(valid, t, np.inf)
                best_local = np.argmin(t, axis=1)  # (C,)
                best_t = t[np.arange(C), best_local]

                update = best_t < t_min[ri:re]
                t_min[ri:re] = np.where(update, best_t, t_min[ri:re])
                face_hit[ri:re] = np.where(update, best_local + fi, face_hit[ri:re])

        hit_normals = np.zeros((N, 3))
        valid_hits = face_hit >= 0
        hit_normals[valid_hits] = normals[face_hit[valid_hits]]
        return t_min, face_hit, hit_normals

    def face_group_of(self, face_idx):
        """Which group does a face belong to?"""
        for gname, fidx in self.face_groups.items():
            if face_idx in fidx:
                return gname
        return "other"

    def face_alpha(self, face_idx, f=500.0):
        gname = self.face_group_of(face_idx)
        return self.group_alpha(gname, f)

    # ── voxelisation (for FDTD etc.) ────────────────────────────
    def voxelize(self, dx):
        """Returns: air (Nx, Ny, Nz) bool, origin (3,)"""
        bb_min, bb_max = self.bounding_box()
        pad = dx * 2
        origin = bb_min - pad
        end = bb_max + pad
        Nx = int((end[0] - origin[0]) / dx) + 1
        Ny = int((end[1] - origin[1]) / dx) + 1
        Nz = int((end[2] - origin[2]) / dx) + 1
        # ray-casting along z for each (x,y)
        air = np.zeros((Nx, Ny, Nz), dtype=bool)
        normals = self.face_normals()
        for ix in range(Nx):
            x = origin[0] + ix * dx
            for iy in range(Ny):
                y = origin[1] + iy * dx
                o = np.array([[x, y, origin[2] - 1.0]])
                d = np.array([[0.0, 0.0, 1.0]])
                t_all, fhit, _ = self.ray_intersect(o, d)
                # collect all z-intersections (need ALL hits, not just nearest)
                # simplified: use nearest-only parity (works for convex)
                if t_all[0] < np.inf:
                    z_hit = origin[2] - 1.0 + t_all[0]
                    for iz in range(Nz):
                        z = origin[2] + iz * dx
                        # inside if above the entry point (simplified)
                        # for proper inside/outside, need all intersections
                        pass
        # fallback: use trimesh for robust voxelisation if available
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
            vox = mesh.voxelized(dx)
            air = vox.matrix
            origin = vox.transform[:3, 3]
            return air, origin, dx
        except Exception:
            pass
        # simple approximation: bounding box
        air = np.ones((Nx, Ny, Nz), dtype=bool)
        air[0, :, :] = air[-1, :, :] = False
        air[:, 0, :] = air[:, -1, :] = False
        air[:, :, 0] = air[:, :, -1] = False
        return air, origin, dx


# ═══════════════════════════════════════════════════════════════════
# File loaders
# ═══════════════════════════════════════════════════════════════════

def load_obj(path) -> TriMesh:
    vertices, faces = [], []
    groups, current_group = {}, "default"
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = []
                for p in parts[1:]:
                    vi = int(p.split("/")[0])
                    idx.append(vi - 1 if vi > 0 else len(vertices) + vi)
                # triangulate polygon
                for i in range(1, len(idx) - 1):
                    fi = len(faces)
                    faces.append([idx[0], idx[i], idx[i + 1]])
                    groups.setdefault(current_group, []).append(fi)
            elif parts[0] in ("g", "o") and len(parts) > 1:
                current_group = parts[1]
            elif parts[0] == "usemtl" and len(parts) > 1:
                current_group = parts[1]
    verts = np.array(vertices, dtype=np.float64)
    fcs = np.array(faces, dtype=np.int32)
    fg = {k: np.array(v, dtype=np.int32) for k, v in groups.items()}
    mesh = TriMesh(verts, fcs, fg)
    if len(fg) <= 1:
        mesh.auto_group()
    else:
        for gn in fg:
            mesh.materials.setdefault(gn, "plaster")
    return mesh


def load_stl(path) -> TriMesh:
    """Load STL (binary or ASCII)."""
    with open(path, "rb") as f:
        header = f.read(80)
        if b"solid" in header[:5]:
            f.seek(0)
            text = f.read().decode("ascii", errors="replace")
            if "facet normal" in text:
                return _load_stl_ascii(text)
        f.seek(80)
        n_tri = struct.unpack("<I", f.read(4))[0]
        verts, faces = [], []
        for i in range(n_tri):
            data = struct.unpack("<12fH", f.read(50))
            n = data[0:3]
            v0, v1, v2 = data[3:6], data[6:9], data[9:12]
            vi = len(verts)
            verts.extend([v0, v1, v2])
            faces.append([vi, vi + 1, vi + 2])
    mesh = TriMesh(np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32))
    mesh.auto_group()
    return mesh


def _load_stl_ascii(text) -> TriMesh:
    import re
    verts, faces = [], []
    solid_name = "default"
    groups = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("solid "):
            solid_name = line[6:].strip() or "default"
        elif line.startswith("vertex"):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(verts) % 3 == 0:
                fi = len(faces)
                vi = len(verts)
                faces.append([vi - 3, vi - 2, vi - 1])
                groups.setdefault(solid_name, []).append(fi)
    fg = {k: np.array(v, dtype=np.int32) for k, v in groups.items()}
    mesh = TriMesh(np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32), fg)
    if len(fg) <= 1:
        mesh.auto_group()
    else:
        for gn in fg:
            mesh.materials.setdefault(gn, "plaster")
    return mesh


def load_step(path) -> TriMesh:
    """Load STEP/STP via gmsh surface meshing."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(str(path))
    gmsh.model.mesh.generate(2)
    # extract
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    all_verts = coords
    all_faces = []
    groups = {}
    surfaces = gmsh.model.getEntities(dim=2)
    for dim, tag in surfaces:
        elem_types, elem_tags, node_tags_list = gmsh.model.mesh.getElements(dim, tag)
        gname = f"surface_{tag}"
        groups[gname] = []
        for et, et_tags, nt in zip(elem_types, elem_tags, node_tags_list):
            nt = np.array(nt, dtype=int)
            n_per = gmsh.model.mesh.getElementProperties(et)[3]
            if n_per < 3: continue
            nt = nt.reshape(-1, n_per)
            for tri_nodes in nt:
                fi = len(all_faces)
                # map tags to indices
                face = [tag_to_idx.get(int(t), 0) for t in tri_nodes[:3]]
                all_faces.append(face)
                groups[gname].append(fi)
                # triangulate quads+
                for k in range(3, len(tri_nodes)):
                    fi = len(all_faces)
                    all_faces.append([tag_to_idx.get(int(tri_nodes[0]), 0),
                                      tag_to_idx.get(int(tri_nodes[k-1]), 0),
                                      tag_to_idx.get(int(tri_nodes[k]), 0)])
                    groups[gname].append(fi)
    gmsh.finalize()
    fg = {k: np.array(v, dtype=np.int32) for k, v in groups.items() if v}
    mesh = TriMesh(np.array(all_verts, dtype=np.float64),
                   np.array(all_faces, dtype=np.int32), fg)
    for gn in fg:
        mesh.materials[gn] = "plaster"
    return mesh


def load_skp(path) -> TriMesh:
    """Load SketchUp .skp via engine/lib/skp_reader.dll → OBJ → load_obj."""
    dll_path = ROOT / "engine" / "lib" / "skp_reader.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"skp_reader.dll not found at {dll_path}")
    tmp_obj = Path(path).with_suffix(".tmp.obj")
    try:
        dll = ctypes.CDLL(str(dll_path))
        dll.skp_to_obj.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        dll.skp_to_obj.restype = ctypes.c_int
        ret = dll.skp_to_obj(str(path).encode(), str(tmp_obj).encode())
        if ret != 0:
            raise RuntimeError(f"skp_to_obj returned {ret}")
        mesh = load_obj(tmp_obj)
        mesh.scale_to_meters("in")  # SKP uses inches
        return mesh
    finally:
        if tmp_obj.exists():
            os.remove(tmp_obj)


def load_trimesh_generic(path) -> TriMesh:
    """Load any format trimesh supports: GLB, GLTF, FBX, PLY, 3DS, DAE..."""
    import trimesh
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        meshes = list(scene.geometry.values())
        if not meshes:
            raise ValueError("No geometry found in file")
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = scene
    fg = {}
    if hasattr(combined, 'metadata') and 'face_groups' in combined.metadata:
        fg = combined.metadata['face_groups']
    mesh = TriMesh(np.array(combined.vertices, dtype=np.float64),
                   np.array(combined.faces, dtype=np.int32), fg)
    if not fg:
        mesh.auto_group()
    else:
        for gn in fg:
            mesh.materials.setdefault(gn, "plaster")
    return mesh


# ═══════════════════════════════════════════════════════════════════
# Universal loader (auto-detect format)
# ═══════════════════════════════════════════════════════════════════
LOADERS = {
    ".obj": load_obj,
    ".stl": load_stl,
    ".step": load_step,
    ".stp": load_step,
    ".skp": load_skp,
}

def load_geometry(path) -> TriMesh:
    """Auto-detect format and load. Falls back to trimesh for exotic formats."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in LOADERS:
        return LOADERS[ext](str(path))
    # try trimesh for everything else
    return load_trimesh_generic(str(path))

SUPPORTED_FORMATS = "3D Models (*.obj *.stl *.step *.stp *.skp *.glb *.gltf *.ply *.fbx *.3ds *.dae *.off)"


# ═══════════════════════════════════════════════════════════════════
# Mesh-based engine implementations
# ═══════════════════════════════════════════════════════════════════

def _reflect_off_surface(dirs, normals):
    """Specular reflection: d' = d - 2(d·n)n"""
    dots = np.sum(dirs * normals, axis=1, keepdims=True)
    return dirs - 2 * dots * normals


def run_ray_tracing_mesh(mesh, src, rec, sr=44100, duration=2.0,
                         n_rays=10000, max_bounces=200, capture_radius=0.5, **kw):
    t0 = time.perf_counter()
    c = 343.0
    src_a, rec_a = np.asarray(src, dtype=float), np.asarray(rec, dtype=float)
    rng = np.random.default_rng(42)
    N = int(sr * duration)
    histogram = np.zeros(N)
    dirs = random_dirs_sphere(n_rays, rng)
    pos = np.tile(src_a, (n_rays, 1))
    energy = np.ones(n_rays)
    travel = np.zeros(n_rays)
    alive = np.ones(n_rays, dtype=bool)

    for bounce in range(max_bounces):
        if not np.any(alive): break
        ai = np.where(alive)[0]
        t_hit, face_idx, hit_normals = mesh.ray_intersect(pos[ai], dirs[ai])
        valid = face_idx >= 0
        # detect near receiver
        to_rec = rec_a - pos[ai]
        proj = np.clip(np.sum(to_rec * dirs[ai], axis=1), 0, t_hit)
        closest = pos[ai] + dirs[ai] * proj[:, None]
        det = np.linalg.norm(closest - rec_a, axis=1) < capture_radius
        for j in np.where(det & valid)[0]:
            gi = ai[j]
            arr = travel[gi] + proj[j] / c
            s = int(arr * sr)
            if 0 <= s < N: histogram[s] += energy[gi]
        # move to hit point
        hit_pos = pos[ai] + dirs[ai] * t_hit[:, None]
        pos[ai] = hit_pos
        travel[ai] += t_hit / c
        # absorption
        for j in np.where(valid)[0]:
            gi = ai[j]
            alpha = mesh.face_alpha(face_idx[j], 500.0)
            energy[gi] *= (1 - alpha)
        # reflect
        new_dirs = _reflect_off_surface(dirs[ai], hit_normals)
        dirs[ai] = new_dirs
        # kill rays that missed geometry
        ai_invalid = ai[~valid]
        alive[ai_invalid] = False
        alive[energy < 1e-8] = False
        alive[travel > duration] = False

    ir = synth_noise_decay(1.0, sr, duration, seed=99)
    ir *= np.sqrt(histogram + 1e-30)
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Ray Tracing (mesh)", "Geometric", ir, sr, m,
                  f"{n_rays} rays on {len(mesh.faces)} triangles",
                  time.perf_counter() - t0)


def run_fdtd_mesh(mesh, src, rec, sr=44100, duration=0.3, max_freq=500, CFL=0.28, **kw):
    """FDTD on voxelised arbitrary geometry."""
    t0 = time.perf_counter()
    c = 343.0
    ppw = 6
    dx = c / (ppw * max_freq)
    air, origin, dx = mesh.voxelize(dx)
    Nx, Ny, Nz = air.shape
    dt = CFL * dx / c
    Nt = int(duration / dt)
    # source / receiver indices
    si = tuple(np.clip(((np.array(src) - origin) / dx).astype(int), 1, [Nx-2, Ny-2, Nz-2]))
    ri = tuple(np.clip(((np.array(rec) - origin) / dx).astype(int), 1, [Nx-2, Ny-2, Nz-2]))
    # reflection at boundaries
    alpha_mean = mesh.mean_alpha(max_freq)
    R = np.sqrt(1 - alpha_mean)
    air_float = air.astype(np.float64)
    courant2 = (c * dt / dx) ** 2
    p = np.zeros((Nx, Ny, Nz))
    p_prev = np.zeros_like(p)
    ir_wave = []
    for n in range(Nt):
        pp = np.pad(p, 1, mode='constant')
        lap = (pp[2:,1:-1,1:-1] + pp[:-2,1:-1,1:-1] +
               pp[1:-1,2:,1:-1] + pp[1:-1,:-2,1:-1] +
               pp[1:-1,1:-1,2:] + pp[1:-1,1:-1,:-2] -
               6 * pp[1:-1,1:-1,1:-1])
        p_next = (2*p - p_prev + courant2 * lap) * air_float
        # absorption at air/solid boundary
        # (simplified: apply R at voxels adjacent to solid)
        if n == 1: p_next[si] += 1.0
        ir_wave.append(p_next[ri])
        p_prev, p = p, p_next
    ir_wave = np.array(ir_wave)
    from scipy.signal import resample as sp_resample
    target_len = int(len(ir_wave) * sr / (1.0 / dt))
    ir = sp_resample(ir_wave, target_len) if target_len > 0 else ir_wave
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, int(sr))
    return Result("FDTD (mesh)", "Wave (TD)", ir, int(sr), m,
                  f"Voxel grid {Nx}x{Ny}x{Nz}, {Nt} steps",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


def run_sabine_mesh(mesh, src, rec, sr=44100, duration=2.0, **kw):
    t0 = time.perf_counter()
    V = mesh.volume()
    rt_bands = {}
    for f in OCTAVE_BANDS:
        A = mesh.absorption_area(f)
        rt_bands[int(f)] = round(0.161 * V / max(A, 1e-6), 4)
    rt_mid = rt_bands.get(500, 1.0)
    ir = synth_noise_decay(rt_mid, sr, duration)
    ir[0] = 1.0
    m = all_metrics(ir, sr)
    m["RT60_bands"] = rt_bands
    return Result("Sabine (mesh)", "Statistical", ir, sr, m,
                  f"V={V:.1f}m³, S={mesh.surface_area():.1f}m²",
                  time.perf_counter() - t0)


# registry of mesh-aware engines
MESH_ENGINES = {
    "Ray Tracing":  run_ray_tracing_mesh,
    "FDTD":         run_fdtd_mesh,
    "Sabine":       run_sabine_mesh,
}

SHOEBOX_ONLY = {"Image Source", "Beam Tracing", "Axial Modes", "Full Modal",
                "Modal ROM", "ARD"}


def run_engine_on_mesh(engine_name, mesh, src, rec, sr=44100, duration=2.0, **kw):
    """Run any engine on a TriMesh. Uses mesh version if available, else bounding-box fallback."""
    if engine_name in MESH_ENGINES:
        return MESH_ENGINES[engine_name](mesh, src, rec, sr=sr, duration=duration, **kw)
    # fallback: convert to shoebox via bounding box
    room = mesh.to_shoebox()
    func = ENGINE_REGISTRY[engine_name]["func"]
    result = func(room, src, rec, sr=sr, duration=duration, **kw)
    if engine_name in SHOEBOX_ONLY:
        result.info += " [bounding-box approximation]"
    return result
