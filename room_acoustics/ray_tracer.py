"""
Stochastic ray tracer for room acoustics.

Traces rays from source through multiple reflections, recording
energy at the receiver. Produces a reflectogram (energy vs time)
that shapes the late reverberant tail.

Each ray:
  1. Starts at source with unit energy
  2. Travels until it hits a surface
  3. Loses energy: E *= (1 - alpha)
  4. Reflects: specular (1-s) + diffuse scatter (s)
  5. Repeat until energy < threshold or max reflections

The receiver collects energy within a capture radius.
The reflectogram is converted to an IR by modulating noise.

Works on arbitrary geometry via triangle mesh ray-triangle intersection.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def _ray_triangle_intersect(origin, direction, v0, v1, v2):
    """Moller-Trumbore ray-triangle intersection.

    Returns (t, u, v) where t is the distance along ray,
    or (-1, 0, 0) if no intersection.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)

    if abs(a) < 1e-10:
        return -1, 0, 0

    f = 1.0 / a
    s = origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return -1, 0, 0

    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return -1, 0, 0

    t = f * np.dot(edge2, q)
    if t > 1e-6:
        return t, u, v
    return -1, 0, 0


class RoomMesh:
    """Triangle mesh for ray tracing. Extracted from the tet mesh boundary."""

    def __init__(self, mesh, ops):
        """Build triangle mesh from boundary faces of the tet/hex mesh."""
        self.vertices = np.column_stack([mesh.x, mesh.y, mesh.z])
        self.triangles = []
        self.normals = []
        self.surface_labels = []
        self.surface_alpha = {}

        # Extract boundary triangles from the mesh
        if hasattr(mesh, '_boundary_faces'):
            for label, faces in mesh._boundary_faces.items():
                face_arr = np.asarray(faces, dtype=int)
                for f in face_arr:
                    if len(f) == 6:
                        # P=2 triangle: use corner vertices (0,1,2)
                        self.triangles.append([f[0], f[1], f[2]])
                        self.surface_labels.append(label)
                    elif len(f) == 3:
                        self.triangles.append([f[0], f[1], f[2]])
                        self.surface_labels.append(label)
                    elif len(f) == 4:
                        # Quad: split into 2 triangles
                        self.triangles.append([f[0], f[1], f[2]])
                        self.surface_labels.append(label)
                        self.triangles.append([f[0], f[2], f[3]])
                        self.surface_labels.append(label)

        self.triangles = np.array(self.triangles, dtype=int)
        self.n_triangles = len(self.triangles)

        # Compute normals
        for tri in self.triangles:
            v0, v1, v2 = self.vertices[tri]
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 0:
                self.normals.append(n / norm)
            else:
                self.normals.append(np.array([0, 0, 1.0]))
        self.normals = np.array(self.normals)

    def set_alpha(self, label, alpha):
        """Set absorption coefficient for a surface label."""
        self.surface_alpha[label] = alpha

    def _build_accel(self):
        """Precompute triangle vertices for vectorized intersection."""
        self._v0 = self.vertices[self.triangles[:, 0]]  # (N, 3)
        self._v1 = self.vertices[self.triangles[:, 1]]
        self._v2 = self.vertices[self.triangles[:, 2]]
        self._e1 = self._v1 - self._v0  # (N, 3)
        self._e2 = self._v2 - self._v0

    def intersect(self, origin, direction):
        """Find closest intersection — vectorized over all triangles."""
        if not hasattr(self, '_v0'):
            self._build_accel()

        # Vectorized Moller-Trumbore
        h = np.cross(direction, self._e2)           # (N, 3)
        a = np.sum(self._e1 * h, axis=1)            # (N,)

        valid = np.abs(a) > 1e-10
        f = np.zeros(self.n_triangles)
        f[valid] = 1.0 / a[valid]

        s = origin - self._v0                        # (N, 3)
        u = f * np.sum(s * h, axis=1)               # (N,)
        valid &= (u >= 0) & (u <= 1)

        q = np.cross(s, self._e1)                    # (N, 3)
        v = f * np.sum(direction * q, axis=1)        # (N,)
        valid &= (v >= 0) & (u + v <= 1)

        t = f * np.sum(self._e2 * q, axis=1)        # (N,)
        valid &= (t > 1e-6)

        if not np.any(valid):
            return None

        t[~valid] = 1e30
        best_idx = np.argmin(t)
        best_t = t[best_idx]

        hit = origin + best_t * direction
        normal = self.normals[best_idx]
        return best_t, best_idx, hit, normal


def trace_rays(room_mesh, source, receiver, n_rays=5000,
               max_order=200, capture_radius=0.5,
               scatter_coeff=0.1, c=343.0, sr=44100, T=3.5):
    """
    Trace rays from source, record energy arriving near receiver.

    Parameters
    ----------
    room_mesh : RoomMesh
    source : (x, y, z)
    receiver : (x, y, z)
    n_rays : number of rays to cast
    max_order : max reflection order
    capture_radius : receiver capture sphere radius [m]
    scatter_coeff : surface scattering coefficient (0=specular, 1=diffuse)
    c : speed of sound [m/s]
    sr : sample rate [Hz]
    T : max time [s]

    Returns
    -------
    reflectogram : energy histogram (n_bins,)
    t_bins : time bin centers [s]
    """
    n_bins = int(T * sr)
    dt = 1.0 / sr
    reflectogram = np.zeros(n_bins)

    src = np.array(source, dtype=float)
    rec = np.array(receiver, dtype=float)

    # Direct sound
    dist_direct = np.linalg.norm(rec - src)
    t_direct = dist_direct / c
    n_direct = int(t_direct * sr)
    if 0 <= n_direct < n_bins:
        reflectogram[n_direct] += 1.0 / (4 * np.pi * dist_direct**2)

    rng = np.random.RandomState(42)

    for ray_i in range(n_rays):
        # Random direction (uniform on sphere)
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

        origin = src.copy()
        energy = 1.0
        total_dist = 0.0

        for bounce in range(max_order):
            result = room_mesh.intersect(origin, direction)
            if result is None:
                break  # ray escaped (shouldn't happen in closed room)

            t_hit, tri_idx, hit_point, normal = result
            total_dist += t_hit

            # Get absorption
            label = room_mesh.surface_labels[tri_idx]
            alpha = room_mesh.surface_alpha.get(label, 0.05)

            # Absorb
            energy *= (1.0 - alpha)

            if energy < 1e-8:
                break

            # Check if ray passes near receiver
            # Project receiver onto ray path
            ray_to_rec = rec - origin
            proj_len = np.dot(ray_to_rec, direction)
            if 0 < proj_len < t_hit:
                closest = origin + proj_len * direction
                dist_to_rec = np.linalg.norm(closest - rec)
                if dist_to_rec < capture_radius:
                    arrival_time = total_dist / c
                    n_bin = int(arrival_time * sr)
                    if 0 <= n_bin < n_bins:
                        # Energy contribution weighted by solid angle
                        weight = energy / (4 * np.pi * max(dist_to_rec, 0.01)**2)
                        weight *= capture_radius**2  # normalize by capture area
                        reflectogram[n_bin] += weight / n_rays

            # Reflect
            if rng.random() < scatter_coeff:
                # Diffuse reflection: random direction in hemisphere
                # oriented along surface normal
                u1, u2 = rng.random(), rng.random()
                cos_t = np.sqrt(u1)
                sin_t = np.sqrt(1 - u1)
                phi_r = 2 * np.pi * u2

                # Build local frame from normal
                if abs(normal[0]) < 0.9:
                    tangent = np.cross(normal, np.array([1, 0, 0]))
                else:
                    tangent = np.cross(normal, np.array([0, 1, 0]))
                tangent /= np.linalg.norm(tangent)
                bitangent = np.cross(normal, tangent)

                direction = (cos_t * normal +
                            sin_t * np.cos(phi_r) * tangent +
                            sin_t * np.sin(phi_r) * bitangent)
            else:
                # Specular reflection
                direction = direction - 2 * np.dot(direction, normal) * normal

            direction /= np.linalg.norm(direction)
            origin = hit_point + 1e-6 * direction  # offset to avoid self-hit

        if (ray_i + 1) % (n_rays // 5) == 0:
            print(f"  rays: {ray_i+1}/{n_rays}", end='', flush=True)

    print(" done")
    return reflectogram, np.arange(n_bins) * dt


def reflectogram_to_ir(reflectogram, sr=44100):
    """Convert energy reflectogram to pressure IR.

    Modulates Gaussian noise by the square root of the energy envelope.
    """
    np.random.seed(123)
    noise = np.random.randn(len(reflectogram))

    # Smooth the reflectogram (energy envelope)
    from scipy.ndimage import uniform_filter1d
    window = max(1, int(0.001 * sr))  # 1ms smoothing
    envelope = uniform_filter1d(reflectogram, window)
    envelope = np.maximum(envelope, 0)

    ir = noise * np.sqrt(envelope)
    return ir
