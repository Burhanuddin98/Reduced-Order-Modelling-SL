"""
Room — the complete room acoustics simulation platform.

Hybrid engine:
  - Modal ROM (0 to f_cross): eigenmodes with analytical time evolution
  - ISM (f_cross to 20kHz): image source for early + Eyring tail for late

Usage:
    room = Room.from_box(8.4, 6.7, 3.0)
    room.set_material("floor", "carpet_thick")
    room.set_material("ceiling", "acoustic_panel")
    room.set_material_default("plaster")
    room.build()

    ir = room.impulse_response(source=(2, 3.35, 1.5), receiver=(6, 2, 1.2))
    print(f"T30={ir.T30:.2f}s  C80={ir.C80:.1f}dB")
    ir.save_wav("room_ir.wav")
    ir.auralize("dry_speech.wav", "output.wav")
"""

import numpy as np
import os
import time as _time


# ===================================================================
# Impulse Response
# ===================================================================

class ImpulseResponse:
    """Holds an impulse response and computes metrics on demand."""

    def __init__(self, data, sr=44100, ir_modal=None, ir_ism=None):
        self.data = np.asarray(data, dtype=float)
        self.sr = sr
        self.dt = 1.0 / sr
        self.ir_modal = ir_modal  # low-freq component
        self.ir_ism = ir_ism      # high-freq component
        self._metrics = None

    @property
    def metrics(self):
        if self._metrics is None:
            from .acoustics_metrics import all_metrics
            self._metrics = all_metrics(self.data, self.dt)
        return self._metrics

    @property
    def T30(self): return self.metrics['T30_s']
    @property
    def T20(self): return self.metrics['T20_s']
    @property
    def EDT(self): return self.metrics['EDT_s']
    @property
    def C80(self): return self.metrics['C80_dB']
    @property
    def D50(self): return self.metrics['D50']
    @property
    def TS(self): return self.metrics['TS_ms']
    @property
    def duration(self): return len(self.data) / self.sr

    def save_wav(self, path, normalize=True):
        """Save IR as WAV file."""
        import scipy.io.wavfile as wavfile
        ir = self.data.copy()
        if normalize and np.max(np.abs(ir)) > 0:
            ir = ir / np.max(np.abs(ir)) * 0.95
        wavfile.write(path, self.sr, (ir * 32767).astype(np.int16))
        print(f"  Saved: {path} ({self.duration:.2f}s)")

    def auralize(self, audio_path, output_path):
        """Convolve IR with dry audio and save."""
        import scipy.io.wavfile as wavfile
        from scipy.signal import fftconvolve
        sr_in, audio = wavfile.read(audio_path)
        audio = audio.astype(float)
        if audio.ndim > 1: audio = audio[:, 0]
        if sr_in != self.sr:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * self.sr / sr_in))
        audio /= max(np.max(np.abs(audio)), 1e-10)
        wet = fftconvolve(audio, self.data, mode='full')[:len(audio) + self.sr]
        wet = wet / max(np.max(np.abs(wet)), 1e-10) * 0.95
        wavfile.write(output_path, self.sr, (wet * 32767).astype(np.int16))
        print(f"  Auralized: {output_path}")

    def summary(self):
        """Print all metrics."""
        m = self.metrics
        print(f"  T30  = {m['T30_s']:.3f} s (R2={m['T30_R2']:.3f})")
        print(f"  EDT  = {m['EDT_s']:.3f} s")
        print(f"  C80  = {m['C80_dB']:.1f} dB")
        print(f"  D50  = {m['D50']:.3f}")
        print(f"  TS   = {m['TS_ms']:.1f} ms")


# ===================================================================
# Room
# ===================================================================

class Room:
    """
    Room acoustics simulation platform.

    Supports box rooms, Gmsh .geo files, STL/OBJ imports, and
    floor plan extrusion. Per-surface material assignment.
    Hybrid IR: modal ROM (low freq) + ISM (high freq).
    """

    def __init__(self):
        self.mesh = None
        self.ops = None
        self._geometry_type = None
        self._materials = {}
        self._default_material = 'plaster'
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
        self._parallel_pairs = None  # cached parallel surface pairs
        self._built = False
        self._build_time = 0
        self._volume = 0
        self._surface_area = 0
        self._dimensions = None  # for box rooms: (Lx, Ly, Lz)
        self.sr = 44100

    # ============================================================
    # Construction
    # ============================================================

    @classmethod
    def from_box(cls, Lx, Ly, Lz, P=4, ppw=6, f_target=500):
        """Create from rectangular box. Auto-sizes mesh for f_target."""
        room = cls()
        room._geometry_type = 'box'
        room._dimensions = (Lx, Ly, Lz)
        c = 343.0
        lam = c / f_target
        h = lam / ppw * P
        Nex = max(2, int(np.ceil(Lx / h)))
        Ney = max(2, int(np.ceil(Ly / h)))
        Nez = max(2, int(np.ceil(Lz / h)))
        room._box_params = (Nex, Ney, Nez, P)
        return room

    @classmethod
    def from_geo(cls, geo_path, h_min=0.3, h_max=0.8):
        """Create from Gmsh .geo file."""
        room = cls()
        room._geometry_type = 'geo'
        room._geo_path = geo_path
        room._h_min = h_min
        room._h_max = h_max
        return room

    @classmethod
    def from_stl(cls, stl_path, h_target=0.5):
        """Create from STL/OBJ surface mesh."""
        room = cls()
        room._geometry_type = 'stl'
        room._stl_path = stl_path
        room._h_target = h_target
        return room

    @classmethod
    def from_polygon(cls, vertices, Lz, h_target=0.4, P=4):
        """Create from 2D floor plan polygon extruded to height Lz."""
        room = cls()
        room._geometry_type = 'polygon'
        room._poly_verts = vertices
        room._poly_Lz = Lz
        room._poly_h = h_target
        room._poly_P = P
        return room

    # ============================================================
    # Materials
    # ============================================================

    def set_material(self, surface_label, material):
        """Assign a material to a surface.

        Parameters
        ----------
        surface_label : str
            Surface identifier (e.g., 'floor', 'ceiling', 'left').
        material : str or MaterialFunction
            Material name from the database (FI impedance), or a
            MaterialFunction for frequency-dependent absorption.
        """
        from .material_function import MaterialFunction
        if isinstance(material, MaterialFunction):
            self._materials[surface_label] = material
        else:
            from .materials import get_material
            get_material(material)  # validate
            self._materials[surface_label] = material

    def set_material_default(self, material):
        """Set default material for unlabeled surfaces.

        Parameters
        ----------
        material : str or MaterialFunction
        """
        from .material_function import MaterialFunction
        if isinstance(material, MaterialFunction):
            self._default_material = material
        else:
            from .materials import get_material
            get_material(material)
            self._default_material = material

    def list_surfaces(self):
        """Print available surface labels and their materials."""
        labels = self._get_labels()
        for label in labels:
            mat = self._materials.get(label, self._default_material)
            print(f"  {label}: {mat}")

    # ============================================================
    # Build
    # ============================================================

    def build(self, n_modes=400):
        """
        Mesh + assemble + eigensolve. One-time cost.
        After this, impulse_response() is near-instant.
        """
        t0 = _time.perf_counter()
        print("Building room...")

        self._build_mesh()
        n_el = getattr(self.mesh, 'N_el', 0)
        print(f"  Mesh: {self.mesh.N_dof} DOFs, {n_el} elements")

        self._assemble()
        self._volume = float(self.ops['M_diag'].sum())
        self._surface_area = float(self.ops['B_total'].diagonal().sum())
        print(f"  Volume: {self._volume:.1f} m3, Surface: {self._surface_area:.1f} m2")

        labels = self._get_labels()
        print(f"  Surfaces: {list(labels)}")

        n_modes = min(n_modes, self.mesh.N_dof - 2)
        print(f"  Eigenmodes ({n_modes})...", end='', flush=True)
        from .modal_rom import compute_room_modes
        self._eigenvalues, self._eigenvectors, self._frequencies = \
            compute_room_modes(self.ops, n_modes)
        f_max = self._frequencies[-1]
        print(f" f_max={f_max:.0f} Hz")

        # Precompute per-surface modal coupling weights for fast
        # material changes (gamma_i = sum_s(w_s_i / Z_s(f_i)))
        if self._geometry_type == 'box' and self._dimensions is not None:
            from .calibrate_absorption import precompute_surface_gamma_weights
            self._surface_weights = precompute_surface_gamma_weights(
                self.mesh, self.ops, self._eigenvectors, self._dimensions)
        else:
            self._surface_weights = None

        # Build ray tracing mesh from boundary faces
        print("  Ray trace mesh...", end='', flush=True)
        from .ray_tracer import RoomMesh
        self._rt_mesh = RoomMesh(self.mesh, self.ops)
        if self._rt_mesh.n_triangles == 0 and self._geometry_type == 'box':
            # Box mesh doesn't have _boundary_faces — build from box geometry
            self._rt_mesh = self._build_box_rt_mesh()
        print(f" {self._rt_mesh.n_triangles} triangles")

        # Detect parallel surfaces for axial mode engine
        from .axial_modes import detect_parallel_surfaces, detect_parallel_surfaces_box
        if self._geometry_type == 'box' and self._dimensions is not None:
            self._parallel_pairs = detect_parallel_surfaces_box(self._dimensions)
        else:
            self._parallel_pairs = detect_parallel_surfaces(self._rt_mesh)
        n_pairs = len(self._parallel_pairs)
        if n_pairs > 0:
            dists = [f"{p.distance:.2f}m" for p in self._parallel_pairs]
            print(f"  Parallel pairs: {n_pairs} ({', '.join(dists)})")
        else:
            print("  Parallel pairs: none detected")

        self._built = True
        self._build_time = _time.perf_counter() - t0
        print(f"  Build: {self._build_time:.1f}s")

    # ============================================================
    # Impulse Response
    # ============================================================

    def impulse_response(self, source, receiver, T=3.5, sigma=0.3,
                         n_rays=2000, max_bounces=150):
        """
        Compute hybrid IR: modal ROM (low) + ray tracing (high).

        Three engines:
          1. Modal ROM: 0 to f_cross (eigenmodes, dispersion-free)
          2. Ray tracer: shapes the full-bandwidth late reverberant tail
          3. ISM (box rooms only): early specular reflections

        Parameters
        ----------
        source : (x, y, z)
        receiver : (x, y, z)
        T : IR duration [s]
        sigma : source pulse width [m]
        n_rays : rays for ray tracer (more = smoother tail)
        max_bounces : max reflection order for ray tracer
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        sr = self.sr
        dt = 1.0 / sr
        n_samples = int(T * sr)
        f_cross = float(self._frequencies[-1]) * 0.85
        nyq = sr / 2

        rec_idx = self.mesh.nearest_node(*receiver)

        # Try spectral (frequency-dependent) decay first
        gamma_spectral = self._compute_spectral_decay()
        use_spectral = gamma_spectral is not None

        if use_spectral:
            print(f"  IR: modal 0-{f_cross:.0f}Hz (spectral alpha), "
                  f"axial {f_cross:.0f}-8000Hz, ray tracer {f_cross:.0f}-{nyq:.0f}Hz")
        else:
            print(f"  IR: modal 0-{f_cross:.0f}Hz, axial {f_cross:.0f}-8000Hz, "
                  f"ray tracer {f_cross:.0f}-{nyq:.0f}Hz")

        # === ENGINE 1: Modal ROM (low frequencies) ===
        t0 = _time.perf_counter()
        if use_spectral:
            # Spectral path: per-mode decay from alpha(f_i)
            from .calibrate_absorption import synthesize_ir_fast
            c = 343.0
            omega = np.sqrt(np.maximum(self._eigenvalues, 0)) * c
            M = self.ops['M_diag']
            r2 = ((self.mesh.x - source[0]) ** 2 +
                  (self.mesh.y - source[1]) ** 2 +
                  (self.mesh.z - source[2]) ** 2)
            p0 = np.exp(-r2 / sigma ** 2)
            modal_amps = self._eigenvectors.T @ (M * p0)
            ir_modal = synthesize_ir_fast(
                self._eigenvalues, self._eigenvectors, omega,
                modal_amps, gamma_spectral, rec_idx, dt, T)[:n_samples]
        else:
            # Legacy FI path
            Z = self._build_impedance()
            from .modal_rom import modal_ir
            res = modal_ir(self.mesh, self.ops,
                           self._eigenvalues, self._eigenvectors,
                           'FI', {'Z_per_node': Z},
                           *source, sigma, dt, T, rec_idx)
            ir_modal = res['ir'][:n_samples]
        t_modal = _time.perf_counter() - t0

        # Low-pass at crossover
        from scipy.signal import butter, filtfilt
        b_lo, a_lo = butter(6, f_cross / nyq, btype='low')
        ir_low = filtfilt(b_lo, a_lo, ir_modal)

        # === ENGINE 2: Axial modes (coherent resonances, f_cross to 8kHz) ===
        # Computed here but level-matched AFTER the ray tracer (needs its energy)
        t0 = _time.perf_counter()
        ir_axial = np.zeros(n_samples)
        axial_info = []
        n_axial_modes = 0
        if self._parallel_pairs:
            from .axial_modes import axial_mode_ir
            ir_axial_raw, axial_info = axial_mode_ir(
                self._parallel_pairs, source, receiver,
                self._materials, self._default_material,
                T=T, sr=sr, f_min=f_cross, f_max=8000,
                room_volume=self._volume,
                room_surface_area=self._surface_area)
            ir_axial_raw = ir_axial_raw[:n_samples]

            # Band-pass filter: f_cross to 8kHz
            f_axial_hi = min(8000, nyq * 0.95)
            b_ax, a_ax = butter(4, [f_cross / nyq, f_axial_hi / nyq], btype='band')
            ir_axial = filtfilt(b_ax, a_ax, ir_axial_raw)
            n_axial_modes = len(axial_info)
        t_axial = _time.perf_counter() - t0

        # === ENGINE 3: Ray tracer (late reverberation, all frequencies) ===
        t0 = _time.perf_counter()

        # Set material absorption on ray trace mesh
        # Use alpha at 1 kHz as representative for ray tracer (broadband geometric)
        from .material_function import MaterialFunction
        from .acoustics_metrics import impedance_to_alpha
        for label in self._get_labels():
            mat_ref = self._materials.get(label, self._default_material)
            if isinstance(mat_ref, MaterialFunction):
                self._rt_mesh.set_alpha(label, mat_ref(1000.0))
            else:
                from .materials import get_material
                mat = get_material(mat_ref)
                self._rt_mesh.set_alpha(label, impedance_to_alpha(mat['Z']))

        from .ray_tracer import reflectogram_to_ir
        reflecto = self._ray_trace_c(source, receiver, n_rays, max_bounces, T)
        ir_rt = reflectogram_to_ir(reflecto, sr)

        # High-pass the ray tracer output at crossover
        b_hi, a_hi = butter(6, f_cross / nyq, btype='high')
        ir_high = filtfilt(b_hi, a_hi, ir_rt[:n_samples])

        # Level-match: scale ray tracer to be subordinate to modal ROM.
        # The modal ROM gives the correct decay rate — the ray tracer
        # provides high-frequency texture but shouldn't dominate the
        # Schroeder decay curve.
        n_start = int(0.05 * sr)
        n_end = int(0.15 * sr)
        rms_low = np.sqrt(np.mean(ir_low[n_start:n_end]**2))
        rms_high = np.sqrt(np.mean(ir_high[n_start:n_end]**2))
        if rms_high > 1e-30 and rms_low > 1e-30:
            # Ray tracer at 15% of modal level — provides texture, not energy
            scale = 0.15 * rms_low / rms_high
            ir_high *= scale

        t_rt = _time.perf_counter() - t0

        # === ENGINE 4: ISM (early reflections, box rooms only) ===
        t0 = _time.perf_counter()
        ir_ism_early = np.zeros(n_samples)
        if self._geometry_type == 'box' and self._dimensions is not None:
            Lx, Ly, Lz = self._dimensions
            alpha_w = self._get_wall_alpha()
            from .image_source import image_sources_shoebox
            ir_ism_raw, _ = image_sources_shoebox(
                Lx, Ly, Lz, source, receiver,
                max_order=5, alpha_walls=alpha_w, sr=sr, T=T)
            n = min(len(ir_ism_raw), n_samples)
            # Only keep first 80ms of ISM (early reflections)
            n_early = min(int(0.08 * sr), n)
            fade = int(0.02 * sr)
            window = np.ones(n)
            window[n_early:n_early+fade] = np.linspace(1, 0, min(fade, n-n_early))
            window[n_early+fade:] = 0
            ir_ism_bp = filtfilt(b_hi, a_hi, ir_ism_raw[:n] * window[:n])
            ir_ism_early[:n] = ir_ism_bp
        t_ism = _time.perf_counter() - t0

        # === Level-match axial modes to ray tracer ===
        # The axial modes add coherent resonant peaks; the ray tracer
        # provides the diffuse energy envelope. Scale axial to be at
        # the same energy level as the ray tracer (they occupy the
        # same frequency range), so they add spectral detail without
        # dominating the overall energy.
        if n_axial_modes > 0:
            n_start = int(0.02 * sr)
            n_end = min(int(0.15 * sr), n_samples)
            rms_rt = np.sqrt(np.mean(ir_high[n_start:n_end]**2))
            rms_axial = np.sqrt(np.mean(ir_axial[n_start:n_end]**2))
            if rms_axial > 1e-30 and rms_rt > 1e-30:
                # Axial at same level as ray tracer — they combine additively
                ir_axial *= rms_rt / rms_axial

        # === BLEND ===
        ir_total = ir_low + ir_axial + ir_high + ir_ism_early

        print(f"  Modal: {t_modal:.2f}s, Axial: {t_axial:.3f}s ({n_axial_modes} modes), "
              f"Ray trace: {t_rt:.1f}s, ISM: {t_ism:.2f}s")

        return ImpulseResponse(ir_total, sr, ir_modal=ir_low, ir_ism=ir_high)

    # ============================================================
    # Internal: mesh building
    # ============================================================

    def _build_mesh(self):
        if self._geometry_type == 'box':
            from .sem import BoxMesh3D, assemble_3d_operators
            Nex, Ney, Nez, P = self._box_params
            Lx, Ly, Lz = self._dimensions
            self.mesh = BoxMesh3D(Lx, Ly, Lz, Nex, Ney, Nez, P)

        elif self._geometry_type == 'geo':
            import gmsh
            from .gmsh_tet_import import _extract_mesh_data
            from .tet_sem import TetMesh3D
            gmsh.initialize()
            gmsh.option.setNumber('General.Terminal', 0)
            gmsh.open(self._geo_path)
            gmsh.option.setNumber('Mesh.CharacteristicLengthMin', self._h_min)
            gmsh.option.setNumber('Mesh.CharacteristicLengthMax', self._h_max)
            gmsh.option.setNumber('Mesh.Algorithm3D', 1)
            gmsh.option.setNumber('Mesh.ElementOrder', 2)
            gmsh.option.setNumber('Mesh.Optimize', 1)
            gmsh.model.mesh.generate(3)
            data = _extract_mesh_data(2)
            gmsh.finalize()
            self.mesh = TetMesh3D(data['nodes'], data['tets'], data['boundary'])

        elif self._geometry_type == 'stl':
            from .gmsh_tet_import import import_surface_mesh
            from .tet_sem import TetMesh3D
            data = import_surface_mesh(self._stl_path, self._h_target, P=2)
            self.mesh = TetMesh3D(data['nodes'], data['tets'], data['boundary'])

        elif self._geometry_type == 'polygon':
            from .geometry import RoomGeometry, generate_quad_mesh, extrude_quad_mesh
            from .unstructured_sem import UnstructuredHexMesh3D
            geom = RoomGeometry(self._poly_verts)
            raw = generate_quad_mesh(geom, self._poly_h, self._poly_P)
            raw3d = extrude_quad_mesh(raw, self._poly_Lz,
                                      max(2, int(self._poly_Lz / self._poly_h)))
            self.mesh = UnstructuredHexMesh3D(raw3d, raw['nodes'],
                                              raw['quads'], self._poly_P)

        if hasattr(self.mesh, '_ensure_coords'):
            self.mesh._ensure_coords()

    def _assemble(self):
        if self._geometry_type == 'box':
            from .sem import assemble_3d_operators
            self.ops = assemble_3d_operators(self.mesh)
        elif self._geometry_type in ('geo', 'stl'):
            from .tet_sem import assemble_tet_3d_operators
            self.ops = assemble_tet_3d_operators(self.mesh)
        elif self._geometry_type == 'polygon':
            from .unstructured_sem import assemble_unstructured_3d_operators
            self.ops = assemble_unstructured_3d_operators(self.mesh)

    def _get_labels(self):
        if hasattr(self.mesh, '_boundary_nodes_per_label'):
            return list(self.mesh._boundary_nodes_per_label.keys())
        return []

    def _resolve_material_function(self, label):
        """Get MaterialFunction for a surface label."""
        from .material_function import MaterialFunction
        mat = self._materials.get(label, self._default_material)
        if isinstance(mat, MaterialFunction):
            return mat
        # Legacy string name -> convert to MaterialFunction via FI impedance
        from .materials import get_material
        m = get_material(mat)
        return MaterialFunction.from_impedance_scalar(m['Z'], name=mat)

    def _compute_spectral_decay(self):
        """Compute per-mode decay rates using frequency-dependent absorption.

        Each mode's decay uses alpha(f_i) at its own eigenfrequency.
        Returns gamma array (n_modes,).
        """
        from .material_function import compute_modal_decay_spectral

        if self._surface_weights is None:
            # Fallback: use FI impedance path
            return None

        mat_funcs = {}
        for label in self._surface_weights:
            mat_funcs[label] = self._resolve_material_function(label)

        return compute_modal_decay_spectral(
            self._surface_weights, mat_funcs, self._frequencies, c=343.0)

    def _build_impedance(self):
        """Build per-node impedance from material assignments (FI legacy path)."""
        from .material_function import MaterialFunction
        from .solvers import RHO_AIR, C_AIR
        rho_c = RHO_AIR * C_AIR

        def _get_Z(mat_ref):
            """Get FI impedance from material ref (string or MaterialFunction)."""
            if isinstance(mat_ref, MaterialFunction):
                return mat_ref.impedance(500.0, rho_c=rho_c)
            from .materials import get_material
            return get_material(mat_ref)['Z']

        N = self.mesh.N_dof
        Z = np.full(N, 1e15)

        if self._geometry_type == 'box':
            Lx, Ly, Lz = self._dimensions
            tol = 1e-6
            face_map = {
                'floor': self.mesh.z < tol,
                'ceiling': self.mesh.z > Lz - tol,
                'left': self.mesh.x < tol,
                'right': self.mesh.x > Lx - tol,
                'front': self.mesh.y < tol,
                'back': self.mesh.y > Ly - tol,
            }
            for label, mask in face_map.items():
                mat_ref = self._materials.get(label, self._default_material)
                Z[mask] = _get_Z(mat_ref)
        else:
            for label in self._get_labels():
                mat_ref = self._materials.get(label, self._default_material)
                nodes = self.mesh.boundary_nodes(label)
                Z[nodes] = _get_Z(mat_ref)

        return Z

    def _get_wall_alpha(self):
        """Get absorption per wall for ISM (box rooms only)."""
        from .material_function import MaterialFunction
        from .acoustics_metrics import impedance_to_alpha
        wall_labels = {'x0': 'left', 'x1': 'right',
                       'y0': 'front', 'y1': 'back',
                       'z0': 'floor', 'z1': 'ceiling'}
        alpha = {}
        for wall, label in wall_labels.items():
            mat_ref = self._materials.get(label, self._default_material)
            if isinstance(mat_ref, MaterialFunction):
                alpha[wall] = mat_ref(500.0)
            else:
                from .materials import get_material
                mat = get_material(mat_ref)
                alpha[wall] = impedance_to_alpha(mat['Z'])
        return alpha

    def _build_box_rt_mesh(self):
        """Build ray trace mesh for box rooms (6 faces = 12 triangles)."""
        from .ray_tracer import RoomMesh
        Lx, Ly, Lz = self._dimensions
        # 8 vertices of the box
        verts = np.array([
            [0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0],
            [0,0,Lz],[Lx,0,Lz],[Lx,Ly,Lz],[0,Ly,Lz],
        ], dtype=float)
        # 12 triangles (2 per face)
        tris = np.array([
            [0,1,2],[0,2,3],  # floor
            [4,6,5],[4,7,6],  # ceiling
            [0,4,5],[0,5,1],  # front
            [2,6,7],[2,7,3],  # back
            [0,3,7],[0,7,4],  # left
            [1,5,6],[1,6,2],  # right
        ], dtype=int)
        labels = ['floor','floor','ceiling','ceiling',
                  'front','front','back','back',
                  'left','left','right','right']
        normals = []
        for t in tris:
            e1 = verts[t[1]]-verts[t[0]]
            e2 = verts[t[2]]-verts[t[0]]
            n = np.cross(e1,e2)
            normals.append(n/np.linalg.norm(n))

        rt = RoomMesh.__new__(RoomMesh)
        rt.vertices = verts
        rt.triangles = tris
        rt.normals = np.array(normals)
        rt.n_triangles = 12
        rt.surface_labels = labels
        rt.surface_alpha = {}
        return rt

    def _ray_trace_c(self, source, receiver, n_rays, max_bounces, T):
        """Run the C ray tracer. Falls back to Python if DLL not found."""
        import ctypes
        from .material_function import MaterialFunction

        rt = self._rt_mesh
        n_tris = rt.n_triangles
        sr = self.sr
        n_bins = int(T * sr)

        # Build per-triangle alpha and scatter arrays from materials
        tri_alpha = np.zeros(n_tris, dtype=np.float64)
        tri_scatter = np.zeros(n_tris, dtype=np.float64)
        for i in range(n_tris):
            label = rt.surface_labels[i]
            tri_alpha[i] = rt.surface_alpha.get(label, 0.05)
            mat_ref = self._materials.get(label, self._default_material)
            if isinstance(mat_ref, MaterialFunction):
                tri_scatter[i] = mat_ref.scatter(1000.0)
            else:
                from .materials import get_material
                mat = get_material(mat_ref)
                tri_scatter[i] = mat.get('scatter', 0.02)

        verts = np.ascontiguousarray(rt.vertices.ravel(), dtype=np.float64)
        tris = np.ascontiguousarray(rt.triangles.ravel(), dtype=np.int32)
        reflecto = np.zeros(n_bins, dtype=np.float64)

        def _ptr(arr, dt=ctypes.c_double):
            return arr.ctypes.data_as(ctypes.POINTER(dt))

        try:
            dll_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'engine', 'lib')
            os.add_dll_directory(dll_dir)
            lib = ctypes.CDLL(os.path.join(dll_dir, 'ray_tracer.dll'))
            lib.ray_trace.restype = ctypes.c_int

            n_out = ctypes.c_int(0)
            lib.ray_trace(
                n_tris,
                _ptr(verts), _ptr(tris, ctypes.c_int),
                _ptr(tri_alpha), _ptr(tri_scatter),
                ctypes.c_double(source[0]), ctypes.c_double(source[1]),
                ctypes.c_double(source[2]),
                ctypes.c_double(receiver[0]), ctypes.c_double(receiver[1]),
                ctypes.c_double(receiver[2]),
                ctypes.c_double(0.1),  # tight capture radius for accuracy
                n_rays, max_bounces,
                ctypes.c_double(343.0), sr, ctypes.c_double(T),
                _ptr(reflecto), ctypes.byref(n_out),
            )
            return reflecto

        except (OSError, AttributeError) as e:
            print(f"  C ray tracer not available ({e}), using Python fallback")
            from .ray_tracer import trace_rays
            reflecto_py, _ = trace_rays(rt, source, receiver,
                                         n_rays=n_rays, max_order=max_bounces,
                                         capture_radius=0.3, scatter_coeff=0.15,
                                         T=T)
            return reflecto_py

    def _ism_component(self, source, receiver, f_cross, T, n_samples):
        """High-frequency IR via ISM early + Eyring late per band."""
        from scipy.signal import butter, filtfilt
        from .acoustics_metrics import eyring_rt60, impedance_to_alpha
        from .materials import get_material
        from .solvers import C_AIR

        sr = self.sr
        nyq = sr / 2
        ir_hf = np.zeros(n_samples)
        np.random.seed(42)

        V = self._volume
        S = self._surface_area
        t_arr = np.arange(n_samples) / sr
        t_mix = 0.08  # mixing time: ISM early, Eyring late
        n_mix = int(t_mix * sr)

        # Get mean alpha from default material
        mat = get_material(self._default_material)
        default_alpha = impedance_to_alpha(mat['Z'])

        is_box = (self._geometry_type == 'box' and self._dimensions is not None)

        for fc in [500, 1000, 2000, 4000, 8000, 16000]:
            if fc < f_cross:
                continue
            fl = fc / np.sqrt(2)
            fh = min(fc * np.sqrt(2), nyq * 0.95)
            if fl >= nyq * 0.95:
                continue

            # Effective absorption for this band
            mean_alpha = default_alpha
            # Add air absorption
            m_air = 5.5e-4 * (50 / 40) * (fc / 1000) ** 1.7
            a_eff = min(mean_alpha + m_air * 4 * V / S, 0.99)
            t60 = eyring_rt60(V, S, a_eff)
            decay = 6.91 / max(t60, 0.01)

            if is_box:
                # ISM for early reflections
                Lx, Ly, Lz = self._dimensions
                alpha_w = self._get_wall_alpha()
                from .image_source import image_sources_shoebox
                ir_early, _ = image_sources_shoebox(
                    Lx, Ly, Lz, source, receiver,
                    max_order=8, alpha_walls=alpha_w, sr=sr, T=T)
                n = min(len(ir_early), n_samples)

                # Eyring late tail
                noise = np.random.randn(n_samples)
                ir_late = noise * np.exp(-decay * t_arr)

                # Level match at crossover
                n_win = int(0.01 * sr)
                s1 = max(0, n_mix - n_win)
                rms_e = np.sqrt(np.mean(ir_early[s1:n_mix] ** 2)) if n_mix > s1 else 0
                rms_l = np.sqrt(np.mean(ir_late[n_mix:n_mix + n_win] ** 2))
                if rms_l > 0 and rms_e > 0:
                    ir_late *= rms_e / rms_l

                # Crossfade
                fade = int(0.02 * sr)
                fo = np.ones(n_samples)
                fi = np.zeros(n_samples)
                fo[n_mix:n_mix + fade] = np.linspace(1, 0, fade)
                fo[n_mix + fade:] = 0
                fi[n_mix:n_mix + fade] = np.linspace(0, 1, fade)
                fi[n_mix + fade:] = 1
                ir_band = ir_early[:n] * fo[:n] + ir_late[:n] * fi[:n]
            else:
                # Non-box: Eyring tail only
                noise = np.random.randn(n_samples)
                envelope = np.exp(-decay * t_arr)
                # Delay by direct sound travel time
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(source, receiver)))
                n_delay = max(int(dist / C_AIR * sr), 1)
                envelope[:n_delay] = 0
                ir_band = noise * envelope * 0.001  # scale down

            # Bandpass filter
            b, a = butter(4, [fl / nyq, fh / nyq], btype='band')
            ir_hf += filtfilt(b, a, ir_band[:n_samples])

        return ir_hf
