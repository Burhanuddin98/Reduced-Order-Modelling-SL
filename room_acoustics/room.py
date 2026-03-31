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

    def set_material(self, surface_label, material_name):
        """Assign a material to a surface."""
        from .materials import get_material
        get_material(material_name)  # validate
        self._materials[surface_label] = material_name

    def set_material_default(self, material_name):
        """Set default material for unlabeled surfaces."""
        from .materials import get_material
        get_material(material_name)
        self._default_material = material_name

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
        print(f"  Mesh: {self.mesh.N_dof} DOFs, {self.mesh.N_el} elements")

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

        self._built = True
        self._build_time = _time.perf_counter() - t0
        print(f"  Build: {self._build_time:.1f}s")

    # ============================================================
    # Impulse Response
    # ============================================================

    def impulse_response(self, source, receiver, T=3.5, sigma=0.3):
        """
        Compute hybrid IR: modal ROM (low) + ISM+Eyring (high).
        Near-instant after build().
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        sr = self.sr
        dt = 1.0 / sr
        n_samples = int(T * sr)
        f_cross = float(self._frequencies[-1]) * 0.85

        rec_idx = self.mesh.nearest_node(*receiver)
        Z = self._build_impedance()

        print(f"  IR: modal 0-{f_cross:.0f}Hz + ISM {f_cross:.0f}-20kHz")

        # === Modal ROM ===
        t0 = _time.perf_counter()
        from .modal_rom import modal_ir
        res = modal_ir(self.mesh, self.ops,
                       self._eigenvalues, self._eigenvectors,
                       'FI', {'Z_per_node': Z},
                       *source, sigma, dt, T, rec_idx)
        ir_modal = res['ir'][:n_samples]
        t_modal = _time.perf_counter() - t0

        # Low-pass at crossover
        from scipy.signal import butter, filtfilt
        nyq = sr / 2
        if f_cross < nyq * 0.95:
            b, a = butter(6, f_cross / nyq, btype='low')
            ir_low = filtfilt(b, a, ir_modal)
        else:
            ir_low = ir_modal

        # === ISM + Eyring tail (high frequencies) ===
        t0 = _time.perf_counter()
        ir_high = self._ism_component(source, receiver, f_cross, T, n_samples)
        t_ism = _time.perf_counter() - t0

        # === Blend ===
        ir_total = ir_low + ir_high

        print(f"  Modal: {t_modal:.2f}s, ISM: {t_ism:.2f}s")

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

    def _build_impedance(self):
        """Build per-node impedance from material assignments."""
        from .materials import get_material
        from .solvers import RHO_AIR, C_AIR
        rho_c = RHO_AIR * C_AIR

        def a2z(a):
            a = np.clip(a, 0.001, 0.999)
            R = np.sqrt(1.0 - a)
            return rho_c * (1 + R) / (1 - R)

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
                mat = get_material(self._materials.get(label, self._default_material))
                Z[mask] = mat['Z']
        else:
            for label in self._get_labels():
                mat = get_material(self._materials.get(label, self._default_material))
                nodes = self.mesh.boundary_nodes(label)
                Z[nodes] = mat['Z']

        return Z

    def _get_wall_alpha(self):
        """Get absorption per wall for ISM (box rooms only)."""
        from .materials import get_material
        from .acoustics_metrics import impedance_to_alpha
        wall_labels = {'x0': 'left', 'x1': 'right',
                       'y0': 'front', 'y1': 'back',
                       'z0': 'floor', 'z1': 'ceiling'}
        alpha = {}
        for wall, label in wall_labels.items():
            mat = get_material(self._materials.get(label, self._default_material))
            alpha[wall] = impedance_to_alpha(mat['Z'])
        return alpha

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
