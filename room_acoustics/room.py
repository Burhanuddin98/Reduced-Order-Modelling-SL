"""
Room — the unified API for room acoustics simulation.

Usage:
    room = Room.from_box(8.4, 6.7, 3.0)
    room.set_material("floor", "wood_floor")
    room.set_material("ceiling", "acoustic_panel")
    room.set_material_default("plaster")
    room.build(f_max=500)

    ir = room.impulse_response(source=(2,3.35,1.5), receiver=(6,2,1.2))
    print(f"T30={ir.T30:.2f}s  C80={ir.C80:.1f}dB")
    ir.save_wav("room_ir.wav")
    ir.auralize("dry_speech.wav", "output.wav")
"""

import numpy as np
import os
import time as _time


class ImpulseResponse:
    """Holds an impulse response and computes metrics on demand."""

    def __init__(self, ir, sr=44100):
        self.data = np.asarray(ir, dtype=float)
        self.sr = sr
        self.dt = 1.0 / sr
        self._metrics = None

    @property
    def metrics(self):
        if self._metrics is None:
            from .acoustics_metrics import all_metrics
            self._metrics = all_metrics(self.data, self.dt)
        return self._metrics

    @property
    def T30(self):
        return self.metrics['T30_s']

    @property
    def T20(self):
        return self.metrics['T20_s']

    @property
    def EDT(self):
        return self.metrics['EDT_s']

    @property
    def C80(self):
        return self.metrics['C80_dB']

    @property
    def D50(self):
        return self.metrics['D50']

    @property
    def TS(self):
        return self.metrics['TS_ms']

    @property
    def duration(self):
        return len(self.data) / self.sr

    def save_wav(self, path, normalize=True):
        """Save IR as WAV file."""
        import scipy.io.wavfile as wavfile
        ir = self.data.copy()
        if normalize and np.max(np.abs(ir)) > 0:
            ir = ir / np.max(np.abs(ir)) * 0.95
        ir_16 = (ir * 32767).astype(np.int16)
        wavfile.write(path, self.sr, ir_16)
        print(f"  Saved IR: {path} ({self.duration:.2f}s, {self.sr}Hz)")

    def auralize(self, audio_path, output_path):
        """Convolve IR with dry audio and save."""
        import scipy.io.wavfile as wavfile
        from scipy.signal import fftconvolve

        sr_in, audio = wavfile.read(audio_path)
        audio = audio.astype(float)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr_in != self.sr:
            from scipy.signal import resample
            n_out = int(len(audio) * self.sr / sr_in)
            audio = resample(audio, n_out)

        # Normalize audio
        audio = audio / max(np.max(np.abs(audio)), 1e-10)

        # Convolve
        wet = fftconvolve(audio, self.data, mode='full')
        wet = wet[:len(audio) + self.sr]  # trim to audio + 1s tail

        # Normalize output
        wet = wet / max(np.max(np.abs(wet)), 1e-10) * 0.95
        wet_16 = (wet * 32767).astype(np.int16)
        wavfile.write(output_path, self.sr, wet_16)
        print(f"  Auralized: {output_path}")


class Room:
    """
    Room acoustics simulation engine.

    Supports:
    - Box rooms (structured hex mesh, fast)
    - STL/OBJ import (tet mesh, any geometry)
    - Floor plan extrusion (hex mesh, any floor plan)
    - Per-surface material assignment
    - Hybrid IR: modal ROM (low freq) + ISM (high freq)
    """

    def __init__(self):
        self.mesh = None
        self.ops = None
        self.geometry_type = None
        self.dimensions = None
        self._materials = {}
        self._default_material = 'plaster'
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
        self._built = False
        self._build_time = 0
        self.P = 4
        self.sr = 44100

    # ============================================================
    # Construction
    # ============================================================

    @classmethod
    def from_box(cls, Lx, Ly, Lz, P=4):
        """Create from rectangular box dimensions."""
        room = cls()
        room.geometry_type = 'box'
        room.dimensions = (Lx, Ly, Lz)
        room.P = P
        return room

    @classmethod
    def from_stl(cls, path, h_target=0.4, P=2):
        """Create from STL/OBJ file."""
        room = cls()
        room.geometry_type = 'stl'
        room._stl_path = path
        room._h_target = h_target
        room.P = P
        return room

    @classmethod
    def from_polygon(cls, vertices, Lz, h_target=0.4, P=4):
        """Create from 2D floor plan polygon extruded to height Lz."""
        room = cls()
        room.geometry_type = 'polygon'
        room._vertices = vertices
        room._Lz = Lz
        room._h_target = h_target
        room.P = P
        return room

    # ============================================================
    # Materials
    # ============================================================

    def set_material(self, surface_label, material_name):
        """Assign a material to a surface by label."""
        self._materials[surface_label] = material_name
        # Invalidate eigenmode cache if materials change after build
        if self._built:
            self._eigenvalues = None
            self._eigenvectors = None

    def set_material_default(self, material_name):
        """Set default material for unlabeled surfaces."""
        self._default_material = material_name

    def list_surfaces(self):
        """Print available surface labels."""
        if self.mesh is None:
            print("  Room not built yet. Call build() first.")
            return
        if hasattr(self.mesh, '_boundary_nodes_per_label'):
            labels = list(self.mesh._boundary_nodes_per_label.keys())
        else:
            labels = ['bottom', 'top', 'left', 'right', 'front', 'back']
        print(f"  Surfaces: {labels}")
        for label in labels:
            mat = self._materials.get(label, self._default_material)
            print(f"    {label}: {mat}")

    # ============================================================
    # Build
    # ============================================================

    def build(self, n_modes=400, f_max_target=None):
        """
        Mesh the geometry, assemble operators, compute eigenmodes.

        This is the expensive one-time step. After this, changing
        source/receiver/materials is near-instant.

        Parameters
        ----------
        n_modes : int — number of eigenmodes to compute
        f_max_target : float or None — target max frequency for modal ROM.
                       If set, overrides n_modes with an estimate.
        """
        t0 = _time.perf_counter()
        print("Building room...")

        # Step 1: Mesh
        self._build_mesh()
        print(f"  Mesh: N={self.mesh.N_dof}")
        print(f"  Surfaces: {list(self._get_surface_labels())}")

        # Step 2: Assemble operators
        self._assemble_operators()
        vol = self.ops['M_diag'].sum()
        print(f"  Volume: {vol:.1f} m^3")

        # Step 3: Eigenmodes
        if f_max_target is not None:
            # Estimate modes needed: N_modes ~ V * (4pi/3) * (f/c)^3
            c = 343.0
            V = vol
            n_modes = max(50, int(V * 4 * np.pi / 3 * (f_max_target / c)**3 * 1.2))
            n_modes = min(n_modes, self.mesh.N_dof - 2)

        print(f"  Computing {n_modes} eigenmodes...", end='', flush=True)
        self._compute_eigenmodes(n_modes)
        print(f" f_max={self._frequencies[-1]:.0f} Hz")

        self._built = True
        self._build_time = _time.perf_counter() - t0
        print(f"  Build time: {self._build_time:.1f}s")

    # ============================================================
    # Impulse Response
    # ============================================================

    def impulse_response(self, source, receiver, T=3.5, sigma=0.5):
        """
        Compute the impulse response between source and receiver.

        Uses hybrid approach:
        - Modal ROM for 0 to f_cross (eigenmode frequency range)
        - ISM + Eyring for f_cross to 20 kHz

        Parameters
        ----------
        source : (x, y, z) — source position [m]
        receiver : (x, y, z) — receiver position [m]
        T : float — IR duration [s]
        sigma : float — source pulse width [m]

        Returns
        -------
        ImpulseResponse object
        """
        if not self._built:
            raise RuntimeError("Call room.build() first")

        sr = self.sr
        dt = 1.0 / sr
        n_samples = int(T * sr)
        f_cross = self._frequencies[-1] * 0.9  # slight margin

        rec_idx = self.mesh.nearest_node(*receiver)

        # Get impedance per node
        Z_per_node = self._get_impedance_per_node()

        print(f"  Computing IR: src={source}, rec=({self.mesh.x[rec_idx]:.1f},"
              f"{self.mesh.y[rec_idx]:.1f},{self.mesh.z[rec_idx]:.1f})")
        print(f"  Modal range: 0-{f_cross:.0f} Hz, ISM: {f_cross:.0f}-20000 Hz")

        # === MODAL ROM (low frequencies) ===
        from .modal_rom import modal_ir
        t0 = _time.perf_counter()
        res_modal = modal_ir(self.mesh, self.ops,
                              self._eigenvalues, self._eigenvectors,
                              'FI', {'Z_per_node': Z_per_node},
                              *source, sigma, dt, T, rec_idx)
        ir_modal = res_modal['ir'][:n_samples]
        t_modal = _time.perf_counter() - t0

        # Low-pass filter modal IR at crossover
        from scipy.signal import butter, filtfilt
        nyq = sr / 2
        if f_cross < nyq * 0.95:
            b_lo, a_lo = butter(6, f_cross / nyq, btype='low')
            ir_low = filtfilt(b_lo, a_lo, ir_modal)
        else:
            ir_low = ir_modal

        # === ISM (high frequencies) ===
        t0 = _time.perf_counter()
        ir_high = self._compute_ism_component(source, receiver, f_cross,
                                               T, sr, n_samples)
        t_ism = _time.perf_counter() - t0

        # === Blend ===
        ir_total = ir_low + ir_high

        print(f"  Modal: {t_modal:.2f}s, ISM: {t_ism:.2f}s")

        return ImpulseResponse(ir_total, sr)

    # ============================================================
    # Internal methods
    # ============================================================

    def _build_mesh(self):
        if self.geometry_type == 'box':
            from .sem import BoxMesh3D
            Lx, Ly, Lz = self.dimensions
            P = self.P
            # Auto element count: ~6 elements per wavelength at 500 Hz
            c = 343.0
            lam = c / 500
            h = lam / 6 * P
            Nex = max(3, int(np.ceil(Lx / h)))
            Ney = max(3, int(np.ceil(Ly / h)))
            Nez = max(2, int(np.ceil(Lz / h)))
            self.mesh = BoxMesh3D(Lx, Ly, Lz, Nex, Ney, Nez, P)

        elif self.geometry_type == 'stl':
            from .gmsh_tet_import import import_surface_mesh
            from .tet_sem import TetMesh3D
            data = import_surface_mesh(self._stl_path, self._h_target, self.P)
            self.mesh = TetMesh3D(data['nodes'], data['tets'], data['boundary'])

        elif self.geometry_type == 'polygon':
            from .geometry import RoomGeometry, generate_quad_mesh, extrude_quad_mesh
            from .unstructured_sem import UnstructuredHexMesh3D
            geom = RoomGeometry(self._vertices)
            raw_2d = generate_quad_mesh(geom, self._h_target, self.P)
            raw_3d = extrude_quad_mesh(raw_2d, self._Lz,
                                        max(2, int(self._Lz / self._h_target)))
            self.mesh = UnstructuredHexMesh3D(raw_3d, raw_2d['nodes'],
                                              raw_2d['quads'], self.P)

        if hasattr(self.mesh, '_ensure_coords'):
            self.mesh._ensure_coords()

    def _assemble_operators(self):
        if self.geometry_type == 'box':
            from .sem import assemble_3d_operators
            self.ops = assemble_3d_operators(self.mesh)
        elif self.geometry_type == 'stl':
            from .tet_sem import assemble_tet_3d_operators
            self.ops = assemble_tet_3d_operators(self.mesh)
        elif self.geometry_type == 'polygon':
            from .unstructured_sem import assemble_unstructured_3d_operators
            self.ops = assemble_unstructured_3d_operators(self.mesh)

    def _compute_eigenmodes(self, n_modes):
        from .modal_rom import compute_room_modes
        self._eigenvalues, self._eigenvectors, self._frequencies = \
            compute_room_modes(self.ops, n_modes)

    def _get_surface_labels(self):
        if hasattr(self.mesh, '_boundary_nodes_per_label'):
            return self.mesh._boundary_nodes_per_label.keys()
        return []

    def _get_impedance_per_node(self):
        """Build per-node impedance from material assignments."""
        from .materials import get_material
        from .solvers import RHO_AIR, C_AIR

        rho_c = RHO_AIR * C_AIR
        N = self.mesh.N_dof
        Z = np.full(N, 1e15)

        def alpha_to_Z(a):
            a = np.clip(a, 0.001, 0.999)
            R = np.sqrt(1.0 - a)
            return rho_c * (1 + R) / (1 - R)

        if self.geometry_type == 'box':
            # Box mesh: identify surfaces by coordinate
            Lx, Ly, Lz = self.dimensions
            tol = 1e-6
            surface_map = {
                'floor': self.mesh.z < tol,
                'ceiling': self.mesh.z > Lz - tol,
                'left': self.mesh.x < tol,
                'right': self.mesh.x > Lx - tol,
                'front': self.mesh.y < tol,
                'back': self.mesh.y > Ly - tol,
            }
            for label, mask in surface_map.items():
                mat_name = self._materials.get(label, self._default_material)
                mat = get_material(mat_name)
                Z[mask] = mat['Z']
        else:
            # Mesh with labeled boundaries
            for label in self._get_surface_labels():
                mat_name = self._materials.get(label, self._default_material)
                mat = get_material(mat_name)
                nodes = self.mesh.boundary_nodes(label)
                Z[nodes] = mat['Z']

        return Z

    def _compute_ism_component(self, source, receiver, f_cross, T, sr, n_samples):
        """Compute high-frequency IR component using ISM + Eyring tail."""
        from .image_source import image_sources_shoebox
        from .acoustics_metrics import eyring_rt60
        from .materials import get_material
        from .solvers import C_AIR
        from scipy.signal import butter, filtfilt

        nyq = sr / 2
        ir_hf = np.zeros(n_samples)
        np.random.seed(42)

        if self.geometry_type != 'box':
            # For non-box, use Eyring tail only (no ISM)
            V = self.ops['M_diag'].sum()
            S_total = self.ops['B_total'].diagonal().sum()
            # Estimate mean alpha from materials
            mean_alpha = 0.05  # fallback
            mat_name = self._default_material
            mat = get_material(mat_name)
            from .acoustics_metrics import impedance_to_alpha
            mean_alpha = impedance_to_alpha(mat['Z'])

            for fc in [500, 1000, 2000, 4000, 8000]:
                if fc < f_cross:
                    continue
                t60 = eyring_rt60(V, S_total, mean_alpha)
                decay = 6.91 / max(t60, 0.01)
                # Air absorption
                m_air = 5.5e-4 * (50/40) * (fc/1000)**1.7
                decay += m_air * C_AIR

                t_arr = np.arange(n_samples) / sr
                noise = np.random.randn(n_samples)
                envelope = np.exp(-decay * t_arr)
                n_delay = int(0.01 * sr)
                envelope[:n_delay] = 0
                ir_band = noise * envelope * 0.001

                fl = fc / np.sqrt(2)
                fh = min(fc * np.sqrt(2), nyq * 0.95)
                if fl >= nyq * 0.95:
                    continue
                b, a = butter(4, [fl/nyq, fh/nyq], btype='band')
                ir_hf += filtfilt(b, a, ir_band)

            return ir_hf

        # Box room: use ISM for early + Eyring for late per band
        Lx, Ly, Lz = self.dimensions
        V = Lx * Ly * Lz
        S_total = 2 * (Lx*Ly + Lx*Lz + Ly*Lz)
        areas = {'x0': Ly*Lz, 'x1': Ly*Lz, 'y0': Lx*Lz,
                 'y1': Lx*Lz, 'z0': Lx*Ly, 'z1': Lx*Ly}

        wall_to_label = {'x0': 'left', 'x1': 'right', 'y0': 'front',
                         'y1': 'back', 'z0': 'floor', 'z1': 'ceiling'}

        # Get alpha per wall
        def _get_wall_alpha():
            alpha_w = {}
            for wall, label in wall_to_label.items():
                mat_name = self._materials.get(label, self._default_material)
                mat = get_material(mat_name)
                from .acoustics_metrics import impedance_to_alpha
                alpha_w[wall] = impedance_to_alpha(mat['Z'])
            return alpha_w

        alpha_w = _get_wall_alpha()
        t_mix = 0.08
        n_mix = int(t_mix * sr)

        for fc in [500, 1000, 2000, 4000, 8000]:
            if fc < f_cross:
                continue

            # ISM early reflections
            ir_early, _ = image_sources_shoebox(
                Lx, Ly, Lz, source, receiver, max_order=10,
                alpha_walls=alpha_w, sr=sr, T=T)

            # Eyring late tail
            mean_a = sum(alpha_w[k] * areas[k] for k in alpha_w) / S_total
            m_air = 5.5e-4 * (50/40) * (fc/1000)**1.7
            a_eff = mean_a + m_air * 4*V/S_total
            a_eff = min(a_eff, 0.99)
            t60 = eyring_rt60(V, S_total, a_eff)
            decay = 6.91 / max(t60, 0.01)

            t_arr = np.arange(n_samples) / sr
            noise = np.random.randn(n_samples)
            ir_late = noise * np.exp(-decay * t_arr)

            # Match level at crossover
            n_win = int(0.01 * sr)
            rms_e = np.sqrt(np.mean(ir_early[max(0,n_mix-n_win):n_mix]**2))
            rms_l = np.sqrt(np.mean(ir_late[n_mix:n_mix+n_win]**2))
            if rms_l > 0 and rms_e > 0:
                ir_late *= rms_e / rms_l

            # Crossfade
            fade_len = int(0.02 * sr)
            fo = np.ones(n_samples)
            fi = np.zeros(n_samples)
            fo[n_mix:n_mix+fade_len] = np.linspace(1, 0, fade_len)
            fo[n_mix+fade_len:] = 0
            fi[n_mix:n_mix+fade_len] = np.linspace(0, 1, fade_len)
            fi[n_mix+fade_len:] = 1

            n = min(len(ir_early), n_samples)
            ir_band = ir_early[:n] * fo[:n] + ir_late[:n] * fi[:n]

            # Bandpass
            fl = fc / np.sqrt(2)
            fh = min(fc * np.sqrt(2), nyq * 0.95)
            if fl >= nyq * 0.95:
                continue
            b, a = butter(4, [fl/nyq, fh/nyq], btype='band')
            ir_hf += filtfilt(b, a, ir_band)

        return ir_hf
