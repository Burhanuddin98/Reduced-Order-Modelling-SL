"""High-level Room API for Laplace-domain ROM acoustics."""

import numpy as np
from romacoustics.sem import RectMesh2D, BoxMesh3D, assemble_2d, assemble_3d
from romacoustics.solver import (
    C_AIR, RHO_AIR,
    weeks_s_values, laplace_to_ir,
    ifft_frequencies, ifft_to_ir,
    sweep_fi, sweep_fd,
    sweep_fi_fullfield, sweep_fd_fullfield,
    build_basis, project_operators,
    rom_sweep_fi, rom_sweep_fd,
    _alpha_to_Z_internal,
)
from romacoustics.ir import ImpulseResponse
from romacoustics.materials import absorption_to_impedance, get_material


class ROM:
    """Pre-built reduced order model for fast parametric queries."""

    def __init__(self, rom_ops, s_vals, sigma_w, b_w, fs, t_max, bc_type,
                 sigma_flow=None):
        self.rom_ops = rom_ops
        self.s_vals = s_vals
        self.sigma_w = sigma_w
        self.b_w = b_w
        self.fs = fs
        self.t_max = t_max
        self.bc_type = bc_type
        self.sigma_flow = sigma_flow
        self.t_eval = np.arange(0, t_max, 1.0/fs)
        self.Nrb = rom_ops['M_r'].shape[0]

    def solve(self, Zs=None, d_mat=None):
        """Evaluate ROM at new parameter value. Returns ImpulseResponse."""
        if self.bc_type == 'FI':
            if Zs is None:
                raise ValueError('Zs required for FI boundary')
            H = rom_sweep_fi(self.rom_ops, self.s_vals, Zs)
            label = f'ROM Nrb={self.Nrb}, Zs={Zs}'
        elif self.bc_type == 'FD':
            if d_mat is None:
                raise ValueError('d_mat required for FD boundary')
            H = rom_sweep_fd(self.rom_ops, self.s_vals, self.sigma_flow, d_mat)
            label = f'ROM Nrb={self.Nrb}, d={d_mat}m'
        else:
            raise ValueError(f'Unknown bc_type: {self.bc_type}')

        ir = laplace_to_ir(H, self.sigma_w, self.b_w, self.t_eval)
        return ImpulseResponse(ir, self.fs, label)


class Room:
    """Room acoustics solver with Laplace-domain FOM and ROM.

    Example:
        room = Room.box_2d(2.0, 2.0)
        room.set_source(1.0, 1.0)
        room.set_receiver(0.2, 0.2)
        room.set_boundary_fi(Zs=5000)
        ir = room.solve(t_max=0.1)
    """

    def __init__(self, mesh, ops):
        self.mesh = mesh
        self.ops = ops
        self.N = mesh.N_dof
        self._src = None
        self._rec_idx = None
        self._p0 = None
        self._bc_type = None
        self._bc_params = {}
        self._c2S = (C_AIR**2 * ops['S']).tocsc()
        self._M = ops['M_diag']
        self._B = np.array(ops['B_total'].diagonal())

    # ── Constructors ─────────────────────────────────────────

    @classmethod
    def box_2d(cls, Lx, Ly, ne=20, order=4):
        """2D rectangular room."""
        mesh = RectMesh2D(Lx, Ly, ne, ne, order)
        return cls(mesh, assemble_2d(mesh))

    @classmethod
    def box_3d(cls, Lx, Ly, Lz, ne=8, order=4):
        """3D box room."""
        mesh = BoxMesh3D(Lx, Ly, Lz, ne, ne, ne, order)
        return cls(mesh, assemble_3d(mesh))

    @classmethod
    def from_gmsh(cls, path, lc=None, f_max=500, order=4):
        """Build room from Gmsh .geo or .msh file.

        Physical Surface names become material labels.
        If lc is None, auto-computed from f_max and PPW=6.

        Example:
            room = Room.from_gmsh('concert_hall.geo', f_max=500)
        """
        import gmsh
        from romacoustics.unstructured import TetMesh, assemble_tet

        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.open(path)

        if path.endswith('.geo'):
            if lc is None:
                lc = C_AIR / f_max / 6  # PPW=6, P=1 for tets
            gmsh.option.setNumber('Mesh.CharacteristicLengthMax', lc)
            gmsh.model.mesh.generate(3)

        # Extract nodes
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # Extract tets
        elem_types, _, elem_nodes = gmsh.model.mesh.getElements(3)
        if 4 not in elem_types:
            gmsh.finalize()
            raise RuntimeError('No tet elements found. Ensure 3D mesh is generated.')
        tet_idx = list(elem_types).index(4)
        tet_raw = elem_nodes[tet_idx].reshape(-1, 4)
        tets = np.array([[tag_to_idx[int(n)] for n in t] for t in tet_raw])

        # Extract boundary faces per physical surface
        boundary = {}
        for dim, phys_tag in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(2, phys_tag)
            if not name:
                continue
            entities = gmsh.model.getEntitiesForPhysicalGroup(2, phys_tag)
            faces = []
            for ent in entities:
                _, _, en = gmsh.model.mesh.getElements(2, ent)
                if en:
                    for face in en[0].reshape(-1, 3):
                        faces.append([tag_to_idx[int(n)] for n in face])
            if faces:
                boundary[name] = np.array(faces)

        gmsh.finalize()

        mesh = TetMesh(nodes, tets, boundary)
        ops = assemble_tet(mesh)
        room = cls(mesh, ops)
        room._surface_labels = list(boundary.keys())
        return room

    # ── Source / Receiver ────────────────────────────────────

    def set_source(self, *pos, sigma=0.2):
        """Set Gaussian pulse source. pos = (x,y) or (x,y,z)."""
        self._src = pos
        if hasattr(self.mesh, '_ensure_coords'):
            self.mesh._ensure_coords()
        if self.mesh.ndim == 2:
            r2 = (self.mesh.x-pos[0])**2 + (self.mesh.y-pos[1])**2
        else:
            r2 = ((self.mesh.x-pos[0])**2 + (self.mesh.y-pos[1])**2
                   + (self.mesh.z-pos[2])**2)
        self._p0 = np.exp(-r2/sigma**2)

    def set_receiver(self, *pos):
        """Set receiver position. pos = (x,y) or (x,y,z)."""
        if self.mesh.ndim == 2:
            self._rec_idx = self.mesh.nearest_node(pos[0], pos[1])
        else:
            self._rec_idx = self.mesh.nearest_node(pos[0], pos[1], pos[2])

    # ── Boundary conditions ──────────────────────────────────

    def set_boundary_fi(self, Zs):
        """Frequency-independent boundary (constant impedance, uniform)."""
        self._bc_type = 'FI'
        self._bc_params = {'Zs': Zs}

    def set_boundary_fd(self, sigma_flow=10000, d_mat=0.05):
        """Frequency-dependent boundary (Miki porous absorber, uniform)."""
        self._bc_type = 'FD'
        self._bc_params = {'sigma_flow': sigma_flow, 'd_mat': d_mat}

    def set_material(self, surface, material_or_Z):
        """Set material on a named surface.

        Args:
            surface: surface label (e.g. 'floor', 'ceiling', 'x_min')
            material_or_Z: material name from database (str) or impedance value (float)

        Example:
            room.set_material('floor', 'carpet_thick')
            room.set_material('ceiling', 'acoustic_panel')
            room.set_material('walls', 8000)  # raw impedance
        """
        if not hasattr(self, '_material_map'):
            self._material_map = {}
        if isinstance(material_or_Z, str):
            Z = get_material(material_or_Z)['Z']
        else:
            Z = float(material_or_Z)
        self._material_map[surface] = Z
        self._bc_type = 'PER_SURFACE'

    # ── Solve ────────────────────────────────────────────────

    def solve(self, t_max=0.1, fs=44100, f_max=500, Ns=None, method='auto'):
        """Compute impulse response.

        Args:
            t_max: IR duration [s]
            fs: sample rate [Hz]
            f_max: upper frequency limit [Hz]
            Ns: number of frequency points (auto if None)
            method: 'ifft' (recommended), 'weeks', or 'auto'

        Returns ImpulseResponse object.
        """
        self._check_ready()
        Ns = Ns or 500

        # Auto-select method: IFFT for long IRs, Weeks for short
        if method == 'auto':
            method = 'ifft' if t_max > 0.15 else 'weeks'

        if method == 'ifft':
            return self._solve_ifft(t_max, fs, f_max, Ns)
        else:
            return self._solve_weeks(t_max, fs, Ns)

    def _solve_ifft(self, t_max, fs, f_max, Ns):
        """Solve via IFFT reconstruction."""
        from scipy.sparse.linalg import spsolve
        from scipy import sparse

        s_vals, freqs, sigma = ifft_frequencies(f_max, Ns)
        B_labels = self.ops.get('B_labels', {})
        H = np.zeros(Ns, dtype=complex)

        for i, s in enumerate(s_vals):
            Br_diag = self._build_Br(s, B_labels)
            sig, omg = s.real, s.imag
            Kr = self._c2S + sparse.diags((sig**2-omg**2)*self._M + sig*Br_diag, format='csc')
            Kc = sparse.diags(2*sig*omg*self._M + omg*Br_diag, format='csc')
            A = sparse.bmat([[Kr,-Kc],[Kc,Kr]], format='csc')
            rhs = np.concatenate([sig*self._p0*self._M, omg*self._p0*self._M])
            x = spsolve(A, rhs)
            H[i] = x[self._rec_idx] + 1j*x[self.N + self._rec_idx]
            if (i+1) % max(1, Ns//10) == 0:
                print(f'  {i+1}/{Ns}', end='', flush=True)
        print(' done')

        ir, t = ifft_to_ir(H, freqs, sigma, t_max, fs)
        return ImpulseResponse(ir, fs, self._label())

    def _solve_weeks(self, t_max, fs, Ns):
        """Solve via Weeks/Laguerre reconstruction (short IRs only)."""
        sigma_w = 10.0 if self.mesh.ndim == 2 else 20.0
        b_w = 1000.0 if self.mesh.ndim == 2 else 800.0
        s_vals, _ = weeks_s_values(sigma_w, b_w, Ns)
        t_eval = np.arange(0, t_max, 1.0/fs)

        if self._bc_type == 'FI':
            H = sweep_fi(self._c2S, self._M, self._B, self._p0, self.N,
                         s_vals, self._bc_params['Zs'], self._rec_idx)
        elif self._bc_type == 'FD':
            H = sweep_fd(self._c2S, self._M, self._B, self._p0, self.N,
                         s_vals, self._bc_params['sigma_flow'],
                         self._bc_params['d_mat'], self._rec_idx)
        else:
            raise ValueError('Weeks method only supports FI/FD. Use IFFT for per-surface.')

        ir = laplace_to_ir(H, sigma_w, b_w, t_eval)
        return ImpulseResponse(ir, fs, self._label())

    def _build_Br(self, s, B_labels):
        """Build boundary impedance diagonal for a given s value."""
        Br = np.zeros(self.N)
        if self._bc_type == 'PER_SURFACE' and hasattr(self, '_material_map'):
            for surface, Z in self._material_map.items():
                if surface in B_labels:
                    Br += C_AIR**2 * RHO_AIR * B_labels[surface] / Z
        elif self._bc_type == 'FI':
            Br = C_AIR**2 * RHO_AIR * self._B / self._bc_params['Zs']
        elif self._bc_type == 'FD':
            f = max(abs(s.imag)/(2*np.pi), 1.0)
            from romacoustics.solver import miki_admittance_scalar
            Ys = miki_admittance_scalar(f, self._bc_params['sigma_flow'],
                                         self._bc_params['d_mat'])
            Br = C_AIR**2 * RHO_AIR * Ys * self._B
        return Br

    def _label(self):
        if self._bc_type == 'PER_SURFACE':
            return 'FOM per-surface'
        elif self._bc_type == 'FI':
            return f'FOM Zs={self._bc_params["Zs"]}'
        elif self._bc_type == 'FD':
            return f'FOM d={self._bc_params["d_mat"]}m'
        return 'FOM'

    # ── ROM ──────────────────────────────────────────────────

    def build_rom(self, Z_train=None, d_train=None, Ns=None,
                  sigma_w=None, b_w=None, eps_pod=1e-6, fs=44100,
                  t_max=0.1):
        """Build parametric ROM from training parameter values.

        For FI: provide Z_train (list of impedance values).
        For FD: provide d_train (list of thickness values).

        Returns ROM object for fast parametric queries.
        """
        self._check_ready()

        if self.mesh.ndim == 2:
            Ns = Ns or 1000
            sigma_w = sigma_w or 10.0
            b_w = b_w or 1000.0
        else:
            Ns = Ns or 500
            sigma_w = sigma_w or 20.0
            b_w = b_w or 800.0

        s_vals, _ = weeks_s_values(sigma_w, b_w, Ns)

        # Collect snapshots
        all_snaps = []
        if self._bc_type == 'FI':
            if Z_train is None:
                raise ValueError('Z_train required for FI ROM')
            for Zs in Z_train:
                print(f'  Snapshots Zs={Zs}...')
                all_snaps.extend(
                    sweep_fi_fullfield(self._c2S, self._M, self._B,
                                       self._p0, self.N, s_vals, Zs))
        elif self._bc_type == 'FD':
            if d_train is None:
                raise ValueError('d_train required for FD ROM')
            for d in d_train:
                print(f'  Snapshots d={d}...')
                all_snaps.extend(
                    sweep_fd_fullfield(self._c2S, self._M, self._B,
                                       self._p0, self.N, s_vals,
                                       self._bc_params['sigma_flow'], d))

        # Build basis
        print('  SVD...')
        Psi, Nrb, sv = build_basis(all_snaps, eps_pod)
        print(f'  Nrb = {Nrb}')
        del all_snaps

        # Project
        rom_ops = project_operators(self.ops, Psi, self._p0, self._rec_idx)

        return ROM(rom_ops, s_vals, sigma_w, b_w, fs, t_max,
                   self._bc_type,
                   sigma_flow=self._bc_params.get('sigma_flow'))

    def _check_ready(self):
        if self._p0 is None:
            raise RuntimeError('Call set_source() first')
        if self._rec_idx is None:
            raise RuntimeError('Call set_receiver() first')
        if self._bc_type is None:
            raise RuntimeError('Call set_boundary_fi(), set_boundary_fd(), or set_material() first')
