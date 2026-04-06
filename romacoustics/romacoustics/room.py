"""High-level Room API for Laplace-domain ROM acoustics.

Supports:
  - Box rooms (structured hex SEM): Room.box_2d(), Room.box_3d()
  - Arbitrary geometry from Gmsh: Room.from_gmsh()
  - Arbitrary geometry from STL: Room.from_stl()
  - Arbitrary geometry from OBJ: Room.from_obj()
"""

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


# ── Helper functions for arbitrary geometry ──────────────────

def _auto_boundary_from_tets(nodes, tets):
    """Extract boundary faces from a tet mesh (faces shared by 1 tet only)."""
    from collections import Counter
    face_count = Counter()
    face_to_nodes = {}
    for tet in tets:
        for combo in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
            face = tuple(sorted(tet[list(combo)]))
            face_count[face] += 1
            face_to_nodes[face] = [tet[combo[0]], tet[combo[1]], tet[combo[2]]]
    boundary_faces = [face_to_nodes[f] for f, c in face_count.items() if c == 1]
    if not boundary_faces:
        return {'boundary': np.zeros((0, 3), dtype=int)}
    boundary = _rename_boundary_by_normal(nodes,
        {'all': np.array(boundary_faces, dtype=int)})
    return boundary


def _rename_boundary_by_normal(nodes, boundary):
    """Rename boundary groups by dominant normal direction (floor/ceiling/walls)."""
    canonical = {
        'floor':      np.array([0, 0, -1.0]),
        'ceiling':    np.array([0, 0,  1.0]),
        'wall_north': np.array([0,  1.0, 0]),
        'wall_south': np.array([0, -1.0, 0]),
        'wall_east':  np.array([ 1.0, 0, 0]),
        'wall_west':  np.array([-1.0, 0, 0]),
    }
    cos_thresh = np.cos(np.radians(30))
    new_boundary = {}
    ungrouped = []

    for label, faces in boundary.items():
        if len(faces) == 0:
            continue
        # Compute mean normal for this group
        for face in faces:
            v = nodes[face]
            n = np.cross(v[1] - v[0], v[2] - v[0])
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            n /= norm
            # Find best canonical match
            matched = False
            for cname, cdir in canonical.items():
                if np.dot(n, cdir) > cos_thresh:
                    new_boundary.setdefault(cname, []).append(face)
                    matched = True
                    break
            if not matched:
                ungrouped.append(face)

    if ungrouped:
        new_boundary['walls'] = np.array(ungrouped, dtype=int)

    # Convert lists to arrays
    for k in new_boundary:
        if isinstance(new_boundary[k], list):
            new_boundary[k] = np.array(new_boundary[k], dtype=int)

    return new_boundary


class ROM:
    """Pre-built reduced order model for fast parametric queries.

    Supports per-surface material variation via affine decomposition:
    Online assembly: Br_r = sum_label (1/Z_label) * Br_r_label
    """

    def __init__(self, rom_ops, s_vals, freqs, sigma, fs, t_max,
                 bc_type='PER_SURFACE', material_map=None, B_labels_keys=None):
        self.rom_ops = rom_ops
        self.s_vals = s_vals
        self.freqs = freqs
        self.sigma = sigma
        self.fs = fs
        self.t_max = t_max
        self.bc_type = bc_type
        self.material_map = material_map or {}
        self.B_labels_keys = B_labels_keys or []
        self.Nrb = rom_ops['M_r'].shape[0]

    def solve(self, Zs=None, d_mat=None, **material_overrides):
        """Evaluate ROM at new parameter values. Returns ImpulseResponse.

        For FI: provide Zs
        For per-surface: provide keyword args matching surface names
            rom.solve(ceiling=500, floor=8000)
        """
        M_r = self.rom_ops['M_r']
        S_r = self.rom_ops['S_r']
        f_r = self.rom_ops['f_r']
        obs = self.rom_ops['obs']
        Br_per = self.rom_ops.get('Br_r_per_surface', {})

        # Build material map: start with training defaults, override
        Z_map = dict(self.material_map)
        if Zs is not None:
            # Uniform impedance on all surfaces
            for k in self.B_labels_keys:
                Z_map[k] = Zs
        Z_map.update(material_overrides)

        Ns = len(self.s_vals)
        H = np.zeros(Ns, dtype=complex)

        for i, s in enumerate(self.s_vals):
            # Affine assembly: Br_r = sum (1/Z_label) * Br_r_label
            Br_r = np.zeros_like(M_r)
            for label, Br_r_label in Br_per.items():
                Z = Z_map.get(label, 50000)
                Br_r += Br_r_label / Z

            A_r = s**2 * M_r + C_AIR**2 * S_r + s * Br_r
            a = np.linalg.solve(A_r, s * f_r)
            H[i] = obs @ a

        ir, _ = ifft_to_ir(H, self.freqs, self.sigma, self.t_max, self.fs)
        label = f'ROM Nrb={self.Nrb}'
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

        # If no physical surfaces found, auto-detect boundary faces
        if not boundary:
            boundary = _auto_boundary_from_tets(nodes, tets)

        mesh = TetMesh(nodes, tets, boundary)
        ops = assemble_tet(mesh)
        room = cls(mesh, ops)
        room._surface_labels = list(boundary.keys())
        return room

    @classmethod
    def from_stl(cls, path, f_max=500, lc=None):
        """Build room from STL file (arbitrary closed geometry).

        The STL surface is auto-grouped by face normal direction.
        Interior is meshed with tetrahedra via Gmsh.

        Example:
            room = Room.from_stl('concert_hall.stl', f_max=400)
            room.set_material('floor', 'carpet_thick')
            room.set_material('ceiling', 'acoustic_panel')
        """
        import gmsh
        from romacoustics.unstructured import TetMesh, assemble_tet

        if lc is None:
            lc = C_AIR / f_max / 6

        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.merge(str(path))

        # Classify and create volume from STL surface
        angle = 40  # degrees for surface classification
        forceParametrizablePatches = True
        gmsh.model.mesh.classifySurfaces(
            angle * np.pi / 180, True, forceParametrizablePatches)
        gmsh.model.mesh.createGeometry()

        # Create volume from surface loop
        s = gmsh.model.getEntities(2)
        sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
        gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', lc)
        gmsh.model.mesh.generate(3)

        # Extract mesh
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        elem_types, _, elem_nodes = gmsh.model.mesh.getElements(3)
        if 4 not in elem_types:
            gmsh.finalize()
            raise RuntimeError('No tet elements. STL may not be watertight.')
        tet_idx = list(elem_types).index(4)
        tet_raw = elem_nodes[tet_idx].reshape(-1, 4)
        tets = np.array([[tag_to_idx[int(n)] for n in t] for t in tet_raw])

        # Extract boundary faces per surface entity
        boundary = {}
        surfaces = gmsh.model.getEntities(2)
        for dim, tag in surfaces:
            _, _, en = gmsh.model.mesh.getElements(2, tag)
            if not en:
                continue
            faces = []
            for f_nodes in en[0].reshape(-1, 3):
                faces.append([tag_to_idx[int(n)] for n in f_nodes])
            if faces:
                boundary[f'surface_{tag}'] = np.array(faces)

        gmsh.finalize()

        # Auto-name surfaces by normal direction
        boundary = _rename_boundary_by_normal(nodes, boundary)

        mesh = TetMesh(nodes, tets, boundary)
        ops = assemble_tet(mesh)
        room = cls(mesh, ops)
        room._surface_labels = list(boundary.keys())
        return room

    @classmethod
    def from_obj(cls, path, f_max=500, lc=None):
        """Build room from OBJ file (arbitrary closed geometry).

        OBJ groups/materials become surface labels.
        Interior is meshed with tetrahedra via Gmsh.

        Example:
            room = Room.from_obj('bedroom.obj', f_max=300)
        """
        # Convert OBJ to STL via Gmsh, then use from_stl pipeline
        import gmsh
        from romacoustics.unstructured import TetMesh, assemble_tet

        if lc is None:
            lc = C_AIR / f_max / 6

        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.merge(str(path))

        angle = 40
        gmsh.model.mesh.classifySurfaces(
            angle * np.pi / 180, True, True)
        gmsh.model.mesh.createGeometry()

        s = gmsh.model.getEntities(2)
        if not s:
            gmsh.finalize()
            raise RuntimeError('No surfaces found in OBJ.')
        sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
        gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', lc)
        gmsh.model.mesh.generate(3)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        elem_types, _, elem_nodes = gmsh.model.mesh.getElements(3)
        if 4 not in elem_types:
            gmsh.finalize()
            raise RuntimeError('No tet elements. Mesh may not be watertight.')
        tet_idx = list(elem_types).index(4)
        tet_raw = elem_nodes[tet_idx].reshape(-1, 4)
        tets = np.array([[tag_to_idx[int(n)] for n in t] for t in tet_raw])

        boundary = {}
        for dim, tag in gmsh.model.getEntities(2):
            _, _, en = gmsh.model.mesh.getElements(2, tag)
            if not en:
                continue
            faces = [[tag_to_idx[int(n)] for n in f] for f in en[0].reshape(-1, 3)]
            if faces:
                boundary[f'surface_{tag}'] = np.array(faces)

        gmsh.finalize()
        boundary = _rename_boundary_by_normal(nodes, boundary)

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

        # Auto-select method: IFFT for per-surface or long IRs
        if method == 'auto':
            if self._bc_type == 'PER_SURFACE' or t_max > 0.15:
                method = 'ifft'
            else:
                method = 'weeks'

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

    def build_rom(self, Z_train=None, d_train=None, material_variations=None,
                  Ns=None, f_max=500, eps_pod=1e-6, fs=44100, t_max=1.0):
        """Build parametric ROM from training parameter values.

        For FI: provide Z_train (list of impedance values).
        For FD: provide d_train (list of thickness values).
        For per-surface: provide material_variations dict:
            {'ceiling': [500, 1000, 2000], 'floor': [3000, 8000]}
            Training uses all combinations.

        Returns ROM object for fast parametric queries.
        """
        self._check_ready()
        Ns = Ns or 500

        # Use IFFT frequencies for long IRs
        s_vals, freqs, sigma = ifft_frequencies(f_max, Ns)
        B_labels = self.ops.get('B_labels', {})

        from scipy.sparse.linalg import spsolve
        from scipy import sparse

        all_snaps = []

        if self._bc_type == 'PER_SURFACE' and material_variations:
            # Generate training configs from material_variations
            import itertools
            surfaces = list(material_variations.keys())
            Z_lists = [material_variations[s] for s in surfaces]
            combos = list(itertools.product(*Z_lists))
            print(f'  {len(combos)} material combinations x {Ns} frequencies')

            for combo in combos:
                Z_map = dict(zip(surfaces, combo))
                # Fill in non-varied surfaces from current material_map
                for s, Z in getattr(self, '_material_map', {}).items():
                    if s not in Z_map:
                        Z_map[s] = Z
                print(f'  Training: {Z_map}')
                for s in s_vals:
                    Br = np.zeros(self.N)
                    for face, Z in Z_map.items():
                        if face in B_labels:
                            Br += C_AIR**2 * RHO_AIR * B_labels[face] / Z
                    sig, omg = s.real, s.imag
                    Kr = self._c2S + sparse.diags((sig**2-omg**2)*self._M + sig*Br, format='csc')
                    Kc = sparse.diags(2*sig*omg*self._M + omg*Br, format='csc')
                    A = sparse.bmat([[Kr,-Kc],[Kc,Kr]], format='csc')
                    rhs = np.concatenate([sig*self._p0*self._M, omg*self._p0*self._M])
                    x = spsolve(A, rhs)
                    all_snaps.append(x[:self.N] + 1j*x[self.N:])

        elif self._bc_type == 'FI' and Z_train:
            for Zs in Z_train:
                print(f'  Snapshots Zs={Zs}...')
                for s in s_vals:
                    Br = C_AIR**2 * RHO_AIR * self._B / Zs
                    sig, omg = s.real, s.imag
                    Kr = self._c2S + sparse.diags((sig**2-omg**2)*self._M + sig*Br, format='csc')
                    Kc = sparse.diags(2*sig*omg*self._M + omg*Br, format='csc')
                    A = sparse.bmat([[Kr,-Kc],[Kc,Kr]], format='csc')
                    rhs = np.concatenate([sig*self._p0*self._M, omg*self._p0*self._M])
                    x = spsolve(A, rhs)
                    all_snaps.append(x[:self.N] + 1j*x[self.N:])
        else:
            raise ValueError('Provide Z_train, d_train, or material_variations')

        print(f'  SVD ({len(all_snaps)} snapshots)...')
        Psi, Nrb, sv = build_basis(all_snaps, eps_pod)
        print(f'  Nrb = {Nrb}')
        del all_snaps

        # Project operators: M_r, S_r, and per-surface Br_r
        M_r = Psi.T @ (self._M[:, None] * Psi)
        S_r = Psi.T @ self.ops['S'].dot(Psi)
        Br_r_per_surface = {}
        for label, B_diag in B_labels.items():
            Br_r_per_surface[label] = Psi.T @ (
                C_AIR**2 * RHO_AIR * B_diag[:, None] * Psi)
        f_r = Psi.T @ (self._p0 * self._M)
        obs = Psi[self._rec_idx, :].copy()

        rom_ops = dict(M_r=M_r, S_r=S_r, Br_r_per_surface=Br_r_per_surface,
                       f_r=f_r, obs=obs)

        return ROM(rom_ops, s_vals, freqs, sigma, fs, t_max,
                   bc_type=self._bc_type,
                   material_map=getattr(self, '_material_map', {}),
                   B_labels_keys=list(B_labels.keys()))

    def _check_ready(self):
        if self._p0 is None:
            raise RuntimeError('Call set_source() first')
        if self._rec_idx is None:
            raise RuntimeError('Call set_receiver() first')
        if self._bc_type is None:
            raise RuntimeError('Call set_boundary_fi(), set_boundary_fd(), or set_material() first')
