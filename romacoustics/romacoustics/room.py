"""High-level Room API for Laplace-domain ROM acoustics."""

import numpy as np
from romacoustics.sem import RectMesh2D, BoxMesh3D, assemble_2d, assemble_3d
from romacoustics.solver import (
    C_AIR, RHO_AIR,
    weeks_s_values, laplace_to_ir,
    sweep_fi, sweep_fd,
    sweep_fi_fullfield, sweep_fd_fullfield,
    build_basis, project_operators,
    rom_sweep_fi, rom_sweep_fd,
)
from romacoustics.ir import ImpulseResponse


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

    # ── Source / Receiver ────────────────────────────────────

    def set_source(self, *pos, sigma=0.2):
        """Set Gaussian pulse source. pos = (x,y) or (x,y,z)."""
        self._src = pos
        if self.mesh.ndim == 2:
            r2 = (self.mesh.x-pos[0])**2 + (self.mesh.y-pos[1])**2
        else:
            self.mesh._ensure_coords()
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
        """Frequency-independent boundary (constant impedance)."""
        self._bc_type = 'FI'
        self._bc_params = {'Zs': Zs}

    def set_boundary_fd(self, sigma_flow=10000, d_mat=0.05):
        """Frequency-dependent boundary (Miki porous absorber)."""
        self._bc_type = 'FD'
        self._bc_params = {'sigma_flow': sigma_flow, 'd_mat': d_mat}

    # ── Solve ────────────────────────────────────────────────

    def solve(self, t_max=0.1, fs=44100, Ns=None, sigma_w=None, b_w=None):
        """Compute impulse response via Laplace-domain FOM + Weeks ILT.

        Returns ImpulseResponse object.
        """
        self._check_ready()

        # Auto-select Weeks parameters based on dimensionality
        if self.mesh.ndim == 2:
            Ns = Ns or 1000
            sigma_w = sigma_w or 10.0
            b_w = b_w or 1000.0
        else:
            Ns = Ns or 500
            sigma_w = sigma_w or 20.0
            b_w = b_w or 800.0

        s_vals, _ = weeks_s_values(sigma_w, b_w, Ns)
        t_eval = np.arange(0, t_max, 1.0/fs)

        if self._bc_type == 'FI':
            H = sweep_fi(self._c2S, self._M, self._B, self._p0, self.N,
                         s_vals, self._bc_params['Zs'], self._rec_idx)
            label = f'FOM Zs={self._bc_params["Zs"]}'
        elif self._bc_type == 'FD':
            H = sweep_fd(self._c2S, self._M, self._B, self._p0, self.N,
                         s_vals, self._bc_params['sigma_flow'],
                         self._bc_params['d_mat'], self._rec_idx)
            label = f'FOM d={self._bc_params["d_mat"]}m'
        else:
            raise ValueError(f'Unknown BC type: {self._bc_type}')

        ir = laplace_to_ir(H, sigma_w, b_w, t_eval)
        return ImpulseResponse(ir, fs, label)

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
            raise RuntimeError('Call set_boundary_fi() or set_boundary_fd() first')
