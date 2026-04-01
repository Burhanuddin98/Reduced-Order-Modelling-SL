"""
FDTD acoustic solver — finite-difference time-domain wave equation.

Solves the 3D wave equation on a voxel grid using a 27-point Laplacian
stencil with per-voxel material absorption, scattering, and sponge
absorbing boundary conditions.

Supports CPU (numpy) and GPU (CuPy) backends with identical interface.

Usage:
    from room_acoustics.fdtd import FDTDSolver

    solver = FDTDSolver(air, dx=0.05, c=343.0)
    solver.set_materials(alpha_3d, scatter_3d)

    # Compute impulse response
    ir, fs = solver.impulse_response(source_ijk, receiver_ijk, duration=1.0)

    # Compute RMS pressure field (for visualization)
    rms = solver.compute_rms(source_ijk, freq=500, steps=200)
"""

import numpy as np


def _get_backend(use_gpu):
    """Get numpy or cupy as compute backend."""
    if use_gpu:
        try:
            import cupy as cp
            return cp
        except ImportError:
            pass
    return np


class FDTDSolver:
    """
    3D FDTD acoustic wave equation solver.

    Uses a 27-point Laplacian stencil (faces + edges + corners)
    for improved isotropy compared to the standard 7-point stencil.
    """

    def __init__(self, air, dx, c=343.0, CFL=0.25, use_gpu=False):
        """
        Parameters
        ----------
        air : ndarray (nx, ny, nz), uint8
            1 = air, 0 = solid.
        dx : float
            Voxel size [m] (isotropic).
        c : float
            Speed of sound [m/s].
        CFL : float
            Courant number (stability: CFL <= 1/sqrt(3) ≈ 0.577 for 3D).
        use_gpu : bool
            Use CuPy GPU backend if available.
        """
        self.xp = _get_backend(use_gpu)
        self.use_gpu = (self.xp is not np)
        xp = self.xp

        self.nx, self.ny, self.nz = air.shape
        self.dx = float(dx)
        self.c = float(c)
        self.CFL = float(CFL)
        self.dt = CFL * dx / c
        self.fs = 1.0 / self.dt

        self.air = xp.asarray(air, dtype=xp.uint8)
        self.air_mask = (self.air == 1)
        self.solid_mask = ~self.air_mask

        # Sponge absorbing boundary
        self._sponge = self._build_sponge(thickness=10, alpha=0.08)

        # Material masks (default: no absorption, no scattering)
        self._R_mask = None  # reflection factor sqrt(1-alpha)
        self._S_mask = None  # scattering coefficient

    def set_materials(self, alpha_3d=None, scatter_3d=None):
        """
        Set per-voxel material absorption and scattering.

        Parameters
        ----------
        alpha_3d : ndarray (nx, ny, nz), float32 or None
            Absorption coefficient per voxel [0, 1]. Only boundary
            voxels should have nonzero values.
        scatter_3d : ndarray (nx, ny, nz), float32 or None
            Scattering coefficient per voxel [0, 1].
        """
        xp = self.xp
        if alpha_3d is not None:
            alpha = xp.asarray(alpha_3d, dtype=xp.float32)
            self._R_mask = xp.sqrt(xp.clip(1.0 - alpha, 0.0, 1.0))
        else:
            self._R_mask = None

        if scatter_3d is not None:
            self._S_mask = xp.asarray(scatter_3d, dtype=xp.float32)
        else:
            self._S_mask = None

    def impulse_response(self, source_ijk, receiver_ijk, duration=1.0,
                         warmup_steps=16, stencil_radius=1):
        """
        Compute impulse response between source and receiver.

        Parameters
        ----------
        source_ijk : (int, int, int)
            Source voxel indices.
        receiver_ijk : (int, int, int)
            Receiver voxel indices.
        duration : float
            IR duration [seconds].
        warmup_steps : int
            Silent steps before impulse injection.
        stencil_radius : int
            Receiver averaging neighborhood radius.

        Returns
        -------
        ir : ndarray (n_samples,), float32
            Impulse response.
        fs : int
            Sample rate [Hz] (determined by CFL and dx).
        """
        xp = self.xp
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dt, c = self.dx, self.dt, self.c

        total_steps = int(np.round(duration * self.fs))
        total_steps = max(total_steps, warmup_steps + 64)

        si, sj, sk = map(int, source_ijk)
        ri, rj, rk = map(int, receiver_ijk)

        # Allocate fields
        p_nm1 = xp.zeros((nx, ny, nz), dtype=xp.float32)
        p_n = xp.zeros_like(p_nm1)

        # Build receiver stencil mask
        recv_mask = self._build_receiver_mask(ri, rj, rk, stencil_radius)

        # Sponge
        sponge = self._sponge

        # Record buffer (after warmup)
        rec_len = total_steps - warmup_steps
        rec = xp.zeros(rec_len, dtype=xp.float32)

        dx2 = dx * dx
        c2dt2 = (c * dt) ** 2

        for t in range(total_steps):
            p_np1 = self._step_27pt(p_nm1, p_n, c2dt2, dx2)

            # Inject impulse once after warmup
            if t == warmup_steps:
                if (0 <= si < nx and 0 <= sj < ny and 0 <= sk < nz
                        and self.air_mask[si, sj, sk]):
                    p_np1[si, sj, sk] += 1.0

            # Neumann BC at domain edges
            p_np1[0, :, :] = p_np1[1, :, :]
            p_np1[-1, :, :] = p_np1[-2, :, :]
            p_np1[:, 0, :] = p_np1[:, 1, :]
            p_np1[:, -1, :] = p_np1[:, -2, :]
            p_np1[:, :, 0] = p_np1[:, :, 1]
            p_np1[:, :, -1] = p_np1[:, :, -2]

            # Rigid solids
            p_np1[self.solid_mask] = p_n[self.solid_mask]

            # Material absorption
            if self._R_mask is not None:
                p_np1 *= self._R_mask

            # Surface scattering
            if self._S_mask is not None:
                self._apply_scattering(p_np1)

            # Sponge ABC
            p_np1 *= sponge

            # Record
            if t >= warmup_steps:
                idx = t - warmup_steps
                if idx < rec_len:
                    if recv_mask is not None:
                        vals = p_np1[recv_mask]
                        if vals.size:
                            rec[idx] = xp.mean(vals)
                    else:
                        if (0 <= ri < nx and 0 <= rj < ny and 0 <= rk < nz):
                            rec[idx] = p_np1[ri, rj, rk]

            # Rotate buffers
            p_nm1 = p_n
            p_n = p_np1

        # Back to CPU
        if self.use_gpu:
            rec = self.xp.asnumpy(rec)

        return rec.astype(np.float32), int(np.round(self.fs))

    def compute_rms(self, source_ijk, freq=500.0, steps=200, damp=1e-3):
        """
        Compute RMS pressure field for a continuous sinusoidal source.

        Parameters
        ----------
        source_ijk : (int, int, int)
            Source voxel indices.
        freq : float
            Drive frequency [Hz].
        steps : int
            Number of time steps.
        damp : float
            Damping factor per step.

        Returns
        -------
        rms : ndarray (nx, ny, nz), float32
            RMS pressure at each voxel.
        """
        xp = self.xp
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dt, c = self.dx, self.dt, self.c

        # Auto-extend for low frequencies
        period = 1.0 / max(freq, 1.0)
        min_cycles = 3
        warmup = int(np.ceil(2 * period / dt))
        accum_steps = int(np.ceil(min_cycles * period / dt))
        steps = max(steps, warmup + accum_steps)

        si, sj, sk = map(int, source_ijk)
        p_nm1 = xp.zeros((nx, ny, nz), dtype=xp.float32)
        p_n = xp.zeros_like(p_nm1)
        acc = xp.zeros_like(p_nm1)

        omega = 2 * np.pi * freq
        drive_amp = 3e-3 * np.sqrt(max(1.0, 240.0 / freq))
        dx2 = dx * dx
        c2dt2 = (c * dt) ** 2
        sponge = self._sponge
        t = 0.0

        for step in range(steps):
            p_np1 = self._step_27pt(p_nm1, p_n, c2dt2, dx2)

            # Damped wave equation
            p_np1 = (2 - damp) * p_n - (1 - damp) * p_nm1 + c2dt2 * (
                (p_np1 - (2 - damp) * p_n + (1 - damp) * p_nm1) / c2dt2)
            # Simplify: this is just the standard FDTD with damping
            # p_np1 = (2-d)*p_n - (1-d)*p_nm1 + c2dt2*lap
            # The _step_27pt already computed 2*p_n - p_nm1 + c2dt2*lap
            # So we need to redo with damping. Let me just rewrite:

            lap = self._laplacian_27pt(p_n, dx2)
            p_np1 = (2 - damp) * p_n - (1 - damp) * p_nm1 + c2dt2 * lap

            # Drive
            s = float(xp.sin(omega * t)) if self.use_gpu else np.sin(omega * t)
            if (0 <= si < nx and 0 <= sj < ny and 0 <= sk < nz
                    and self.air_mask[si, sj, sk]):
                p_np1[si, sj, sk] += drive_amp * s

            # Boundaries
            p_np1[0, :, :] = p_np1[1, :, :]
            p_np1[-1, :, :] = p_np1[-2, :, :]
            p_np1[:, 0, :] = p_np1[:, 1, :]
            p_np1[:, -1, :] = p_np1[:, -2, :]
            p_np1[:, :, 0] = p_np1[:, :, 1]
            p_np1[:, :, -1] = p_np1[:, :, -2]

            p_np1[self.solid_mask] = p_n[self.solid_mask]

            if self._R_mask is not None:
                p_np1 *= xp.clip(self._R_mask, 0.4, 1.0)
            if self._S_mask is not None:
                self._apply_scattering(p_np1)
            p_np1 *= sponge

            p_nm1 = p_n
            p_n = p_np1
            if step >= warmup:
                acc += p_n * p_n
            t += dt

        denom = max(1, steps - warmup)
        rms = xp.sqrt(acc / denom).astype(xp.float32)

        if self.use_gpu:
            rms = self.xp.asnumpy(rms)
        return rms

    # ===============================================================
    # Internal
    # ===============================================================

    def _laplacian_27pt(self, p, dx2):
        """Compute 27-point discrete Laplacian."""
        xp = self.xp
        lap = xp.zeros_like(p)

        pc = p[1:-1, 1:-1, 1:-1]

        # Face neighbors (6), weight 1.0
        faces = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
                 p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
                 p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])

        # Edge neighbors (12), weight 0.5
        edges = (
            p[2:, 2:, 1:-1] + p[2:, :-2, 1:-1] +
            p[:-2, 2:, 1:-1] + p[:-2, :-2, 1:-1] +
            p[2:, 1:-1, 2:] + p[2:, 1:-1, :-2] +
            p[:-2, 1:-1, 2:] + p[:-2, 1:-1, :-2] +
            p[1:-1, 2:, 2:] + p[1:-1, 2:, :-2] +
            p[1:-1, :-2, 2:] + p[1:-1, :-2, :-2])

        # Corner neighbors (8), weight 0.25
        corners = (
            p[2:, 2:, 2:] + p[2:, 2:, :-2] +
            p[2:, :-2, 2:] + p[2:, :-2, :-2] +
            p[:-2, 2:, 2:] + p[:-2, 2:, :-2] +
            p[:-2, :-2, 2:] + p[:-2, :-2, :-2])

        # center_coeff = -(6*1.0 + 12*0.5 + 8*0.25) = -14.0
        lap[1:-1, 1:-1, 1:-1] = (
            1.0 * faces + 0.5 * edges + 0.25 * corners - 14.0 * pc
        ) / dx2

        return lap

    def _step_27pt(self, p_nm1, p_n, c2dt2, dx2):
        """One FDTD time step using 27-point Laplacian."""
        lap = self._laplacian_27pt(p_n, dx2)
        return 2.0 * p_n - p_nm1 + c2dt2 * lap

    def _build_sponge(self, thickness=10, alpha=0.08):
        """Build sponge absorbing boundary layer."""
        xp = self.xp
        nx, ny, nz = self.nx, self.ny, self.nz

        sponge = xp.ones((nx, ny, nz), dtype=xp.float32)
        for ax, L in enumerate((nx, ny, nz)):
            pad = min(thickness, L // 2)
            if pad > 0:
                ramp = xp.ones(L, dtype=xp.float32)
                inner = xp.linspace(0.0, 1.0, pad, dtype=xp.float32)
                rise = inner * inner  # quadratic
                ramp[:pad] = rise
                ramp[-pad:] = rise[::-1]
                shape = [1, 1, 1]
                shape[ax] = L
                sponge *= ramp.reshape(shape)

        # Apply exponential taper
        sponge = xp.exp(-alpha * (1.0 - sponge))
        # Don't apply sponge inside solid
        sponge = xp.where(self.air_mask, sponge, xp.ones_like(sponge))
        return sponge

    def _build_receiver_mask(self, ri, rj, rk, radius):
        """Build 3D receiver stencil mask."""
        if radius <= 0:
            return None
        xp = self.xp
        mask = xp.zeros((self.nx, self.ny, self.nz), dtype=xp.bool_)
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    ii, jj, kk = ri + di, rj + dj, rk + dk
                    if (0 <= ii < self.nx and 0 <= jj < self.ny
                            and 0 <= kk < self.nz and self.air_mask[ii, jj, kk]):
                        mask[ii, jj, kk] = True
        return mask

    def _apply_scattering(self, p):
        """Apply surface scattering (local smoothing at boundary voxels)."""
        xp = self.xp
        S = self._S_mask
        core = (slice(1, -1), slice(1, -1), slice(1, -1))
        neigh_avg = (
            p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
            p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
            p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
        ) / 6.0
        S_core = S[core]
        p[core] = (1.0 - S_core) * p[core] + S_core * neigh_avg
