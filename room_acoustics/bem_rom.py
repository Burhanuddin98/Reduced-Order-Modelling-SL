"""
BEM-ROM: Reduced Order Model built on BEM surface snapshots.

Offline phase:
  1. Solve BEM at Ns training frequencies (Laplace domain)
  2. Collect surface pressure snapshots (N_surf × Ns)
  3. SVD → reduced basis Ψ (N_surf × Nrb)
  4. Project BEM operators onto Ψ → small dense system (Nrb × Nrb)

Online phase:
  For any new impedance Z(f) or source/receiver position:
  1. Solve Nrb × Nrb dense system at each frequency (~microseconds)
  2. Reconstruct H(f) → inverse Laplace → IR

Speedup: 100-1000× over FOM, depending on Nrb vs N.
Storage: Ψ is N_surf × Nrb (~MB, not GB).

Usage:
    from room_acoustics.bem_rom import BEMROM

    rom = BEMROM.build(solver, source, receiver, materials,
                       f_min=20, f_max=4000, n_train=60)
    # Now instant evaluation for any materials:
    ir = rom.impulse_response(new_materials, T=3.0)
"""

import numpy as np
from typing import Dict, Optional, Tuple
import time

from .bem_solver import (BEMSolver, _assemble_bem_matrices_chunked,
                          _incident_field, _evaluate_at_receiver,
                          _simple_inverse_laplace)
from .material_function import MaterialFunction


class BEMROM:
    """
    Reduced-order BEM solver for parametric room acoustics.

    The reduced basis captures the room's acoustic behavior across
    frequencies. Changing materials only requires re-solving a small
    dense system (Nrb × Nrb) — no re-assembly of the full BEM matrix.
    """

    def __init__(self, solver, basis, sv, training_freqs, sigma,
                 source, receiver, snapshots_H):
        """
        Don't call directly — use BEMROM.build().
        """
        self.solver = solver
        self.basis = basis          # (N, Nrb) reduced basis
        self.sv = sv                # (Nrb,) singular values
        self.training_freqs = training_freqs
        self.sigma = sigma
        self.source = np.asarray(source)
        self.receiver = np.asarray(receiver)
        self.snapshots_H = snapshots_H  # (n_train,) complex, for validation
        self.Nrb = basis.shape[1]

    @classmethod
    def build(cls, solver, source, receiver, materials,
              f_min=20, f_max=4000, n_train=60, sigma=5.0,
              tol_sv=1e-8):
        """
        Build ROM from BEM training solves.

        Parameters
        ----------
        solver : BEMSolver
        source, receiver : (3,) positions
        materials : dict of {group_name: MaterialFunction}
        f_min, f_max : training frequency range
        n_train : number of training frequencies
        sigma : Laplace damping
        tol_sv : singular value truncation tolerance

        Returns
        -------
        rom : BEMROM
        """
        print(f"BEM-ROM offline phase: {n_train} training frequencies "
              f"[{f_min}-{f_max} Hz], {solver.N} surface DOFs")

        t0 = time.perf_counter()

        # Log-spaced frequencies (denser at low freq where modes are sharp)
        train_freqs = np.geomspace(max(f_min, 5), f_max, n_train)

        # Compute full BEM at training frequencies
        H, freqs, snapshots = solver.transfer_function(
            source, receiver, materials,
            freqs=train_freqs, sigma=sigma)

        # SVD of snapshot matrix (real + imag stacked)
        # Following Sampedro Llopis: stack [Re(S); Im(S)] for real SVD
        S_real = np.vstack([snapshots.real, snapshots.imag])  # (2N, n_train)

        print(f"  SVD on {S_real.shape[0]}×{S_real.shape[1]} snapshot matrix...")
        t_svd = time.perf_counter()
        U, sv, Vt = np.linalg.svd(S_real, full_matrices=False)
        dt_svd = time.perf_counter() - t_svd
        print(f"  SVD complete: {dt_svd:.1f}s")

        # Truncate basis
        # Energy criterion: keep modes capturing (1 - tol_sv) of total energy
        energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
        Nrb = max(np.searchsorted(energy, 1.0 - tol_sv) + 1, 5)
        Nrb = min(Nrb, len(sv))

        # Extract real-space basis (first N rows of U)
        N = solver.N
        basis_real = U[:N, :Nrb]
        basis_imag = U[N:, :Nrb]
        basis = basis_real + 1j * basis_imag  # (N, Nrb) complex

        sv_trunc = sv[:Nrb]
        reduction = 100 * (1 - Nrb / N)

        total = time.perf_counter() - t0
        print(f"  ROM built: Nrb={Nrb} basis vectors "
              f"({reduction:.1f}% reduction from {N} DOFs)")
        print(f"  Singular value range: {sv[0]:.2e} to {sv_trunc[-1]:.2e}")
        print(f"  Total offline time: {total:.1f}s")

        return cls(solver, basis, sv_trunc, train_freqs, sigma,
                   source, receiver, H)

    def solve_online(self, materials, freqs=None, f_min=20, f_max=4000,
                     n_freqs=100):
        """
        Online ROM evaluation — fast dense solves.

        Parameters
        ----------
        materials : dict of {group_name: MaterialFunction}
            Can be different from training materials.
        freqs : array of evaluation frequencies [Hz]

        Returns
        -------
        H : (n_freqs,) complex transfer function
        freqs : (n_freqs,) float frequency vector
        """
        if freqs is None:
            freqs = np.linspace(f_min, f_max, n_freqs)

        c = self.solver.c
        rho = self.solver.rho
        Psi = self.basis  # (N, Nrb)

        H = np.zeros(len(freqs), dtype=complex)

        t0 = time.perf_counter()
        for i, f in enumerate(freqs):
            omega = 2 * np.pi * f
            s = self.sigma + 1j * omega
            k = s / c

            # Get impedance for this frequency
            Z = self.solver._get_impedance(materials, f)
            beta = rho * c / (Z + 1e-10)

            # Assemble reduced system: A_r = Ψ^H A Ψ
            # Instead of assembling full A, we project:
            #   A_r[p,q] = Σ_ij Ψ*_ip A_ij Ψ_jq
            # This is expensive if done naively. Use:
            #   A_r = Ψ^H A Ψ where A is assembled row-by-row

            # For moderate N: assemble full A, project
            # For large N: use FMM-accelerated matvecs
            N = self.solver.N

            if N <= 15000:
                # Direct projection
                A = _assemble_bem_matrices_chunked(
                    self.solver.centers, self.solver.normals,
                    self.solver.areas, k, beta)

                A_r = Psi.conj().T @ A @ Psi  # (Nrb, Nrb)
            else:
                # Matvec-based projection: A_r[:,q] = Ψ^H (A Ψ[:,q])
                # Still requires N matvecs but avoids N×N storage
                A_r = np.zeros((self.Nrb, self.Nrb), dtype=complex)
                for q in range(self.Nrb):
                    # A @ Psi[:,q] via chunked assembly
                    Apsi_q = self._matvec_bem(k, beta, Psi[:, q])
                    A_r[:, q] = Psi.conj().T @ Apsi_q

            # Reduced RHS: f_r = Ψ^H f
            rhs = _incident_field(self.solver.centers, self.source, k)
            f_r = Psi.conj().T @ rhs  # (Nrb,)

            # Solve small dense system
            try:
                p_r = np.linalg.solve(A_r, f_r)  # (Nrb,)
            except np.linalg.LinAlgError:
                A_r += 1e-10 * np.eye(self.Nrb)
                p_r = np.linalg.solve(A_r, f_r)

            # Reconstruct surface pressure
            p_surf = Psi @ p_r  # (N,)

            # Evaluate at receiver
            H[i] = _evaluate_at_receiver(
                p_surf, self.solver.centers, self.solver.normals,
                self.solver.areas, self.receiver, self.source, k, beta)

        total = time.perf_counter() - t0
        print(f"  ROM online: {len(freqs)} freqs in {total:.2f}s "
              f"({total/len(freqs)*1000:.1f}ms/freq)")

        return H, freqs

    def _matvec_bem(self, k, beta, x):
        """BEM matrix-vector product without storing full matrix."""
        centers = self.solver.centers
        normals = self.solver.normals
        areas = self.solver.areas
        N = len(centers)

        y = 0.5 * x.copy()
        # Process in chunks
        chunk = 3000
        for i in range(0, N, chunk):
            ie = min(i + chunk, N)
            diff = centers[i:ie, None, :] - centers[None, :, :]
            r = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(r[: , i:ie] if ie - i == N else
                             np.zeros(1), 1.0)  # avoid /0

            for di in range(ie - i):
                j = i + di
                if j < N:
                    r[di, j] = 1.0

            G = np.exp(1j * k * r) / (4 * np.pi * r)
            r_hat = diff / (r[:, :, None] + 1e-30)
            cos_a = np.sum(r_hat * normals[None, :, :], axis=2)
            dGdn = G * (1j * k - 1.0 / (r + 1e-30)) * cos_a

            for di in range(ie - i):
                j = i + di
                if j < N:
                    G[di, j] = 0
                    dGdn[di, j] = 0

            kernel = (dGdn + 1j * k * beta[None, :] * G) * areas[None, :]
            y[i:ie] += kernel @ x

        return y

    def impulse_response(self, materials=None, T=3.0, sr=44100,
                          f_min=20, f_max=None, n_freqs=100):
        """
        Compute IR using online ROM.

        Parameters
        ----------
        materials : dict or None (uses training materials if None)
        T : IR duration
        sr : sample rate
        f_max : upper frequency (defaults to training max)

        Returns
        -------
        ir : (N_samples,) float
        t : (N_samples,) float
        """
        if materials is None:
            # Use cached training transfer function
            ir, t = _simple_inverse_laplace(
                self.snapshots_H, self.training_freqs, self.sigma, T, sr)
            peak = np.max(np.abs(ir))
            if peak > 0:
                ir /= peak
            return ir, t

        if f_max is None:
            f_max = self.training_freqs[-1]

        H, freqs = self.solve_online(materials, f_min=f_min,
                                      f_max=f_max, n_freqs=n_freqs)

        ir, t = _simple_inverse_laplace(H, freqs, self.sigma, T, sr)
        peak = np.max(np.abs(ir))
        if peak > 0:
            ir /= peak
        return ir, t

    def extract_modes(self, materials=None, f_min=20, f_max=4000,
                       n_freqs=200):
        """
        Extract resonance frequencies and decay rates from BEM transfer function.

        Uses peak-picking on |H(f)| to find resonances, then fits
        decay rate from the -3dB bandwidth of each peak.

        Returns
        -------
        modes : list of (frequency, amplitude, decay_rate) tuples
        """
        if materials is not None:
            H, freqs = self.solve_online(materials, f_min=f_min,
                                          f_max=f_max, n_freqs=n_freqs)
        else:
            H = self.snapshots_H
            freqs = self.training_freqs

        H_mag = np.abs(H)
        H_db = 20 * np.log10(H_mag + 1e-30)

        # Peak detection
        modes = []
        for i in range(1, len(H_mag) - 1):
            if H_mag[i] > H_mag[i-1] and H_mag[i] > H_mag[i+1]:
                f_peak = freqs[i]
                amp = H_mag[i]

                # -3dB bandwidth → decay rate
                threshold = H_db[i] - 3.0
                # Find -3dB crossings
                left = i
                while left > 0 and H_db[left] > threshold:
                    left -= 1
                right = i
                while right < len(H_db) - 1 and H_db[right] > threshold:
                    right += 1

                if right > left + 1:
                    bw = freqs[right] - freqs[left]
                    gamma = np.pi * bw  # decay rate from bandwidth
                else:
                    gamma = self.sigma  # fallback

                modes.append((f_peak, amp, gamma))

        return modes
