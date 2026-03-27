#!/usr/bin/env python
"""
3-D Room Acoustics Validation — targeting 100x ROM speedup.

Reproduces Paper 2 (Sampedro Llopis 2022) 3D test case:
  1m x 1m x 1m cube, freq-dependent boundaries, f_max = 1 kHz.

Memory budget: < 4 GB peak (safe for 16 GB system).
GPU: RTX 2060 Max-Q (6 GB VRAM) for FOM sparse matvec.
"""

import os, sys, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from room_acoustics.sem import BoxMesh3D, assemble_3d_operators
from room_acoustics.solvers import (
    fom_pphi_3d_gpu, rom_pphi_3d, build_psd_basis, build_modified_psd_basis,
    C_AIR, RHO_AIR,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'results')
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 62)
    print("  3-D Room Acoustics — 100x Speedup Target")
    print("  Paper 2: 1m cube, FI boundaries, f_max=1kHz")
    print("=" * 62)

    # --- Mesh (Paper 2 section III.B.3) ---
    Lx = Ly = Lz = 1.0          # 1m cube
    P = 4
    f_max = 1000.0
    PPW = 10                     # points per wavelength
    lam_min = C_AIR / f_max      # 0.343 m
    h_target = lam_min / PPW * P  # ~0.137 m
    Ne = max(2, int(np.ceil(Lx / h_target)))   # elements per direction

    print(f"\n  Target: {Ne} elements/dir, P={P}")
    mesh = BoxMesh3D(Lx, Ly, Lz, Ne, Ne, Ne, P)
    print(f"  Mesh: {Ne}x{Ne}x{Ne} hex elements, N_dof = {mesh.N_dof}")
    print(f"  Element size: h = {mesh.hx:.4f} m")
    mem_state = mesh.N_dof * 8 * 2 / 1e6
    print(f"  State vectors (p + Phi): {mem_state:.1f} MB")

    # --- Operators ---
    print("\n  Assembling 3D operators...")
    t0 = time.perf_counter()
    ops = assemble_3d_operators(mesh)
    t_asm = time.perf_counter() - t0
    nnz = ops['S'].nnz
    mem_ops = nnz * 12 / 1e6   # ~12 bytes per entry (CSR)
    print(f"  Assembly: {t_asm:.1f}s, S nnz={nnz:,}, ops memory ~{mem_ops:.0f} MB")

    # --- Source / receiver ---
    src = (0.5, 0.5, 0.5)       # center
    rec = (0.25, 0.1, 0.8)
    rec_idx = mesh.nearest_node(*rec)
    sigma = 0.2                  # Gaussian width
    T = 0.15                     # 150 ms — long enough to expose instability

    # CFL: dt <= CFL * h / (c * P^2)  — conservative for RK4
    dt = 0.15 * mesh.hx / (C_AIR * P**2)
    dt = round(dt, 8)
    Nt = int(round(T / dt))
    print(f"\n  dt = {dt:.2e} s, Nt = {Nt}, T = {T} s")

    # Snapshot stride: cap at ~2000 snapshots, stay under 2 GB
    snap_stride = max(1, Nt // 2000)
    n_snaps_est = Nt // snap_stride + 1
    mem_snaps = mesh.N_dof * n_snaps_est * 8 * 2 / 1e9
    print(f"  Snapshot plan: stride={snap_stride}, ~{n_snaps_est} snaps, ~{mem_snaps:.2f} GB")

    # ===== TEST A: Perfectly Reflecting (stable, best speedup) =====
    for bc_label, bc_type, bc_params in [
        ("PR (rigid walls)", 'PR', {}),
        ("FI (Z=2000)",      'FI', {'Z': 2000}),
    ]:
        print(f"\n  --- {bc_label} ---")
        store_bp = (bc_type == 'FI')
        print(f"  Running 3D FOM (p-Phi, store_bp={store_bp})...")
        t0 = time.perf_counter()
        res_fom = fom_pphi_3d_gpu(mesh, ops, bc_type, bc_params, *src, sigma,
                                   dt, T, rec_idx=rec_idx,
                                   store_snapshots=True, snap_stride=snap_stride,
                                   store_boundary_pressure=store_bp)
        t_fom = time.perf_counter() - t0
        print(f"  FOM time: {t_fom:.2f}s ({Nt} steps, {t_fom/Nt*1000:.3f} ms/step)")
        n_snaps = len(res_fom['snaps_p'])
        print(f"  Snapshots: {n_snaps}")

        # Build basis — modified for FI (boundary energy enrichment)
        print("  Building PSD basis...")
        t0 = time.perf_counter()
        if store_bp and 'snaps_pb' in res_fom:
            print("    -> Modified p-Phi (boundary energy enrichment)")
            Psi_H, sigmas, Nrb_auto = build_modified_psd_basis(
                res_fom['snaps_p'], res_fom['snaps_Phi'],
                res_fom['snaps_pb'], eps_pod=1e-8)
            del res_fom['snaps_pb']
        else:
            Psi_H, sigmas, Nrb_auto = build_psd_basis(
                res_fom['snaps_p'], res_fom['snaps_Phi'], eps_pod=1e-8)
        t_svd = time.perf_counter() - t0
        del res_fom['snaps_p'], res_fom['snaps_Phi']; gc.collect()
        print(f"  SVD: {t_svd:.1f}s, Nrb_auto={Nrb_auto}, basis={Psi_H.shape}")

        # ROM sweep
        nrb_list = [5, 10, 20, 40, 60, 80, 100, 150]
        nrb_list = [n for n in nrb_list if n <= Psi_H.shape[1]]
        speedups, errors = [], []

        print(f"  {'Nrb':>6s} {'ROM_t':>8s} {'Speedup':>10s} {'Error':>12s}")
        for Nrb in nrb_list:
            t0 = time.perf_counter()
            res_rom = rom_pphi_3d(mesh, ops, Psi_H, bc_type, bc_params,
                                  *src, sigma, dt, T,
                                  rec_idx=rec_idx, Nrb_override=Nrb)
            t_rom = time.perf_counter() - t0
            eps = np.max(np.abs(res_fom['ir'] - res_rom['ir']))
            sp = t_fom / t_rom
            speedups.append(sp)
            errors.append(eps)
            tag = " ***" if sp >= 100 else ""
            print(f"  {Nrb:6d} {t_rom:8.2f}s {sp:10.1f}x {eps:12.2e}{tag}")

        # Plot for this BC
        t_vec = np.arange(len(res_fom['ir'])) * dt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(t_vec*1e3, res_fom['ir'], 'b-', lw=1, label='FOM')
        if len(nrb_list) > 0 and len(errors) > 0:
            best_idx = len(errors) - 1  # highest Nrb
            res_best = rom_pphi_3d(mesh, ops, Psi_H, bc_type, bc_params,
                                   *src, sigma, dt, T,
                                   rec_idx=rec_idx, Nrb_override=nrb_list[best_idx])
            ax1.plot(t_vec*1e3, res_best['ir'], 'r--', lw=1,
                     label=f'ROM Nrb={nrb_list[best_idx]}')
        ax1.set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
                title=f'3D {bc_label}')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        valid = [(s, e, n) for s, e, n in zip(speedups, errors, nrb_list) if e < 1e6]
        if valid:
            sp_v, er_v, nr_v = zip(*valid)
            ax2.loglog(er_v, sp_v, 'bo-', markersize=6)
            for sv, ev, nv in zip(sp_v, er_v, nr_v):
                ax2.annotate(f'{nv}', (ev, sv), fontsize=7, ha='left')
        ax2.axhline(100, color='g', ls='--', lw=1, label='100x')
        ax2.set(xlabel='Error [Pa]', ylabel='Speedup',
                title=f'Speedup (N={mesh.N_dof})')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f'test_3d_{bc_type.lower()}.png'
        plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot: {fname}")

        max_sp = max(speedups) if speedups else 0
        tag = "ACHIEVED" if max_sp >= 100 else f"{max_sp:.0f}x"
        print(f"  Peak speedup: {tag}")
        del Psi_H; gc.collect()

    print("\n  Done.")


if __name__ == '__main__':
    main()
