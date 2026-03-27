#!/usr/bin/env python
"""
Automated validation of all 2-D room acoustics test cases.

Reproduces results from:
  Paper 1 — Bonthu et al. (2026): Stable MOR for TD Room Acoustics
  Paper 2 — Sampedro Llopis et al. (2022): RBM for Room Acoustic Simulations

Test cases
----------
  1  PR  FOM comparison   (p-v vs p-Φ vs analytical)
  2  FI  FOM comparison   (Z = 600, 16 000)
  3  LR  FOM comparison   (σ_mat=10k, d=0.05 / 0.2)
  4  Energy conservation  (all BC types)
  5  ROM short (T=50 ms)  PR / FI / LR
  6  ROM long  (T=200 ms) stability: p-v vs p-Φ vs mod-pΦ
  7  Eigenvalue analysis   PR / FI
  8  Speedup analysis
"""

import os, sys, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# add parent to path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from room_acoustics.sem import RectMesh2D, assemble_2d_operators
from room_acoustics.solvers import (
    fom_pphi, fom_pv, analytical_rigid_rect,
    build_psd_basis, build_modified_psd_basis, build_pod_basis,
    rom_pphi, rom_pv, eigenvalue_analysis,
    C_AIR, RHO_AIR,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved {path}")


# ---------------------------------------------------------------------------
# Common parameters  (Paper 1, §3.1)
# ---------------------------------------------------------------------------
LX, LY       = 4.0, 2.0          # domain  [m]
SRC_X, SRC_Y  = 1.7, 1.2         # source
REC_X, REC_Y  = 0.25, 0.22       # receiver
SIGMA         = 0.3               # Gaussian pulse width  [m]
P_ORDER       = 4                 # polynomial order
PPW           = 10                # points per wavelength at f_max (paper: 7, raised for accuracy)
F_MAX         = 700.0             # upper cutoff  [Hz]
DT            = 1.0e-5            # time step  [s] (smaller for higher resolution)
CFL           = 0.2

# derived
LAM_MIN  = C_AIR / F_MAX
H_TARGET = LAM_MIN / PPW * P_ORDER
NEX      = max(2, int(np.ceil(LX / H_TARGET)))
NEY      = max(2, int(np.ceil(LY / H_TARGET)))


def make_mesh():
    mesh = RectMesh2D(LX, LY, NEX, NEY, P_ORDER)
    print(f"  Mesh: {NEX}×{NEY} elements, P={P_ORDER}, "
          f"N_dof={mesh.N_dof}, hx={mesh.hx:.4f}, hy={mesh.hy:.4f}")
    return mesh


# ===================================================================
#  TEST 1 — PR boundaries: FOM comparison (p-v vs p-Φ vs analytical)
# ===================================================================

def test1_pr_fom():
    print("\n" + "="*70)
    print("TEST 1: PR boundaries — FOM verification (T=50 ms)")
    print("="*70)
    mesh = make_mesh()
    ops  = assemble_2d_operators(mesh)
    rec  = mesh.nearest_node(REC_X, REC_Y)
    T    = 0.05

    t0 = time.perf_counter()
    res_pphi = fom_pphi(mesh, ops, 'PR', {}, SRC_X, SRC_Y, SIGMA,
                        DT, T, rec_idx=rec)
    t_pphi = time.perf_counter() - t0
    print(f"  p-Φ FOM: {t_pphi:.2f} s")

    t0 = time.perf_counter()
    res_pv = fom_pv(mesh, ops, 'PR', {}, SRC_X, SRC_Y, SIGMA,
                    DT, T, rec_idx=rec)
    t_pv = time.perf_counter() - t0
    print(f"  p-v FOM: {t_pv:.2f} s  (speedup p-Φ/p-v = {t_pv/t_pphi:.2f}×)")

    # Evaluate analytical solution at the actual grid point (not requested coords)
    actual_rx = mesh.x[rec]
    actual_ry = mesh.y[rec]
    print(f"  Receiver: requested ({REC_X},{REC_Y}), actual grid ({actual_rx:.4f},{actual_ry:.4f})")
    t_ana, ir_ana = analytical_rigid_rect(mesh, SRC_X, SRC_Y, SIGMA,
                                          actual_rx, actual_ry, DT, T, n_modes=120)

    # errors
    ir_pphi = res_pphi['ir']
    ir_pv   = res_pv['ir']
    t_vec   = np.arange(len(ir_pphi)) * DT

    eps_ana_pv   = np.max(np.abs(ir_ana - ir_pv))   / (np.max(np.abs(ir_ana)) + 1e-30)
    eps_ana_pphi = np.max(np.abs(ir_ana - ir_pphi))  / (np.max(np.abs(ir_ana)) + 1e-30)
    eps_form     = np.max(np.abs(ir_pv  - ir_pphi))  / (np.max(np.abs(ir_pv))  + 1e-30)

    print(f"  ε(ana−p-v)   = {eps_ana_pv:.6f}")
    print(f"  ε(ana−p-Φ)   = {eps_ana_pphi:.6f}")
    print(f"  ε(p-v−p-Φ)   = {eps_form:.6e}")

    # plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_vec*1e3, ir_ana,  'b-',  lw=1.2, label='Analytical')
    ax.plot(t_vec*1e3, ir_pv,   'r--', lw=1.0, label='p-v')
    ax.plot(t_vec*1e3, ir_pphi, 'g:',  lw=1.0, label='p-Φ')
    ax.set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
           title='Test 1: PR boundaries — FOM verification')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    savefig('test1_pr_fom.png')

    return dict(eps_ana_pv=eps_ana_pv, eps_ana_pphi=eps_ana_pphi,
                eps_form=eps_form, speedup=t_pv/t_pphi)


# ===================================================================
#  TEST 2 — FI boundaries: FOM comparison
# ===================================================================

def test2_fi_fom():
    print("\n" + "="*70)
    print("TEST 2: FI boundaries — FOM comparison (T=50 ms)")
    print("="*70)
    mesh = make_mesh()
    ops  = assemble_2d_operators(mesh)
    rec  = mesh.nearest_node(REC_X, REC_Y)
    T    = 0.05

    results = {}
    for Z in [600, 16000]:
        print(f"\n  Z = {Z} N·s/m⁴:")
        bc = {'Z': Z}

        t0 = time.perf_counter()
        res_pphi = fom_pphi(mesh, ops, 'FI', bc, SRC_X, SRC_Y, SIGMA,
                            DT, T, rec_idx=rec)
        t_pphi = time.perf_counter() - t0

        t0 = time.perf_counter()
        res_pv = fom_pv(mesh, ops, 'FI', bc, SRC_X, SRC_Y, SIGMA,
                        DT, T, rec_idx=rec)
        t_pv = time.perf_counter() - t0

        eps = np.max(np.abs(res_pv['ir'] - res_pphi['ir'])) / \
              (np.max(np.abs(res_pv['ir'])) + 1e-30)
        print(f"    p-Φ: {t_pphi:.2f}s, p-v: {t_pv:.2f}s, ε(form)={eps:.2e}")
        results[Z] = dict(ir_pphi=res_pphi['ir'], ir_pv=res_pv['ir'], eps=eps)

    # plot
    t_vec = np.arange(len(results[600]['ir_pv'])) * DT
    fig, ax = plt.subplots(figsize=(8, 3))
    for Z, ls in [(16000, '-'), (600, '--')]:
        ax.plot(t_vec*1e3, results[Z]['ir_pv'],   f'r{ls}', lw=1, label=f'p-v (Z={Z})')
        ax.plot(t_vec*1e3, results[Z]['ir_pphi'],  f'b{ls}', lw=1, label=f'p-Φ (Z={Z})')
    ax.set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
           title='Test 2: FI boundaries — FOM comparison')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    savefig('test2_fi_fom.png')
    return results


# ===================================================================
#  TEST 3 — LR boundaries: FOM comparison
# ===================================================================

def test3_lr_fom():
    print("\n" + "="*70)
    print("TEST 3: LR boundaries — FOM comparison (T=50 ms)")
    print("="*70)
    mesh = make_mesh()
    ops  = assemble_2d_operators(mesh)
    rec  = mesh.nearest_node(REC_X, REC_Y)
    T    = 0.05

    results = {}
    for d_mat in [0.2, 0.05]:
        print(f"\n  d_mat = {d_mat} m:")
        bc = {'sigma_mat': 10000, 'd_mat': d_mat}

        t0 = time.perf_counter()
        res_pphi = fom_pphi(mesh, ops, 'LR', bc, SRC_X, SRC_Y, SIGMA,
                            DT, T, rec_idx=rec)
        t_pphi = time.perf_counter() - t0

        t0 = time.perf_counter()
        res_pv = fom_pv(mesh, ops, 'LR', bc, SRC_X, SRC_Y, SIGMA,
                        DT, T, rec_idx=rec)
        t_pv = time.perf_counter() - t0

        eps = np.max(np.abs(res_pv['ir'] - res_pphi['ir'])) / \
              (np.max(np.abs(res_pv['ir'])) + 1e-30)
        print(f"    p-Φ: {t_pphi:.2f}s, p-v: {t_pv:.2f}s, ε(form)={eps:.2e}")
        results[d_mat] = dict(ir_pphi=res_pphi['ir'], ir_pv=res_pv['ir'], eps=eps)

    # plot
    t_vec = np.arange(len(results[0.2]['ir_pv'])) * DT
    fig, ax = plt.subplots(figsize=(8, 3))
    for d, ls in [(0.2, '-'), (0.05, '--')]:
        ax.plot(t_vec*1e3, results[d]['ir_pv'],  f'r{ls}', lw=1, label=f'p-v (d={d})')
        ax.plot(t_vec*1e3, results[d]['ir_pphi'], f'b{ls}', lw=1, label=f'p-Φ (d={d})')
    ax.set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
           title='Test 3: LR boundaries — FOM comparison')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    savefig('test3_lr_fom.png')
    return results


# ===================================================================
#  TEST 4 — Energy conservation
# ===================================================================

def test4_energy():
    print("\n" + "="*70)
    print("TEST 4: Energy conservation (T=50 ms)")
    print("="*70)
    mesh = make_mesh()
    ops  = assemble_2d_operators(mesh)
    T    = 0.05

    cases = [
        ('PR', {}),
        ('FI', {'Z': 16000}),
        ('LR', {'sigma_mat': 10000, 'd_mat': 0.05}),
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    t_vec = np.arange(int(round(T/DT)) + 1) * DT

    for bc_type, bc_params in cases:
        res_pphi = fom_pphi(mesh, ops, bc_type, bc_params,
                            SRC_X, SRC_Y, SIGMA, DT, T)
        res_pv = fom_pv(mesh, ops, bc_type, bc_params,
                        SRC_X, SRC_Y, SIGMA, DT, T)
        E0_pphi = res_pphi['energy'][0]
        E0_pv   = res_pv['energy'][0]
        label_extra = ''
        if bc_type == 'FI':
            label_extra = f" (Z={bc_params['Z']})"
        elif bc_type == 'LR':
            label_extra = f" (d={bc_params['d_mat']})"

        ax.plot(t_vec*1e3, res_pphi['energy']/E0_pphi, '-',  lw=1.2,
                label=f'{bc_type}{label_extra} p-Φ')
        ax.plot(t_vec*1e3, res_pv['energy']/E0_pv,     '--', lw=1.0,
                label=f'{bc_type}{label_extra} p-v')

    ax.set(xlabel='Time [ms]', ylabel='E / E₀',
           title='Test 4: Energy conservation')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    savefig('test4_energy.png')
    print("  done")


# ===================================================================
#  TEST 5 — ROM short impulse (T=50 ms)
# ===================================================================

def test5_rom_short():
    print("\n" + "="*70)
    print("TEST 5: ROM short impulse (T=50 ms)")
    print("="*70)
    mesh = make_mesh()
    ops  = assemble_2d_operators(mesh)
    rec  = mesh.nearest_node(REC_X, REC_Y)
    T    = 0.05

    cases = [
        ('PR', {},                'test5_rom_pr.png'),
        ('FI', {'Z': 16000},     'test5_rom_fi.png'),
    ]

    for bc_type, bc_params, fname in cases:
        print(f"\n  {bc_type} (short):")

        # FOM with snapshots (stride=5 for short runs)
        t0 = time.perf_counter()
        res_fom = fom_pphi(mesh, ops, bc_type, bc_params,
                           SRC_X, SRC_Y, SIGMA, DT, T,
                           rec_idx=rec, store_snapshots=True, snap_stride=5)
        t_fom = time.perf_counter() - t0

        # build PSD basis
        Psi_H, sigma_vals, Nrb = build_psd_basis(
            res_fom['snaps_p'], res_fom['snaps_Phi'], eps_pod=1e-6)
        del res_fom['snaps_p'], res_fom['snaps_Phi']; gc.collect()
        Nrb_use = min(Nrb, 122)
        print(f"    PSD basis: Nrb_auto={Nrb}, using Nrb={Nrb_use}")
        print(f"    FOM time: {t_fom:.2f}s")

        # ROM
        t0 = time.perf_counter()
        res_rom = rom_pphi(mesh, ops, Psi_H, bc_type, bc_params,
                           SRC_X, SRC_Y, SIGMA, DT, T,
                           rec_idx=rec, Nrb_override=Nrb_use)
        t_rom = time.perf_counter() - t0
        print(f"    ROM time: {t_rom:.2f}s  (speedup={t_fom/t_rom:.1f}×)")

        eps = np.max(np.abs(res_fom['ir'] - res_rom['ir']))
        print(f"    ε(FOM-ROM) = {eps:.2e}")

        # plot
        t_vec = np.arange(len(res_fom['ir'])) * DT
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax1.plot(t_vec*1e3, res_fom['ir'], 'b-', lw=1, label='FOM')
        ax1.plot(t_vec*1e3, res_rom['ir'], 'r--', lw=1, label=f'ROM (Nrb={Nrb_use})')
        ax1.set(ylabel='Pressure [Pa]',
                title=f'Test 5: {bc_type} ROM — short impulse')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2.plot(t_vec*1e3, res_fom['ir'] - res_rom['ir'], 'k-', lw=0.8)
        ax2.set(xlabel='Time [ms]', ylabel='Error [Pa]')
        ax2.grid(True, alpha=0.3)
        savefig(fname)


# ===================================================================
#  TEST 6 — ROM long impulse (T=200 ms) — stability
# ===================================================================

def test6_rom_long():
    print("\n" + "="*70)
    print("TEST 6: ROM long impulse — stability (T=200 ms)")
    print("="*70)
    mesh = make_mesh()
    ops  = assemble_2d_operators(mesh)
    rec  = mesh.nearest_node(REC_X, REC_Y)
    T    = 0.20

    for bc_type, bc_params, Nrb_use in [
        ('PR', {},            122),
        ('FI', {'Z': 16000},  122),
    ]:
        print(f"\n  {bc_type} (long, Nrb={Nrb_use}):")

        # FOM snapshots — stride=10 to cap memory for long runs
        snap_stride = 10
        t0 = time.perf_counter()
        res_fom = fom_pphi(mesh, ops, bc_type, bc_params,
                           SRC_X, SRC_Y, SIGMA, DT, T,
                           rec_idx=rec, store_snapshots=True,
                           store_boundary_pressure=(bc_type != 'PR'),
                           snap_stride=snap_stride)
        t_fom = time.perf_counter() - t0
        n_snaps = len(res_fom['snaps_p'])
        print(f"    FOM: {t_fom:.2f}s, {n_snaps} snapshots (stride={snap_stride})")

        # standard PSD basis
        Psi_std, _, _ = build_psd_basis(
            res_fom['snaps_p'], res_fom['snaps_Phi'], eps_pod=1e-8)

        # modified PSD (with boundary energy) for damped cases
        Psi_mod = None
        if bc_type != 'PR' and 'snaps_pb' in res_fom:
            Psi_mod, _, _ = build_modified_psd_basis(
                res_fom['snaps_p'], res_fom['snaps_Phi'],
                res_fom['snaps_pb'], eps_pod=1e-8)
        # free snapshot memory
        for k in ('snaps_p', 'snaps_Phi', 'snaps_pb'):
            res_fom.pop(k, None)
        gc.collect()

        # ROM — standard
        t0 = time.perf_counter()
        res_std = rom_pphi(mesh, ops, Psi_std, bc_type, bc_params,
                           SRC_X, SRC_Y, SIGMA, DT, T,
                           rec_idx=rec, Nrb_override=Nrb_use)
        t_std = time.perf_counter() - t0

        eps_std = np.max(np.abs(res_fom['ir'] - res_std['ir']))
        print(f"    ROM std:  t={t_std:.2f}s, ε={eps_std:.2e}")

        # ROM — modified (if available)
        res_mod_ir = None
        if Psi_mod is not None:
            Nrb_mod = min(Nrb_use, Psi_mod.shape[1])
            res_mod = rom_pphi(mesh, ops, Psi_mod, bc_type, bc_params,
                               SRC_X, SRC_Y, SIGMA, DT, T,
                               rec_idx=rec, Nrb_override=Nrb_mod)
            eps_mod = np.max(np.abs(res_fom['ir'] - res_mod['ir']))
            print(f"    ROM mod:  ε={eps_mod:.2e}")
            res_mod_ir = res_mod['ir']

        # plot
        t_vec = np.arange(len(res_fom['ir'])) * DT
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax1.plot(t_vec*1e3, res_fom['ir'], 'b-', lw=1, label='FOM')
        ax1.plot(t_vec*1e3, res_std['ir'], 'r--', lw=1, label='ROM (std p-Φ)')
        if res_mod_ir is not None:
            ax1.plot(t_vec*1e3, res_mod_ir, 'g:', lw=1, label='ROM (mod p-Φ)')
        ax1.set(ylabel='Pressure [Pa]',
                title=f'Test 6: {bc_type} ROM — long impulse (T=200ms)')
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

        ax2.plot(t_vec*1e3, res_fom['ir'] - res_std['ir'], 'r-', lw=0.7,
                 label='error (std)')
        if res_mod_ir is not None:
            ax2.plot(t_vec*1e3, res_fom['ir'] - res_mod_ir, 'g-', lw=0.7,
                     label='error (mod)')
        ax2.set(xlabel='Time [ms]', ylabel='Error [Pa]')
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
        savefig(f'test6_rom_long_{bc_type.lower()}.png')


# ===================================================================
#  TEST 7 — Eigenvalue analysis
# ===================================================================

def test7_eigenvalues():
    print("\n" + "="*70)
    print("TEST 7: Eigenvalue analysis (stability)")
    print("="*70)

    # use a smaller mesh for eigenvalue computation
    mesh_small = RectMesh2D(LX, LY, 6, 3, P_ORDER)
    ops_small  = assemble_2d_operators(mesh_small)
    N_small = mesh_small.N_dof
    print(f"  Small mesh for eigenvalues: N={N_small}")

    # FOM eigenvalues
    eig_fom = eigenvalue_analysis(ops_small)
    print(f"  FOM eigenvalues: max Re = {np.max(np.real(eig_fom)):.6e}")

    # ROM eigenvalues for different Nrb
    T_snap = 0.05
    res = fom_pphi(mesh_small, ops_small, 'PR', {},
                   SRC_X, SRC_Y, SIGMA, DT, T_snap,
                   store_snapshots=True)
    Psi_H, _, _ = build_psd_basis(res['snaps_p'], res['snaps_Phi'], eps_pod=1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if eig_fom is not None:
        axes[0].scatter(np.real(eig_fom), np.imag(eig_fom),
                        s=3, c='blue', alpha=0.5, label='FOM')

    for Nrb, color in [(50, 'red'), (100, 'green')]:
        if Nrb <= Psi_H.shape[1]:
            eig_rom = eigenvalue_analysis(ops_small, Psi_H[:, :Nrb])
            axes[0].scatter(np.real(eig_rom), np.imag(eig_rom),
                            s=10, c=color, marker='x', label=f'ROM {Nrb}')
            print(f"  ROM Nrb={Nrb}: max Re = {np.max(np.real(eig_rom)):.6e}")

    axes[0].axvline(0, color='k', lw=0.5)
    axes[0].set(xlabel='Re(λ)', ylabel='Im(λ)', title='PR: Eigenvalues (p-Φ)')
    axes[0].legend(fontsize=7)

    # FI case
    res_fi = fom_pphi(mesh_small, ops_small, 'FI', {'Z': 16000},
                      SRC_X, SRC_Y, SIGMA, DT, T_snap,
                      store_snapshots=True,
                      store_boundary_pressure=True)
    Psi_fi, _, _ = build_psd_basis(res_fi['snaps_p'], res_fi['snaps_Phi'],
                                   eps_pod=1e-10)

    eig_fom_fi = eigenvalue_analysis(ops_small)  # same FOM ops (BC is in RHS)
    if eig_fom_fi is not None:
        axes[1].scatter(np.real(eig_fom_fi), np.imag(eig_fom_fi),
                        s=3, c='blue', alpha=0.5, label='FOM')
    for Nrb, color in [(50, 'red'), (100, 'green')]:
        if Nrb <= Psi_fi.shape[1]:
            eig_r = eigenvalue_analysis(ops_small, Psi_fi[:, :Nrb])
            axes[1].scatter(np.real(eig_r), np.imag(eig_r),
                            s=10, c=color, marker='x', label=f'ROM {Nrb}')

    axes[1].axvline(0, color='k', lw=0.5)
    axes[1].set(xlabel='Re(λ)', ylabel='Im(λ)', title='FI: Eigenvalues (p-Φ)')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    savefig('test7_eigenvalues.png')


# ===================================================================
#  TEST 8 — Speedup analysis
# ===================================================================

def test8_speedup():
    print("\n" + "="*70)
    print("TEST 8: Speedup analysis")
    print("="*70)
    mesh = make_mesh()
    ops  = assemble_2d_operators(mesh)
    rec  = mesh.nearest_node(REC_X, REC_Y)
    T    = 0.05

    # build ROM basis
    res_fom = fom_pphi(mesh, ops, 'PR', {}, SRC_X, SRC_Y, SIGMA,
                       DT, T, rec_idx=rec, store_snapshots=True)
    Psi_H, sigma_vals, _ = build_psd_basis(
        res_fom['snaps_p'], res_fom['snaps_Phi'], eps_pod=1e-10)

    # time FOM
    t0 = time.perf_counter()
    _ = fom_pphi(mesh, ops, 'PR', {}, SRC_X, SRC_Y, SIGMA, DT, T, rec_idx=rec)
    t_fom = time.perf_counter() - t0

    nrb_list = [10, 20, 40, 80, 120]
    nrb_list = [n for n in nrb_list if n <= Psi_H.shape[1]]
    speedups = []
    errors   = []

    for Nrb in nrb_list:
        t0 = time.perf_counter()
        res_r = rom_pphi(mesh, ops, Psi_H, 'PR', {},
                         SRC_X, SRC_Y, SIGMA, DT, T,
                         rec_idx=rec, Nrb_override=Nrb)
        t_rom = time.perf_counter() - t0
        eps = np.max(np.abs(res_fom['ir'] - res_r['ir']))
        sp = t_fom / t_rom
        speedups.append(sp)
        errors.append(eps)
        print(f"  Nrb={Nrb:4d}: speedup={sp:6.1f}×, error={eps:.2e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(nrb_list, speedups, 'bo-')
    ax1.set(xlabel='N_rb', ylabel='Speedup', title='Speedup vs ROM size')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(nrb_list, errors, 'rs-')
    ax2.set(xlabel='N_rb', ylabel='Error (L∞)', title='Error vs ROM size')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('test8_speedup.png')


# ===================================================================
#  TEST 9 — Large domain speedup (2 kHz, high N)
# ===================================================================

def test9_large_speedup():
    print("\n" + "="*70)
    print("TEST 9: Large domain ROM speedup (2 kHz)")
    print("="*70)

    # Paper 2 setup: 2m x 2m, f_max=2kHz, PPW=6
    Lx9, Ly9 = 2.0, 2.0
    f_max9 = 2000.0
    ppw9 = 6
    P9 = 4
    lam9 = C_AIR / f_max9
    h9 = lam9 / ppw9 * P9
    nex9 = max(2, int(np.ceil(Lx9 / h9)))
    ney9 = max(2, int(np.ceil(Ly9 / h9)))

    mesh9 = RectMesh2D(Lx9, Ly9, nex9, ney9, P9)
    ops9 = assemble_2d_operators(mesh9)
    print(f"  Mesh: {nex9}x{ney9}, P={P9}, N_dof={mesh9.N_dof}")

    src9 = (1.0, 1.0)
    rec9_idx = mesh9.nearest_node(0.2, 0.2)
    sigma9 = 0.2
    dt9 = 5.9e-6       # CFL for this resolution
    T9 = 0.05

    # FOM with snapshots — stride=10 to cap memory
    print("  Running FOM (p-Phi)...")
    t0 = time.perf_counter()
    res_fom9 = fom_pphi(mesh9, ops9, 'FI', {'Z': 2000}, *src9, sigma9,
                        dt9, T9, rec_idx=rec9_idx, store_snapshots=True,
                        snap_stride=10)
    t_fom9 = time.perf_counter() - t0
    print(f"  FOM time: {t_fom9:.2f}s, {len(res_fom9['snaps_p'])} snapshots")

    # Build PSD basis
    Psi9, sig9, Nrb9 = build_psd_basis(
        res_fom9['snaps_p'], res_fom9['snaps_Phi'], eps_pod=1e-6)
    # free snapshot memory
    del res_fom9['snaps_p'], res_fom9['snaps_Phi']
    gc.collect()
    print(f"  PSD basis: Nrb_auto={Nrb9}")

    # Sweep ROM sizes
    nrb_list9 = [10, 20, 40, 80, 150, 300]
    nrb_list9 = [n for n in nrb_list9 if n <= Psi9.shape[1]]
    speedups9, errors9 = [], []

    for Nrb in nrb_list9:
        t0 = time.perf_counter()
        res_r = rom_pphi(mesh9, ops9, Psi9, 'FI', {'Z': 2000},
                         *src9, sigma9, dt9, T9,
                         rec_idx=rec9_idx, Nrb_override=Nrb)
        t_rom = time.perf_counter() - t0
        eps = np.max(np.abs(res_fom9['ir'] - res_r['ir']))
        sp = t_fom9 / t_rom
        speedups9.append(sp)
        errors9.append(eps)
        print(f"  Nrb={Nrb:4d}: speedup={sp:6.1f}x, error={eps:.2e}, "
              f"ROM_time={t_rom:.2f}s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.loglog(errors9, speedups9, 'bo-')
    ax1.set(xlabel='Error (L-inf)', ylabel='Speedup',
            title=f'Large domain speedup (N={mesh9.N_dof})')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(nrb_list9, errors9, 'rs-')
    ax2.set(xlabel='N_rb', ylabel='Error', title='Error convergence')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('test9_large_speedup.png')
    return dict(N_dof=mesh9.N_dof, speedups=speedups9, errors=errors9)


# ===================================================================
#  MAIN — run all tests
# ===================================================================

def main():
    print("=" * 62)
    print("  Room Acoustics 2D Validation Suite")
    print("  Based on Bonthu (2026) & Sampedro Llopis (2022)")
    print("=" * 62)
    print(f"  Output: {OUT_DIR}")

    results = {}
    t_total = time.perf_counter()

    results['test1'] = test1_pr_fom();   gc.collect()
    results['test2'] = test2_fi_fom();   gc.collect()
    results['test3'] = test3_lr_fom();   gc.collect()
    test4_energy();                       gc.collect()
    test5_rom_short();                    gc.collect()
    test6_rom_long();                     gc.collect()
    test7_eigenvalues();                  gc.collect()
    test8_speedup();                      gc.collect()
    results['test9'] = test9_large_speedup(); gc.collect()

    elapsed = time.perf_counter() - t_total

    # summary report
    print("\n" + "="*70)
    print("  VALIDATION SUMMARY")
    print("="*70)
    r1 = results['test1']
    print(f"  Test 1 (PR FOM):  ε(ana-pΦ)={r1['eps_ana_pphi']:.4e}  "
          f"ε(pv-pΦ)={r1['eps_form']:.4e}  speedup={r1['speedup']:.2f}×")
    for Z in [600, 16000]:
        if Z in results.get('test2', {}):
            print(f"  Test 2 (FI Z={Z:5d}):  ε(pv-pΦ)="
                  f"{results['test2'][Z]['eps']:.4e}")
    for d in [0.05, 0.2]:
        if d in results.get('test3', {}):
            print(f"  Test 3 (LR d={d}):  ε(pv-pΦ)="
                  f"{results['test3'][d]['eps']:.4e}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Plots saved to: {OUT_DIR}")

    # write text report
    report_path = os.path.join(OUT_DIR, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Room Acoustics 2D Validation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mesh: {NEX}x{NEY} elements, P={P_ORDER}\n")
        f.write(f"Domain: {LX}x{LY} m\n")
        f.write(f"dt={DT}, CFL={CFL}\n\n")
        f.write(f"Test 1 (PR): eps_ana_pphi={r1['eps_ana_pphi']:.6e}\n")
        f.write(f"             eps_form={r1['eps_form']:.6e}\n")
        f.write(f"             speedup(pv/pphi)={r1['speedup']:.2f}\n")
        for Z in [600, 16000]:
            if Z in results.get('test2', {}):
                f.write(f"Test 2 (FI Z={Z}): eps={results['test2'][Z]['eps']:.6e}\n")
        for d in [0.05, 0.2]:
            if d in results.get('test3', {}):
                f.write(f"Test 3 (LR d={d}): eps={results['test3'][d]['eps']:.6e}\n")
        f.write(f"\nTotal time: {elapsed:.1f}s\n")
    print(f"  Report: {report_path}")


if __name__ == '__main__':
    main()
