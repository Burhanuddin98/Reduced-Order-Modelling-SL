#!/usr/bin/env python
"""
Validation suite for 3D unstructured (extruded) SEM.

Tests:
  1. Box cross-validation: extruded vs Kronecker (machine-precision match)
  2. L-shaped room 3D: FOM energy + FI decay
  3. L-shaped room 3D: ROM accuracy and speedup
"""

import os, sys, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Block cupy to avoid NVRTC issues — GPU path tested separately
sys.modules.setdefault('cupy', None)
sys.modules.setdefault('cupyx', None)
sys.modules.setdefault('cupyx.scipy', None)
sys.modules.setdefault('cupyx.scipy.sparse', None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from room_acoustics.geometry import (
    rectangular_room, l_shaped_room, t_shaped_room,
    generate_quad_mesh, extrude_quad_mesh,
)
from room_acoustics.unstructured_sem import (
    UnstructuredHexMesh3D, assemble_unstructured_3d_operators,
)
from room_acoustics.sem import BoxMesh3D, assemble_3d_operators
from room_acoustics.solvers import (
    fom_pphi_3d_gpu, rom_pphi_3d, build_psd_basis,
    build_modified_psd_basis, C_AIR, RHO_AIR,
)
from room_acoustics.results_io import save_result

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')
P = 4


def savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved {path}")


def _energy(p, phi, ops):
    M = ops['M_diag']; S = ops['S']
    return (0.5 / (RHO_AIR * C_AIR**2) * np.dot(p, M * p)
            + 0.5 * RHO_AIR * np.dot(phi, S.dot(phi)))


# ==================================================================
# TEST 1: Box cross-validation
# ==================================================================

def test1_box_cross():
    print("\n" + "=" * 70)
    print("TEST 1: Box cross-validation (extruded vs Kronecker)")
    print("=" * 70)

    Lx, Ly, Lz = 1.0, 1.0, 1.0
    Ne = 2

    mesh_s = BoxMesh3D(Lx, Ly, Lz, Ne, Ne, Ne, P)
    ops_s = assemble_3d_operators(mesh_s)

    geom = rectangular_room(Lx, Ly)
    raw_2d = generate_quad_mesh(geom, Lx / Ne, P)
    raw_3d = extrude_quad_mesh(raw_2d, Lz, Ne)
    mesh_u = UnstructuredHexMesh3D(raw_3d, raw_2d['nodes'],
                                    raw_2d['quads'], P)
    ops_u = assemble_unstructured_3d_operators(mesh_u)

    print(f"  Structured N={mesh_s.N_dof}, Unstructured N={mesh_u.N_dof}")

    rec_s = mesh_s.nearest_node(0.3, 0.3, 0.3)
    rec_u = mesh_u.nearest_node(0.3, 0.3, 0.3)
    dt = 1e-5; T = 0.005

    res_s = fom_pphi_3d_gpu(mesh_s, ops_s, 'PR', {}, 0.5, 0.5, 0.5,
                             0.15, dt, T, rec_idx=rec_s)
    res_u = fom_pphi_3d_gpu(mesh_u, ops_u, 'PR', {}, 0.5, 0.5, 0.5,
                             0.15, dt, T, rec_idx=rec_u)

    err = np.max(np.abs(res_s['ir'] - res_u['ir']))
    peak = np.max(np.abs(res_s['ir']))
    print(f"  IR error: {err:.2e} (relative {err/peak:.2e})")

    t_vec = np.arange(len(res_s['ir'])) * dt * 1e3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t_vec, res_s['ir'], 'b-', lw=1, label='Kronecker')
    ax1.plot(t_vec, res_u['ir'], 'r--', lw=1, label='Extruded')
    ax1.set(ylabel='Pressure [Pa]',
            title='3D Box: Kronecker vs Extruded')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.plot(t_vec, res_s['ir'] - res_u['ir'], 'k-', lw=0.8)
    ax2.set(xlabel='Time [ms]', ylabel='Error [Pa]')
    ax2.grid(True, alpha=0.3)
    savefig('unstruct3d_test1_cross.png')

    save_result('unstruct3d_test1_cross', {
        'domain': {'Lx': Lx, 'Ly': Ly, 'Lz': Lz, 'type': 'box'},
        'P': P, 'N_elements_per_dir': Ne,
        'N_dof_structured': mesh_s.N_dof,
        'N_dof_unstructured': mesh_u.N_dof,
        'ir_max_error': err,
        'ir_relative_error': err / peak,
        'ir_peak': peak,
        'dt': dt, 'T': T,
    }, suite='3d_unstructured')

    return dict(error=err, relative=err / peak)


# ==================================================================
# TEST 2: L-shaped room 3D FOM
# ==================================================================

def test2_lshape_3d_fom():
    print("\n" + "=" * 70)
    print("TEST 2: L-shaped room 3D FOM")
    print("=" * 70)

    geom = l_shaped_room(6.0, 4.0, 3.0, 2.0)
    h = 0.5; Lz = 2.5; n_layers = 2

    raw_2d = generate_quad_mesh(geom, h, P)
    raw_3d = extrude_quad_mesh(raw_2d, Lz, n_layers)
    mesh = UnstructuredHexMesh3D(raw_3d, raw_2d['nodes'],
                                  raw_2d['quads'], P)
    ops = assemble_unstructured_3d_operators(mesh)

    vol = ops['M_diag'].sum()
    surf = ops['B_total'].diagonal().sum()
    print(f"  L-shape 3D: {mesh.N_el} hexes, N_dof={mesh.N_dof}")
    print(f"  Volume: {vol:.2f} (expected 45.0)")
    print(f"  Surface: {surf:.2f} (expected 86.0)")

    src = (1.5, 1.5, 1.25)
    rec_idx = mesh.nearest_node(4.0, 0.5, 1.0)
    dt = 0.15 * h / (C_AIR * P**2)
    dt = round(dt, 8)
    T = 0.04

    # PR
    print(f"  PR FOM: dt={dt:.2e}, Nt={int(T/dt)}")
    res_pr = fom_pphi_3d_gpu(mesh, ops, 'PR', {}, *src, 0.3, dt, T,
                              rec_idx=rec_idx, store_snapshots=True,
                              snap_stride=5)
    E_pr = [_energy(res_pr['snaps_p'][k], res_pr['snaps_Phi'][k], ops)
            for k in range(len(res_pr['snaps_p']))]
    E_pr = np.array(E_pr)
    drift = abs(E_pr[-1] - E_pr[0]) / E_pr[0]
    print(f"  PR energy drift: {drift:.2e}")

    # FI
    res_fi = fom_pphi_3d_gpu(mesh, ops, 'FI', {'Z': 2000}, *src, 0.3,
                              dt, T, rec_idx=rec_idx, store_snapshots=True,
                              snap_stride=5)
    E_fi = [_energy(res_fi['snaps_p'][k], res_fi['snaps_Phi'][k], ops)
            for k in range(len(res_fi['snaps_p']))]
    E_fi = np.array(E_fi)
    print(f"  FI energy decay: E_end/E_start = {E_fi[-1]/E_fi[0]:.4f}")

    # Plot
    t_vec = np.arange(len(res_pr['ir'])) * dt * 1e3
    t_snap = np.arange(len(E_pr)) * dt * 5 * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(t_vec, res_pr['ir'], 'b-', lw=0.8)
    axes[0, 0].set(ylabel='Pressure [Pa]',
                   title='L-shape 3D PR: impulse response')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_snap, E_pr / E_pr[0], 'b-', label='PR')
    axes[0, 1].plot(t_snap, E_fi / E_fi[0], 'r-', label='FI (Z=2000)')
    axes[0, 1].set(ylabel='E(t)/E(0)', title='Energy evolution')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_vec, res_fi['ir'], 'r-', lw=0.8)
    axes[1, 0].set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
                   title='L-shape 3D FI: impulse response')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(0.5, 0.5,
                    f"N_dof = {mesh.N_dof}\n"
                    f"{mesh.N_el} hex elements\n"
                    f"P = {P}\n"
                    f"Volume = {vol:.1f} m³\n"
                    f"Surface = {surf:.1f} m²\n"
                    f"PR drift = {drift:.1e}\n"
                    f"FI E_end/E_start = {E_fi[-1]/E_fi[0]:.4f}",
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace')
    axes[1, 1].axis('off')

    plt.tight_layout()
    savefig('unstruct3d_test2_lshape_fom.png')

    save_result('unstruct3d_test2_lshape_fom', {
        'domain': {'type': 'L-shape extruded', 'Lx': 6.0, 'Ly': 4.0,
                   'notch_x': 3.0, 'notch_y': 2.0, 'Lz': Lz,
                   'n_layers': n_layers},
        'mesh': {'N_dof': mesh.N_dof, 'N_el': mesh.N_el, 'P': P, 'h': h},
        'volume': {'computed': vol, 'expected': 45.0},
        'surface': {'computed': surf, 'expected': 86.0},
        'stiffness_asymmetry': float(abs(ops['S'] - ops['S'].T).max()),
        'pr_energy_drift': drift,
        'fi_energy_ratio': float(E_fi[-1] / E_fi[0]),
        'dt': dt, 'T': T,
        'source': src,
    }, suite='3d_unstructured')

    return dict(mesh=mesh, ops=ops, res_pr=res_pr, src=src,
                rec_idx=rec_idx, dt=dt, T=T)


# ==================================================================
# TEST 3: L-shaped room 3D ROM
# ==================================================================

def test3_lshape_3d_rom(data):
    print("\n" + "=" * 70)
    print("TEST 3: L-shaped room 3D ROM")
    print("=" * 70)

    mesh = data['mesh']; ops = data['ops']
    res_fom = data['res_pr']
    src = data['src']; rec_idx = data['rec_idx']
    dt = data['dt']; T = data['T']

    Psi, sig, Nrb_auto = build_psd_basis(
        res_fom['snaps_p'], res_fom['snaps_Phi'], eps_pod=1e-10)
    print(f"  PSD basis: Nrb_auto={Nrb_auto}")

    # Clean FOM timing
    t0 = time.perf_counter()
    _ = fom_pphi_3d_gpu(mesh, ops, 'PR', {}, *src, 0.3, dt, T,
                         rec_idx=rec_idx)
    t_fom = time.perf_counter() - t0
    print(f"  FOM time: {t_fom:.2f}s")

    nrb_list = [5, 10, 20, 40, 60, 80]
    nrb_list = [n for n in nrb_list if n <= Psi.shape[1]]

    speedups, errors = [], []
    for nrb in nrb_list:
        t0 = time.perf_counter()
        rr = rom_pphi_3d(mesh, ops, Psi, 'PR', {}, *src, 0.3, dt, T,
                          rec_idx=rec_idx, Nrb_override=nrb)
        t_rom = time.perf_counter() - t0
        err = np.max(np.abs(res_fom['ir'] - rr['ir']))
        sp = t_fom / t_rom
        speedups.append(sp); errors.append(err)
        print(f"    Nrb={nrb:3d}: speedup={sp:6.1f}x, error={err:.2e}")

    # Best ROM for overlay
    best_idx = min(len(nrb_list) - 1, 3)
    best_nrb = nrb_list[best_idx]
    res_best = rom_pphi_3d(mesh, ops, Psi, 'PR', {}, *src, 0.3, dt, T,
                            rec_idx=rec_idx, Nrb_override=best_nrb)

    t_vec = np.arange(len(res_fom['ir'])) * dt * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(t_vec, res_fom['ir'], 'b-', lw=1, label='FOM')
    axes[0, 0].plot(t_vec, res_best['ir'], 'r--', lw=1,
                    label=f'ROM (Nrb={best_nrb})')
    axes[0, 0].set(ylabel='Pressure [Pa]',
                   title='L-shape 3D: FOM vs ROM')
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_vec, res_fom['ir'] - res_best['ir'], 'k-', lw=0.8)
    axes[0, 1].set(ylabel='Error [Pa]',
                   title=f'ROM error (Nrb={best_nrb})')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(nrb_list, speedups, 'bo-')
    axes[1, 0].set(xlabel='Nrb', ylabel='Speedup',
                   title=f'Speedup (N={mesh.N_dof})')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(nrb_list, errors, 'rs-')
    axes[1, 1].set(xlabel='Nrb', ylabel='Error (L-inf)',
                   title='Error convergence')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('unstruct3d_test3_lshape_rom.png')

    save_result('unstruct3d_test3_lshape_rom', {
        'N_dof': mesh.N_dof,
        'N_el': mesh.N_el,
        'Nrb_auto': Nrb_auto,
        'fom_time_s': t_fom,
        'sweep': [{'Nrb': nrb, 'speedup': sp, 'error_Linf': err}
                  for nrb, sp, err in zip(nrb_list, speedups, errors)],
    }, suite='3d_unstructured')


# ==================================================================
# Main
# ==================================================================

def main():
    print("=" * 70)
    print("  3D Unstructured (Extruded) SEM Validation Suite")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_start = time.perf_counter()

    test1_box_cross(); gc.collect()
    d2 = test2_lshape_3d_fom(); gc.collect()
    test3_lshape_3d_rom(d2); gc.collect()

    print(f"\n  Total time: {time.perf_counter() - t_start:.1f}s")
    print(f"  Plots saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
