#!/usr/bin/env python
"""
Validation suite for unstructured SEM on arbitrary 2D geometries.

Tests:
  1. Rectangular room: unstructured vs structured operators (machine-precision match)
  2. L-shaped room: FOM energy conservation (PR) and decay (FI)
  3. L-shaped room: ROM accuracy and speedup
  4. T-shaped room: FOM + ROM demonstration
"""

import os, sys, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from room_acoustics.geometry import (
    rectangular_room, l_shaped_room, t_shaped_room, room_with_column,
    generate_quad_mesh,
)
from room_acoustics.unstructured_sem import (
    UnstructuredQuadMesh2D, assemble_unstructured_2d_operators,
)
from room_acoustics.sem import RectMesh2D, assemble_2d_operators
from room_acoustics.solvers import (
    fom_pphi, build_psd_basis, build_modified_psd_basis,
    rom_pphi, C_AIR, RHO_AIR,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')

def savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved {path}")

P = 4

# ===================================================================
# TEST 1: Cross-validation on rectangular domain
# ===================================================================

def test1_cross_validation():
    print("\n" + "=" * 70)
    print("TEST 1: Rectangular domain cross-validation")
    print("=" * 70)

    Lx, Ly = 2.0, 1.0
    Nex, Ney = 4, 2

    # Structured
    mesh_s = RectMesh2D(Lx, Ly, Nex, Ney, P)
    ops_s = assemble_2d_operators(mesh_s)

    # Unstructured
    geom = rectangular_room(Lx, Ly)
    raw = generate_quad_mesh(geom, Lx / Nex, P)
    mesh_u = UnstructuredQuadMesh2D(raw['nodes'], raw['quads'],
                                     raw['boundary'], P)
    ops_u = assemble_unstructured_2d_operators(mesh_u)

    print(f"  Structured N={mesh_s.N_dof}, Unstructured N={mesh_u.N_dof}")

    # Run FOM on both
    rec_s = mesh_s.nearest_node(0.5, 0.3)
    rec_u = mesh_u.nearest_node(0.5, 0.3)
    dt = 1e-5; T = 0.02

    res_s = fom_pphi(mesh_s, ops_s, 'PR', {}, 1.0, 0.5, 0.2, dt, T, rec_idx=rec_s)
    res_u = fom_pphi(mesh_u, ops_u, 'PR', {}, 1.0, 0.5, 0.2, dt, T, rec_idx=rec_u)

    err = np.max(np.abs(res_s['ir'] - res_u['ir']))
    peak = np.max(np.abs(res_s['ir']))
    print(f"  IR max error: {err:.2e} (peak={peak:.2e}, relative={err/peak:.2e})")

    # Plot
    t_vec = np.arange(len(res_s['ir'])) * dt * 1e3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(t_vec, res_s['ir'], 'b-', lw=1, label='Structured (Kronecker)')
    ax1.plot(t_vec, res_u['ir'], 'r--', lw=1, label='Unstructured (Gmsh)')
    ax1.set(ylabel='Pressure [Pa]', title='Test 1: Cross-validation (rectangular room)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(t_vec, res_s['ir'] - res_u['ir'], 'k-', lw=0.8)
    ax2.set(xlabel='Time [ms]', ylabel='Error [Pa]')
    ax2.grid(True, alpha=0.3)
    savefig('unstruct_test1_cross.png')

    return dict(error=err, relative=err/peak)


# ===================================================================
# TEST 2: L-shaped room FOM (energy conservation + FI decay)
# ===================================================================

def test2_lshape_fom():
    print("\n" + "=" * 70)
    print("TEST 2: L-shaped room FOM (energy)")
    print("=" * 70)

    geom = l_shaped_room(6.0, 4.0, 3.0, 2.0)
    h = 0.35
    raw = generate_quad_mesh(geom, h, P)
    mesh = UnstructuredQuadMesh2D(raw['nodes'], raw['quads'],
                                   raw['boundary'], P)
    ops = assemble_unstructured_2d_operators(mesh)
    print(f"  L-shape: {raw['quads'].shape[0]} quads, N_dof={mesh.N_dof}")
    print(f"  Area: {ops['M_diag'].sum():.2f} (expected 18.0)")

    src_x, src_y = 1.5, 1.5
    rec_idx = mesh.nearest_node(4.0, 0.5)
    dt = 0.15 * h / (C_AIR * P**2)
    dt = round(dt, 8)
    T = 0.05

    # PR
    print(f"  PR FOM: dt={dt:.2e}, Nt={int(T/dt)}")
    res_pr = fom_pphi(mesh, ops, 'PR', {}, src_x, src_y, 0.3, dt, T,
                      rec_idx=rec_idx, store_snapshots=True, snap_stride=5)

    M = ops['M_diag']; S = ops['S']
    def energy(p, phi):
        return (0.5/(RHO_AIR*C_AIR**2) * np.dot(p, M*p)
                + 0.5*RHO_AIR * np.dot(phi, S.dot(phi)))

    E_pr = [energy(res_pr['snaps_p'][k], res_pr['snaps_Phi'][k])
            for k in range(len(res_pr['snaps_p']))]
    E_pr = np.array(E_pr)
    drift = abs(E_pr[-1] - E_pr[0]) / E_pr[0]
    print(f"  PR energy drift: {drift:.2e}")

    # FI
    res_fi = fom_pphi(mesh, ops, 'FI', {'Z': 2000}, src_x, src_y, 0.3,
                      dt, T, rec_idx=rec_idx, store_snapshots=True,
                      snap_stride=5)
    E_fi = [energy(res_fi['snaps_p'][k], res_fi['snaps_Phi'][k])
            for k in range(len(res_fi['snaps_p']))]
    E_fi = np.array(E_fi)
    print(f"  FI energy decay: E_end/E_start = {E_fi[-1]/E_fi[0]:.4f}")
    print(f"  FI monotonic: {all(np.diff(E_fi) <= 1e-15)}")

    # Plot
    t_snap = np.arange(len(E_pr)) * dt * 5 * 1e3
    t_vec = np.arange(len(res_pr['ir'])) * dt * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(t_vec, res_pr['ir'], 'b-', lw=0.8)
    axes[0, 0].set(ylabel='Pressure [Pa]',
                   title='L-shape PR: impulse response')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_snap, E_pr / E_pr[0], 'b-', lw=1, label='PR')
    axes[0, 1].plot(t_snap, E_fi / E_fi[0], 'r-', lw=1, label='FI (Z=2000)')
    axes[0, 1].set(ylabel='E(t)/E(0)', title='Energy evolution')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_vec, res_fi['ir'], 'r-', lw=0.8)
    axes[1, 0].set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
                   title='L-shape FI: impulse response')
    axes[1, 0].grid(True, alpha=0.3)

    # Mesh visualization
    ax = axes[1, 1]
    nodes = mesh.corner_nodes
    for quad in raw['quads']:
        poly = plt.Polygon(nodes[quad], fill=False, edgecolor='gray',
                           lw=0.3)
        ax.add_patch(poly)
    bnd = mesh.all_boundary_nodes()
    ax.plot(mesh.x[bnd], mesh.y[bnd], 'k.', ms=1)
    ax.plot(src_x, src_y, 'r*', ms=10, label='Source')
    ax.plot(mesh.x[rec_idx], mesh.y[rec_idx], 'bs', ms=6, label='Receiver')
    ax.set(xlabel='x [m]', ylabel='y [m]', title='L-shaped room mesh',
           aspect='equal')
    ax.legend(fontsize=7)

    plt.tight_layout()
    savefig('unstruct_test2_lshape_fom.png')

    return dict(mesh=mesh, ops=ops, res_pr=res_pr, res_fi=res_fi,
                src=(src_x, src_y), rec_idx=rec_idx, dt=dt, T=T,
                energy_drift_pr=drift)


# ===================================================================
# TEST 3: L-shaped room ROM accuracy + speedup
# ===================================================================

def test3_lshape_rom(test2_data):
    print("\n" + "=" * 70)
    print("TEST 3: L-shaped room ROM (accuracy + speedup)")
    print("=" * 70)

    mesh = test2_data['mesh']
    ops = test2_data['ops']
    res_fom = test2_data['res_pr']
    src_x, src_y = test2_data['src']
    rec_idx = test2_data['rec_idx']
    dt = test2_data['dt']
    T = test2_data['T']

    # Build basis
    Psi, sig, Nrb_auto = build_psd_basis(
        res_fom['snaps_p'], res_fom['snaps_Phi'], eps_pod=1e-10)
    print(f"  PSD basis: Nrb_auto={Nrb_auto}")

    # Time FOM (clean run)
    t0 = time.perf_counter()
    _ = fom_pphi(mesh, ops, 'PR', {}, src_x, src_y, 0.3, dt, T,
                 rec_idx=rec_idx)
    t_fom = time.perf_counter() - t0
    print(f"  FOM time: {t_fom:.2f}s")

    nrb_list = [5, 10, 20, 40, 60, 80, 100]
    nrb_list = [n for n in nrb_list if n <= Psi.shape[1]]

    speedups, errors = [], []
    for nrb in nrb_list:
        t0 = time.perf_counter()
        res_rom = rom_pphi(mesh, ops, Psi, 'PR', {}, src_x, src_y, 0.3,
                           dt, T, rec_idx=rec_idx, Nrb_override=nrb)
        t_rom = time.perf_counter() - t0
        err = np.max(np.abs(res_fom['ir'] - res_rom['ir']))
        sp = t_fom / t_rom
        speedups.append(sp)
        errors.append(err)
        print(f"    Nrb={nrb:4d}: speedup={sp:6.1f}x, error={err:.2e}")

    # Best ROM for overlay plot
    best_nrb = nrb_list[min(len(nrb_list)-1, 4)]
    res_best = rom_pphi(mesh, ops, Psi, 'PR', {}, src_x, src_y, 0.3,
                        dt, T, rec_idx=rec_idx, Nrb_override=best_nrb)

    # Plot
    t_vec = np.arange(len(res_fom['ir'])) * dt * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(t_vec, res_fom['ir'], 'b-', lw=1, label='FOM')
    axes[0, 0].plot(t_vec, res_best['ir'], 'r--', lw=1,
                    label=f'ROM (Nrb={best_nrb})')
    axes[0, 0].set(ylabel='Pressure [Pa]',
                   title='L-shape: FOM vs ROM')
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
    savefig('unstruct_test3_lshape_rom.png')


# ===================================================================
# TEST 4: T-shaped room + room with column
# ===================================================================

def test4_complex_geometries():
    print("\n" + "=" * 70)
    print("TEST 4: Complex geometries (T-shape, column)")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    geom_cases = []

    # T-shaped room
    geom_t = t_shaped_room(6.0, 5.0, 2.0, 3.0)
    h_t = 0.4
    raw_t = generate_quad_mesh(geom_t, h_t, P)
    mesh_t = UnstructuredQuadMesh2D(raw_t['nodes'], raw_t['quads'],
                                     raw_t['boundary'], P)
    ops_t = assemble_unstructured_2d_operators(mesh_t)
    geom_cases.append(('T-shape', mesh_t, ops_t, raw_t, h_t,
                        1.0, 3.5, mesh_t.nearest_node(3.0, 4.5)))

    # Room with column
    geom_c = room_with_column(5.0, 4.0, (2.5, 2.0), 0.4, n_col_sides=8)
    h_c = 0.35
    raw_c = generate_quad_mesh(geom_c, h_c, P)
    mesh_c = UnstructuredQuadMesh2D(raw_c['nodes'], raw_c['quads'],
                                     raw_c['boundary'], P)
    ops_c = assemble_unstructured_2d_operators(mesh_c)
    geom_cases.append(('Column', mesh_c, ops_c, raw_c, h_c,
                        1.0, 1.0, mesh_c.nearest_node(4.0, 3.0)))

    for idx, (name, mesh, ops, raw, h, sx, sy, rec) in enumerate(geom_cases):
        dt = 0.15 * h / (C_AIR * P**2)
        dt = round(dt, 8)
        T = 0.05

        print(f"\n  {name}: {raw['quads'].shape[0]} quads, N_dof={mesh.N_dof}")
        print(f"  Area: {ops['M_diag'].sum():.2f}")

        # Mesh plot
        ax = axes[idx, 0]
        nodes = mesh.corner_nodes
        for quad in raw['quads']:
            poly = plt.Polygon(nodes[quad], fill=False, edgecolor='gray', lw=0.3)
            ax.add_patch(poly)
        bnd = mesh.all_boundary_nodes()
        ax.plot(mesh.x[bnd], mesh.y[bnd], 'k.', ms=1)
        ax.plot(sx, sy, 'r*', ms=10)
        ax.plot(mesh.x[rec], mesh.y[rec], 'bs', ms=6)
        ax.set(xlabel='x [m]', ylabel='y [m]', title=f'{name} mesh',
               aspect='equal')

        # FOM
        t0 = time.perf_counter()
        res = fom_pphi(mesh, ops, 'FI', {'Z': 3000}, sx, sy, 0.3, dt, T,
                       rec_idx=rec, store_snapshots=True,
                       store_boundary_pressure=True, snap_stride=5)
        t_fom = time.perf_counter() - t0
        print(f"  FOM time: {t_fom:.2f}s")

        t_vec = np.arange(len(res['ir'])) * dt * 1e3
        axes[idx, 1].plot(t_vec, res['ir'], 'b-', lw=0.8)
        axes[idx, 1].set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
                         title=f'{name} FI impulse response')
        axes[idx, 1].grid(True, alpha=0.3)

        # ROM
        Psi, _, Nrb = build_modified_psd_basis(
            res['snaps_p'], res['snaps_Phi'], res['snaps_pb'],
            eps_pod=1e-8)

        # Clean FOM timing
        t0 = time.perf_counter()
        _ = fom_pphi(mesh, ops, 'FI', {'Z': 3000}, sx, sy, 0.3, dt, T, rec_idx=rec)
        t_fom_clean = time.perf_counter() - t0

        nrb_list = [5, 10, 20, 40, 60]
        nrb_list = [n for n in nrb_list if n <= Psi.shape[1]]
        sp_list, err_list = [], []
        for nrb in nrb_list:
            t0 = time.perf_counter()
            rr = rom_pphi(mesh, ops, Psi, 'FI', {'Z': 3000}, sx, sy, 0.3,
                          dt, T, rec_idx=rec, Nrb_override=nrb)
            t_rom = time.perf_counter() - t0
            err = np.max(np.abs(res['ir'] - rr['ir']))
            sp = t_fom_clean / t_rom
            sp_list.append(sp); err_list.append(err)
            print(f"    Nrb={nrb:3d}: speedup={sp:6.1f}x, error={err:.2e}")

        axes[idx, 2].semilogy(nrb_list, err_list, 'rs-')
        ax2 = axes[idx, 2].twinx()
        ax2.plot(nrb_list, sp_list, 'bo-', alpha=0.7)
        axes[idx, 2].set(xlabel='Nrb', ylabel='Error (L-inf)',
                         title=f'{name} ROM convergence')
        ax2.set_ylabel('Speedup', color='blue')
        axes[idx, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('unstruct_test4_complex.png')


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  Unstructured SEM Validation Suite")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_start = time.perf_counter()

    test1_cross_validation();  gc.collect()
    d2 = test2_lshape_fom();   gc.collect()
    test3_lshape_rom(d2);      gc.collect()
    test4_complex_geometries(); gc.collect()

    print(f"\n  Total time: {time.perf_counter() - t_start:.1f}s")
    print(f"  Plots saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
