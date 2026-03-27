#!/usr/bin/env python
"""
High-quality visualizations of room acoustics simulations.

Produces:
  1. Pressure field snapshots showing wave propagation in L-shaped room
  2. Side-by-side FOM vs ROM field comparison
  3. T-shape and column room wave propagation
  4. 3D extruded L-shape slice visualization
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from room_acoustics.geometry import (
    l_shaped_room, t_shaped_room, room_with_column,
    generate_quad_mesh, extrude_quad_mesh,
)
from room_acoustics.unstructured_sem import (
    UnstructuredQuadMesh2D, assemble_unstructured_2d_operators,
    UnstructuredHexMesh3D, assemble_unstructured_3d_operators,
)
from room_acoustics.solvers import (
    fom_pphi, build_psd_basis, build_modified_psd_basis, rom_pphi,
    C_AIR, RHO_AIR,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

P = 4


def savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    saved {path}")


def _triangulate_mesh(mesh):
    """Build a Triangulation from quad mesh for plotting."""
    # Split each quad into 2 triangles for matplotlib
    triangles = []
    n1d = P + 1
    for e in range(mesh.N_el):
        dof = mesh.elem_dof[e]
        for j in range(P):
            for i in range(P):
                n0 = dof[i + j * n1d]
                n1 = dof[(i+1) + j * n1d]
                n2 = dof[(i+1) + (j+1) * n1d]
                n3 = dof[i + (j+1) * n1d]
                triangles.append([n0, n1, n2])
                triangles.append([n0, n2, n3])
    return Triangulation(mesh.x, mesh.y, triangles)


def _plot_field(ax, tri, field, mesh, title, vmax=None, cmap='RdBu_r'):
    """Plot a pressure field on the mesh."""
    if vmax is None:
        vmax = max(abs(field.max()), abs(field.min()), 1e-10)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    tc = ax.tripcolor(tri, field, cmap=cmap, norm=norm, shading='gouraud')
    # Draw boundary
    bnd = mesh.all_boundary_nodes()
    # Sort boundary nodes to draw outline
    ax.plot(mesh.x[bnd], mesh.y[bnd], 'k.', ms=0.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=8)
    ax.set_ylabel('y [m]', fontsize=8)
    ax.tick_params(labelsize=7)
    return tc


# ==================================================================
# VIS 1: L-shaped room wave propagation snapshots
# ==================================================================

def vis1_lshape_propagation():
    print("\n  VIS 1: L-shaped room wave propagation...")

    geom = l_shaped_room(6.0, 4.0, 3.0, 2.0)
    raw = generate_quad_mesh(geom, 0.25, P)
    mesh = UnstructuredQuadMesh2D(raw['nodes'], raw['quads'],
                                   raw['boundary'], P)
    ops = assemble_unstructured_2d_operators(mesh)
    print(f"    Mesh: {mesh.N_dof} DOFs")

    src_x, src_y = 1.5, 1.5
    sigma = 0.2
    dt = round(0.15 * 0.25 / (C_AIR * P**2), 8)
    T = 0.04

    # Run FOM collecting snapshots at specific times
    snap_times = [0.001, 0.004, 0.008, 0.012, 0.018, 0.025, 0.032, 0.040]
    snap_stride = max(1, int(snap_times[0] / dt / 2))

    # Use stride that captures the desired frames (every ~20 steps)
    snap_stride = max(1, int(0.001 / dt))
    res = fom_pphi(mesh, ops, 'FI', {'Z': 3000}, src_x, src_y, sigma,
                   dt, T, store_snapshots=True, snap_stride=snap_stride)

    tri = _triangulate_mesh(mesh)
    snaps = res['snaps_p']
    n_snaps = len(snaps)

    # Pick 8 time frames (indices into the strided snapshot list)
    frame_indices = []
    for t_want in snap_times:
        idx = int(round(t_want / (dt * snap_stride)))
        idx = min(idx, n_snaps - 1)
        frame_indices.append(idx)

    # Find global vmax for consistent colorbar
    vmax = max(np.max(np.abs(snaps[i])) for i in frame_indices) * 0.8

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle('L-Shaped Room — Wave Propagation (FI boundaries, Z=3000)',
                 fontsize=14, fontweight='bold', y=0.98)

    for k, (ax, fi) in enumerate(zip(axes.flat, frame_indices)):
        t_ms = fi * dt * snap_stride * 1e3
        tc = _plot_field(ax, tri, snaps[fi], mesh,
                         f't = {t_ms:.1f} ms', vmax=vmax)
        # Mark source
        ax.plot(src_x, src_y, 'k*', ms=8, zorder=5)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(tc, cax=cbar_ax)
    cb.set_label('Pressure [Pa]', fontsize=10)

    plt.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.06,
                        wspace=0.25, hspace=0.3)
    savefig('vis_lshape_propagation.png')

    return mesh, ops, tri, res


# ==================================================================
# VIS 2: FOM vs ROM field comparison
# ==================================================================

def vis2_fom_vs_rom(mesh, ops, tri, res_fom):
    print("\n  VIS 2: FOM vs ROM field comparison...")

    src_x, src_y = 1.5, 1.5
    sigma = 0.2
    dt = round(0.15 * 0.25 / (C_AIR * P**2), 8)
    T = 0.04

    # Build ROM basis
    Psi, _, Nrb = build_psd_basis(res_fom['snaps_p'], res_fom['snaps_Phi'],
                                   eps_pod=1e-8)
    Nrb_use = min(Nrb, 60)

    # Run ROM collecting snapshots
    # ROM only gives IR at receiver, not full field. We need to reconstruct.
    # Run FOM at a specific time, then project ROM coefficients to get field.

    # Get ROM coefficients over time
    p0_full = res_fom['snaps_p'][0]
    a_p = Psi[:, :Nrb_use].T @ p0_full
    a_Phi = np.zeros(Nrb_use)

    # Build propagator manually
    from room_acoustics.solvers import _stabilize_propagator
    from scipy.linalg import schur
    rho, c = RHO_AIR, C_AIR
    rc2 = rho * c**2
    M_inv = ops['M_inv']
    S_full = ops['S']
    B_diag = np.array(ops['B_total'].diagonal())

    K1Psi = rc2 * (M_inv[:, None] * S_full.dot(Psi[:, :Nrb_use]))
    K1_r = Psi[:, :Nrb_use].T @ K1Psi
    K2 = -1.0 / rho

    A_r = np.zeros((2*Nrb_use, 2*Nrb_use))
    A_r[:Nrb_use, Nrb_use:] = K1_r
    A_r[Nrb_use:, :Nrb_use] = K2 * np.eye(Nrb_use)

    Z = 3000
    fi_vec = -rc2 / Z * M_inv * B_diag
    K3_r = Psi[:, :Nrb_use].T @ (fi_vec[:, None] * Psi[:, :Nrb_use])
    A_r[:Nrb_use, :Nrb_use] += K3_r

    dtA = dt * A_r
    dtA2 = dtA @ dtA
    Prop = (np.eye(2*Nrb_use) + dtA + dtA2/2 + dtA2@dtA/6 + dtA2@dtA2/24)
    Prop = _stabilize_propagator(Prop, 'FI')

    # Step ROM to specific times and reconstruct fields
    state = np.concatenate([a_p, a_Phi])
    compare_times = [0.008, 0.018, 0.032]
    compare_steps = [int(round(t / dt)) for t in compare_times]

    rom_fields = {}
    step = 0
    for target in compare_steps:
        while step < target:
            state = Prop @ state
            step += 1
        rom_fields[target] = Psi[:, :Nrb_use] @ state[:Nrb_use]

    # Plot: 3 rows x 3 cols: FOM, ROM, Error
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'FOM vs ROM (Nrb={Nrb_use}) — L-Shaped Room with Absorbing Walls',
                 fontsize=14, fontweight='bold', y=0.98)

    # FOM snapshots are strided — find closest snapshot index
    snap_stride_fom = max(1, int(0.001 / dt))
    snaps = res_fom['snaps_p']
    for col, (t_target, step_idx) in enumerate(zip(compare_times, compare_steps)):
        # Map absolute step to strided snapshot index
        snap_idx = min(step_idx // snap_stride_fom, len(snaps) - 1)
        t_ms = t_target * 1e3
        fom_field = snaps[snap_idx]
        rom_field = rom_fields[step_idx]
        error_field = fom_field - rom_field
        vmax = max(np.max(np.abs(fom_field)), 1e-10) * 0.9

        _plot_field(axes[0, col], tri, fom_field, mesh,
                    f'FOM — t={t_ms:.0f} ms', vmax=vmax)
        axes[0, col].plot(1.5, 1.5, 'k*', ms=8)

        tc = _plot_field(axes[1, col], tri, rom_field, mesh,
                         f'ROM (Nrb={Nrb_use}) — t={t_ms:.0f} ms', vmax=vmax)
        axes[1, col].plot(1.5, 1.5, 'k*', ms=8)

        err_max = max(np.max(np.abs(error_field)), 1e-10)
        _plot_field(axes[2, col], tri, error_field, mesh,
                    f'Error — max={err_max:.2e} Pa', vmax=err_max)

    plt.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.04,
                        wspace=0.3, hspace=0.35)
    cbar_ax = fig.add_axes([0.93, 0.38, 0.015, 0.25])
    fig.colorbar(tc, cax=cbar_ax, label='Pressure [Pa]')
    savefig('vis_fom_vs_rom.png')


# ==================================================================
# VIS 3: T-shape and column room wave propagation
# ==================================================================

def vis3_complex_rooms():
    print("\n  VIS 3: Complex room wave propagation...")

    cases = []

    # T-shaped room (coarser mesh to save memory)
    geom_t = t_shaped_room(6.0, 5.0, 2.0, 3.0)
    raw_t = generate_quad_mesh(geom_t, 0.4, P)
    mesh_t = UnstructuredQuadMesh2D(raw_t['nodes'], raw_t['quads'],
                                     raw_t['boundary'], P)
    ops_t = assemble_unstructured_2d_operators(mesh_t)
    cases.append(('T-Shaped Room', mesh_t, ops_t, 3.0, 4.0, 0.2, 0.4))

    # Room with column (coarser mesh)
    geom_c = room_with_column(5.0, 4.0, (2.5, 2.0), 0.5, n_col_sides=8)
    raw_c = generate_quad_mesh(geom_c, 0.4, P)
    mesh_c = UnstructuredQuadMesh2D(raw_c['nodes'], raw_c['quads'],
                                     raw_c['boundary'], P)
    ops_c = assemble_unstructured_2d_operators(mesh_c)
    cases.append(('Room with Column', mesh_c, ops_c, 1.0, 1.0, 0.15, 0.4))

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))

    for row, (name, mesh, ops, sx, sy, sigma, h) in enumerate(cases):
        dt = round(0.15 * h / (C_AIR * P**2), 8)
        T = 0.035
        snap_stride = max(1, int(0.001 / dt))

        print(f"    {name}: N={mesh.N_dof}, snap_stride={snap_stride}")
        res = fom_pphi(mesh, ops, 'FI', {'Z': 2000}, sx, sy, sigma,
                       dt, T, store_snapshots=True, snap_stride=snap_stride)

        tri = _triangulate_mesh(mesh)
        snaps = res['snaps_p']
        n_snaps = len(snaps)
        del res  # free memory

        times = [0.002, 0.006, 0.012, 0.020, 0.030]
        vmax = 0
        frame_indices = []
        for t in times:
            idx = min(int(round(t / (dt * snap_stride))), n_snaps - 1)
            frame_indices.append(idx)
            vmax = max(vmax, np.max(np.abs(snaps[idx])))
        vmax *= 0.7

        for col, fi in enumerate(frame_indices):
            ax = axes[row, col]
            t_ms = fi * dt * snap_stride * 1e3
            tc = _plot_field(ax, tri, snaps[fi], mesh,
                             f'{name}\nt={t_ms:.1f} ms' if col == 0
                             else f't={t_ms:.1f} ms',
                             vmax=vmax)
            ax.plot(sx, sy, 'k*', ms=6, zorder=5)
            if col > 0:
                ax.set_ylabel('')

    plt.subplots_adjust(left=0.04, right=0.92, top=0.95, bottom=0.05,
                        wspace=0.2, hspace=0.3)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(tc, cax=cbar_ax, label='Pressure [Pa]')
    fig.suptitle('Wave Propagation in Complex Geometries (FI boundaries)',
                 fontsize=14, fontweight='bold')
    savefig('vis_complex_rooms.png')


# ==================================================================
# VIS 4: Energy decay comparison across room shapes
# ==================================================================

def vis4_energy_comparison():
    print("\n  VIS 4: Energy decay comparison...")

    from room_acoustics.geometry import rectangular_room
    rooms = [
        ('Rectangle 4x2', rectangular_room(4.0, 2.0), 2.0, 1.0),
        ('L-shape', l_shaped_room(6.0, 4.0, 3.0, 2.0), 1.5, 1.5),
        ('T-shape', t_shaped_room(6.0, 5.0, 2.0, 3.0), 3.0, 4.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    colors = ['#2196F3', '#F44336', '#4CAF50']

    for idx, (name, geom, sx, sy) in enumerate(rooms):
        h = 0.35
        raw = generate_quad_mesh(geom, h, P)
        mesh = UnstructuredQuadMesh2D(raw['nodes'], raw['quads'],
                                       raw['boundary'], P)
        ops = assemble_unstructured_2d_operators(mesh)
        dt = round(0.15 * h / (C_AIR * P**2), 8)
        T = 0.06

        import gc
        print(f"    {name}: N={mesh.N_dof}")
        snap_stride = max(10, int(0.002 / dt))

        def run_and_energy(bc_type, bc_params):
            M = ops['M_diag']; S = ops['S']
            res = fom_pphi(mesh, ops, bc_type, bc_params, sx, sy, 0.3,
                           dt, T, store_snapshots=True,
                           snap_stride=snap_stride)
            E = []
            for k in range(len(res['snaps_p'])):
                p = res['snaps_p'][k]; phi = res['snaps_Phi'][k]
                E.append(0.5/(RHO_AIR*C_AIR**2)*np.dot(p, M*p)
                         + 0.5*RHO_AIR*np.dot(phi, S.dot(phi)))
            del res; gc.collect()
            return np.array(E)

        E_pr = run_and_energy('PR', {})
        E_fi1 = run_and_energy('FI', {'Z': 1000})
        E_fi2 = run_and_energy('FI', {'Z': 5000})
        t_snap = np.arange(len(E_pr)) * dt * snap_stride * 1e3

        ax = axes[idx]
        ax.plot(t_snap, E_pr/E_pr[0], 'b-', lw=1.5, label='Rigid (PR)')
        ax.plot(t_snap, E_fi1/E_fi1[0], 'r-', lw=1.5, label='Z=1000')
        ax.plot(t_snap, E_fi2/E_fi2[0], 'g-', lw=1.5, label='Z=5000')
        ax.set(xlabel='Time [ms]', ylabel='E(t)/E(0)',
               title=f'{name} (N={mesh.N_dof})')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    plt.suptitle('Energy Decay: Room Shape vs Wall Absorption',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    savefig('vis_energy_comparison.png')


# ==================================================================

def main():
    print("=" * 60)
    print("  Room Acoustics — High-Quality Visualizations")
    print("=" * 60)

    t0 = time.perf_counter()

    mesh, ops, tri, res = vis1_lshape_propagation()
    vis2_fom_vs_rom(mesh, ops, tri, res)
    vis3_complex_rooms()
    vis4_energy_comparison()

    print(f"\n  Total: {time.perf_counter()-t0:.1f}s")
    print(f"  Results in: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
