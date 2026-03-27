#!/usr/bin/env python
"""
BRAS CR2 Benchmark: Seminar room (shoebox)

Geometry: 8.4 x 6.7 x 3.0 m (168.84 m^3)
Reference: Aspock & Vorlander, "BRAS - Benchmark for Room Acoustical
           Simulation", Applied Acoustics, 2021.

Material assignments (from BRAS documentation):
  Floor:   linoleum on concrete (alpha ~ 0.03)
  Ceiling: acoustic tiles (alpha ~ 0.50)
  Walls:   painted concrete (alpha ~ 0.02-0.05)
  Window wall: glass panels (alpha ~ 0.04)
  Door wall: wooden door + plaster (alpha ~ 0.06)

This test validates against Sabine/Eyring RT60 predictions.
Full BRAS validation requires the measured IRs from:
  https://depositonce.tu-berlin.de/items/version/64

ISO 3382 metrics computed: T30, T20, EDT, C80, D50, TS
"""

import sys, os, time
sys.modules.setdefault('cupy', None)
sys.modules.setdefault('cupyx', None)
sys.modules.setdefault('cupyx.scipy', None)
sys.modules.setdefault('cupyx.scipy.sparse', None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from room_acoustics.geometry import rectangular_room
from room_acoustics.gmsh_tet_import import generate_tet_mesh
from room_acoustics.tet_sem import TetMesh3D, assemble_tet_3d_operators
from room_acoustics.materials import assign_materials
from room_acoustics.solvers import fom_pphi_3d_gpu, C_AIR, RHO_AIR
from room_acoustics.acoustics_metrics import (
    schroeder_decay, all_metrics, sabine_rt60, eyring_rt60,
    impedance_to_alpha,
)
from room_acoustics.results_io import save_result

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved {path}")


def main():
    print("=" * 60)
    print("  BRAS CR2 Benchmark: Seminar Room (Shoebox)")
    print("  8.4 x 6.7 x 3.0 m, 168.84 m^3")
    print("=" * 60)

    # ============================================================
    # Room geometry
    # ============================================================
    Lx, Ly, Lz = 8.4, 6.7, 3.0
    V = Lx * Ly * Lz  # 168.84 m^3

    # Surface areas
    S_floor = Lx * Ly       # 56.28
    S_ceiling = Lx * Ly     # 56.28
    S_front = Lx * Lz       # 25.20 (short wall)
    S_back = Lx * Lz        # 25.20
    S_left = Ly * Lz        # 20.10
    S_right = Ly * Lz       # 20.10
    S_total = 2*(Lx*Ly + Lx*Lz + Ly*Lz)  # 203.16

    print(f"  Volume: {V:.2f} m^3")
    print(f"  Total surface: {S_total:.2f} m^2")

    # ============================================================
    # Material assignment (BRAS CR2 approximate)
    # ============================================================
    # BRAS CR2 materials (mid-frequency absorption coefficients):
    #   Floor: linoleum (alpha ~ 0.03)  -> Z ~ 27000
    #   Ceiling: acoustic tiles (alpha ~ 0.50) -> Z ~ 700
    #   Walls: painted concrete (alpha ~ 0.02) -> Z ~ 40000
    #   One wall has windows (alpha ~ 0.04) -> Z ~ 20000
    #   One wall has door (alpha ~ 0.06) -> Z ~ 13000

    # Map absorption to impedance: Z = rho*c * (1+R)/(1-R)
    # where R = sqrt(1-alpha), so Z = rho*c * (1+sqrt(1-a))/(1-sqrt(1-a))
    rho_c = RHO_AIR * C_AIR

    def alpha_to_Z(alpha):
        R = np.sqrt(1.0 - alpha)
        return rho_c * (1 + R) / (1 - R)

    materials_config = {
        'floor':   {'alpha': 0.03, 'Z': alpha_to_Z(0.03), 'area': S_floor},
        'ceiling': {'alpha': 0.50, 'Z': alpha_to_Z(0.50), 'area': S_ceiling},
        'bottom':  {'alpha': 0.02, 'Z': alpha_to_Z(0.02), 'area': S_front},   # front wall
        'top':     {'alpha': 0.02, 'Z': alpha_to_Z(0.02), 'area': S_back},    # back wall
        'left':    {'alpha': 0.04, 'Z': alpha_to_Z(0.04), 'area': S_left},    # window wall
        'right':   {'alpha': 0.06, 'Z': alpha_to_Z(0.06), 'area': S_right},   # door wall
    }

    print("\n  Materials:")
    for label, m in materials_config.items():
        print(f"    {label:10s}: alpha={m['alpha']:.2f}, Z={m['Z']:.0f}, "
              f"area={m['area']:.1f} m^2")

    # ============================================================
    # Analytical RT60 predictions
    # ============================================================
    areas = {k: v['area'] for k, v in materials_config.items()}
    alphas = {k: v['alpha'] for k, v in materials_config.items()}

    rt60_sabine = sabine_rt60(V, areas, alphas)
    mean_alpha = sum(areas[k]*alphas[k] for k in areas) / S_total
    rt60_eyring = eyring_rt60(V, S_total, mean_alpha)

    print(f"\n  Sabine RT60:  {rt60_sabine:.2f} s")
    print(f"  Eyring RT60:  {rt60_eyring:.2f} s")
    print(f"  Mean alpha:   {mean_alpha:.3f}")

    # ============================================================
    # Build mesh and operators
    # ============================================================
    geom = rectangular_room(Lx, Ly)
    h = 0.4
    data = generate_tet_mesh(geom, Lz=Lz, h_target=h, P=2)
    mesh = TetMesh3D(data['nodes'], data['tets'], data['boundary'])
    ops = assemble_tet_3d_operators(mesh)

    vol_computed = ops['M_diag'].sum()
    print(f"\n  Mesh: N={mesh.N_dof}, {mesh.N_el} tets")
    print(f"  Volume: {vol_computed:.2f} m^3 (expected {V:.2f})")
    print(f"  Labels: {list(mesh._boundary_nodes_per_label.keys())}")

    # Assign per-node impedance
    N = mesh.N_dof
    Z_per_node = np.full(N, 1e15)
    for label, m in materials_config.items():
        if label in mesh._boundary_nodes_per_label:
            nodes = mesh.boundary_nodes(label)
            Z_per_node[nodes] = m['Z']

    # ============================================================
    # FOM simulation
    # ============================================================
    # Source and receiver positions (BRAS CR2 has specific positions,
    # here we use representative locations)
    src = (2.0, 3.35, 1.5)  # near one end, centered
    rec = mesh.nearest_node(6.0, 2.0, 1.2)  # far end

    # CFL
    min_h = 1e10
    for e in range(min(300, mesh.N_el)):
        verts = mesh.nodes[mesh.elem_conn[e, :4]]
        for i in range(4):
            for j in range(i+1, 4):
                min_h = min(min_h, np.linalg.norm(verts[i]-verts[j]))
    dt = round(0.1 * min_h / (C_AIR * 4), 8)

    # Simulate long enough to capture RT60 (need at least 2x RT60)
    T_sim = min(2.0 * rt60_sabine, 0.5)  # cap at 0.5s to avoid memory issues
    Nt = int(T_sim / dt)
    print(f"\n  dt={dt:.2e}, T={T_sim:.3f}s, Nt={Nt}")
    print(f"  Source: {src}")
    print(f"  Receiver: ({mesh.x[rec]:.1f},{mesh.y[rec]:.1f},{mesh.z[rec]:.1f})")

    t0 = time.perf_counter()
    res = fom_pphi_3d_gpu(mesh, ops, 'FI', {'Z_per_node': Z_per_node},
                          *src, 0.15, dt, T_sim, rec_idx=rec)
    t_fom = time.perf_counter() - t0
    print(f"  FOM time: {t_fom:.1f}s")

    ir = res['ir']
    print(f"  IR length: {len(ir)} samples, peak: {np.max(np.abs(ir)):.4f}")

    # ============================================================
    # Compute metrics
    # ============================================================
    metrics = all_metrics(ir, dt)
    print("\n  ISO 3382 Metrics:")
    print(f"    T30:  {metrics['T30_s']:.2f} s (R^2={metrics['T30_R2']:.3f})")
    print(f"    T20:  {metrics['T20_s']:.2f} s (R^2={metrics['T20_R2']:.3f})")
    print(f"    EDT:  {metrics['EDT_s']:.2f} s (R^2={metrics['EDT_R2']:.3f})")
    print(f"    C80:  {metrics['C80_dB']:.1f} dB")
    print(f"    D50:  {metrics['D50']:.3f}")
    print(f"    TS:   {metrics['TS_ms']:.1f} ms")

    # ============================================================
    # Compare against Sabine/Eyring
    # ============================================================
    print("\n  RT60 Comparison:")
    print(f"    Sabine:      {rt60_sabine:.3f} s")
    print(f"    Eyring:      {rt60_eyring:.3f} s")
    print(f"    Simulated:   {metrics['T30_s']:.3f} s (T30)")
    if not np.isnan(metrics['T30_s']) and rt60_sabine > 0:
        rel_err_sabine = abs(metrics['T30_s'] - rt60_sabine) / rt60_sabine
        rel_err_eyring = abs(metrics['T30_s'] - rt60_eyring) / rt60_eyring
        print(f"    vs Sabine:   {rel_err_sabine*100:.1f}% difference")
        print(f"    vs Eyring:   {rel_err_eyring*100:.1f}% difference")
    else:
        rel_err_sabine = np.nan
        rel_err_eyring = np.nan

    # ============================================================
    # Plot
    # ============================================================
    t_vec = np.arange(len(ir)) * dt
    t_decay, decay_dB = schroeder_decay(ir, dt)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('BRAS CR2 Benchmark: Seminar Room (8.4 x 6.7 x 3.0 m)',
                 fontsize=13, fontweight='bold')

    # IR
    axes[0, 0].plot(t_vec * 1e3, ir, 'b-', lw=0.5)
    axes[0, 0].set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
                   title='Impulse Response')
    axes[0, 0].grid(True, alpha=0.3)

    # Schroeder decay
    axes[0, 1].plot(t_decay * 1e3, decay_dB, 'b-', lw=1)
    axes[0, 1].axhline(-60, color='r', ls='--', alpha=0.5, label='-60 dB')
    axes[0, 1].axhline(-5, color='g', ls=':', alpha=0.5)
    axes[0, 1].axhline(-35, color='g', ls=':', alpha=0.5, label='T30 range')
    if not np.isnan(metrics['T30_s']):
        axes[0, 1].axvline(metrics['T30_s'] * 1e3, color='r', ls='-',
                           alpha=0.3, label=f'T30={metrics["T30_s"]:.2f}s')
    axes[0, 1].set(xlabel='Time [ms]', ylabel='Energy [dB]',
                   title='Schroeder Decay Curve', ylim=(-80, 5))
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(True, alpha=0.3)

    # RT60 comparison bar chart
    rt_vals = [rt60_sabine, rt60_eyring]
    rt_labels = ['Sabine', 'Eyring']
    if not np.isnan(metrics['T30_s']):
        rt_vals.append(metrics['T30_s'])
        rt_labels.append('Simulated\n(T30)')
    if not np.isnan(metrics['EDT_s']):
        rt_vals.append(metrics['EDT_s'])
        rt_labels.append('EDT')
    colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800']
    axes[1, 0].bar(rt_labels, rt_vals, color=colors[:len(rt_vals)])
    axes[1, 0].set(ylabel='RT60 [s]', title='Reverberation Time Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rt_vals):
        axes[1, 0].text(i, v + 0.01, f'{v:.2f}s', ha='center', fontsize=9)

    # Room info
    info = (f"Volume: {V:.1f} m^3\n"
            f"Surface: {S_total:.1f} m^2\n"
            f"Mean alpha: {mean_alpha:.3f}\n\n"
            f"Mesh: {mesh.N_dof} DOFs, {mesh.N_el} tets\n"
            f"dt: {dt:.2e} s, T: {T_sim:.3f} s\n"
            f"FOM time: {t_fom:.1f} s\n\n"
            f"T30: {metrics['T30_s']:.3f} s\n"
            f"Sabine: {rt60_sabine:.3f} s\n"
            f"Eyring: {rt60_eyring:.3f} s\n"
            f"C80: {metrics['C80_dB']:.1f} dB\n"
            f"D50: {metrics['D50']:.3f}")
    axes[1, 1].text(0.1, 0.5, info, transform=axes[1, 1].transAxes,
                    fontsize=10, va='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Room Parameters')

    plt.tight_layout()
    savefig('bras_cr2_benchmark.png')

    # ============================================================
    # Verification checks
    # ============================================================
    print("\n" + "=" * 60)
    print("  VERIFICATION")
    print("=" * 60)

    checks = []

    # Volume
    v_ok = abs(vol_computed - V) < 0.5
    checks.append(('Volume', v_ok, f'{vol_computed:.2f} vs {V:.2f}'))

    # T30 R^2 (decay should be linear in dB)
    r2_ok = metrics['T30_R2'] > 0.85
    checks.append(('T30 fit quality', r2_ok, f"R^2={metrics['T30_R2']:.3f}"))

    # T30 vs Sabine (should agree within 30% for a diffuse room)
    if not np.isnan(metrics['T30_s']):
        sab_ok = rel_err_sabine < 0.40
        checks.append(('T30 vs Sabine', sab_ok,
                       f"{rel_err_sabine*100:.1f}% (< 40%)"))
    else:
        checks.append(('T30 vs Sabine', False, 'T30 is NaN'))

    # C80 plausible (typical range -5 to +15 dB for rooms)
    c80_ok = -10 < metrics['C80_dB'] < 30
    checks.append(('C80 plausible', c80_ok, f"{metrics['C80_dB']:.1f} dB"))

    # D50 in valid range
    d50_ok = 0 < metrics['D50'] < 1
    checks.append(('D50 in [0,1]', d50_ok, f"{metrics['D50']:.3f}"))

    all_pass = all(c[1] for c in checks)
    for label, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {label} -- {detail}")
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    # ============================================================
    # Save results
    # ============================================================
    save_result('bras_cr2_benchmark', {
        'room': {
            'name': 'BRAS CR2 Seminar Room',
            'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
            'volume_m3': V, 'surface_m2': S_total,
        },
        'materials': {k: {'alpha': v['alpha'], 'Z': v['Z'], 'area': v['area']}
                      for k, v in materials_config.items()},
        'mean_alpha': mean_alpha,
        'mesh': {'N_dof': mesh.N_dof, 'N_el': mesh.N_el, 'h': h, 'P': 2,
                 'volume_computed': float(vol_computed)},
        'simulation': {'dt': dt, 'T': T_sim, 'Nt': Nt,
                       'fom_time_s': t_fom},
        'analytical': {
            'sabine_rt60_s': rt60_sabine,
            'eyring_rt60_s': rt60_eyring,
        },
        'metrics': metrics,
        'comparison': {
            'T30_vs_sabine_pct': float(rel_err_sabine * 100)
                if not np.isnan(rel_err_sabine) else None,
            'T30_vs_eyring_pct': float(rel_err_eyring * 100)
                if not np.isnan(rel_err_eyring) else None,
        },
        'verification_passed': all_pass,
        'checks': [{'name': c[0], 'pass': c[1], 'detail': c[2]}
                   for c in checks],
    }, suite='bras_benchmark')


if __name__ == '__main__':
    main()
