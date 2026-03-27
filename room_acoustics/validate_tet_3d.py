#!/usr/bin/env python
"""
Validation suite for P=2 tetrahedral elements.

Tests:
  1. Unit tests: basis functions, gradients, quadrature exactness
  2. Assembly checks: volume, surface, symmetry, patch test, positive mass
  3. Box eigenfrequencies vs analytical
  4. L-shaped room FOM: energy conservation (PR) + decay (FI)
  5. L-shaped room ROM: accuracy and speedup
"""

import os, sys, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Block cupy
sys.modules.setdefault('cupy', None)
sys.modules.setdefault('cupyx', None)
sys.modules.setdefault('cupyx.scipy', None)
sys.modules.setdefault('cupyx.scipy.sparse', None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from room_acoustics.geometry import rectangular_room, l_shaped_room
from room_acoustics.gmsh_tet_import import generate_tet_mesh
from room_acoustics.tet_sem import (
    TetMesh3D, assemble_tet_3d_operators,
    tet_shape_p2, tet_shape_grad_p2, tet_quadrature, _tet_ref_nodes,
)
from room_acoustics.solvers import (
    fom_pphi_3d_gpu, rom_pphi_3d, build_psd_basis, C_AIR, RHO_AIR,
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


def _energy(p, phi, ops):
    M = ops['M_diag']; S = ops['S']
    return (0.5 / (RHO_AIR * C_AIR**2) * np.dot(p, M * p)
            + 0.5 * RHO_AIR * np.dot(phi, S.dot(phi)))


# ==================================================================
# TEST 1: Reference element unit tests
# ==================================================================

def test1_reference_element():
    print("\n" + "=" * 70)
    print("TEST 1: Reference element unit tests")
    print("=" * 70)

    results = {}

    # 1a. Partition of unity + Kronecker delta
    ref = _tet_ref_nodes()
    pou_ok = True
    kron_ok = True
    for i, (r, s, t) in enumerate(ref):
        N = tet_shape_p2(r, s, t)
        if abs(N.sum() - 1.0) > 1e-14:
            pou_ok = False
        for j in range(10):
            expected = 1.0 if i == j else 0.0
            if abs(N[j] - expected) > 1e-14:
                kron_ok = False

    # POU at random points
    for _ in range(20):
        r, s, t = np.random.rand(3) * 0.3
        N = tet_shape_p2(r, s, t)
        if abs(N.sum() - 1.0) > 1e-13:
            pou_ok = False

    results['partition_of_unity'] = pou_ok
    results['kronecker_delta'] = kron_ok
    print(f"  Partition of unity: {'PASS' if pou_ok else 'FAIL'}")
    print(f"  Kronecker delta:    {'PASS' if kron_ok else 'FAIL'}")

    # 1b. Gradient finite difference check
    eps = 1e-7
    grad_ok = True
    max_grad_err = 0.0
    for _ in range(10):
        r0, s0, t0 = np.random.rand(3) * 0.25
        dN_ana = tet_shape_grad_p2(r0, s0, t0)
        for d in range(3):
            shift = np.zeros(3); shift[d] = eps
            N_p = tet_shape_p2(r0 + shift[0], s0 + shift[1], t0 + shift[2])
            N_m = tet_shape_p2(r0 - shift[0], s0 - shift[1], t0 - shift[2])
            dN_fd = (N_p - N_m) / (2 * eps)
            err = np.max(np.abs(dN_ana[d] - dN_fd))
            max_grad_err = max(max_grad_err, err)
            if err > 1e-5:
                grad_ok = False

    results['gradient_fd_check'] = grad_ok
    results['gradient_max_error'] = max_grad_err
    print(f"  Gradient FD check:  {'PASS' if grad_ok else 'FAIL'} "
          f"(max err {max_grad_err:.2e})")

    # 1c. Quadrature exactness
    quad_ok = True
    quad_results = {}

    # Volume of tet = 1/6
    for deg in [1, 2, 3, 5]:
        pts, wts = tet_quadrature(deg)
        vol = wts.sum()
        err = abs(vol - 1.0 / 6.0)
        quad_results[f'volume_deg{deg}'] = {'value': vol, 'error': err}
        if err > 1e-14:
            quad_ok = False

    # Integral of r = 1/24
    pts5, wts5 = tet_quadrature(5)
    I_r = sum(w * p[0] for p, w in zip(pts5, wts5))
    quad_results['integral_r'] = {'value': I_r, 'expected': 1/24,
                                   'error': abs(I_r - 1/24)}

    # Integral of r^2 = 1/60
    I_r2 = sum(w * p[0]**2 for p, w in zip(pts5, wts5))
    quad_results['integral_r2'] = {'value': I_r2, 'expected': 1/60,
                                    'error': abs(I_r2 - 1/60)}

    # Integral of r*s = 1/120
    I_rs = sum(w * p[0] * p[1] for p, w in zip(pts5, wts5))
    quad_results['integral_rs'] = {'value': I_rs, 'expected': 1/120,
                                    'error': abs(I_rs - 1/120)}

    for key, val in quad_results.items():
        if 'error' in val and val['error'] > 1e-12:
            quad_ok = False

    results['quadrature'] = quad_results
    results['quadrature_pass'] = quad_ok
    print(f"  Quadrature:         {'PASS' if quad_ok else 'FAIL'}")

    save_result('tet_test1_reference', results, suite='tet_validation')
    return results


# ==================================================================
# TEST 2: Assembly on unit cube
# ==================================================================

def test2_assembly():
    print("\n" + "=" * 70)
    print("TEST 2: Assembly checks (unit cube)")
    print("=" * 70)

    geom = rectangular_room(1.0, 1.0)
    data = generate_tet_mesh(geom, Lz=1.0, h_target=0.3, P=2)
    mesh = TetMesh3D(data['nodes'], data['tets'], data['boundary'])
    ops = assemble_tet_3d_operators(mesh)

    results = {
        'mesh': {'N_dof': mesh.N_dof, 'N_el': mesh.N_el, 'h_target': 0.3},
    }

    # Volume
    vol = ops['M_diag'].sum()
    results['volume'] = {'computed': vol, 'expected': 1.0,
                          'error': abs(vol - 1.0)}
    print(f"  Volume: {vol:.6f} (expected 1.0, err={abs(vol-1):.2e})")

    # Surface area
    surf = ops['B_total'].diagonal().sum()
    results['surface'] = {'computed': surf, 'expected': 6.0,
                           'error': abs(surf - 6.0)}
    print(f"  Surface: {surf:.4f} (expected 6.0, err={abs(surf-6):.2e})")

    # Symmetry
    S = ops['S']
    asym = float(abs(S - S.T).max())
    results['stiffness_asymmetry'] = asym
    print(f"  S asymmetry: {asym:.2e}")

    # S @ ones = 0 (zero row sums)
    rowsum = float(abs(S.dot(np.ones(mesh.N_dof))).max())
    results['stiffness_rowsum'] = rowsum
    print(f"  S @ ones: {rowsum:.2e}")

    # Patch test (interior only)
    p_lin = 2 * mesh.x + 3 * mesh.y + 5 * mesh.z + 1
    resid = S.dot(p_lin)
    bnd = set(mesh.all_boundary_nodes().tolist())
    interior_resid = max(abs(resid[i]) for i in range(mesh.N_dof)
                         if i not in bnd)
    results['patch_test_interior'] = interior_resid
    print(f"  Patch test (interior): {interior_resid:.2e}")

    # Positive mass
    min_mass = float(ops['M_diag'].min())
    n_neg = int((ops['M_diag'] < 0).sum())
    results['min_lumped_mass'] = min_mass
    results['negative_mass_entries'] = n_neg
    print(f"  Min lumped mass: {min_mass:.2e} (neg entries: {n_neg})")

    # SPD check
    v = np.random.randn(mesh.N_dof); v -= v.mean()
    vtSv = float(v @ S.dot(v))
    results['v_S_v'] = vtSv
    print(f"  v^T S v: {vtSv:.4f} (should be > 0)")

    all_pass = (abs(vol - 1.0) < 1e-6 and abs(surf - 6.0) < 1e-2
                and asym < 1e-12 and interior_resid < 1e-10
                and n_neg == 0 and vtSv > 0)
    results['all_pass'] = all_pass
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    save_result('tet_test2_assembly', results, suite='tet_validation')
    return mesh, ops


# ==================================================================
# TEST 3: Box eigenfrequencies vs analytical
# ==================================================================

def test3_eigenfrequencies(mesh, ops):
    print("\n" + "=" * 70)
    print("TEST 3: Box eigenfrequencies vs analytical")
    print("=" * 70)

    from scipy.sparse import diags
    M_sp = diags(ops['M_diag'])
    n_eigs = min(15, mesh.N_dof - 1)
    eigenvalues, _ = eigsh(ops['S'], k=n_eigs, M=M_sp, sigma=0, which='LM')
    eigenvalues = np.sort(eigenvalues)

    # Analytical: lambda_{l,m,n} = pi^2 * (l^2 + m^2 + n^2) for unit cube
    ana = []
    for l in range(10):
        for m in range(10):
            for n in range(10):
                ana.append(np.pi**2 * (l**2 + m**2 + n**2))
    ana = np.sort(ana)[:n_eigs]

    print(f"  {'Mode':>4s}  {'Analytical':>14s}  {'Computed':>14s}  "
          f"{'Rel Error':>10s}  {'Freq [Hz]':>10s}")
    print("  " + "-" * 60)

    errors = []
    for i in range(n_eigs):
        a = ana[i]
        c = eigenvalues[i]
        rel = abs(c - a) / max(a, 1e-10) if a > 1e-10 else abs(c)
        freq = np.sqrt(max(a, 0)) * C_AIR / (2 * np.pi) if a > 0 else 0
        errors.append(rel)
        print(f"  {i:4d}  {a:14.4f}  {c:14.4f}  {rel:10.2e}  {freq:10.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    idx = np.arange(n_eigs)
    ax1.plot(idx, ana, 'ko', ms=6, label='Analytical')
    ax1.plot(idx, eigenvalues, 'r^', ms=5, label=f'P=2 Tet (N={mesh.N_dof})')
    ax1.set(xlabel='Mode index', ylabel='Eigenvalue',
            title='Unit cube eigenfrequencies')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.semilogy(idx[1:], errors[1:], 'ro-', ms=5)
    ax2.set(xlabel='Mode index', ylabel='Relative error',
            title='Eigenvalue error (P=2 tets)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('tet_test3_eigenfreq.png')

    save_result('tet_test3_eigenfreq', {
        'domain': 'unit cube',
        'N_dof': mesh.N_dof, 'N_el': mesh.N_el,
        'analytical_eigenvalues': ana,
        'computed_eigenvalues': eigenvalues,
        'relative_errors': errors,
    }, suite='tet_validation')

    return errors


# ==================================================================
# TEST 4: L-shaped room FOM
# ==================================================================

def test4_lshape_fom():
    print("\n" + "=" * 70)
    print("TEST 4: L-shaped room 3D FOM (tets)")
    print("=" * 70)

    geom = l_shaped_room(4.0, 3.0, 2.0, 1.5)
    data = generate_tet_mesh(geom, Lz=2.0, h_target=0.4, P=2)
    mesh = TetMesh3D(data['nodes'], data['tets'], data['boundary'])
    ops = assemble_tet_3d_operators(mesh)

    vol = ops['M_diag'].sum()
    vol_expected = (4*3 - 2*1.5) * 2.0  # = 18
    surf_expected = 2 * (4*3 - 2*1.5) + (4+3+2+1.5+2+1.5) * 2.0  # = 2*9 + 14*2 = 46
    surf = ops['B_total'].diagonal().sum()
    print(f"  Mesh: N={mesh.N_dof}, {mesh.N_el} tets")
    print(f"  Volume: {vol:.2f} (expected {vol_expected})")
    print(f"  Surface: {surf:.2f} (expected {surf_expected})")

    src = (1.0, 1.0, 1.0)
    rec_idx = mesh.nearest_node(3.0, 0.5, 1.0)
    dt = 3e-5
    T = 0.03

    # PR
    print("  Running PR FOM...")
    res_pr = fom_pphi_3d_gpu(mesh, ops, 'PR', {}, *src, 0.2, dt, T,
                              rec_idx=rec_idx, store_snapshots=True,
                              snap_stride=20)
    E_pr = [_energy(res_pr['snaps_p'][k], res_pr['snaps_Phi'][k], ops)
            for k in range(len(res_pr['snaps_p']))]
    E_pr = np.array(E_pr)
    drift = abs(E_pr[-1] - E_pr[0]) / E_pr[0]
    print(f"  PR energy drift: {drift:.2e}")

    # FI
    print("  Running FI FOM...")
    res_fi = fom_pphi_3d_gpu(mesh, ops, 'FI', {'Z': 2000}, *src, 0.2, dt, T,
                              rec_idx=rec_idx, store_snapshots=True,
                              snap_stride=20)
    E_fi = [_energy(res_fi['snaps_p'][k], res_fi['snaps_Phi'][k], ops)
            for k in range(len(res_fi['snaps_p']))]
    E_fi = np.array(E_fi)
    fi_ratio = E_fi[-1] / E_fi[0]
    print(f"  FI energy ratio: {fi_ratio:.4f}")

    # Monotonic decay check
    mono = all(E_fi[i+1] <= E_fi[i] * 1.001 for i in range(len(E_fi)-1))
    print(f"  FI monotonic decay: {'YES' if mono else 'NO'}")

    # Plot
    t_vec = np.arange(len(res_pr['ir'])) * dt * 1e3
    t_snap = np.arange(len(E_pr)) * dt * 20 * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(t_vec, res_pr['ir'], 'b-', lw=0.8)
    axes[0, 0].set(ylabel='Pressure [Pa]', title='L-shape 3D PR (tets)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_snap, E_pr / E_pr[0], 'b-', label='PR')
    axes[0, 1].plot(t_snap, E_fi / E_fi[0], 'r-', label='FI (Z=2000)')
    axes[0, 1].set(ylabel='E(t)/E(0)', title='Energy evolution')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_vec, res_fi['ir'], 'r-', lw=0.8)
    axes[1, 0].set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
                   title='L-shape 3D FI (tets)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(0.5, 0.5,
                    f"N_dof = {mesh.N_dof}\n"
                    f"{mesh.N_el} tet elements\n"
                    f"P = 2\n"
                    f"Volume = {vol:.1f} m³\n"
                    f"Surface = {surf:.1f} m²\n"
                    f"PR drift = {drift:.1e}\n"
                    f"FI E_end/E_start = {fi_ratio:.4f}\n"
                    f"FI monotonic = {mono}",
                    transform=axes[1, 1].transAxes, fontsize=10,
                    va='center', ha='center', fontfamily='monospace')
    axes[1, 1].axis('off')

    plt.tight_layout()
    savefig('tet_test4_lshape_fom.png')

    save_result('tet_test4_lshape_fom', {
        'domain': {'type': 'L-shape extruded', 'Lx': 4.0, 'Ly': 3.0,
                   'notch_x': 2.0, 'notch_y': 1.5, 'Lz': 2.0},
        'mesh': {'N_dof': mesh.N_dof, 'N_el': mesh.N_el, 'P': 2,
                 'h_target': 0.4},
        'volume': {'computed': vol, 'expected': vol_expected},
        'surface': {'computed': surf, 'expected': surf_expected},
        'pr_energy_drift': drift,
        'fi_energy_ratio': fi_ratio,
        'fi_monotonic_decay': mono,
        'dt': dt, 'T': T,
    }, suite='tet_validation')

    return mesh, ops, res_pr, src, rec_idx, dt, T


# ==================================================================
# TEST 5: L-shaped room ROM
# ==================================================================

def test5_lshape_rom(mesh, ops, res_fom, src, rec_idx, dt, T):
    print("\n" + "=" * 70)
    print("TEST 5: L-shaped room 3D ROM (tets)")
    print("=" * 70)

    Psi, _, Nrb_auto = build_psd_basis(
        res_fom['snaps_p'], res_fom['snaps_Phi'], eps_pod=1e-8)
    print(f"  PSD basis: Nrb_auto={Nrb_auto}")

    # FOM timing
    t0 = time.perf_counter()
    _ = fom_pphi_3d_gpu(mesh, ops, 'PR', {}, *src, 0.2, dt, T,
                         rec_idx=rec_idx)
    t_fom = time.perf_counter() - t0
    print(f"  FOM time: {t_fom:.2f}s")

    nrb_list = [5, 10, 20, 40, 60]
    nrb_list = [n for n in nrb_list if n <= Psi.shape[1]]

    speedups, errors = [], []
    for nrb in nrb_list:
        t0 = time.perf_counter()
        rr = rom_pphi_3d(mesh, ops, Psi, 'PR', {}, *src, 0.2, dt, T,
                          rec_idx=rec_idx, Nrb_override=nrb)
        t_rom = time.perf_counter() - t0
        err = np.max(np.abs(res_fom['ir'] - rr['ir']))
        sp = t_fom / t_rom
        speedups.append(sp); errors.append(err)
        print(f"    Nrb={nrb:3d}: speedup={sp:6.1f}x, error={err:.2e}")

    # Plot
    best_idx = min(len(nrb_list) - 1, 2)
    best_nrb = nrb_list[best_idx]
    res_best = rom_pphi_3d(mesh, ops, Psi, 'PR', {}, *src, 0.2, dt, T,
                            rec_idx=rec_idx, Nrb_override=best_nrb)

    t_vec = np.arange(len(res_fom['ir'])) * dt * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(t_vec, res_fom['ir'], 'b-', lw=1, label='FOM')
    axes[0, 0].plot(t_vec, res_best['ir'], 'r--', lw=1,
                    label=f'ROM (Nrb={best_nrb})')
    axes[0, 0].set(ylabel='Pressure [Pa]',
                   title='L-shape 3D Tets: FOM vs ROM')
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
    savefig('tet_test5_lshape_rom.png')

    save_result('tet_test5_lshape_rom', {
        'N_dof': mesh.N_dof, 'N_el': mesh.N_el,
        'Nrb_auto': Nrb_auto,
        'fom_time_s': t_fom,
        'sweep': [{'Nrb': nrb, 'speedup': sp, 'error_Linf': err}
                  for nrb, sp, err in zip(nrb_list, speedups, errors)],
    }, suite='tet_validation')


# ==================================================================
# Main
# ==================================================================

def main():
    print("=" * 70)
    print("  P=2 Tetrahedral Element Validation Suite")
    print("=" * 70)

    t_start = time.perf_counter()

    test1_reference_element(); gc.collect()
    mesh, ops = test2_assembly(); gc.collect()
    test3_eigenfrequencies(mesh, ops); gc.collect()
    d4 = test4_lshape_fom(); gc.collect()
    mesh4, ops4, res_pr4, src4, rec4, dt4, T4 = d4
    test5_lshape_rom(mesh4, ops4, res_pr4, src4, rec4, dt4, T4); gc.collect()

    print(f"\n  Total time: {time.perf_counter() - t_start:.1f}s")
    print(f"  Plots saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
