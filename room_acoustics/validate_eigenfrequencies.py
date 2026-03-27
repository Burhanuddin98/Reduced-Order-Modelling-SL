#!/usr/bin/env python
"""
Eigenfrequency validation — the definitive physics test.

Computes eigenvalues of the discrete Laplacian (M^{-1} S) and compares
against known analytical/reference values. This directly validates that
the assembled operators encode the correct physics.

Tests:
  1. Rectangular room: analytical eigenfrequencies (exact)
  2. L-shaped domain: Neumann eigenvalues from Dauge benchmark (11-digit reference)
  3. Mesh convergence: eigenvalue error vs mesh resolution
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from room_acoustics.geometry import rectangular_room, generate_quad_mesh
from room_acoustics.unstructured_sem import (
    UnstructuredQuadMesh2D, assemble_unstructured_2d_operators,
)
from room_acoustics.sem import RectMesh2D, assemble_2d_operators
from room_acoustics.solvers import C_AIR

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')


def savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved {path}")


def compute_eigenvalues(ops, n_eigs=20):
    """Compute smallest eigenvalues of the generalized problem S v = lam M v.

    For Neumann BCs (rigid walls), the smallest eigenvalue is 0 (constant mode).
    Returns sorted eigenvalues.
    """
    from scipy.sparse import diags
    M_diag = ops['M_diag']
    S = ops['S']
    # Generalized eigenvalue problem: S v = lam M v
    # Equivalent to: M^{-1} S v = lam v
    # Use shift-invert mode with sigma=0 to find smallest eigenvalues
    M_sp = diags(M_diag)
    # eigsh needs symmetric matrices — S is symmetric, M is diagonal positive
    eigenvalues, _ = eigsh(S, k=n_eigs, M=M_sp, sigma=0, which='LM')
    return np.sort(eigenvalues)


# ==================================================================
# TEST 1: Rectangular room — analytical eigenfrequencies
# ==================================================================

def test1_rectangular():
    print("\n" + "=" * 70)
    print("TEST 1: Rectangular room eigenfrequencies (analytical)")
    print("=" * 70)

    Lx, Ly = 2.0, 1.0
    P = 6

    # Analytical eigenvalues for Neumann BCs on [0,Lx] x [0,Ly]:
    # lambda_{m,n} = pi^2 * ((m/Lx)^2 + (n/Ly)^2)
    # Sorted, first 20 non-trivial
    ana_eigs = []
    for m in range(20):
        for n in range(20):
            lam = np.pi**2 * ((m / Lx)**2 + (n / Ly)**2)
            ana_eigs.append(lam)
    ana_eigs = np.sort(ana_eigs)[:20]  # first 20 including lambda=0

    # Numerical (structured)
    Nex = max(4, int(np.ceil(Lx * 2)))
    Ney = max(4, int(np.ceil(Ly * 2)))
    mesh_s = RectMesh2D(Lx, Ly, Nex, Ney, P)
    ops_s = assemble_2d_operators(mesh_s)
    num_eigs_s = compute_eigenvalues(ops_s, n_eigs=20)

    # Numerical (unstructured)
    geom = rectangular_room(Lx, Ly)
    raw = generate_quad_mesh(geom, Lx / Nex, P)
    mesh_u = UnstructuredQuadMesh2D(raw['nodes'], raw['quads'],
                                     raw['boundary'], P)
    ops_u = assemble_unstructured_2d_operators(mesh_u)
    num_eigs_u = compute_eigenvalues(ops_u, n_eigs=20)

    print(f"  Mesh: {Nex}x{Ney}, P={P}, N_struct={mesh_s.N_dof}, "
          f"N_unstruct={mesh_u.N_dof}")
    print(f"\n  {'Mode':>4s}  {'Analytical':>14s}  {'Structured':>14s}  "
          f"{'Err_s':>10s}  {'Unstructured':>14s}  {'Err_u':>10s}")
    print("  " + "-" * 75)

    for i in range(min(15, len(ana_eigs))):
        a = ana_eigs[i]
        s = num_eigs_s[i]
        u = num_eigs_u[i]
        e_s = abs(s - a) / max(a, 1e-10) if a > 1e-10 else abs(s)
        e_u = abs(u - a) / max(a, 1e-10) if a > 1e-10 else abs(u)
        freq_a = np.sqrt(max(a, 0)) * C_AIR / (2 * np.pi) if a > 0 else 0
        print(f"  {i:4d}  {a:14.6f}  {s:14.6f}  {e_s:10.2e}  "
              f"{u:14.6f}  {e_u:10.2e}   [{freq_a:7.1f} Hz]")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    idx = np.arange(len(ana_eigs))
    ax1.plot(idx, ana_eigs, 'ko', ms=6, label='Analytical')
    ax1.plot(idx, num_eigs_s, 'b^', ms=5, label='Structured SEM')
    ax1.plot(idx, num_eigs_u, 'rv', ms=5, label='Unstructured SEM')
    ax1.set(xlabel='Mode index', ylabel='Eigenvalue',
            title=f'Rectangular room {Lx}x{Ly} m')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Relative error
    err_s = [abs(num_eigs_s[i] - ana_eigs[i]) / max(ana_eigs[i], 1e-10)
             for i in range(1, len(ana_eigs))]  # skip lambda=0
    err_u = [abs(num_eigs_u[i] - ana_eigs[i]) / max(ana_eigs[i], 1e-10)
             for i in range(1, len(ana_eigs))]
    ax2.semilogy(range(1, len(ana_eigs)), err_s, 'b^-', ms=5,
                 label='Structured')
    ax2.semilogy(range(1, len(ana_eigs)), err_u, 'rv-', ms=5,
                 label='Unstructured')
    ax2.set(xlabel='Mode index', ylabel='Relative error',
            title='Eigenvalue error')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('eigenfreq_test1_rect.png')

    return dict(ana=ana_eigs, num_s=num_eigs_s, num_u=num_eigs_u)


# ==================================================================
# TEST 2: L-shaped domain — Dauge benchmark (Neumann)
# ==================================================================

def test2_lshape():
    print("\n" + "=" * 70)
    print("TEST 2: L-shaped domain Neumann eigenvalues (Dauge benchmark)")
    print("=" * 70)

    # Reference: Dauge Maxwell benchmark
    # https://perso.univ-rennes1.fr/monique.dauge/benchmax.html
    # L-shape: 3 unit squares, vertices at
    # (0,0), (1,0), (1,1), (-1,1), (-1,-1), (0,-1)
    # == our l_shaped_room(1, 1, 0, 0) rescaled
    #
    # Equivalently: (0,0),(2,0),(2,1),(1,1),(1,2),(0,2) — same shape.
    # We use the latter since our l_shaped_room needs positive notch coords.
    #
    # The eigenvalues scale with geometry. For the unit L (side=1):
    # For a domain of side length L, eigenvalues scale as 1/L^2.
    # The Dauge reference uses the L with bounding box [-1,1]x[-1,1],
    # which has side length 2. Our (0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2)
    # also has bounding box [0,2]x[0,2], side=2. So eigenvalues match directly.
    #
    # Actually, let me reconsider. The Dauge L-shape is:
    # (-1,-1), (1,-1), (1,0), (0,0), (0,1), (-1,1)
    # This is an L made of 3 unit squares with total area = 3.
    # Our l_shaped_room(2,2,1,1) gives vertices:
    # (0,0), (2,0), (2,1), (1,1), (1,2), (0,2)
    # Area = 2*2 - 1*1 = 3. Same area, same shape. Eigenvalues are identical.

    # Dauge Neumann eigenvalues (non-zero):
    ref_eigenvalues = np.array([
        0.0,              # constant mode
        1.47562182408,    # singular at reentrant corner
        3.53403136678,    # regular
        np.pi**2,         # exact (9.8696...)
        np.pi**2,         # double eigenvalue
        11.3894793979,    # singular at reentrant corner
    ])
    ref_labels = [
        'DC (const)',
        'singular (corner)',
        'regular',
        'exact (pi^2)',
        'exact (pi^2, double)',
        'singular (corner)',
    ]

    # Run at several mesh resolutions to show convergence
    P = 6
    h_values = [0.3, 0.2, 0.12, 0.08]
    results = []

    # Build the L-shape matching Dauge: area = 3 unit squares
    # l_shaped_room(Lx=2, Ly=2, notch_x=1, notch_y=1)
    from room_acoustics.geometry import l_shaped_room
    geom = l_shaped_room(2.0, 2.0, 1.0, 1.0)

    for h in h_values:
        raw = generate_quad_mesh(geom, h, P)
        mesh = UnstructuredQuadMesh2D(raw['nodes'], raw['quads'],
                                       raw['boundary'], P)
        ops = assemble_unstructured_2d_operators(mesh)

        n_eigs = min(10, mesh.N_dof - 1)
        num_eigs = compute_eigenvalues(ops, n_eigs=n_eigs)

        # Check area
        area = ops['M_diag'].sum()

        errors = []
        for i in range(min(len(ref_eigenvalues), len(num_eigs))):
            ref = ref_eigenvalues[i]
            num = num_eigs[i]
            if ref > 1e-10:
                errors.append(abs(num - ref) / ref)
            else:
                errors.append(abs(num))

        results.append(dict(h=h, N=mesh.N_dof, n_el=len(raw['quads']),
                            eigs=num_eigs, errors=errors, area=area))

        print(f"\n  h={h:.2f}, {len(raw['quads'])} quads, N={mesh.N_dof}, "
              f"area={area:.4f}")
        print(f"  {'Mode':>4s}  {'Reference':>14s}  {'Computed':>14s}  "
              f"{'Rel Error':>10s}  {'Description'}")
        print("  " + "-" * 70)
        for i in range(min(len(ref_eigenvalues), len(num_eigs))):
            ref = ref_eigenvalues[i]
            num = num_eigs[i]
            err = errors[i]
            label = ref_labels[i] if i < len(ref_labels) else ''
            print(f"  {i:4d}  {ref:14.8f}  {num:14.8f}  {err:10.2e}  {label}")

    # Plot convergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Eigenvalue spectrum comparison (finest mesh)
    finest = results[-1]
    n_plot = min(len(ref_eigenvalues), len(finest['eigs']))
    idx = np.arange(n_plot)
    axes[0].plot(idx, ref_eigenvalues[:n_plot], 'ko', ms=8,
                 label='Dauge reference')
    axes[0].plot(idx, finest['eigs'][:n_plot], 'r^', ms=6,
                 label=f'SEM (h={finest["h"]}, N={finest["N"]})')
    axes[0].set(xlabel='Mode index', ylabel='Eigenvalue',
                title='L-shaped domain Neumann eigenvalues')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Convergence: error vs N for each mode
    modes_to_plot = [1, 2, 3, 5]  # skip mode 0 (DC) and mode 4 (double of 3)
    for mi in modes_to_plot:
        if mi >= len(ref_eigenvalues):
            continue
        Ns = [r['N'] for r in results if mi < len(r['errors'])]
        errs = [r['errors'][mi] for r in results if mi < len(r['errors'])]
        if errs:
            axes[1].loglog(Ns, errs, 'o-', ms=5,
                           label=f'Mode {mi} ({ref_labels[mi][:15]})')
    axes[1].set(xlabel='N_dof', ylabel='Relative error',
                title='Eigenvalue convergence vs mesh size')
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    # Convergence: error vs h for mode 1 (the singular one)
    hs = [r['h'] for r in results if len(r['errors']) > 1]
    e1 = [r['errors'][1] for r in results if len(r['errors']) > 1]
    e2 = [r['errors'][2] for r in results if len(r['errors']) > 2]
    if hs and e1:
        axes[2].loglog(hs, e1, 'ro-', ms=6, label='Mode 1 (singular)')
    if hs and e2:
        axes[2].loglog(hs, e2, 'bs-', ms=6, label='Mode 2 (regular)')
    # Reference slopes
    if len(hs) >= 2:
        h_ref = np.array(hs)
        axes[2].loglog(h_ref, e1[0] * (h_ref / h_ref[0])**2, 'k--',
                        alpha=0.3, label='O(h^2)')
        axes[2].loglog(h_ref, e1[0] * (h_ref / h_ref[0])**(2*P), 'k:',
                        alpha=0.3, label=f'O(h^{2*P})')
    axes[2].set(xlabel='Element size h', ylabel='Relative error',
                title='Convergence rate')
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('eigenfreq_test2_lshape.png')

    return results


# ==================================================================
# TEST 3: Mesh convergence — impulse response
# ==================================================================

def test3_convergence():
    print("\n" + "=" * 70)
    print("TEST 3: Mesh convergence (impulse response)")
    print("=" * 70)

    from room_acoustics.geometry import l_shaped_room
    from room_acoustics.solvers import fom_pphi

    geom = l_shaped_room(4.0, 3.0, 2.0, 1.5)
    P = 4
    h_values = [0.5, 0.3, 0.2, 0.12]
    src_x, src_y = 1.0, 1.0
    sigma = 0.3
    T = 0.02

    irs = []
    Ns = []

    for h in h_values:
        raw = generate_quad_mesh(geom, h, P)
        mesh = UnstructuredQuadMesh2D(raw['nodes'], raw['quads'],
                                       raw['boundary'], P)
        ops = assemble_unstructured_2d_operators(mesh)
        rec_idx = mesh.nearest_node(3.0, 0.5)

        dt = 0.15 * h / (C_AIR * P**2)
        dt = round(dt, 8)

        res = fom_pphi(mesh, ops, 'PR', {}, src_x, src_y, sigma,
                       dt, T, rec_idx=rec_idx)
        ir = res['ir']
        t_vec = np.arange(len(ir)) * dt
        irs.append((t_vec, ir, mesh.N_dof, h, dt))
        Ns.append(mesh.N_dof)
        print(f"  h={h:.2f}: N={mesh.N_dof}, dt={dt:.2e}, Nt={len(ir)-1}")

    # Compare each resolution against the finest
    t_ref, ir_ref, N_ref, _, _ = irs[-1]
    print(f"\n  Reference: N={N_ref} (h={h_values[-1]})")

    errors = []
    for t_vec, ir, N, h, dt in irs[:-1]:
        # Interpolate to reference time axis (truncate to common length)
        t_common = np.linspace(0, min(t_vec[-1], t_ref[-1]), 2000)
        ir_interp = np.interp(t_common, t_vec, ir)
        ir_ref_interp = np.interp(t_common, t_ref, ir_ref)
        err = np.max(np.abs(ir_interp - ir_ref_interp))
        peak = np.max(np.abs(ir_ref))
        rel = err / peak
        errors.append(rel)
        print(f"  h={h:.2f} (N={N:5d}): L-inf error = {err:.2e}, "
              f"relative = {rel:.2e}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for t_vec, ir, N, h, dt in irs:
        axes[0].plot(t_vec * 1e3, ir, lw=0.8, label=f'N={N}')
    axes[0].set(xlabel='Time [ms]', ylabel='Pressure [Pa]',
                title='Impulse response at different resolutions')
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    # Error vs finest
    t_common = np.linspace(0, min(t_ref[-1], min(t[0][-1] for t in irs[:-1])), 2000)
    ir_ref_common = np.interp(t_common, t_ref, ir_ref)
    for i, (t_vec, ir, N, h, dt) in enumerate(irs[:-1]):
        ir_interp = np.interp(t_common, t_vec, ir)
        axes[1].plot(t_common * 1e3, ir_interp - ir_ref_common, lw=0.8,
                     label=f'N={N} - N={N_ref}')
    axes[1].set(xlabel='Time [ms]', ylabel='Error [Pa]',
                title='Difference vs finest mesh')
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    # Convergence rate
    h_coarse = [h for _, _, _, h, _ in irs[:-1]]
    if errors:
        axes[2].loglog(h_coarse, errors, 'ro-', ms=6, label='L-inf rel error')
        # Reference slope
        h_arr = np.array(h_coarse)
        axes[2].loglog(h_arr, errors[0] * (h_arr / h_arr[0])**(2*P),
                        'k--', alpha=0.3, label=f'O(h^{2*P})')
        axes[2].loglog(h_arr, errors[0] * (h_arr / h_arr[0])**2,
                        'k:', alpha=0.3, label='O(h^2)')
    axes[2].set(xlabel='Element size h', ylabel='Relative error',
                title='Mesh convergence rate')
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('eigenfreq_test3_convergence.png')


# ==================================================================
# Main
# ==================================================================

def main():
    print("=" * 70)
    print("  Eigenfrequency Validation Suite")
    print("  Tests against analytical solutions and published benchmarks")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_start = time.perf_counter()

    test1_rectangular()
    test2_lshape()
    test3_convergence()

    print(f"\n  Total time: {time.perf_counter() - t_start:.1f}s")
    print(f"  Plots saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
