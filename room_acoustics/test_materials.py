#!/usr/bin/env python
"""Test case: concert hall with mixed materials + full verification."""

import sys, os, time
sys.modules.setdefault('cupy', None)
sys.modules.setdefault('cupyx', None)
sys.modules.setdefault('cupyx.scipy', None)
sys.modules.setdefault('cupyx.scipy.sparse', None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from room_acoustics.geometry import l_shaped_room
from room_acoustics.gmsh_tet_import import generate_tet_mesh
from room_acoustics.tet_sem import TetMesh3D, assemble_tet_3d_operators
from room_acoustics.materials import assign_materials, MATERIALS
from room_acoustics.solvers import (
    fom_pphi_3d_gpu, rom_pphi_3d, build_psd_basis, C_AIR, RHO_AIR,
)
from room_acoustics.results_io import save_result

print("=" * 60)
print("  Concert Hall Test Case - Mixed Materials")
print("=" * 60)

# Build mesh
geom = l_shaped_room(8.0, 6.0, 4.0, 3.0)
data = generate_tet_mesh(geom, Lz=4.0, h_target=0.4, P=2)
mesh = TetMesh3D(data['nodes'], data['tets'], data['boundary'])
ops = assemble_tet_3d_operators(mesh)

vol = ops['M_diag'].sum()
surf = ops['B_total'].diagonal().sum()
vol_expected = (8*6 - 4*3) * 4.0
labels = list(mesh._boundary_nodes_per_label.keys())
print(f"Mesh: N={mesh.N_dof}, {mesh.N_el} tets")
print(f"Volume: {vol:.1f} m^3 (expected {vol_expected})")
print(f"Surface: {surf:.1f} m^2")
print(f"Labels: {labels}")

# Setup
src = (2.0, 2.0, 2.0)
rec = mesh.nearest_node(6.0, 1.0, 1.5)

min_h = 1e10
for e in range(min(300, mesh.N_el)):
    verts = mesh.nodes[mesh.elem_conn[e, :4]]
    for i in range(4):
        for j in range(i+1, 4):
            min_h = min(min_h, np.linalg.norm(verts[i]-verts[j]))
dt = round(0.1 * min_h / (C_AIR * 4), 8)
T = 0.06
snap_stride = max(1, int(0.002/dt))
print(f"dt={dt:.2e}, Nt={int(T/dt)}, snap_stride={snap_stride}")
print(f"Source: {src}")
print(f"Receiver: ({mesh.x[rec]:.1f},{mesh.y[rec]:.1f},{mesh.z[rec]:.1f})")

# 4 configurations
configs = [
    ("rigid", "PR", {}, "All walls rigid (PR)"),
    ("uniform_concrete", "FI", {"Z": 50000}, "All walls concrete (Z=50000)"),
    ("mixed_concert", "FI",
     {"Z_per_node": assign_materials(mesh, {
         "floor": "wood_floor", "ceiling": "acoustic_panel",
     }, default="plaster")},
     "Floor=wood, ceiling=panel, walls=plaster"),
    ("mixed_absorptive", "FI",
     {"Z_per_node": assign_materials(mesh, {
         "floor": "carpet_thick", "ceiling": "acoustic_panel_thick",
     }, default="curtain_heavy")},
     "Floor=carpet, ceiling=thick_panel, walls=curtain"),
]

all_results = {}
M = ops['M_diag']; S = ops['S']

for name, bc_type, bc_params, desc in configs:
    print(f"\n--- {name}: {desc} ---")
    t0 = time.perf_counter()
    res = fom_pphi_3d_gpu(mesh, ops, bc_type, bc_params,
                          *src, 0.3, dt, T, rec_idx=rec,
                          store_snapshots=True, snap_stride=snap_stride)
    t_fom = time.perf_counter() - t0

    E = []
    for k in range(len(res['snaps_p'])):
        p = res['snaps_p'][k]; phi = res['snaps_Phi'][k]
        E.append(0.5/(RHO_AIR*C_AIR**2)*np.dot(p, M*p)
                 + 0.5*RHO_AIR*np.dot(phi, S.dot(phi)))
    E = np.array(E)

    result = {"desc": desc, "fom_time_s": t_fom,
              "ir_peak": float(np.max(np.abs(res['ir']))),
              "ir_rms": float(np.sqrt(np.mean(res['ir']**2)))}

    if bc_type == "PR":
        drift = abs(E[-1]-E[0])/E[0]
        result["energy_drift"] = drift
        print(f"  FOM: {t_fom:.1f}s, energy drift={drift:.2e}")
    else:
        ratio = E[-1]/E[0]
        mono = all(E[i+1] <= E[i]*1.001 for i in range(len(E)-1))
        result["energy_ratio"] = ratio
        result["monotonic_decay"] = mono
        print(f"  FOM: {t_fom:.1f}s, E_end/E_start={ratio:.4f}, mono={mono}")

        # ROM
        Psi, _, Nrb = build_psd_basis(res['snaps_p'], res['snaps_Phi'],
                                       eps_pod=1e-6)
        t0 = time.perf_counter()
        _ = fom_pphi_3d_gpu(mesh, ops, bc_type, bc_params,
                            *src, 0.3, dt, T, rec_idx=rec)
        t_fom_clean = time.perf_counter() - t0

        rom_sweep = []
        for nrb in [5, 10, 20]:
            if nrb > Psi.shape[1]:
                break
            t0 = time.perf_counter()
            rr = rom_pphi_3d(mesh, ops, Psi, bc_type, bc_params,
                              *src, 0.3, dt, T, rec_idx=rec,
                              Nrb_override=nrb)
            t_rom = time.perf_counter() - t0
            err = np.max(np.abs(res['ir'] - rr['ir']))
            sp = t_fom_clean / t_rom
            rom_sweep.append({"Nrb": nrb, "speedup": float(sp),
                             "error_Linf": float(err)})
            print(f"    ROM Nrb={nrb}: {sp:.0f}x, err={err:.2e}")
        result["rom_sweep"] = rom_sweep
        result["Nrb_auto"] = Nrb

    all_results[name] = result

# Eigenfrequency check
print("\n--- Eigenfrequency check ---")
M_sp = diags(ops['M_diag'])
eigs, _ = eigsh(ops['S'], k=8, M=M_sp, sigma=0, which='LM')
eigs = np.sort(eigs)
freqs = [np.sqrt(max(e,0))*C_AIR/(2*np.pi) if e > 0 else 0 for e in eigs]

print(f"Expected lowest (8m side): ~{C_AIR/(2*8):.0f} Hz")
for i, f in enumerate(freqs):
    print(f"  mode {i}: {f:.1f} Hz")

# Verification summary
print("\n" + "=" * 60)
print("  VERIFICATION SUMMARY")
print("=" * 60)

checks = []

# 1. Volume
v_ok = abs(vol - vol_expected) < 0.1
checks.append(("Volume", v_ok, f"{vol:.1f} (expected {vol_expected})"))

# 2. PR energy conservation
d = all_results["rigid"]["energy_drift"]
e_ok = d < 1e-3
checks.append(("PR energy drift", e_ok, f"{d:.2e}"))

# 3. FI energy decay monotonic
for name in ["uniform_concrete", "mixed_concert", "mixed_absorptive"]:
    m = all_results[name]["monotonic_decay"]
    checks.append((f"{name} monotonic", m, str(m)))

# 4. More absorption = more decay
e_concrete = all_results["uniform_concrete"]["energy_ratio"]
e_concert = all_results["mixed_concert"]["energy_ratio"]
e_absorb = all_results["mixed_absorptive"]["energy_ratio"]
order_ok = e_concrete > e_concert > e_absorb
checks.append(("Absorption ordering", order_ok,
               f"concrete={e_concrete:.3f} > concert={e_concert:.3f} > absorptive={e_absorb:.3f}"))

# 5. Eigenfrequencies plausible
f1_ok = 10 < freqs[1] < 40
checks.append(("Mode 1 plausible", f1_ok, f"{freqs[1]:.1f} Hz"))

# 6. ROM works for all configs
for name in ["uniform_concrete", "mixed_concert", "mixed_absorptive"]:
    if "rom_sweep" in all_results[name]:
        best = all_results[name]["rom_sweep"][-1]
        r_ok = best["error_Linf"] < 0.1
        checks.append((f"{name} ROM", r_ok,
                       f"err={best['error_Linf']:.2e}, sp={best['speedup']:.0f}x"))

all_pass = all(c[1] for c in checks)
for label, ok, detail in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: {label} — {detail}")
print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

# Save
save_result("concert_hall_materials_test", {
    "mesh": {"N_dof": mesh.N_dof, "N_el": mesh.N_el, "h": 0.4, "P": 2,
             "volume_m3": float(vol), "surface_m2": float(surf),
             "volume_expected": vol_expected, "labels": labels},
    "source": src,
    "receiver": [float(mesh.x[rec]), float(mesh.y[rec]), float(mesh.z[rec])],
    "dt": dt, "T": T,
    "eigenfrequencies_Hz": freqs,
    "configs": all_results,
    "verification_passed": all_pass,
    "checks": [{"name": c[0], "pass": c[1], "detail": c[2]} for c in checks],
}, suite="material_test")
