"""
CHORAS solver backend interface for romacoustics.

Entry point: rom_method(json_file_path)

Reads the CHORAS simulation JSON, runs Laplace-domain FOM or ROM,
writes IR + ISO 3382 metrics back into the same JSON file.

Integration:
  1. Add romacoustics as submodule in CHORAS backend/
  2. Import rom_method in simulation_backend/__init__.py
  3. Add TaskType.ROM to app/types/Task.py
  4. Add case branch in simulation_service.py:run_solver()
"""

import json
import os
import numpy as np
import time

from romacoustics.solver import (
    C_AIR, RHO_AIR,
    weeks_s_values, laplace_to_ir,
    sweep_fi, sweep_fd,
    sweep_fi_fullfield, sweep_fd_fullfield,
    build_basis, project_operators, rom_sweep_fi,
    miki_impedance,
)
from romacoustics.ir import ImpulseResponse
from romacoustics.materials import MATERIALS, absorption_to_impedance
from romacoustics.metrics import octave_band_metrics


def rom_method(json_file_path=None):
    """CHORAS-compatible solver entry point.

    Reads simulation config from JSON, runs solver, writes results back.
    """
    if json_file_path is None:
        raise ValueError('json_file_path required')

    # ── 1. Read config ───────────────────────────────────────
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    geo_path = data.get('geo_path', '')
    msh_path = data.get('msh_path', '')
    absorption = data.get('absorption_coefficients', {})
    settings = data.get('simulationSettings', {})
    results_block = data['results'][0]
    source = [results_block['sourceX'],
              results_block['sourceY'],
              results_block['sourceZ']]
    receivers = results_block['responses']
    freq_bands = results_block.get('frequencies', [125, 250, 500, 1000, 2000])

    # ROM-specific settings (with defaults)
    f_max = float(settings.get('rom_f_max', 1000))
    t_max = float(settings.get('rom_ir_length', 1.0))
    poly_order = int(settings.get('rom_poly_order', 4))
    sigma_src = float(settings.get('rom_source_sigma', 0.2))
    fs = int(settings.get('rom_sample_rate', 44100))
    Ns = int(settings.get('rom_num_frequencies', 1000))
    sigma_w = float(settings.get('rom_weeks_sigma', 10.0))
    b_w = float(settings.get('rom_weeks_b', 1000.0))

    _update_progress(data, json_file_path, 5)

    # ── 2. Build mesh ────────────────────────────────────────
    mesh, ops, surface_labels = _build_mesh(geo_path, msh_path, f_max,
                                             poly_order)
    N = mesh.N_dof
    _update_progress(data, json_file_path, 15)

    # ── 3. Parse materials ───────────────────────────────────
    # CHORAS gives absorption per surface as comma-separated string
    # at octave bands. We convert to impedance per surface.
    Z_per_surface = {}
    for label in surface_labels:
        if label in absorption:
            alpha_str = absorption[label]
            if isinstance(alpha_str, str):
                alphas = [float(x.strip()) for x in alpha_str.split(',')]
            else:
                alphas = [float(x) for x in alpha_str]
            # Average absorption across bands → single impedance
            alpha_mean = np.mean(alphas)
            Z_per_surface[label] = absorption_to_impedance(alpha_mean)
        else:
            Z_per_surface[label] = 50000.0  # default: mostly reflective

    # ── 4. Assemble per-surface boundary mass ────────────────
    B_labels = ops.get('B_labels', {})
    if not B_labels:
        # Fallback: single uniform boundary
        B_labels = {'all': np.array(ops['B_total'].diagonal())}
        Z_uniform = np.mean(list(Z_per_surface.values())) if Z_per_surface else 50000.0
        Z_per_surface = {'all': Z_uniform}

    M_diag = ops['M_diag']
    c2S = (C_AIR**2 * ops['S']).tocsc()
    B_total_diag = np.array(ops['B_total'].diagonal())

    _update_progress(data, json_file_path, 20)

    # ── 5. Source ────────────────────────────────────────────
    p0 = _gaussian_source(mesh, source, sigma_src)

    # ── 6. Solve per receiver ────────────────────────────────
    s_vals, z_safe = weeks_s_values(sigma_w, b_w, Ns)
    t_eval = np.arange(0, t_max, 1.0 / fs)

    n_recv = len(receivers)
    for ri, recv in enumerate(receivers):
        rx, ry, rz = recv['x'], recv['y'], recv['z']
        rec_idx = _nearest_node(mesh, rx, ry, rz)

        # Build combined Br diagonal from per-surface materials
        Br_diag = np.zeros(N)
        for label, B_label in B_labels.items():
            Z = Z_per_surface.get(label, 50000.0)
            Br_diag += C_AIR**2 * RHO_AIR * B_label / Z

        # Laplace sweep
        H = _sweep_with_br(c2S, M_diag, Br_diag, p0, N, s_vals, rec_idx)

        # Weeks ILT
        ir_signal = laplace_to_ir(H, sigma_w, b_w, t_eval)
        ir = ImpulseResponse(ir_signal, fs)

        # ISO 3382 metrics per octave band
        metrics = octave_band_metrics(ir_signal, fs, freq_bands)

        # Write results
        recv['receiverResults'] = ir_signal.tolist()
        recv['parameters']['t30'] = metrics['T30']
        recv['parameters']['t20'] = metrics['T20']
        recv['parameters']['edt'] = metrics['EDT']
        recv['parameters']['c80'] = metrics['C80']
        recv['parameters']['d50'] = metrics['D50']
        recv['parameters']['ts'] = metrics['TS']

        pct = 20 + int(80 * (ri + 1) / n_recv)
        _update_progress(data, json_file_path, pct)

    # ── 7. Write final JSON ──────────────────────────────────
    _update_progress(data, json_file_path, 100)


# ── Internal helpers ─────────────────────────────────────────

def _update_progress(data, path, pct):
    """Update percentage in JSON for CHORAS frontend polling."""
    data['results'][0]['percentage'] = pct
    with open(path, 'w') as f:
        json.dump(data, f)


def _build_mesh(geo_path, msh_path, f_max, poly_order):
    """Build SEM mesh from Gmsh files. Returns (mesh, ops, surface_labels)."""
    try:
        import gmsh
        GMSH = True
    except ImportError:
        GMSH = False

    if GMSH and (geo_path or msh_path):
        return _build_gmsh_mesh(geo_path, msh_path, f_max, poly_order)
    else:
        raise RuntimeError(
            'Gmsh required for CHORAS integration. '
            'Install with: pip install gmsh')


def _build_gmsh_mesh(geo_path, msh_path, f_max, poly_order):
    """Build mesh via Gmsh API."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 0)

    if msh_path and os.path.exists(msh_path):
        gmsh.open(msh_path)
    elif geo_path and os.path.exists(geo_path):
        gmsh.open(geo_path)
        # Compute element size from PPW
        wavelength = C_AIR / f_max
        ppw = 6  # points per wavelength (conservative)
        lc = wavelength / ppw / poly_order
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', lc)
        gmsh.model.mesh.generate(3)
    else:
        raise FileNotFoundError(f'No mesh file found: geo={geo_path}, msh={msh_path}')

    # Extract surface labels from physical groups
    surface_labels = []
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        if dim == 2 and name:  # 2D = surface
            surface_labels.append(name)

    # Get element type
    elem_types, _, _ = gmsh.model.mesh.getElements(3)  # dim=3 for volume

    if not elem_types:
        # Try 2D
        elem_types, _, _ = gmsh.model.mesh.getElements(2)

    # Determine mesh type and build
    if 4 in elem_types:  # tetrahedra
        mesh, ops = _build_tet_mesh(gmsh, poly_order, surface_labels)
    else:
        # Try hex/quad
        mesh, ops = _build_hex_mesh(gmsh, poly_order, surface_labels)

    gmsh.finalize()
    return mesh, ops, surface_labels


def _build_tet_mesh(gmsh_api, poly_order, surface_labels):
    """Build tet mesh from Gmsh model."""
    from romacoustics.unstructured import TetMesh, assemble_tet

    # Extract nodes
    node_tags, coords, _ = gmsh_api.model.mesh.getNodes()
    nodes = coords.reshape(-1, 3)
    # Gmsh tags are 1-based
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    # Extract tet elements
    elem_types, elem_tags, elem_nodes = gmsh_api.model.mesh.getElements(3)
    tet_idx = list(elem_types).index(4) if 4 in elem_types else None
    if tet_idx is None:
        raise RuntimeError('No tetrahedral elements found')

    tet_nodes = elem_nodes[tet_idx].reshape(-1, 4)
    tets = np.array([[tag_to_idx[int(n)] for n in tet] for tet in tet_nodes])

    # Extract boundary faces per surface label
    boundary = {}
    for dim, phys_tag in gmsh_api.model.getPhysicalGroups(2):
        name = gmsh_api.model.getPhysicalName(2, phys_tag)
        if not name:
            continue
        entities = gmsh_api.model.getEntitiesForPhysicalGroup(2, phys_tag)
        faces = []
        for ent in entities:
            _, _, face_nodes = gmsh_api.model.mesh.getElements(2, ent)
            if face_nodes:
                fn = face_nodes[0].reshape(-1, 3)
                for face in fn:
                    faces.append([tag_to_idx[int(n)] for n in face])
        if faces:
            boundary[name] = np.array(faces)

    mesh = TetMesh(nodes, tets, boundary)
    ops = assemble_tet(mesh)
    return mesh, ops


def _build_hex_mesh(gmsh_api, poly_order, surface_labels):
    """Build hex mesh from Gmsh model (fallback)."""
    # For now, raise — hex support needs more work
    raise NotImplementedError(
        'Hex mesh support coming in v0.3. '
        'Use tetrahedral meshing (Gmsh default).')


def _gaussian_source(mesh, pos, sigma):
    """Gaussian pulse at source position."""
    if hasattr(mesh, 'z'):
        r2 = ((mesh.x - pos[0])**2 + (mesh.y - pos[1])**2 +
              (mesh.z - pos[2])**2)
    else:
        r2 = (mesh.x - pos[0])**2 + (mesh.y - pos[1])**2
    return np.exp(-r2 / sigma**2)


def _nearest_node(mesh, rx, ry, rz):
    """Find nearest mesh node to receiver."""
    if hasattr(mesh, 'z'):
        return int(np.argmin(
            (mesh.x - rx)**2 + (mesh.y - ry)**2 + (mesh.z - rz)**2))
    else:
        return int(np.argmin((mesh.x - rx)**2 + (mesh.y - ry)**2))


def _sweep_with_br(c2S, M_diag, Br_diag, p0, N, s_vals, rec_idx):
    """Laplace sweep with pre-assembled Br diagonal."""
    from scipy.sparse.linalg import spsolve
    from scipy import sparse

    Ns = len(s_vals)
    H = np.zeros(Ns, dtype=complex)
    for i, s in enumerate(s_vals):
        sig, omg = s.real, s.imag
        Kr = c2S + sparse.diags(
            (sig**2 - omg**2)*M_diag + sig*Br_diag, format='csc')
        Kc = sparse.diags(
            2*sig*omg*M_diag + omg*Br_diag, format='csc')
        A = sparse.bmat([[Kr, -Kc], [Kc, Kr]], format='csc')
        rhs = np.concatenate([sig*p0*M_diag, omg*p0*M_diag])
        x = spsolve(A, rhs)
        H[i] = x[rec_idx] + 1j*x[N + rec_idx]
    return H
