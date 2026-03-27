"""
Acoustic material database and per-surface material assignment.

Maps human-readable material names to impedance values (FI) or
Miki model parameters (LR) for frequency-dependent absorption.

Usage:
    from room_acoustics.materials import MATERIALS, assign_materials

    # Assign different materials to different walls
    mat_map = {'floor': 'carpet_thick', 'ceiling': 'acoustic_panel',
               'wall_0': 'concrete', 'wall_1': 'glass'}
    Z_per_node = assign_materials(mesh, mat_map)

    # Use in solver
    result = fom_pphi_3d_gpu(mesh, ops, 'FI', {'Z_per_node': Z_per_node}, ...)
"""

import numpy as np


# ===================================================================
# Material database
# ===================================================================

# Each material has:
#   Z    : frequency-independent impedance [N*s/m^3] (for FI mode)
#   sigma: flow resistivity [N*s/m^4] (for Miki/LR mode)
#   d    : material thickness [m] (for Miki/LR mode)
#   desc : human-readable description
#
# Impedance values are approximate mid-frequency equivalents.
# For accurate frequency-dependent simulation, use the LR mode
# with sigma and d parameters.
#
# Reference: rho*c = 1.2 * 343 = 411.6 N*s/m^3 (air impedance)
# A perfectly absorbing surface has Z = rho*c = 412.
# Higher Z = more reflective. Z -> infinity = rigid wall.

MATERIALS = {
    # --- Hard surfaces ---
    'concrete': {
        'Z': 50000, 'sigma': 20000, 'd': 0.15,
        'desc': 'Poured concrete (very reflective)',
    },
    'brick': {
        'Z': 30000, 'sigma': 15000, 'd': 0.12,
        'desc': 'Exposed brick wall',
    },
    'marble': {
        'Z': 80000, 'sigma': 50000, 'd': 0.03,
        'desc': 'Polished marble floor/wall',
    },
    'tile': {
        'Z': 60000, 'sigma': 30000, 'd': 0.01,
        'desc': 'Ceramic tile',
    },
    'glass': {
        'Z': 15000, 'sigma': 10000, 'd': 0.006,
        'desc': 'Window glass (some low-freq absorption)',
    },

    # --- Wood ---
    'wood_floor': {
        'Z': 10000, 'sigma': 8000, 'd': 0.02,
        'desc': 'Hardwood floor on joists',
    },
    'wood_panel': {
        'Z': 5000, 'sigma': 5000, 'd': 0.01,
        'desc': 'Thin wood paneling',
    },
    'plywood': {
        'Z': 8000, 'sigma': 6000, 'd': 0.012,
        'desc': 'Plywood panel',
    },

    # --- Soft/absorptive surfaces ---
    'carpet_thin': {
        'Z': 2000, 'sigma': 10000, 'd': 0.005,
        'desc': 'Thin carpet on concrete',
    },
    'carpet_thick': {
        'Z': 1000, 'sigma': 10000, 'd': 0.015,
        'desc': 'Thick carpet with underlay',
    },
    'curtain_light': {
        'Z': 3000, 'sigma': 8000, 'd': 0.003,
        'desc': 'Light curtain/drape',
    },
    'curtain_heavy': {
        'Z': 1200, 'sigma': 8000, 'd': 0.01,
        'desc': 'Heavy velvet curtain',
    },

    # --- Acoustic treatment ---
    'acoustic_panel': {
        'Z': 800, 'sigma': 12000, 'd': 0.05,
        'desc': 'Acoustic absorption panel (50mm fiberglass)',
    },
    'acoustic_foam': {
        'Z': 600, 'sigma': 10000, 'd': 0.05,
        'desc': 'Open-cell acoustic foam',
    },
    'acoustic_panel_thick': {
        'Z': 500, 'sigma': 12000, 'd': 0.1,
        'desc': 'Thick acoustic panel (100mm)',
    },
    'bass_trap': {
        'Z': 600, 'sigma': 15000, 'd': 0.2,
        'desc': 'Bass trap (200mm absorber in corner)',
    },

    # --- Plaster/drywall ---
    'plaster': {
        'Z': 20000, 'sigma': 15000, 'd': 0.02,
        'desc': 'Plaster on masonry',
    },
    'drywall': {
        'Z': 8000, 'sigma': 5000, 'd': 0.013,
        'desc': 'Gypsum drywall (plasterboard)',
    },

    # --- Seating/audience ---
    'upholstered_seats': {
        'Z': 1500, 'sigma': 10000, 'd': 0.08,
        'desc': 'Upholstered theater/cinema seats',
    },
    'audience_seated': {
        'Z': 800, 'sigma': 8000, 'd': 0.15,
        'desc': 'Seated audience (highly absorptive)',
    },

    # --- Special ---
    'rigid': {
        'Z': 1e12, 'sigma': 1e6, 'd': 1.0,
        'desc': 'Perfectly rigid (PR equivalent)',
    },
    'anechoic': {
        'Z': 412, 'sigma': 5000, 'd': 1.0,
        'desc': 'Perfectly absorbing (Z = rho*c)',
    },
}


def get_material(name):
    """Look up a material by name. Returns dict with Z, sigma, d, desc."""
    if name not in MATERIALS:
        available = ', '.join(sorted(MATERIALS.keys()))
        raise ValueError(f"Unknown material '{name}'. Available: {available}")
    return MATERIALS[name]


def list_materials():
    """Print all available materials."""
    print(f"{'Name':<25s} {'Z':>8s}  {'sigma':>8s}  {'d':>6s}  Description")
    print("-" * 85)
    for name, m in sorted(MATERIALS.items()):
        print(f"{name:<25s} {m['Z']:8.0f}  {m['sigma']:8.0f}  "
              f"{m['d']:6.3f}  {m['desc']}")


# ===================================================================
# Per-surface material assignment
# ===================================================================

def assign_materials(mesh, material_map, default='concrete'):
    """
    Create a per-node impedance vector from surface material assignments.

    Parameters
    ----------
    mesh : TetMesh3D / UnstructuredHexMesh3D / etc.
        Must have boundary_nodes(label) method.
    material_map : dict
        Maps surface label -> material name.
        e.g. {'floor': 'carpet_thick', 'wall_0': 'concrete'}
    default : str
        Default material for unlabeled surfaces.

    Returns
    -------
    Z_per_node : ndarray (N_dof,)
        Impedance at each node. Interior nodes get Z=inf (rigid).
        Boundary nodes get the Z from their assigned material.
    """
    N = mesh.N_dof
    Z_per_node = np.full(N, 1e15)  # interior nodes: effectively rigid

    # Get all boundary labels
    if hasattr(mesh, '_boundary_nodes_per_label'):
        labels = mesh._boundary_nodes_per_label.keys()
    else:
        labels = []

    for label in labels:
        mat_name = material_map.get(label, default)
        mat = get_material(mat_name)
        nodes = mesh.boundary_nodes(label)
        Z_per_node[nodes] = mat['Z']

    return Z_per_node


def assign_miki_params(mesh, material_map, default='concrete'):
    """
    Create per-node Miki parameters for frequency-dependent (LR) boundaries.

    Returns
    -------
    sigma_per_node : ndarray (N_dof,) — flow resistivity
    d_per_node : ndarray (N_dof,) — material thickness
    """
    N = mesh.N_dof
    sigma_per_node = np.full(N, 1e6)  # interior: very high resistivity
    d_per_node = np.full(N, 1.0)

    if hasattr(mesh, '_boundary_nodes_per_label'):
        labels = mesh._boundary_nodes_per_label.keys()
    else:
        labels = []

    for label in labels:
        mat_name = material_map.get(label, default)
        mat = get_material(mat_name)
        nodes = mesh.boundary_nodes(label)
        sigma_per_node[nodes] = mat['sigma']
        d_per_node[nodes] = mat['d']

    return sigma_per_node, d_per_node
