"""Material database and absorption-to-impedance conversion."""

import numpy as np
from romacoustics.solver import C_AIR, RHO_AIR

# Normal incidence: alpha = 1 - |( Z - rho*c ) / ( Z + rho*c )|^2
# Solving for Z: Z = rho*c * (1 + sqrt(1-alpha)) / sqrt(alpha)
# (real-valued approximation for frequency-independent case)

RHO_C = RHO_AIR * C_AIR  # 411.6 Pa s/m


def absorption_to_impedance(alpha):
    """Convert normal incidence absorption coefficient to impedance.

    Uses the real-valued approximation:
        Z = rho*c * (1 + sqrt(1-alpha)) / sqrt(alpha)

    For alpha=0 (rigid): returns 1e15
    For alpha=1 (anechoic): returns rho*c = 411.6
    """
    alpha = np.clip(float(alpha), 1e-6, 1.0 - 1e-6)
    return RHO_C * (1 + np.sqrt(1 - alpha)) / np.sqrt(alpha)


def impedance_to_absorption(Z):
    """Convert impedance to normal incidence absorption coefficient."""
    r = (Z - RHO_C) / (Z + RHO_C)
    return 1 - abs(r)**2


# 22-material database
# Z values are representative normal-incidence impedances [Pa s/m]
MATERIALS = {
    # Hard surfaces
    'concrete':           {'Z': 80000, 'desc': 'Poured concrete'},
    'brick':              {'Z': 50000, 'desc': 'Exposed brick'},
    'marble':             {'Z': 70000, 'desc': 'Polished marble/stone'},
    'tile':               {'Z': 60000, 'desc': 'Ceramic tile'},
    'glass':              {'Z': 40000, 'desc': 'Window glass'},

    # Wood
    'wood_floor':         {'Z': 10000, 'desc': 'Hardwood floor'},
    'wood_panel':         {'Z':  5000, 'desc': 'Wood paneling on frame'},
    'plywood':            {'Z':  8000, 'desc': 'Thin plywood panel'},

    # Soft / absorptive
    'carpet_thin':        {'Z':  3000, 'desc': 'Thin carpet on concrete'},
    'carpet_thick':       {'Z':  1500, 'desc': 'Heavy carpet on underlay'},
    'curtain_light':      {'Z':  2500, 'desc': 'Light curtain fabric'},
    'curtain_heavy':      {'Z':  1000, 'desc': 'Heavy velour curtain'},

    # Acoustic treatment
    'acoustic_panel':     {'Z':   800, 'desc': 'Acoustic absorber panel'},
    'foam':               {'Z':   600, 'desc': 'Acoustic foam (50mm)'},
    'bass_trap':          {'Z':   500, 'desc': 'Bass trap / thick absorber'},

    # Plaster / drywall
    'plaster':            {'Z': 20000, 'desc': 'Plaster on solid wall'},
    'drywall':            {'Z':  8000, 'desc': 'Drywall / plasterboard'},

    # Seating
    'upholstered_seats':  {'Z':  1500, 'desc': 'Upholstered theater seats'},
    'audience_seated':    {'Z':   800, 'desc': 'Audience in upholstered seats'},

    # Special
    'rigid':              {'Z':  1e12, 'desc': 'Perfectly rigid (testing)'},
    'anechoic':           {'Z': RHO_C, 'desc': 'Perfectly absorbing (rho*c)'},
    'open':               {'Z': RHO_C, 'desc': 'Open boundary'},
}


def get_material(name):
    """Look up material by name. Returns dict with 'Z' and 'desc'."""
    if name not in MATERIALS:
        available = ', '.join(sorted(MATERIALS.keys()))
        raise KeyError(f'Unknown material "{name}". Available: {available}')
    return MATERIALS[name]


def list_materials():
    """Print all available materials."""
    print(f'{"Name":<22s} {"Z [Pa s/m]":>12s}  Description')
    print('-' * 60)
    for name, m in sorted(MATERIALS.items()):
        print(f'{name:<22s} {m["Z"]:>12.0f}  {m["desc"]}')
