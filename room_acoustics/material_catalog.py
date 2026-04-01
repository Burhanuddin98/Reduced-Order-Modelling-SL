"""
Material catalog — standard acoustic materials with frequency-dependent
absorption and scattering coefficients.

Provides catalog-grade starting values for absorption calibration.
Values from common acoustic reference sources (Mechel, Cox & D'Antonio,
ISO 354 measurements). Stored as MaterialFunction objects for seamless
integration with the simulation pipeline.

Usage:
    from room_acoustics.material_catalog import CATALOG, get_catalog_material

    ceiling_mat = get_catalog_material('acoustic_tile')
    alpha_at_1k = ceiling_mat(1000.0)

    # List all available materials
    list_catalog()
"""

import numpy as np
from .material_function import MaterialFunction


# Standard octave-band center frequencies
OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000]

# Catalog of common acoustic materials
# alpha: absorption coefficients at octave bands [125, 250, 500, 1000, 2000, 4000] Hz
# scatter: scattering coefficients at same bands
CATALOG = {
    # --- Hard surfaces ---
    'concrete_painted': {
        'alpha': [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'desc': 'Painted poured concrete',
    },
    'concrete_rough': {
        'alpha': [0.02, 0.03, 0.03, 0.03, 0.04, 0.07],
        'scatter': [0.10, 0.10, 0.12, 0.15, 0.20, 0.25],
        'desc': 'Rough/unpainted concrete',
    },
    'brick_painted': {
        'alpha': [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
        'scatter': [0.05, 0.05, 0.05, 0.10, 0.10, 0.15],
        'desc': 'Painted brick wall',
    },
    'brick_exposed': {
        'alpha': [0.02, 0.03, 0.03, 0.04, 0.05, 0.07],
        'scatter': [0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
        'desc': 'Exposed brick wall',
    },
    'marble': {
        'alpha': [0.01, 0.01, 0.01, 0.01, 0.02, 0.02],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'desc': 'Polished marble/granite',
    },
    'tile_ceramic': {
        'alpha': [0.01, 0.01, 0.01, 0.01, 0.02, 0.02],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'desc': 'Glazed ceramic tile',
    },

    # --- Glass ---
    'glass_window': {
        'alpha': [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'desc': 'Single-pane window glass',
    },
    'glass_double': {
        'alpha': [0.15, 0.10, 0.07, 0.05, 0.03, 0.02],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'desc': 'Double-pane insulated glass',
    },

    # --- Wood ---
    'wood_floor': {
        'alpha': [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.10, 0.10],
        'desc': 'Hardwood floor on joists',
    },
    'wood_panel': {
        'alpha': [0.42, 0.21, 0.10, 0.08, 0.06, 0.06],
        'scatter': [0.05, 0.05, 0.05, 0.10, 0.10, 0.15],
        'desc': 'Thin wood paneling over air gap',
    },
    'plywood': {
        'alpha': [0.28, 0.22, 0.17, 0.09, 0.10, 0.11],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.10, 0.10],
        'desc': 'Plywood panel',
    },

    # --- Plaster/drywall ---
    'plaster_on_masonry': {
        'alpha': [0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.10, 0.10],
        'desc': 'Plaster on solid masonry',
    },
    'drywall': {
        'alpha': [0.29, 0.10, 0.05, 0.04, 0.07, 0.09],
        'scatter': [0.05, 0.05, 0.05, 0.10, 0.10, 0.15],
        'desc': 'Gypsum drywall on studs',
    },

    # --- Floors ---
    'linoleum': {
        'alpha': [0.02, 0.03, 0.03, 0.03, 0.03, 0.02],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'desc': 'Linoleum on concrete',
    },
    'carpet_thin': {
        'alpha': [0.02, 0.06, 0.14, 0.37, 0.60, 0.65],
        'scatter': [0.05, 0.05, 0.10, 0.20, 0.30, 0.40],
        'desc': 'Thin carpet on concrete',
    },
    'carpet_thick': {
        'alpha': [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],
        'scatter': [0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        'desc': 'Heavy carpet with underlay',
    },
    'parquet': {
        'alpha': [0.04, 0.04, 0.07, 0.06, 0.06, 0.07],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.10, 0.10],
        'desc': 'Parquet floor on concrete',
    },

    # --- Ceiling tiles ---
    'acoustic_tile': {
        'alpha': [0.20, 0.40, 0.70, 0.80, 0.60, 0.40],
        'scatter': [0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        'desc': 'Mineral fiber acoustic tile',
    },
    'acoustic_tile_perforated': {
        'alpha': [0.50, 0.70, 0.60, 0.70, 0.70, 0.50],
        'scatter': [0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
        'desc': 'Perforated metal + mineral wool ceiling',
    },

    # --- Absorbers ---
    'acoustic_panel_50mm': {
        'alpha': [0.10, 0.40, 0.80, 0.90, 0.80, 0.60],
        'scatter': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        'desc': '50mm fiberglass panel',
    },
    'acoustic_panel_100mm': {
        'alpha': [0.30, 0.70, 0.90, 0.95, 0.90, 0.85],
        'scatter': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        'desc': '100mm fiberglass panel',
    },
    'acoustic_foam': {
        'alpha': [0.05, 0.20, 0.50, 0.80, 0.90, 0.85],
        'scatter': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'desc': 'Open-cell acoustic foam',
    },
    'bass_trap': {
        'alpha': [0.60, 0.80, 0.90, 0.80, 0.50, 0.40],
        'scatter': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        'desc': 'Corner bass trap (200mm absorber)',
    },

    # --- Curtains/fabric ---
    'curtain_light': {
        'alpha': [0.03, 0.04, 0.11, 0.17, 0.24, 0.35],
        'scatter': [0.05, 0.05, 0.10, 0.15, 0.20, 0.25],
        'desc': 'Light curtain/drape',
    },
    'curtain_heavy': {
        'alpha': [0.14, 0.35, 0.55, 0.72, 0.70, 0.65],
        'scatter': [0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        'desc': 'Heavy velvet curtain',
    },

    # --- Seating ---
    'seats_upholstered': {
        'alpha': [0.19, 0.37, 0.56, 0.67, 0.61, 0.59],
        'scatter': [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        'desc': 'Upholstered theater seats (unoccupied)',
    },
    'audience_seated': {
        'alpha': [0.39, 0.57, 0.80, 0.94, 0.92, 0.87],
        'scatter': [0.30, 0.40, 0.50, 0.60, 0.70, 0.70],
        'desc': 'Seated audience on upholstered seats',
    },

    # --- Special ---
    'rigid': {
        'alpha': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'desc': 'Perfectly rigid surface',
    },
    'anechoic': {
        'alpha': [0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
        'scatter': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'desc': 'Perfectly absorbing surface',
    },
}


def get_catalog_material(name):
    """Get a MaterialFunction from the catalog by name.

    Parameters
    ----------
    name : str
        Material name (see list_catalog() for available names).

    Returns
    -------
    MaterialFunction
        With octave-band alpha and scattering, interpolated to any frequency.
    """
    if name not in CATALOG:
        available = ', '.join(sorted(CATALOG.keys()))
        raise ValueError(f"Unknown catalog material '{name}'. Available: {available}")

    m = CATALOG[name]
    return MaterialFunction(
        OCTAVE_BANDS, m['alpha'], name=name,
        scattering=m['scatter'])


def list_catalog():
    """Print all available catalog materials."""
    print(f"{'Name':<28s} {'125':>5s} {'250':>5s} {'500':>5s} {'1k':>5s} "
          f"{'2k':>5s} {'4k':>5s}  Description")
    print("-" * 95)
    for name in sorted(CATALOG.keys()):
        m = CATALOG[name]
        a = m['alpha']
        print(f"{name:<28s} {a[0]:5.2f} {a[1]:5.2f} {a[2]:5.2f} "
              f"{a[3]:5.2f} {a[4]:5.2f} {a[5]:5.2f}  {m['desc']}")


def load_json_catalog(path):
    """Load a material catalog from JSON file (modal-clouds format).

    Expected format:
    {
      "materials": {
        "material_name": {
          "alpha": [a1, a2, ...],
          "scatter": [s1, s2, ...]
        }
      },
      "_meta": {"bands_hz": [125, 250, 500, 1000, 2000, 4000]}
    }

    Returns dict of name -> MaterialFunction.
    """
    import json
    with open(path) as f:
        data = json.load(f)

    bands = data.get('_meta', {}).get('bands_hz', OCTAVE_BANDS)
    materials = {}

    for name, props in data.get('materials', {}).items():
        alpha = props.get('alpha', [0.05] * len(bands))
        scatter = props.get('scatter', [0.05] * len(bands))
        materials[name] = MaterialFunction(
            bands, alpha, name=name, scattering=scatter)

    return materials
