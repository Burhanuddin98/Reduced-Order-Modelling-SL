"""
Structured data export for validation results.

Every validation test saves a JSON file alongside its PNG plot,
containing all numerical results needed to audit the simulation
without rerunning it.
"""

import json
import os
import numpy as np
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'results')


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_result(name, data, suite=None):
    """Save a validation result as JSON.

    Parameters
    ----------
    name : str
        Result filename (without extension), e.g. 'eigenfreq_test1_rect'
    data : dict
        All numerical results. Will be augmented with metadata.
    suite : str, optional
        Suite name for grouping, e.g. 'eigenfrequency', '2d_validation'
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    record = {
        'metadata': {
            'name': name,
            'suite': suite or 'unknown',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.0',
        },
        'results': data,
    }

    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2, cls=_NumpyEncoder)
    print(f"    data -> {path}")
    return path


def load_result(name):
    """Load a saved validation result."""
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path) as f:
        return json.load(f)
