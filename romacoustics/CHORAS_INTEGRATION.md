# CHORAS Integration Guide

## What is this?

`romacoustics` is a Laplace-domain Reduced Basis Method solver for room acoustics. It computes impulse responses with parametric boundary conditions at 100-10000x speedup over full-order wave solvers.

This document describes how to integrate it as a **CHORAS solver backend**.

## CHORAS Architecture

CHORAS uses:
- **Celery** task queue for async simulation dispatch
- **Docker** containers for each solver backend
- Each solver registers a configuration schema (JSON)

## Integration

### 1. Celery Task

```python
# choras_romacoustics/tasks.py
from celery import Celery
from romacoustics import Room

app = Celery('romacoustics')

@app.task
def compute_ir(config):
    """CHORAS-compatible task.

    config = {
        'geometry': {'type': 'box', 'dimensions': [Lx, Ly, Lz]},
        'source': {'position': [x, y, z], 'sigma': 0.2},
        'receiver': {'position': [x, y, z]},
        'boundary': {
            'type': 'frequency_independent',  # or 'frequency_dependent'
            'impedance': 5000,                 # for FI
            'flow_resistivity': 10000,         # for FD
            'thickness': 0.05,                 # for FD
        },
        'simulation': {
            't_max': 0.1,
            'fs': 44100,
        }
    }
    """
    dims = config['geometry']['dimensions']
    if len(dims) == 2:
        room = Room.box_2d(dims[0], dims[1])
    else:
        room = Room.box_3d(dims[0], dims[1], dims[2])

    src = config['source']['position']
    room.set_source(*src, sigma=config['source'].get('sigma', 0.2))
    room.set_receiver(*config['receiver']['position'])

    bc = config['boundary']
    if bc['type'] == 'frequency_independent':
        room.set_boundary_fi(Zs=bc['impedance'])
    elif bc['type'] == 'frequency_dependent':
        room.set_boundary_fd(sigma_flow=bc['flow_resistivity'],
                             d_mat=bc['thickness'])

    sim = config['simulation']
    ir = room.solve(t_max=sim['t_max'], fs=sim['fs'])

    return {
        'impulse_response': ir.signal.tolist(),
        'fs': ir.fs,
        'metrics': {
            'T30': ir.T30,
            'T20': ir.T20,
            'EDT': ir.EDT,
            'C80': ir.C80,
            'D50': ir.D50,
        }
    }
```

### 2. Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -e . celery redis
CMD ["celery", "-A", "choras_romacoustics.tasks", "worker", "--loglevel=info"]
```

### 3. Configuration Schema

```json
{
    "solver_name": "romacoustics",
    "version": "0.1.0",
    "description": "Laplace-domain ROM for parametric room acoustics",
    "parameters": {
        "geometry": {
            "type": "object",
            "properties": {
                "type": {"enum": ["box"]},
                "dimensions": {"type": "array", "items": {"type": "number"}}
            }
        },
        "boundary": {
            "type": "object",
            "properties": {
                "type": {"enum": ["frequency_independent", "frequency_dependent"]},
                "impedance": {"type": "number"},
                "flow_resistivity": {"type": "number"},
                "thickness": {"type": "number"}
            }
        }
    },
    "outputs": ["impulse_response", "T30", "T20", "EDT", "C80", "D50"]
}
```

## ROM Mode (Parametric)

For CHORAS's parametric design workflow (testing many material configurations):

```python
@app.task
def build_rom(config, training_params):
    """Build ROM once, return serialized ROM."""
    room = _setup_room(config)
    rom = room.build_rom(**training_params)
    # Serialize ROM operators for storage
    return serialize_rom(rom)

@app.task
def query_rom(rom_data, query_param):
    """Query pre-built ROM — instant response."""
    rom = deserialize_rom(rom_data)
    ir = rom.solve(**query_param)
    return {'impulse_response': ir.signal.tolist(), ...}
```

## Limitations

- Box geometries only (no arbitrary meshes yet)
- Uniform boundary conditions (same material on all surfaces)
- No source directivity
- Gaussian pulse excitation only

## Contact

Burhanuddin Sakarwala — https://github.com/Burhanuddin98
