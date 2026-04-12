# romacoustics

<p align="center">
  <img src="logo.png" alt="romacoustics" width="300">
</p>

Laplace-domain reduced basis method for computing room impulse responses with parametric boundary conditions.

Based on:
> Sampedro Llopis, H., Engsig-Karup, A.P., Jeong, C.-H., Pind, F., Hesthaven, J.S. (2022).
> *Reduced basis methods for numerical room acoustic simulations with parametrized boundaries*.
> J. Acoust. Soc. Am. 152(2), pp. 851-865.
> [DOI: 10.1121/10.0012696](https://doi.org/10.1121/10.0012696)

Python implementation. No compiled dependencies.

---

## Install

```bash
git clone https://github.com/Burhanuddin98/Reduced-Order-Modelling-SL.git
cd Reduced-Order-Modelling-SL/romacoustics
pip install -e .
```

For arbitrary geometry (STL/OBJ/Gmsh):
```bash
pip install gmsh
```

Requires: Python 3.8+, numpy, scipy, matplotlib.

---

## Overview

The solver discretises the acoustic wave equation using finite elements and solves in the Laplace domain (s = sigma + i*omega). The positive real shift sigma regularises resonances, avoiding near-singular systems that occur with the standard Helmholtz equation at real frequencies.

```
Input: room geometry + surface impedances + source/receiver positions
                            |
                            v
              Finite element mesh (Gmsh)
              P4 hex (box) or P1 tet (arbitrary)
                            |
                            v
              Laplace-domain solve at N frequencies:
              (s^2 M + c^2 S + s Br) p = s M p0
                            |
                            v
              Time reconstruction via IFFT
              with Laplace shift correction
                            |
                            v
              Impulse response + ISO 3382 metrics
              (T30, T20, EDT, C80, D50)
```

For parametric studies, a reduced basis is constructed from SVD of solution snapshots. The online phase solves a small dense system (typically 15-20 unknowns) instead of the full sparse system.

---

## Usage

### Box rooms

```python
from romacoustics import Room

# 2D
room = Room.box_2d(2.0, 2.0, ne=20, order=4)
room.set_source(1.0, 1.0, sigma=0.2)
room.set_receiver(0.2, 0.2)
room.set_boundary_fi(Zs=5000)
ir = room.solve(t_max=0.1)

# 3D with per-surface materials
room = Room.box_3d(4.0, 3.0, 2.5, ne=6, order=4)
room.set_source(1.0, 1.0, 1.2)
room.set_receiver(3.0, 2.0, 1.0)
room.set_material('floor', 'carpet_thick')
room.set_material('ceiling', 'acoustic_panel')
ir = room.solve(t_max=0.3, f_max=300, Ns=200)
```

### Arbitrary geometry

Requires gmsh. Input geometry must be a closed (watertight) surface.

```python
room = Room.from_gmsh('room.geo', f_max=200)
room = Room.from_stl('room.stl', f_max=200)
room = Room.from_obj('room.obj', f_max=200)
```

The interior is meshed with P1 tetrahedra. Mesh resolution is determined by f_max at 6 points per wavelength. Boundary surfaces are grouped automatically by face normal direction (floor, ceiling, wall_north, wall_south, wall_east, wall_west).

### Parametric ROM

Builds a reduced model from a small number of full-order solves at different impedance values. Subsequent evaluations at new impedance values solve only the reduced system.

```python
rom = room.build_rom(Z_train=[500, 8000, 15500])
ir = rom.solve(Zs=3000)
```

Currently limited to uniform impedance variation. Per-surface parametric sweeps are not yet supported.

---

## Validation

### Full-order solver

| Test | Reference | Result |
|------|-----------|--------|
| 2D eigenfrequencies (rigid rectangle) | Analytical | 9/10 within 2.3 Hz |
| Laplace vs time-domain (FI, Z=5000) | RK4 p-Phi solver | Relative error 6.85e-4 |
| Laplace vs time-domain (rigid) | Three-way comparison | Relative error 2.49e-2 |
| 3D BRAS Scene 9 eigenfrequencies | Measured RIR peaks | 15/15 match |

### Reduced-order model

Tested with uniform impedance boundaries.

| Case | FOM DOFs | ROM basis | Relative error |
|------|----------|-----------|---------------|
| 2D, Zs=5000 | 6,561 | 17 | 0.5% |
| 2D, Zs=15000 | 6,561 | 17 | 0.6% |
| 3D FD, d=0.05m | 35,937 | 16 | 0.8% |
| 3D FD, d=0.15m | 35,937 | 16 | 0.8% |

### Arbitrary geometry

| Geometry | DOFs | Surfaces detected | Solve time (150 freqs) |
|----------|------|-------------------|----------------------|
| L-shaped room (.geo) | 3,662 | 6 | 75 s |
| BRAS CR2 seminar room (.obj) | 6,697 | 7 | 289 s |

---

## Limitations

- **Frequency range** is limited by mesh resolution. A 170 m^3 room at f_max=500 Hz requires approximately 35,000 DOFs and several minutes of solve time.
- **STL/OBJ import** requires watertight (closed, manifold) geometry. Non-watertight meshes will fail during volume meshing.
- **Parametric ROM** currently supports uniform impedance sweeps only.
- **Source**: Gaussian pulse excitation only. No directivity.
- **Boundaries**: locally-reacting impedance. No extended reaction or structural coupling.

---

## API

### Room

| Method | Description |
|--------|-------------|
| `Room.box_2d(Lx, Ly, ne, order)` | 2D rectangular room |
| `Room.box_3d(Lx, Ly, Lz, ne, order)` | 3D box room |
| `Room.from_gmsh(path, f_max)` | From Gmsh .geo/.msh file |
| `Room.from_stl(path, f_max)` | From STL (watertight) |
| `Room.from_obj(path, f_max)` | From OBJ (watertight) |
| `.set_source(*pos, sigma)` | Source position |
| `.set_receiver(*pos)` | Receiver position |
| `.set_boundary_fi(Zs)` | Uniform impedance |
| `.set_boundary_fd(sigma_flow, d_mat)` | Miki porous absorber |
| `.set_material(surface, name_or_Z)` | Per-surface material |
| `.solve(t_max, fs, f_max, Ns)` | Compute impulse response |
| `.build_rom(Z_train=)` | Build reduced model |

### ImpulseResponse

| Property / Method | Description |
|-------------------|-------------|
| `.signal`, `.t`, `.fs` | Time-domain data |
| `.T30`, `.T20`, `.EDT` | Reverberation times (s) |
| `.C80`, `.D50` | Clarity (dB), definition |
| `.to_wav(path)` | Export WAV |
| `.to_npz(path)` | Export numpy archive |
| `.plot()` | Waveform + decay curve |

### Materials

22 built-in materials. `list_materials()` prints available options.

---

## CHORAS integration

See [CHORAS_INTEGRATION.md](CHORAS_INTEGRATION.md) for solver backend configuration.

---

## Cite

```bibtex
@article{sampedro2022,
  author  = {Sampedro Llopis, H. and Engsig-Karup, A.P. and Jeong, C.-H. and Pind, F. and Hesthaven, J.S.},
  title   = {Reduced basis methods for numerical room acoustic simulations with parametrized boundaries},
  journal = {J. Acoust. Soc. Am.},
  volume  = {152},
  number  = {2},
  pages   = {851--865},
  year    = {2022},
  doi     = {10.1121/10.0012696}
}
```

## License

MIT
