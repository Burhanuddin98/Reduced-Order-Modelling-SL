<p align="center">
  <img src="logo.png" alt="romacoustics" width="400">
</p>

# romacoustics

Laplace-domain Reduced Basis Method for room acoustic impulse response computation. Based on:

> Sampedro Llopis et al. (2022), "Reduced basis methods for numerical room acoustic simulations with parametrized boundaries", JASA 152(2), pp. 851-865. [DOI: 10.1121/10.0012696](https://doi.org/10.1121/10.0012696)

Pure Python. No compiled code. Optional GPU acceleration via CuPy.

---

## What it does

Computes room impulse responses by solving the wave equation in the Laplace domain using finite elements. Supports parametric reduced-order models for fast material sweeps.

```
Geometry (box / STL / OBJ / Gmsh)
    |
    v
Tet mesh (Gmsh, P1 elements)
    |
    v
FEM assembly (M, S, B operators)
    |
    v
Laplace-domain FOM: solve (s^2 M + c^2 S + s Br) p = s M p0
at N_freq complex frequencies s = sigma + i*omega
    |
    v
Time reconstruction (IFFT with Laplace shift correction)
    |
    v
Impulse response -> T30, EDT, C80, D50 (ISO 3382)
```

For parametric studies (testing many materials on the same room):

```
FOM solves at 3 training impedances (offline, minutes)
    |
    v
SVD of snapshot matrix -> reduced basis (15-20 vectors)
    |
    v
Project operators onto basis -> Nrb x Nrb dense system
    |
    v
Online: solve Nrb x Nrb system per frequency (milliseconds)
```

---

## Install

```bash
git clone https://github.com/Burhanuddin98/Reduced-Order-Modelling-SL.git
cd Reduced-Order-Modelling-SL/romacoustics
pip install -e .
```

For arbitrary geometry support (STL/OBJ/Gmsh files):
```bash
pip install gmsh
```

Requirements: Python 3.8+, numpy, scipy, matplotlib.

---

## Quick start

### 2D rectangular room

```python
from romacoustics import Room

room = Room.box_2d(2.0, 2.0, ne=20, order=4)   # 6561 DOFs
room.set_source(1.0, 1.0, sigma=0.2)
room.set_receiver(0.2, 0.2)
room.set_boundary_fi(Zs=5000)

ir = room.solve(t_max=0.1)

print(f'T30 = {ir.T30:.3f}s')
print(f'C80 = {ir.C80:.1f} dB')

ir.to_wav('room_impulse.wav')
ir.plot()
```

### 3D box room with per-surface materials

```python
room = Room.box_3d(4.0, 3.0, 2.5, ne=6, order=4)
room.set_source(1.0, 1.0, 1.2)
room.set_receiver(3.0, 2.0, 1.0)
room.set_material('floor', 'carpet_thick')
room.set_material('ceiling', 'acoustic_panel')
room.set_material('x_min', 'plaster')
room.set_material('x_max', 'plaster')

ir = room.solve(t_max=0.3, f_max=300, Ns=200)
```

### Arbitrary geometry

Requires `gmsh`. Input must be a closed (watertight) surface.

```python
# From Gmsh .geo file
room = Room.from_gmsh('L_shaped_room.geo', f_max=200)

# From OBJ or STL (must be watertight)
room = Room.from_obj('room.obj', f_max=200)
room = Room.from_stl('room.stl', f_max=200)

# Surfaces are auto-detected by face normal direction
# Typical labels: floor, ceiling, wall_north, wall_south, wall_east, wall_west
room.set_material('floor', 'carpet_thick')
room.set_material('ceiling', 'plaster')

ir = room.solve(t_max=0.3, f_max=200, Ns=150)
```

The interior is meshed with P1 tetrahedra. Mesh resolution is set by `f_max` at 6 points per wavelength.

### Parametric ROM

Build the reduced model once from a few training solves, then query at any impedance value:

```python
room = Room.box_2d(2.0, 2.0, ne=15, order=4)
room.set_source(1.0, 1.0, sigma=0.2)
room.set_receiver(0.2, 0.2)
room.set_boundary_fi(Zs=5000)

# Offline: 3 FOM solves
rom = room.build_rom(Z_train=[500, 8000, 15500])

# Online: milliseconds each
ir1 = rom.solve(Zs=3000)
ir2 = rom.solve(Zs=12000)
```

ROM is currently supported for uniform impedance sweeps only.

---

## Validation

### FOM accuracy

| Test | Reference | Error |
|------|-----------|-------|
| Eigenfrequencies (2D rigid rectangle) | Analytical | 9/10 within 2.3 Hz |
| Laplace vs time-domain (FI boundary) | RK4 p-Phi solver | 6.85e-4 relative |
| Laplace vs time-domain (rigid) | Three-way comparison | 2.49e-2 relative |
| BRAS Scene 9 eigenfrequencies (3D) | Measured RIRs | 15/15 match |

### ROM accuracy (uniform impedance)

| Case | FOM DOFs | ROM basis | Error | Speedup |
|------|----------|-----------|-------|---------|
| 2D, Zs=5000 | 6,561 | 17 | 0.5% | 9,493x |
| 2D, Zs=15000 | 6,561 | 17 | 0.6% | 11,557x |
| 3D, d=0.05m | 35,937 | 16 | 0.8% | 6,750x |
| 3D, d=0.15m | 35,937 | 16 | 0.8% | 9,202x |

### Arbitrary geometry

| Test | DOFs | Surfaces | Solve time | Result |
|------|------|----------|------------|--------|
| L-shaped room (.geo) | 3,662 | 6 auto-detected | 75s | T30=0.204s |
| BRAS CR2 room (.obj) | 6,697 | 7 auto-detected | 289s | T30=0.326s |

---

## Limitations

- **Frequency range**: limited by mesh resolution. At f_max=500 Hz in 3D, expect ~35K DOFs and several minutes of solve time. Higher frequencies require finer meshes and proportionally longer solves.
- **Arbitrary geometry**: STL and OBJ files must be watertight (closed, manifold). Non-watertight meshes will fail during volume meshing.
- **Parametric ROM**: currently supports uniform impedance sweeps only. Per-surface parametric ROM is not yet implemented.
- **Source model**: Gaussian pulse only. No source directivity.
- **Boundary model**: locally-reacting impedance (frequency-independent or Miki model). No extended reaction or structural coupling.

---

## API reference

### Room

| Method | Description |
|--------|-------------|
| `Room.box_2d(Lx, Ly, ne, order)` | 2D rectangular room (structured SEM) |
| `Room.box_3d(Lx, Ly, Lz, ne, order)` | 3D box room (structured SEM) |
| `Room.from_gmsh(path, f_max)` | 3D room from Gmsh .geo/.msh file |
| `Room.from_stl(path, f_max)` | 3D room from STL file (watertight) |
| `Room.from_obj(path, f_max)` | 3D room from OBJ file (watertight) |
| `.set_source(*pos, sigma)` | Gaussian pulse source position |
| `.set_receiver(*pos)` | Receiver position |
| `.set_boundary_fi(Zs)` | Uniform frequency-independent impedance |
| `.set_boundary_fd(sigma_flow, d_mat)` | Uniform Miki porous absorber |
| `.set_material(surface, name_or_Z)` | Per-surface material (by name or impedance) |
| `.solve(t_max, fs, f_max, Ns)` | Compute impulse response |
| `.build_rom(Z_train=)` | Build parametric ROM (uniform impedance only) |

### ImpulseResponse

| Property / Method | Description |
|-------------------|-------------|
| `.signal`, `.t`, `.fs` | Time-domain data |
| `.T30`, `.T20`, `.EDT` | Reverberation times (s) |
| `.C80`, `.D50` | Clarity (dB), definition |
| `.to_wav(path)` | Export 16-bit WAV |
| `.to_npz(path)` | Export numpy archive |
| `.plot()` | Waveform + energy decay curve |

### Materials

22 built-in materials. Use `list_materials()` to see all options.

```python
from romacoustics import list_materials
list_materials()
```

---

## How it works

1. **Spatial discretization**: Spectral Element Method (box rooms, GLL quadrature, P=4) or P1 tetrahedral FEM (arbitrary geometry via Gmsh). Box rooms use Kronecker product assembly for a diagonal mass matrix.

2. **Laplace-domain formulation**: the wave equation is transformed to complex frequency s = sigma + i*omega. The positive real shift sigma regularises resonances. One sparse linear solve per frequency.

3. **Time reconstruction**: IFFT of the one-sided transfer function with exponential correction for the Laplace shift.

4. **Reduced basis (optional)**: SVD of the solution snapshot matrix across training parameters. Galerkin projection reduces the system to a dense Nrb x Nrb problem.

---

## CHORAS integration

See [CHORAS_INTEGRATION.md](CHORAS_INTEGRATION.md) for solver backend configuration:
- Celery task wrapper
- Dockerfile
- JSON configuration schema (supports box, STL, OBJ, Gmsh geometry types)

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
