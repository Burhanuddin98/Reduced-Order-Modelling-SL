# romacoustics

Open-source Reduced Order Modelling for Room Acoustics.

Implements the Laplace-domain Reduced Basis Method from:

> Sampedro Llopis et al. (2022), "Reduced basis methods for numerical room acoustic simulations with parametrized boundaries", *JASA* 152(2), pp. 851-865.

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`.

## Quick Start

```python
from romacoustics import Room

# Create a 2D room (2m x 2m)
room = Room.box_2d(2.0, 2.0, ne=20, order=4)
room.set_source(1.0, 1.0, sigma=0.2)
room.set_receiver(0.2, 0.2)
room.set_boundary_fi(Zs=5000)

# Full-order solve → impulse response
ir = room.solve(t_max=0.1)
print(ir.T30, ir.C80)  # ISO 3382 metrics
ir.to_wav('output.wav')
ir.plot()
```

## Parametric ROM (100-10000x speedup)

Build the ROM once from training data, then query instantly:

```python
# Build ROM from 3 training impedances (~10 min)
rom = room.build_rom(Z_train=[500, 8000, 15500])

# Query at ANY impedance (instant)
ir1 = rom.solve(Zs=3000)   # ~0.03s
ir2 = rom.solve(Zs=12000)  # ~0.03s
ir3 = rom.solve(Zs=7777)   # ~0.03s
```

## 3D with frequency-dependent boundaries

```python
room = Room.box_3d(1.0, 1.0, 1.0, ne=8, order=4)
room.set_source(0.5, 0.5, 0.5, sigma=0.2)
room.set_receiver(0.25, 0.1, 0.8)
room.set_boundary_fd(sigma_flow=10000, d_mat=0.05)

ir = room.solve(t_max=0.1)
ir.plot_spectrogram()

# Parametric ROM over material thickness
rom = room.build_rom(d_train=[0.02, 0.12, 0.22])
ir2 = rom.solve(d_mat=0.07)
```

## Validated Results

FOM validated against:
- Analytical eigenfrequencies (rigid rect): 9/10 peaks match within 2.3 Hz
- Time-domain RK4 solver: relative error 6.85e-4

ROM validated against FOM:
- 2D (N=6561): Nrb=17, speedup 9493x, relative error 0.5%
- 3D (N=35937): Nrb=16, speedup 6750x, relative error 0.8%

## API

### `Room`
- `Room.box_2d(Lx, Ly, ne=20, order=4)` — 2D rectangular room
- `Room.box_3d(Lx, Ly, Lz, ne=8, order=4)` — 3D box room
- `.set_source(*pos, sigma=0.2)` — Gaussian pulse source
- `.set_receiver(*pos)` — receiver position
- `.set_boundary_fi(Zs)` — frequency-independent impedance
- `.set_boundary_fd(sigma_flow, d_mat)` — Miki porous absorber
- `.solve(t_max, fs, Ns)` → `ImpulseResponse`
- `.build_rom(Z_train or d_train)` → `ROM`

### `ROM`
- `.solve(Zs=... or d_mat=...)` → `ImpulseResponse`

### `ImpulseResponse`
- `.signal`, `.t`, `.fs` — raw data
- `.T30`, `.T20`, `.EDT`, `.C80`, `.D50` — ISO 3382 metrics
- `.edc_db` — energy decay curve
- `.to_wav(path)`, `.to_npz(path)` — export
- `.plot()`, `.plot_spectrogram()` — visualization

## Method

1. **SEM mesh** — Gauss-Lobatto-Legendre spectral elements (Kronecker assembly)
2. **Laplace-domain FOM** — solve `(s²M + c²S + s·Br)p = s·M·p0` at complex frequencies
3. **Weeks ILT** — Laguerre polynomial expansion to reconstruct time-domain IR
4. **SVD basis** — cotangent-lift snapshot matrix → truncated SVD
5. **ROM projection** — Galerkin projection onto reduced basis (Nrb << N)
6. **Online query** — dense Nrb × Nrb solve per frequency (instant)

## License

MIT
