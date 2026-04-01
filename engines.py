"""
engines.py  –  22 room-acoustic simulation engines, unified interface.
Each engine: run_xxx(room, src, rec, sr, duration, **kw) -> Result
"""
import numpy as np, time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
OCTAVE_BANDS = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
C_DEFAULT = 343.0
RHO_DEFAULT = 1.225

# ═══════════════════════════════════════════════════════════════════
# Material database  (absorption coefficients at 6 octave bands)
# ═══════════════════════════════════════════════════════════════════
MATERIALS: Dict[str, List[float]] = {
    "concrete":        [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
    "brick":           [0.02, 0.03, 0.03, 0.04, 0.05, 0.07],
    "marble":          [0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
    "glass":           [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],
    "plaster":         [0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
    "gypsum_board":    [0.29, 0.10, 0.05, 0.04, 0.07, 0.09],
    "wood_floor":      [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
    "wood_panel":      [0.28, 0.22, 0.17, 0.09, 0.10, 0.11],
    "parquet":         [0.04, 0.04, 0.07, 0.06, 0.06, 0.07],
    "linoleum":        [0.02, 0.02, 0.03, 0.04, 0.04, 0.05],
    "carpet_thin":     [0.02, 0.06, 0.14, 0.37, 0.60, 0.65],
    "carpet_thick":    [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],
    "curtain_light":   [0.03, 0.04, 0.11, 0.17, 0.24, 0.35],
    "curtain_heavy":   [0.14, 0.35, 0.55, 0.72, 0.70, 0.65],
    "acoustic_panel":  [0.28, 0.67, 0.98, 0.86, 0.93, 0.87],
    "acoustic_foam":   [0.08, 0.25, 0.65, 0.95, 0.90, 0.85],
    "acoustic_tile":   [0.50, 0.70, 0.60, 0.70, 0.70, 0.50],
    "audience_seated": [0.39, 0.57, 0.80, 0.94, 0.92, 0.87],
    "water":           [0.01, 0.01, 0.01, 0.02, 0.02, 0.03],
    "anechoic":        [0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    "rigid":           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
}

SURFACE_NAMES = ["floor", "ceiling", "left", "right", "front", "back"]
SURFACE_AREAS_BOX = {
    "floor":   lambda r: r.Lx * r.Ly,
    "ceiling": lambda r: r.Lx * r.Ly,
    "left":    lambda r: r.Ly * r.Lz,
    "right":   lambda r: r.Ly * r.Lz,
    "front":   lambda r: r.Lx * r.Lz,
    "back":    lambda r: r.Lx * r.Lz,
}

# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Room:
    Lx: float; Ly: float; Lz: float
    surfaces: Dict[str, str] = field(default_factory=lambda: {s: "plaster" for s in SURFACE_NAMES})
    c: float = C_DEFAULT
    rho: float = RHO_DEFAULT

    @property
    def volume(self): return self.Lx * self.Ly * self.Lz
    @property
    def dims(self): return np.array([self.Lx, self.Ly, self.Lz])
    @property
    def total_surface(self):
        return 2*(self.Lx*self.Ly + self.Ly*self.Lz + self.Lx*self.Lz)

    def alpha_at(self, surface: str, f: float) -> float:
        mat = self.surfaces.get(surface, "plaster")
        a = np.array(MATERIALS.get(mat, MATERIALS["plaster"]), dtype=float)
        return float(np.interp(f, OCTAVE_BANDS, a))

    def mean_alpha_at(self, f: float) -> float:
        total_a, total_s = 0.0, 0.0
        for sn in SURFACE_NAMES:
            area = SURFACE_AREAS_BOX[sn](self)
            total_a += self.alpha_at(sn, f) * area
            total_s += area
        return total_a / total_s

    def absorption_area(self, f: float) -> float:
        return sum(self.alpha_at(sn, f) * SURFACE_AREAS_BOX[sn](self) for sn in SURFACE_NAMES)


@dataclass
class Result:
    name: str
    category: str
    ir: Optional[np.ndarray]
    sr: int
    metrics: Dict
    info: str
    compute_time: float
    band_limited: bool = False
    max_freq: float = 22050.0


# ═══════════════════════════════════════════════════════════════════
# Metrics  (ISO 3382 compliant)
# ═══════════════════════════════════════════════════════════════════
def schroeder_decay(ir, sr):
    e = ir ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc /= edc[0] + 1e-30
    return 10.0 * np.log10(edc + 1e-30)

def _rt_from_decay(edc_db, sr, lo=-5.0, hi=-35.0):
    n = len(edc_db)
    t = np.arange(n) / sr
    i0 = np.searchsorted(-edc_db, -lo)
    i1 = np.searchsorted(-edc_db, -hi)
    if i1 <= i0 + 2 or i1 >= n:
        return 0.0, 0.0
    seg_t, seg_db = t[i0:i1], edc_db[i0:i1]
    A = np.vstack([seg_t, np.ones(len(seg_t))]).T
    try:
        (slope, intercept), res, _, _ = np.linalg.lstsq(A, seg_db, rcond=None)
    except Exception:
        return 0.0, 0.0
    if slope >= 0:
        return 0.0, 0.0
    rt = -60.0 / slope
    ss_res = np.sum((seg_db - (slope * seg_t + intercept)) ** 2)
    ss_tot = np.sum((seg_db - seg_db.mean()) ** 2) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return max(rt, 0.0), r2

def compute_t30(ir, sr):
    return _rt_from_decay(schroeder_decay(ir, sr), sr, -5, -35)

def compute_t20(ir, sr):
    return _rt_from_decay(schroeder_decay(ir, sr), sr, -5, -25)

def compute_edt(ir, sr):
    return _rt_from_decay(schroeder_decay(ir, sr), sr, 0, -10)

def compute_c80(ir, sr):
    n80 = int(0.08 * sr)
    if n80 >= len(ir): return 0.0
    early = np.sum(ir[:n80] ** 2) + 1e-30
    late = np.sum(ir[n80:] ** 2) + 1e-30
    return 10.0 * np.log10(early / late)

def compute_d50(ir, sr):
    n50 = int(0.05 * sr)
    if n50 >= len(ir): return 1.0
    return float(np.sum(ir[:n50] ** 2) / (np.sum(ir ** 2) + 1e-30))

def compute_ts(ir, sr):
    t = np.arange(len(ir)) / sr
    e = ir ** 2
    return float(np.sum(t * e) / (np.sum(e) + 1e-30) * 1000)

def all_metrics(ir, sr):
    if ir is None or len(ir) < 100:
        return {"T30_s": 0, "EDT_s": 0, "C80_dB": 0, "D50": 0, "TS_ms": 0}
    t30, t30r2 = compute_t30(ir, sr)
    edt, edtr2 = compute_edt(ir, sr)
    return {
        "T30_s": round(t30, 4), "T30_R2": round(t30r2, 4),
        "EDT_s": round(edt, 4), "EDT_R2": round(edtr2, 4),
        "C80_dB": round(compute_c80(ir, sr), 2),
        "D50": round(compute_d50(ir, sr), 4),
        "TS_ms": round(compute_ts(ir, sr), 1),
    }


# ═══════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════
def sabine_rt60(room, f=500.0):
    A = room.absorption_area(f)
    return 0.161 * room.volume / max(A, 1e-6)

def eyring_rt60(room, f=500.0):
    a = room.mean_alpha_at(f)
    S = room.total_surface
    if a >= 1.0: return 0.0
    return 0.161 * room.volume / (-S * np.log(1.0 - a + 1e-10))

def air_absorption_coeff(f, humidity=50.0, temp=20.0):
    """ISO 9613-1 simplified: returns m [Np/m]."""
    fr = f / 1000.0
    return 5.5e-4 * (50.0 / max(humidity, 10.0)) * fr ** 1.7 * (temp + 273.15) / 293.15

def synth_noise_decay(rt60, sr, duration, seed=42):
    rng = np.random.default_rng(seed)
    N = int(sr * duration)
    noise = rng.standard_normal(N)
    if rt60 <= 0: return noise * 1e-10
    gamma = 6.91 / rt60
    t = np.arange(N, dtype=np.float64) / sr
    return noise * np.exp(-gamma * t)

def box_exit(origins, dirs, dims):
    """Vectorised: find exit point of rays from box [0,dims]. Returns (t, wall_idx)."""
    N = origins.shape[0]
    t_min = np.full(N, 1e30)
    walls = np.full(N, -1, dtype=np.int32)
    for ax in range(3):
        d = dirs[:, ax]
        # positive wall
        mask = d > 1e-12
        t = np.where(mask, (dims[ax] - origins[:, ax]) / np.where(mask, d, 1.0), 1e30)
        up = mask & (t > 1e-8) & (t < t_min)
        t_min = np.where(up, t, t_min)
        walls = np.where(up, 2 * ax + 1, walls)
        # negative wall
        mask = d < -1e-12
        t = np.where(mask, -origins[:, ax] / np.where(mask, d, -1.0), 1e30)
        up = mask & (t > 1e-8) & (t < t_min)
        t_min = np.where(up, t, t_min)
        walls = np.where(up, 2 * ax, walls)
    return t_min, walls

def reflect_specular(dirs, walls):
    """Reflect direction vectors at box walls."""
    out = dirs.copy()
    for ax in range(3):
        mask = (walls == 2 * ax) | (walls == 2 * ax + 1)
        out[mask, ax] *= -1.0
    return out

def wall_normal(wall_idx):
    """Return outward normal for wall index."""
    normals = np.array([
        [-1, 0, 0], [1, 0, 0],   # left/right  (x=0, x=Lx)
        [0, -1, 0], [0, 1, 0],   # front/back  (y=0, y=Ly)
        [0, 0, -1], [0, 0, 1],   # floor/ceil  (z=0, z=Lz)
    ], dtype=float)
    return normals[wall_idx]

WALL_TO_SURFACE = {0: "left", 1: "right", 2: "front", 3: "back", 4: "floor", 5: "ceiling"}

def random_dirs_sphere(N, rng=None):
    if rng is None: rng = np.random.default_rng()
    z = rng.uniform(-1, 1, N)
    phi = rng.uniform(0, 2 * np.pi, N)
    r = np.sqrt(1 - z ** 2)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def random_dirs_hemisphere(normals, rng=None):
    """Random directions in hemisphere around each normal (Lambert cosine)."""
    if rng is None: rng = np.random.default_rng()
    N = normals.shape[0]
    dirs = random_dirs_sphere(N, rng)
    dots = np.sum(dirs * normals, axis=1)
    dirs[dots < 0] *= -1
    return dirs

def octave_bandpass(signal, sr, fc):
    from scipy.signal import butter, sosfiltfilt
    fl = fc / np.sqrt(2)
    fh = fc * np.sqrt(2)
    fh = min(fh, sr / 2 - 1)
    if fl >= fh or fl < 1: return signal
    sos = butter(4, [fl, fh], btype='band', fs=sr, output='sos')
    return sosfiltfilt(sos, signal)


# ═══════════════════════════════════════════════════════════════════
#  1. SABINE  (statistical)
# ═══════════════════════════════════════════════════════════════════
def run_sabine(room, src, rec, sr=44100, duration=2.0, **kw):
    t0 = time.perf_counter()
    rt = {}
    for f in OCTAVE_BANDS:
        rt[int(f)] = round(sabine_rt60(room, f), 4)
    rt_mid = sabine_rt60(room, 500.0)
    ir = synth_noise_decay(rt_mid, sr, duration)
    ir[0] = 1.0  # direct sound
    m = all_metrics(ir, sr)
    m["RT60_bands"] = rt
    return Result("Sabine", "Statistical", ir, sr, m,
                  f"Sabine RT60 = {rt_mid:.3f}s (500 Hz)", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
#  2. EYRING  (statistical)
# ═══════════════════════════════════════════════════════════════════
def run_eyring(room, src, rec, sr=44100, duration=2.0, **kw):
    t0 = time.perf_counter()
    rt = {}
    for f in OCTAVE_BANDS:
        rt[int(f)] = round(eyring_rt60(room, f), 4)
    rt_mid = eyring_rt60(room, 500.0)
    ir = synth_noise_decay(rt_mid, sr, duration)
    ir[0] = 1.0
    m = all_metrics(ir, sr)
    m["RT60_bands"] = rt
    return Result("Eyring", "Statistical", ir, sr, m,
                  f"Eyring RT60 = {rt_mid:.3f}s (500 Hz)", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
#  3. SEA — Statistical Energy Analysis
# ═══════════════════════════════════════════════════════════════════
def run_sea(room, src, rec, sr=44100, duration=2.0, **kw):
    t0 = time.perf_counter()
    V, S, c = room.volume, room.total_surface, room.c
    rt_bands = {}
    for f in OCTAVE_BANDS:
        # modal density (Weyl formula)
        n_f = 4 * np.pi * f ** 2 * V / c ** 3 + np.pi * f * S / (2 * c ** 2)
        A = room.absorption_area(f)
        m_air = air_absorption_coeff(f)
        eta_d = c * A / (8 * np.pi * f * V + 1e-10) + m_air * c / (2 * np.pi * f + 1e-10)
        rt60 = 2.2 / (2 * np.pi * f * eta_d + 1e-10)
        rt_bands[int(f)] = round(min(rt60, 30.0), 4)
    rt_mid = rt_bands.get(500, 1.0)
    ir = synth_noise_decay(rt_mid, sr, duration)
    ir[0] = 1.0
    m = all_metrics(ir, sr)
    m["RT60_bands"] = rt_bands
    return Result("SEA", "Statistical", ir, sr, m,
                  f"SEA RT60 = {rt_mid:.3f}s (500 Hz)", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
#  4. AXIAL MODES
# ═══════════════════════════════════════════════════════════════════
def run_axial_modes(room, src, rec, sr=44100, duration=2.0, f_max=4000, **kw):
    t0 = time.perf_counter()
    c = room.c
    dims = room.dims
    src_a, rec_a = np.asarray(src), np.asarray(rec)
    pairs = [
        (0, "left", "right"),
        (1, "front", "back"),
        (2, "floor", "ceiling"),
    ]
    N = int(sr * duration)
    t_arr = np.arange(N, dtype=np.float64) / sr
    ir = np.zeros(N)
    n_modes = 0
    for ax, s1, s2 in pairs:
        L = dims[ax]
        xs, xr = src_a[ax], rec_a[ax]
        a1 = room.alpha_at(s1, 500)
        a2 = room.alpha_at(s2, 500)
        max_n = int(f_max * 2 * L / c) + 1
        for n in range(1, max_n + 1):
            fn = n * c / (2 * L)
            if fn > f_max: break
            alpha1 = room.alpha_at(s1, fn)
            alpha2 = room.alpha_at(s2, fn)
            gamma = -c / (2 * L) * np.log(max((1 - alpha1) * (1 - alpha2), 1e-10))
            gamma += air_absorption_coeff(fn) * c
            # 3D coupling loss
            A_pair = SURFACE_AREAS_BOX[s1](room)
            coupling = 1.0 - 2 * A_pair / room.total_surface
            gamma_room = 6.91 / max(eyring_rt60(room, fn), 0.01)
            gamma_eff = (1 - coupling) * gamma + coupling * gamma_room
            Sn = np.cos(n * np.pi * xs / L)
            Rn = np.cos(n * np.pi * xr / L)
            amp = Sn * Rn * 2.0 / L
            if abs(amp) < 1e-6: continue
            omega = 2 * np.pi * fn
            ir += amp * np.exp(-gamma_eff * t_arr) * np.cos(omega * t_arr)
            n_modes += 1
    if np.max(np.abs(ir)) > 0:
        ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Axial Modes", "Modal", ir, sr, m,
                  f"{n_modes} axial modes up to {f_max} Hz", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
#  5. FULL MODAL (analytical rectangular room modes)
# ═══════════════════════════════════════════════════════════════════
def run_full_modal(room, src, rec, sr=44100, duration=2.0, f_max=2000, **kw):
    t0 = time.perf_counter()
    c = room.c
    Lx, Ly, Lz = room.Lx, room.Ly, room.Lz
    V = room.volume
    xs, ys, zs = src
    xr, yr, zr = rec
    # enumerate modes
    max_nx = int(f_max * 2 * Lx / c) + 1
    max_ny = int(f_max * 2 * Ly / c) + 1
    max_nz = int(f_max * 2 * Lz / c) + 1
    modes_f, modes_amp, modes_gamma = [], [], []
    for nx in range(0, max_nx + 1):
        for ny in range(0, max_ny + 1):
            for nz in range(0, max_nz + 1):
                if nx == 0 and ny == 0 and nz == 0: continue
                fn = c / 2 * np.sqrt((nx / Lx) ** 2 + (ny / Ly) ** 2 + (nz / Lz) ** 2)
                if fn > f_max: continue
                phi_s = np.cos(nx * np.pi * xs / Lx) * np.cos(ny * np.pi * ys / Ly) * np.cos(nz * np.pi * zs / Lz)
                phi_r = np.cos(nx * np.pi * xr / Lx) * np.cos(ny * np.pi * yr / Ly) * np.cos(nz * np.pi * zr / Lz)
                amp = phi_s * phi_r
                if abs(amp) < 1e-6: continue
                # decay: per-surface contribution
                gamma = 0.0
                for sn in SURFACE_NAMES:
                    a_s = room.alpha_at(sn, fn)
                    A_s = SURFACE_AREAS_BOX[sn](room)
                    # coupling factor: 2 if mode index perpendicular to surface > 0
                    if sn in ("left", "right"):   eps = 2.0 if nx > 0 else 0.0
                    elif sn in ("front", "back"): eps = 2.0 if ny > 0 else 0.0
                    else:                         eps = 2.0 if nz > 0 else 0.0
                    gamma += c * a_s * A_s * eps / (4 * V)
                gamma += air_absorption_coeff(fn) * c
                modes_f.append(fn)
                modes_amp.append(amp)
                modes_gamma.append(gamma)
    N = int(sr * duration)
    t_arr = np.arange(N, dtype=np.float64) / sr
    ir = np.zeros(N)
    # vectorised synthesis in chunks
    modes_f = np.array(modes_f)
    modes_amp = np.array(modes_amp)
    modes_gamma = np.array(modes_gamma)
    chunk = 500
    for i in range(0, len(modes_f), chunk):
        f_c = modes_f[i:i + chunk, None]
        a_c = modes_amp[i:i + chunk, None]
        g_c = modes_gamma[i:i + chunk, None]
        ir += np.sum(a_c * np.exp(-g_c * t_arr[None, :]) * np.cos(2 * np.pi * f_c * t_arr[None, :]), axis=0)
    if np.max(np.abs(ir)) > 0:
        ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Full Modal", "Modal", ir, sr, m,
                  f"{len(modes_f)} modes up to {f_max} Hz", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
#  6. MODAL ROM  (reduced-order model from analytical modes)
# ═══════════════════════════════════════════════════════════════════
def run_modal_rom(room, src, rec, sr=44100, duration=2.0, f_max=2000, n_basis=40, **kw):
    t0 = time.perf_counter()
    c = room.c
    Lx, Ly, Lz = room.Lx, room.Ly, room.Lz
    V = room.volume
    xs, ys, zs = src
    xr, yr, zr = rec
    max_nx = int(f_max * 2 * Lx / c) + 1
    max_ny = int(f_max * 2 * Ly / c) + 1
    max_nz = int(f_max * 2 * Lz / c) + 1
    all_modes = []
    for nx in range(0, max_nx + 1):
        for ny in range(0, max_ny + 1):
            for nz in range(0, max_nz + 1):
                if nx == 0 and ny == 0 and nz == 0: continue
                fn = c / 2 * np.sqrt((nx / Lx) ** 2 + (ny / Ly) ** 2 + (nz / Lz) ** 2)
                if fn > f_max: continue
                all_modes.append((fn, nx, ny, nz))
    all_modes.sort(key=lambda x: x[0])
    selected = all_modes[:n_basis]
    N = int(sr * duration)
    t_arr = np.arange(N, dtype=np.float64) / sr
    ir = np.zeros(N)
    for fn, nx, ny, nz in selected:
        phi_s = np.cos(nx*np.pi*xs/Lx)*np.cos(ny*np.pi*ys/Ly)*np.cos(nz*np.pi*zs/Lz)
        phi_r = np.cos(nx*np.pi*xr/Lx)*np.cos(ny*np.pi*yr/Ly)*np.cos(nz*np.pi*zr/Lz)
        amp = phi_s * phi_r
        if abs(amp) < 1e-6: continue
        gamma = 0.0
        for sn in SURFACE_NAMES:
            a_s = room.alpha_at(sn, fn)
            A_s = SURFACE_AREAS_BOX[sn](room)
            if sn in ("left", "right"):   eps = 2.0 if nx > 0 else 0.0
            elif sn in ("front", "back"): eps = 2.0 if ny > 0 else 0.0
            else:                         eps = 2.0 if nz > 0 else 0.0
            gamma += c * a_s * A_s * eps / (4 * V)
        gamma += air_absorption_coeff(fn) * c
        ir += amp * np.exp(-gamma * t_arr) * np.cos(2 * np.pi * fn * t_arr)
    if np.max(np.abs(ir)) > 0:
        ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Modal ROM", "Modal", ir, sr, m,
                  f"{len(selected)}/{len(all_modes)} modes (reduced basis)",
                  time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
#  7. IMAGE SOURCE METHOD (Allen & Berkley 1979)
# ═══════════════════════════════════════════════════════════════════
def _wall_reflections(n):
    an = abs(n)
    if n > 0:  return an // 2, (an + 1) // 2
    elif n < 0: return (an + 1) // 2, an // 2
    return 0, 0

def run_ism(room, src, rec, sr=44100, duration=2.0, max_order=15, **kw):
    t0 = time.perf_counter()
    c = room.c
    Lx, Ly, Lz = room.Lx, room.Ly, room.Lz
    xs, ys, zs = src
    xr, yr, zr = rec
    N = int(sr * duration)
    ir = np.zeros(N)
    n_images = 0
    wall_pairs = [("left", "right"), ("front", "back"), ("floor", "ceiling")]
    for nx in range(-max_order, max_order + 1):
        xi = nx * Lx + (xs if nx % 2 == 0 else Lx - xs)
        for ny in range(-max_order, max_order + 1):
            yi = ny * Ly + (ys if ny % 2 == 0 else Ly - ys)
            order_xy = abs(nx) + abs(ny)
            if order_xy > max_order: continue
            for nz in range(-(max_order - order_xy), max_order - order_xy + 1):
                order = order_xy + abs(nz)
                if order == 0: continue
                zi = nz * Lz + (zs if nz % 2 == 0 else Lz - zs)
                dist = np.sqrt((xi - xr) ** 2 + (yi - yr) ** 2 + (zi - zr) ** 2)
                arrival = dist / c
                if arrival >= duration: continue
                # reflection coefficient product
                R = 1.0
                for ax, (s0, sL) in enumerate(wall_pairs):
                    nn = [nx, ny, nz][ax]
                    n0, nL = _wall_reflections(nn)
                    f_mid = 500.0
                    R *= (1 - room.alpha_at(s0, f_mid)) ** n0 * (1 - room.alpha_at(sL, f_mid)) ** nL
                amp = np.sqrt(max(R, 0.0)) / (4 * np.pi * dist + 1e-10)
                idx = int(arrival * sr)
                if 0 <= idx < N:
                    ir[idx] += amp * ((-1) ** order)
                    n_images += 1
    if np.max(np.abs(ir)) > 0:
        ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Image Source", "Geometric", ir, sr, m,
                  f"{n_images} images, order {max_order}", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
#  8. RAY TRACING (stochastic Monte Carlo)
# ═══════════════════════════════════════════════════════════════════
def run_ray_tracing(room, src, rec, sr=44100, duration=2.0, n_rays=10000,
                    max_bounces=200, capture_radius=0.5, **kw):
    t0 = time.perf_counter()
    c = room.c; dims = room.dims
    src_a, rec_a = np.asarray(src, dtype=float), np.asarray(rec, dtype=float)
    rng = np.random.default_rng(42)
    N = int(sr * duration)
    histogram = np.zeros(N)
    dirs = random_dirs_sphere(n_rays, rng)
    pos = np.tile(src_a, (n_rays, 1))
    energy = np.ones(n_rays)
    travel_time = np.zeros(n_rays)
    alive = np.ones(n_rays, dtype=bool)
    for bounce in range(max_bounces):
        if not np.any(alive): break
        idx_alive = np.where(alive)[0]
        t_hit, walls = box_exit(pos[idx_alive], dirs[idx_alive], dims)
        hit_pos = pos[idx_alive] + dirs[idx_alive] * t_hit[:, None]
        travel_time[idx_alive] += t_hit / c
        # check receiver proximity along path
        to_rec = rec_a - pos[idx_alive]
        proj = np.sum(to_rec * dirs[idx_alive], axis=1)
        proj = np.clip(proj, 0, t_hit)
        closest = pos[idx_alive] + dirs[idx_alive] * proj[:, None]
        dist_to_rec = np.linalg.norm(closest - rec_a, axis=1)
        detected = dist_to_rec < capture_radius
        det_idx = idx_alive[detected]
        for di in det_idx:
            arr_time = travel_time[di] - (t_hit[np.where(idx_alive == di)[0][0]] - proj[np.where(idx_alive == di)[0][0]]) / c
            sample = int(arr_time * sr)
            if 0 <= sample < N:
                histogram[sample] += energy[di]
        # apply absorption at walls
        pos[idx_alive] = hit_pos
        for w in range(6):
            mask_w = walls == w
            if not np.any(mask_w): continue
            sn = WALL_TO_SURFACE[w]
            alpha = room.alpha_at(sn, 500.0)
            full_idx = idx_alive[mask_w]
            energy[full_idx] *= (1 - alpha)
        dirs[idx_alive] = reflect_specular(dirs[idx_alive], walls)
        alive[energy < 1e-8] = False
        alive[travel_time > duration] = False
    # convert histogram to IR
    ir = synth_noise_decay(1.0, sr, duration, seed=99)
    ir *= np.sqrt(histogram + 1e-30)
    if np.max(np.abs(ir)) > 0:
        ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Ray Tracing", "Geometric", ir, sr, m,
                  f"{n_rays} rays, {max_bounces} max bounces", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
#  9. CONE TRACING
# ═══════════════════════════════════════════════════════════════════
def run_cone_tracing(room, src, rec, sr=44100, duration=2.0, n_cones=5000,
                     max_bounces=150, **kw):
    t0 = time.perf_counter()
    c = room.c; dims = room.dims
    src_a, rec_a = np.asarray(src, dtype=float), np.asarray(rec, dtype=float)
    rng = np.random.default_rng(43)
    N = int(sr * duration)
    histogram = np.zeros(N)
    solid_angle = 4 * np.pi / n_cones
    dirs = random_dirs_sphere(n_cones, rng)
    pos = np.tile(src_a, (n_cones, 1))
    energy = np.ones(n_cones)
    travel = np.zeros(n_cones)
    alive = np.ones(n_cones, dtype=bool)
    for bounce in range(max_bounces):
        if not np.any(alive): break
        ai = np.where(alive)[0]
        t_hit, walls = box_exit(pos[ai], dirs[ai], dims)
        hit_pos = pos[ai] + dirs[ai] * t_hit[:, None]
        travel[ai] += t_hit / c
        # cone detection: at distance d, cone covers area = solid_angle * d^2
        dist_to_rec = np.linalg.norm(pos[ai] - rec_a, axis=1)
        cone_radius = np.sqrt(solid_angle / np.pi) * dist_to_rec
        to_rec = rec_a - pos[ai]
        proj = np.sum(to_rec * dirs[ai], axis=1)
        closest_d = np.sqrt(np.maximum(np.sum(to_rec ** 2, axis=1) - proj ** 2, 0))
        detected = (closest_d < cone_radius) & (proj > 0) & (proj < t_hit)
        for j in np.where(detected)[0]:
            gi = ai[j]
            arr = travel[gi] - (t_hit[j] - proj[j]) / c
            s = int(arr * sr)
            if 0 <= s < N: histogram[s] += energy[gi]
        pos[ai] = hit_pos
        for w in range(6):
            mw = walls == w
            if not np.any(mw): continue
            energy[ai[mw]] *= (1 - room.alpha_at(WALL_TO_SURFACE[w], 500.0))
        dirs[ai] = reflect_specular(dirs[ai], walls)
        alive[energy < 1e-8] = False
        alive[travel > duration] = False
    ir = synth_noise_decay(1.0, sr, duration, seed=100) * np.sqrt(histogram + 1e-30)
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Cone Tracing", "Geometric", ir, sr, m,
                  f"{n_cones} cones", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
# 10. BEAM TRACING (shoebox = exact ISM via beam volumes)
# ═══════════════════════════════════════════════════════════════════
def run_beam_tracing(room, src, rec, sr=44100, duration=2.0, max_order=12, **kw):
    t0 = time.perf_counter()
    # For a shoebox, beam tracing is mathematically equivalent to ISM.
    # Each beam corresponds to exactly one image source path.
    # We reuse ISM but track beam solid angles.
    c = room.c
    Lx, Ly, Lz = room.Lx, room.Ly, room.Lz
    xs, ys, zs = src; xr, yr, zr = rec
    N = int(sr * duration)
    ir = np.zeros(N)
    n_beams = 0
    wall_pairs = [("left", "right"), ("front", "back"), ("floor", "ceiling")]
    for nx in range(-max_order, max_order + 1):
        xi = nx * Lx + (xs if nx % 2 == 0 else Lx - xs)
        for ny in range(-max_order, max_order + 1):
            yi = ny * Ly + (ys if ny % 2 == 0 else Ly - ys)
            oxy = abs(nx) + abs(ny)
            if oxy > max_order: continue
            for nz in range(-(max_order - oxy), max_order - oxy + 1):
                order = oxy + abs(nz)
                if order == 0: continue
                zi = nz * Lz + (zs if nz % 2 == 0 else Lz - zs)
                dist = np.sqrt((xi-xr)**2 + (yi-yr)**2 + (zi-zr)**2)
                arr = dist / c
                if arr >= duration: continue
                R = 1.0
                for ax, (s0, sL) in enumerate(wall_pairs):
                    nn = [nx, ny, nz][ax]
                    n0, nL = _wall_reflections(nn)
                    R *= (1-room.alpha_at(s0, 500))**n0 * (1-room.alpha_at(sL, 500))**nL
                # beam solid angle shrinks with distance^2
                amp = np.sqrt(max(R, 0)) / (4*np.pi*dist + 1e-10)
                idx = int(arr * sr)
                if 0 <= idx < N:
                    ir[idx] += amp
                    n_beams += 1
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Beam Tracing", "Geometric", ir, sr, m,
                  f"{n_beams} beams, order {max_order}", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
# 11. PHONON TRACING (energy packets with Lambert scattering)
# ═══════════════════════════════════════════════════════════════════
def run_phonon_tracing(room, src, rec, sr=44100, duration=2.0, n_phonons=10000,
                       max_bounces=200, scatter_coeff=0.3, capture_radius=0.5, **kw):
    t0 = time.perf_counter()
    c = room.c; dims = room.dims
    src_a, rec_a = np.asarray(src, dtype=float), np.asarray(rec, dtype=float)
    rng = np.random.default_rng(44)
    N = int(sr * duration)
    histogram = np.zeros(N)
    dirs = random_dirs_sphere(n_phonons, rng)
    pos = np.tile(src_a, (n_phonons, 1))
    energy = np.ones(n_phonons)
    travel = np.zeros(n_phonons)
    alive = np.ones(n_phonons, dtype=bool)
    for bounce in range(max_bounces):
        if not np.any(alive): break
        ai = np.where(alive)[0]
        t_hit, walls = box_exit(pos[ai], dirs[ai], dims)
        hit_pos = pos[ai] + dirs[ai] * t_hit[:, None]
        travel[ai] += t_hit / c
        # detect near receiver
        to_rec = rec_a - pos[ai]
        proj = np.clip(np.sum(to_rec * dirs[ai], axis=1), 0, t_hit)
        closest = pos[ai] + dirs[ai] * proj[:, None]
        det = np.linalg.norm(closest - rec_a, axis=1) < capture_radius
        for j in np.where(det)[0]:
            s = int((travel[ai[j]] - (t_hit[j] - proj[j]) / c) * sr)
            if 0 <= s < N: histogram[s] += energy[ai[j]]
        pos[ai] = hit_pos
        # absorption + scattering
        for w in range(6):
            mw = walls == w
            if not np.any(mw): continue
            alpha = room.alpha_at(WALL_TO_SURFACE[w], 500.0)
            energy[ai[mw]] *= (1 - alpha)
        # scattering: fraction of phonons get diffuse new direction
        scatter_mask = rng.random(len(ai)) < scatter_coeff
        normals_scat = wall_normal(walls)
        normals_scat = normals_scat[scatter_mask]
        if len(normals_scat) > 0:
            new_dirs = random_dirs_hemisphere(normals_scat, rng)
            full_dirs = dirs[ai].copy()
            full_dirs[scatter_mask] = new_dirs
            dirs[ai] = full_dirs
        # specular for non-scattered
        spec_mask = ~scatter_mask
        if np.any(spec_mask):
            spec_idx = ai[spec_mask]
            dirs[spec_idx] = reflect_specular(dirs[spec_idx], walls[spec_mask])
        alive[energy < 1e-8] = False
        alive[travel > duration] = False
    ir = synth_noise_decay(1.0, sr, duration, seed=101) * np.sqrt(histogram + 1e-30)
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Phonon Tracing", "Geometric", ir, sr, m,
                  f"{n_phonons} phonons, scatter={scatter_coeff}", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
# 12. PATH TRACING (bidirectional Monte Carlo)
# ═══════════════════════════════════════════════════════════════════
def run_path_tracing(room, src, rec, sr=44100, duration=2.0, n_paths=20000,
                     max_bounces=50, connect_radius=0.3, **kw):
    t0 = time.perf_counter()
    c = room.c; dims = room.dims
    src_a, rec_a = np.asarray(src, dtype=float), np.asarray(rec, dtype=float)
    rng = np.random.default_rng(45)
    N = int(sr * duration)
    histogram = np.zeros(N)
    # trace from source
    s_dirs = random_dirs_sphere(n_paths, rng)
    s_pos = np.tile(src_a, (n_paths, 1))
    s_energy = np.ones(n_paths)
    s_time = np.zeros(n_paths)
    s_paths = [(s_pos.copy(), s_energy.copy(), s_time.copy())]
    for b in range(max_bounces):
        t_hit, walls = box_exit(s_pos, s_dirs, dims)
        s_pos = s_pos + s_dirs * t_hit[:, None]
        s_time += t_hit / c
        for w in range(6):
            mw = walls == w
            if np.any(mw):
                s_energy[mw] *= (1 - room.alpha_at(WALL_TO_SURFACE[w], 500.0))
        s_dirs = reflect_specular(s_dirs, walls)
        s_paths.append((s_pos.copy(), s_energy.copy(), s_time.copy()))
        alive = (s_energy > 1e-8) & (s_time < duration)
        if not np.any(alive): break
    # check all path points against receiver
    for pos, eng, tt in s_paths:
        dist = np.linalg.norm(pos - rec_a, axis=1)
        near = dist < connect_radius
        for j in np.where(near)[0]:
            total_t = tt[j] + dist[j] / c
            s = int(total_t * sr)
            if 0 <= s < N:
                histogram[s] += eng[j] / (dist[j] + 0.01)
    ir = synth_noise_decay(1.0, sr, duration, seed=102) * np.sqrt(histogram + 1e-30)
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Path Tracing", "Geometric", ir, sr, m,
                  f"{n_paths} paths, {max_bounces} bounces", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
# 13. RADIOSITY (diffuse energy exchange between patches)
# ═══════════════════════════════════════════════════════════════════
def run_radiosity(room, src, rec, sr=44100, duration=2.0, patches_per_wall=16, **kw):
    t0 = time.perf_counter()
    c = room.c
    Lx, Ly, Lz = room.Lx, room.Ly, room.Lz
    # build patches on each wall
    patch_centers = []
    patch_areas = []
    patch_normals = []
    patch_alpha = []
    wall_defs = [
        ("floor",   lambda u, v: np.array([u*Lx, v*Ly, 0]),     Lx, Ly, np.array([0, 0, 1])),
        ("ceiling", lambda u, v: np.array([u*Lx, v*Ly, Lz]),    Lx, Ly, np.array([0, 0, -1])),
        ("left",    lambda u, v: np.array([0, u*Ly, v*Lz]),      Ly, Lz, np.array([1, 0, 0])),
        ("right",   lambda u, v: np.array([Lx, u*Ly, v*Lz]),     Ly, Lz, np.array([-1, 0, 0])),
        ("front",   lambda u, v: np.array([u*Lx, 0, v*Lz]),      Lx, Lz, np.array([0, 1, 0])),
        ("back",    lambda u, v: np.array([u*Lx, Ly, v*Lz]),     Lx, Lz, np.array([0, -1, 0])),
    ]
    ppw = int(np.sqrt(patches_per_wall))
    for sn, pos_fn, W, H, normal in wall_defs:
        area = W * H / (ppw * ppw)
        for iu in range(ppw):
            for iv in range(ppw):
                u = (iu + 0.5) / ppw
                v = (iv + 0.5) / ppw
                patch_centers.append(pos_fn(u, v))
                patch_areas.append(area)
                patch_normals.append(normal.copy())
                patch_alpha.append(room.alpha_at(sn, 500.0))
    centers = np.array(patch_centers)
    areas = np.array(patch_areas)
    normals = np.array(patch_normals)
    alphas = np.array(patch_alpha)
    rho = 1.0 - alphas
    NP = len(centers)
    # form factors (approximate: point-to-point)
    F = np.zeros((NP, NP))
    for i in range(NP):
        for j in range(NP):
            if i == j: continue
            r_vec = centers[j] - centers[i]
            dist = np.linalg.norm(r_vec) + 1e-10
            r_hat = r_vec / dist
            cos_i = abs(np.dot(normals[i], r_hat))
            cos_j = abs(np.dot(normals[j], -r_hat))
            F[i, j] = cos_i * cos_j * areas[j] / (np.pi * dist ** 2)
    # normalise rows (energy conservation)
    row_sums = F.sum(axis=1, keepdims=True)
    F = F / (row_sums + 1e-10)
    # find source patch (nearest to source) and receiver patch
    src_a = np.asarray(src, dtype=float)
    rec_a = np.asarray(rec, dtype=float)
    # source injects energy to nearest patches
    dist_src = np.linalg.norm(centers - src_a, axis=1)
    src_patch = np.argmin(dist_src)
    rec_patches = np.argsort(np.linalg.norm(centers - rec_a, axis=1))[:4]
    # iterate energy exchange
    E = np.zeros(NP)
    E[src_patch] = 1.0
    n_iter = int(duration * c / (2 * max(Lx, Ly, Lz)))
    n_iter = min(max(n_iter, 50), 500)
    dt_rad = 2 * max(Lx, Ly, Lz) / c
    N = int(sr * duration)
    decay = np.zeros(N)
    for it in range(n_iter):
        E_new = rho * (F.T @ E)
        t_sample = int(it * dt_rad * sr)
        if t_sample < N:
            decay[t_sample] = E_new[rec_patches].sum()
        E = E_new
    # convert decay to IR
    ir = synth_noise_decay(1.0, sr, duration, seed=103)
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(decay, max(int(0.01 * sr), 1))
    ir *= np.sqrt(smooth + 1e-30)
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Radiosity", "Geometric", ir, sr, m,
                  f"{NP} patches, {n_iter} iterations", time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
# WAVE METHOD HELPERS
# ═══════════════════════════════════════════════════════════════════
def _wave_grid(room, max_freq, ppw=6):
    dx = room.c / (ppw * max_freq)
    Nx = max(int(round(room.Lx / dx)), 3)
    Ny = max(int(round(room.Ly / dx)), 3)
    Nz = max(int(round(room.Lz / dx)), 3)
    dx = room.Lx / Nx  # recalculate
    return Nx, Ny, Nz, dx

def _src_rec_idx(room, src, rec, Nx, Ny, Nz, dx):
    si = tuple(np.clip([int(s / dx) for s in src], 1, [Nx-2, Ny-2, Nz-2]))
    ri = tuple(np.clip([int(r / dx) for r in rec], 1, [Nx-2, Ny-2, Nz-2]))
    return si, ri

def _boundary_R(room, Nx, Ny, Nz, f=500.0):
    """Create reflection coefficient mask for box boundaries."""
    R = np.ones((Nx, Ny, Nz), dtype=np.float64)
    a = room.alpha_at("left", f);   R[0, :, :]  *= np.sqrt(1-a)
    a = room.alpha_at("right", f);  R[-1, :, :] *= np.sqrt(1-a)
    a = room.alpha_at("front", f);  R[:, 0, :]  *= np.sqrt(1-a)
    a = room.alpha_at("back", f);   R[:, -1, :] *= np.sqrt(1-a)
    a = room.alpha_at("floor", f);  R[:, :, 0]  *= np.sqrt(1-a)
    a = room.alpha_at("ceiling", f);R[:, :, -1] *= np.sqrt(1-a)
    return R


# ═══════════════════════════════════════════════════════════════════
# 14. FDTD (Finite Difference Time Domain)
# ═══════════════════════════════════════════════════════════════════
def run_fdtd(room, src, rec, sr=44100, duration=0.5, max_freq=500, CFL=0.28, **kw):
    t0 = time.perf_counter()
    Nx, Ny, Nz, dx = _wave_grid(room, max_freq)
    dt = CFL * dx / room.c
    Nt = int(duration / dt)
    si, ri = _src_rec_idx(room, src, rec, Nx, Ny, Nz, dx)
    R = _boundary_R(room, Nx, Ny, Nz, max_freq)
    courant2 = (room.c * dt / dx) ** 2
    p = np.zeros((Nx, Ny, Nz))
    p_prev = np.zeros_like(p)
    ir_wave = []
    for n in range(Nt):
        # Laplacian with Neumann BC via edge padding
        pp = np.pad(p, 1, mode='edge')
        lap = (pp[2:,1:-1,1:-1] + pp[:-2,1:-1,1:-1] +
               pp[1:-1,2:,1:-1] + pp[1:-1,:-2,1:-1] +
               pp[1:-1,1:-1,2:] + pp[1:-1,1:-1,:-2] -
               6 * pp[1:-1,1:-1,1:-1])
        p_next = 2*p - p_prev + courant2 * lap
        p_next *= R
        if n == 1: p_next[si] += 1.0
        ir_wave.append(p_next[ri])
        p_prev, p = p, p_next
    ir_wave = np.array(ir_wave)
    # resample to target sr
    from scipy.signal import resample
    wave_sr = 1.0 / dt
    target_len = int(len(ir_wave) * sr / wave_sr)
    ir = resample(ir_wave, target_len) if target_len > 0 else ir_wave
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, int(sr))
    return Result("FDTD", "Wave (TD)", ir, int(sr), m,
                  f"Grid {Nx}x{Ny}x{Nz}, {Nt} steps, f<{max_freq}Hz",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


# ═══════════════════════════════════════════════════════════════════
# 15. PSTD (Pseudo-Spectral Time Domain, DCT-based)
# ═══════════════════════════════════════════════════════════════════
def run_pstd(room, src, rec, sr=44100, duration=0.5, max_freq=500, CFL=0.35, **kw):
    t0 = time.perf_counter()
    from scipy.fft import dctn, idctn
    ppw = 4  # spectral accuracy needs fewer ppw
    Nx, Ny, Nz, dx = _wave_grid(room, max_freq, ppw=ppw)
    dt = CFL * dx / room.c
    Nt = int(duration / dt)
    si, ri = _src_rec_idx(room, src, rec, Nx, Ny, Nz, dx)
    R = _boundary_R(room, Nx, Ny, Nz, max_freq)
    # wavenumber grid for DCT
    kx = np.pi * np.arange(Nx) / (Nx * dx)
    ky = np.pi * np.arange(Ny) / (Ny * dx)
    kz = np.pi * np.arange(Nz) / (Nz * dx)
    K2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    p = np.zeros((Nx, Ny, Nz))
    p_prev = np.zeros_like(p)
    ir_wave = []
    c2dt2 = (room.c * dt) ** 2
    for n in range(Nt):
        p_hat = dctn(p, type=2, norm='ortho')
        lap = idctn(-K2 * p_hat, type=2, norm='ortho')
        p_next = 2*p - p_prev + c2dt2 * lap
        p_next *= R
        if n == 1: p_next[si] += 1.0
        ir_wave.append(p_next[ri])
        p_prev, p = p, p_next
    ir_wave = np.array(ir_wave)
    from scipy.signal import resample
    target_len = int(len(ir_wave) * sr / (1.0 / dt))
    ir = resample(ir_wave, target_len) if target_len > 0 else ir_wave
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, int(sr))
    return Result("PSTD", "Wave (TD)", ir, int(sr), m,
                  f"Grid {Nx}x{Ny}x{Nz}, spectral, f<{max_freq}Hz",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


# ═══════════════════════════════════════════════════════════════════
# 16. TLM (Transmission Line Matrix)
# ═══════════════════════════════════════════════════════════════════
def run_tlm(room, src, rec, sr=44100, duration=0.5, max_freq=500, **kw):
    t0 = time.perf_counter()
    Nx, Ny, Nz, dx = _wave_grid(room, max_freq)
    dt = dx / (room.c * np.sqrt(3))
    Nt = int(duration / dt)
    si, ri = _src_rec_idx(room, src, rec, Nx, Ny, Nz, dx)
    R_mask = _boundary_R(room, Nx, Ny, Nz, max_freq)
    # 6 port voltages per node: +x, -x, +y, -y, +z, -z
    V = np.zeros((6, Nx, Ny, Nz))
    ir_wave = []
    for n in range(Nt):
        # SCATTER: V_out[i] = (sum_in / 3) - V_in[i]
        S = V.sum(axis=0) / 3.0
        V_out = S[None, :, :, :] - V
        # INJECT source
        if n == 1:
            for port in range(6):
                V_out[port][si] += 1.0 / 6.0
        # CONNECT: swap between neighbours
        V_new = np.zeros_like(V)
        V_new[0, :-1, :, :] = V_out[1, 1:, :, :]   # +x gets -x from right neighbour
        V_new[1, 1:, :, :]  = V_out[0, :-1, :, :]   # -x gets +x from left neighbour
        V_new[2, :, :-1, :] = V_out[3, :, 1:, :]
        V_new[3, :, 1:, :]  = V_out[2, :, :-1, :]
        V_new[4, :, :, :-1] = V_out[5, :, :, 1:]
        V_new[5, :, :, 1:]  = V_out[4, :, :, :-1]
        # BOUNDARY: reflect with absorption
        V_new[1, 0, :, :]   = V_out[0, 0, :, :] * R_mask[0, :, :]
        V_new[0, -1, :, :]  = V_out[1, -1, :, :] * R_mask[-1, :, :]
        V_new[3, :, 0, :]   = V_out[2, :, 0, :] * R_mask[:, 0, :]
        V_new[2, :, -1, :]  = V_out[3, :, -1, :] * R_mask[:, -1, :]
        V_new[5, :, :, 0]   = V_out[4, :, :, 0] * R_mask[:, :, 0]
        V_new[4, :, :, -1]  = V_out[5, :, :, -1] * R_mask[:, :, -1]
        V = V_new
        # pressure = sum of incoming voltages
        pressure = V.sum(axis=0)
        ir_wave.append(pressure[ri])
    ir_wave = np.array(ir_wave)
    from scipy.signal import resample
    target_len = int(len(ir_wave) * sr / (1.0 / dt))
    ir = resample(ir_wave, target_len) if target_len > 0 else ir_wave
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, int(sr))
    return Result("TLM", "Wave (TD)", ir, int(sr), m,
                  f"Grid {Nx}x{Ny}x{Nz}, {Nt} TLM steps, f<{max_freq}Hz",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


# ═══════════════════════════════════════════════════════════════════
# 17. ARD (Adaptive Rectangular Decomposition, DCT modes)
# ═══════════════════════════════════════════════════════════════════
def run_ard(room, src, rec, sr=44100, duration=1.0, max_freq=500, **kw):
    t0 = time.perf_counter()
    c = room.c
    Lx, Ly, Lz = room.Lx, room.Ly, room.Lz
    # single rectangular partition (optimal for shoebox)
    # mode frequencies via DCT: f_{ijk} = c/2 * sqrt((i/Lx)^2 + (j/Ly)^2 + (k/Lz)^2)
    max_i = int(max_freq * 2 * Lx / c) + 1
    max_j = int(max_freq * 2 * Ly / c) + 1
    max_k = int(max_freq * 2 * Lz / c) + 1
    xs, ys, zs = src; xr, yr, zr = rec
    N = int(sr * duration)
    t_arr = np.arange(N, dtype=np.float64) / sr
    ir = np.zeros(N)
    n_modes = 0
    # For ARD, each mode is a cosine standing wave; propagation is exact (no dispersion)
    mean_alpha = room.mean_alpha_at(500.0)
    rt60 = eyring_rt60(room, 500.0)
    gamma_base = 6.91 / max(rt60, 0.01)
    for i in range(0, max_i + 1):
        for j in range(0, max_j + 1):
            for k in range(0, max_k + 1):
                if i == 0 and j == 0 and k == 0: continue
                fn = c / 2 * np.sqrt((i/Lx)**2 + (j/Ly)**2 + (k/Lz)**2)
                if fn > max_freq: continue
                phi_s = np.cos(i*np.pi*xs/Lx)*np.cos(j*np.pi*ys/Ly)*np.cos(k*np.pi*zs/Lz)
                phi_r = np.cos(i*np.pi*xr/Lx)*np.cos(j*np.pi*yr/Ly)*np.cos(k*np.pi*zr/Lz)
                amp = phi_s * phi_r
                if abs(amp) < 1e-6: continue
                gamma = gamma_base + air_absorption_coeff(fn) * c
                ir += amp * np.exp(-gamma * t_arr) * np.cos(2*np.pi*fn * t_arr)
                n_modes += 1
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("ARD", "Wave (TD)", ir, sr, m,
                  f"{n_modes} DCT modes, 1 partition, f<{max_freq}Hz",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


# ═══════════════════════════════════════════════════════════════════
# 18. DG — Discontinuous Galerkin (P0 upwind flux, first-order acoustics)
# ═══════════════════════════════════════════════════════════════════
def run_dg(room, src, rec, sr=44100, duration=0.3, max_freq=300, CFL=0.25, **kw):
    t0 = time.perf_counter()
    c = room.c; rho = room.rho
    Nx, Ny, Nz, dx = _wave_grid(room, max_freq, ppw=5)
    dt = CFL * dx / (c * np.sqrt(3))
    Nt = int(duration / dt)
    si, ri = _src_rec_idx(room, src, rec, Nx, Ny, Nz, dx)
    # state: pressure p, velocity (vx, vy, vz) — cell-centred (P0 DG)
    p = np.zeros((Nx, Ny, Nz))
    vx = np.zeros_like(p); vy = np.zeros_like(p); vz = np.zeros_like(p)
    ir_wave = []
    Z = rho * c  # acoustic impedance
    R_walls = _boundary_R(room, Nx, Ny, Nz, max_freq)
    for n in range(Nt):
        # upwind flux for linearised acoustics:
        # dp/dt = -rho*c^2 * div(v),  dv/dt = -(1/rho) * grad(p)
        # flux at interface: p* = 0.5(pL+pR) - Z/2*(vL-vR)·n
        #                    v* = 0.5(vL+vR) - 1/(2Z)*(pL-pR)*n
        # Net update per cell from 6 faces:
        dp = np.zeros_like(p)
        dvx = np.zeros_like(p); dvy = np.zeros_like(p); dvz = np.zeros_like(p)
        # x-faces
        pL, pR = p[:-1,:,:], p[1:,:,:]
        vL, vR = vx[:-1,:,:], vx[1:,:,:]
        p_star = 0.5*(pL+pR) - 0.5*Z*(vR-vL)
        v_star = 0.5*(vL+vR) - 0.5/Z*(pR-pL)
        flux_p = rho * c**2 * v_star / dx
        flux_v = p_star / (rho * dx)
        dp[:-1,:,:] -= flux_p
        dp[1:,:,:]  += flux_p
        dvx[:-1,:,:] -= flux_v
        dvx[1:,:,:]  += flux_v
        # y-faces
        pL, pR = p[:,:-1,:], p[:,1:,:]
        vL, vR = vy[:,:-1,:], vy[:,1:,:]
        p_star = 0.5*(pL+pR) - 0.5*Z*(vR-vL)
        v_star = 0.5*(vL+vR) - 0.5/Z*(pR-pL)
        flux_p = rho * c**2 * v_star / dx
        flux_v = p_star / (rho * dx)
        dp[:,:-1,:] -= flux_p
        dp[:,1:,:]  += flux_p
        dvy[:,:-1,:] -= flux_v
        dvy[:,1:,:]  += flux_v
        # z-faces
        pL, pR = p[:,:,:-1], p[:,:,1:]
        vL, vR = vz[:,:,:-1], vz[:,:,1:]
        p_star = 0.5*(pL+pR) - 0.5*Z*(vR-vL)
        v_star = 0.5*(vL+vR) - 0.5/Z*(pR-pL)
        flux_p = rho * c**2 * v_star / dx
        flux_v = p_star / (rho * dx)
        dp[:,:,:-1] -= flux_p
        dp[:,:,1:]  += flux_p
        dvz[:,:,:-1] -= flux_v
        dvz[:,:,1:]  += flux_v
        # Euler time step
        p  += dt * dp
        vx += dt * dvx
        vy += dt * dvy
        vz += dt * dvz
        # boundary absorption
        p *= R_walls
        # source
        if n == 1: p[si] += 1.0
        ir_wave.append(p[ri])
    ir_wave = np.array(ir_wave)
    from scipy.signal import resample
    target_len = int(len(ir_wave) * sr / (1.0 / dt))
    ir = resample(ir_wave, target_len) if target_len > 0 else ir_wave
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, int(sr))
    return Result("DG (P0)", "Wave (TD)", ir, int(sr), m,
                  f"Grid {Nx}x{Ny}x{Nz}, DG upwind, f<{max_freq}Hz",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


# ═══════════════════════════════════════════════════════════════════
# 19. LBM — Lattice Boltzmann Method (D3Q19)
# ═══════════════════════════════════════════════════════════════════
def run_lbm(room, src, rec, sr=44100, duration=0.3, max_freq=300, **kw):
    t0 = time.perf_counter()
    Nx, Ny, Nz, dx = _wave_grid(room, max_freq, ppw=5)
    dt_lbm = dx / (np.sqrt(3) * room.c)
    Nt = int(duration / dt_lbm)
    si, ri = _src_rec_idx(room, src, rec, Nx, Ny, Nz, dx)
    # D3Q19 velocities and weights
    e = np.array([
        [0,0,0],
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
        [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
        [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
        [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],
    ], dtype=np.int32)
    w = np.array([1/3] + [1/18]*6 + [1/36]*12)
    opp = [0, 2,1,4,3,6,5, 10,9,8,7, 14,13,12,11, 18,17,16,15]
    cs2 = 1.0 / 3.0
    tau = 0.52  # relaxation (slightly above 0.5 for low viscosity)
    # distribution functions
    f = np.zeros((19, Nx, Ny, Nz))
    rho0 = 1.0
    for i in range(19):
        f[i] = w[i] * rho0
    # boundary mask
    is_wall = np.zeros((Nx, Ny, Nz), dtype=bool)
    is_wall[0,:,:] = is_wall[-1,:,:] = True
    is_wall[:,0,:] = is_wall[:,-1,:] = True
    is_wall[:,:,0] = is_wall[:,:,-1] = True
    ir_wave = []
    for n in range(Nt):
        # macro quantities
        rho_field = f.sum(axis=0)
        # collision (BGK)
        for i in range(19):
            f_eq = w[i] * rho_field  # simplified: no velocity term (acoustic limit)
            f[i] += -(f[i] - f_eq) / tau
        # source injection
        if n == 1:
            for i in range(19):
                f[i][si] += w[i] * 0.01
        # streaming
        f_new = np.zeros_like(f)
        for i in range(19):
            f_new[i] = np.roll(np.roll(np.roll(f[i], e[i,0], axis=0), e[i,1], axis=1), e[i,2], axis=2)
        # bounce-back at walls
        for i in range(1, 19):
            f_new[i][is_wall] = f[opp[i]][is_wall]
        f = f_new
        # record pressure perturbation
        rho_rec = f[:, ri[0], ri[1], ri[2]].sum()
        ir_wave.append(rho_rec - rho0)
    ir_wave = np.array(ir_wave)
    from scipy.signal import resample
    target_len = int(len(ir_wave) * sr / (1.0 / dt_lbm))
    ir = resample(ir_wave, target_len) if target_len > 0 else ir_wave
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, int(sr))
    return Result("LBM (D3Q19)", "Wave (TD)", ir, int(sr), m,
                  f"Grid {Nx}x{Ny}x{Nz}, D3Q19, f<{max_freq}Hz",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


# ═══════════════════════════════════════════════════════════════════
# 20. FEM HELMHOLTZ (frequency-domain, regular hex grid)
# ═══════════════════════════════════════════════════════════════════
def run_fem_helmholtz(room, src, rec, sr=44100, duration=1.0, max_freq=500, n_freqs=60, **kw):
    t0 = time.perf_counter()
    from scipy.sparse import diags, eye, kron
    from scipy.sparse.linalg import spsolve
    c = room.c; rho = room.rho
    Nx, Ny, Nz, dx = _wave_grid(room, max_freq, ppw=6)
    N_dof = Nx * Ny * Nz
    si, ri = _src_rec_idx(room, src, rec, Nx, Ny, Nz, dx)
    si_flat = si[0] * Ny * Nz + si[1] * Nz + si[2]
    ri_flat = ri[0] * Ny * Nz + ri[1] * Nz + ri[2]
    # 1D Laplacian
    def lap1d(n):
        return diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csc') / dx**2
    Ix = eye(Nx, format='csc')
    Iy = eye(Ny, format='csc')
    Iz = eye(Nz, format='csc')
    Lx = lap1d(Nx); Ly = lap1d(Ny); Lz = lap1d(Nz)
    K = -(kron(kron(Lx, Iy), Iz) + kron(kron(Ix, Ly), Iz) + kron(kron(Ix, Iy), Lz))
    M = eye(N_dof, format='csc') * dx**3
    # damping: boundary nodes
    C_diag = np.zeros(N_dof)
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                if x == 0 or x == Nx-1 or y == 0 or y == Ny-1 or z == 0 or z == Nz-1:
                    flat = x * Ny * Nz + y * Nz + z
                    alpha_avg = room.mean_alpha_at(max_freq / 2)
                    Z_wall = rho * c / max(alpha_avg, 0.01)
                    C_diag[flat] = rho * c**2 / Z_wall * dx**2
    C = diags(C_diag, 0, format='csc')
    # source vector
    f_vec = np.zeros(N_dof)
    f_vec[si_flat] = 1.0
    # solve at discrete frequencies
    freqs = np.linspace(20, max_freq, n_freqs)
    H = np.zeros(n_freqs, dtype=complex)
    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        k2 = (omega / c) ** 2
        A = K - k2 * M + 1j * omega * C
        try:
            p_sol = spsolve(A, f_vec)
            H[i] = p_sol[ri_flat]
        except Exception:
            H[i] = 0.0
    # interpolate to full frequency grid and IFFT
    N_ir = int(sr * duration)
    N_fft = N_ir // 2 + 1
    f_fft = np.linspace(0, sr / 2, N_fft)
    H_full = np.interp(f_fft, freqs, np.abs(H)) * np.exp(1j * np.interp(f_fft, freqs, np.angle(H)))
    H_full[f_fft > max_freq] = 0.0
    ir = np.fft.irfft(H_full, n=N_ir)
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("FEM Helmholtz", "Wave (FD)", ir, sr, m,
                  f"{N_dof} DOF, {n_freqs} frequencies, f<{max_freq}Hz",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


# ═══════════════════════════════════════════════════════════════════
# 21. BEM (Boundary Element Method, collocation)
# ═══════════════════════════════════════════════════════════════════
def run_bem(room, src, rec, sr=44100, duration=1.0, max_freq=300, patches_per_wall=25, **kw):
    t0 = time.perf_counter()
    c = room.c; rho = room.rho
    Lx, Ly, Lz = room.Lx, room.Ly, room.Lz
    src_a, rec_a = np.asarray(src), np.asarray(rec)
    # build surface mesh (quad patches per wall)
    centers, normals, areas, alphas = [], [], [], []
    wall_defs = [
        ("floor",   lambda u,v: [u*Lx,v*Ly,0],      [0,0,1],  Lx,Ly),
        ("ceiling", lambda u,v: [u*Lx,v*Ly,Lz],     [0,0,-1], Lx,Ly),
        ("left",    lambda u,v: [0,u*Ly,v*Lz],       [1,0,0],  Ly,Lz),
        ("right",   lambda u,v: [Lx,u*Ly,v*Lz],      [-1,0,0], Ly,Lz),
        ("front",   lambda u,v: [u*Lx,0,v*Lz],       [0,1,0],  Lx,Lz),
        ("back",    lambda u,v: [u*Lx,Ly,v*Lz],      [0,-1,0], Lx,Lz),
    ]
    ppw = int(np.sqrt(patches_per_wall))
    for sn, pos_fn, n_vec, W, H in wall_defs:
        area = W * H / ppw**2
        for iu in range(ppw):
            for iv in range(ppw):
                u, v = (iu+0.5)/ppw, (iv+0.5)/ppw
                centers.append(pos_fn(u, v))
                normals.append(n_vec)
                areas.append(area)
                alphas.append(room.alpha_at(sn, max_freq/2))
    centers = np.array(centers, dtype=float)
    normals_arr = np.array(normals, dtype=float)
    areas_arr = np.array(areas)
    alphas_arr = np.array(alphas)
    NP = len(centers)
    # impedance per patch
    Z_patch = rho * c / np.maximum(alphas_arr, 0.01)
    # solve Helmholtz BIE at discrete frequencies
    n_freqs = 40
    freqs = np.linspace(20, max_freq, n_freqs)
    H_tf = np.zeros(n_freqs, dtype=complex)
    for fi, freq in enumerate(freqs):
        k = 2 * np.pi * freq / c
        # Green's function: G(r) = exp(ikr) / (4*pi*r)
        # BIE (collocation): (I/2 + D + ik*beta*S) p = p_inc
        # where beta = rho*c/Z (admittance)
        beta = rho * c / Z_patch
        # assemble
        A = np.eye(NP) * 0.5
        rhs = np.zeros(NP, dtype=complex)
        for i in range(NP):
            r_src = np.linalg.norm(centers[i] - src_a)
            rhs[i] = np.exp(1j * k * r_src) / (4 * np.pi * r_src + 1e-10)
            for j in range(NP):
                if i == j: continue
                r_vec = centers[i] - centers[j]
                r = np.linalg.norm(r_vec) + 1e-10
                G = np.exp(1j * k * r) / (4 * np.pi * r)
                dGdn = G * (1j*k - 1/r) * np.dot(r_vec/r, normals_arr[j])
                A[i, j] += dGdn * areas_arr[j]
                A[i, j] += 1j * k * beta[j] * G * areas_arr[j]
        try:
            p_surf = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            p_surf = np.zeros(NP, dtype=complex)
        # evaluate at receiver
        p_rec = 0.0 + 0j
        r_direct = np.linalg.norm(rec_a - src_a)
        p_rec += np.exp(1j * k * r_direct) / (4 * np.pi * r_direct + 1e-10)
        for j in range(NP):
            r_vec = rec_a - centers[j]
            r = np.linalg.norm(r_vec) + 1e-10
            G = np.exp(1j * k * r) / (4 * np.pi * r)
            dGdn = G * (1j*k - 1/r) * np.dot(r_vec/r, normals_arr[j])
            p_rec += (dGdn + 1j*k*beta[j]*G) * p_surf[j] * areas_arr[j]
        H_tf[fi] = p_rec
    # IFFT
    N_ir = int(sr * duration)
    N_fft = N_ir // 2 + 1
    f_fft = np.linspace(0, sr/2, N_fft)
    H_full = np.interp(f_fft, freqs, np.abs(H_tf)) * np.exp(1j * np.interp(f_fft, freqs, np.angle(H_tf)))
    H_full[f_fft > max_freq] = 0
    ir = np.fft.irfft(H_full, n=N_ir)
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("BEM", "Wave (FD)", ir, sr, m,
                  f"{NP} surface patches, {n_freqs} freqs, f<{max_freq}Hz",
                  time.perf_counter() - t0, band_limited=True, max_freq=max_freq)


# ═══════════════════════════════════════════════════════════════════
# 22. DIFFUSION EQUATION (energy-based PDE)
# ═══════════════════════════════════════════════════════════════════
def run_diffusion(room, src, rec, sr=44100, duration=2.0, **kw):
    t0 = time.perf_counter()
    c = room.c; V = room.volume
    mean_free_path = 4 * V / room.total_surface
    D = c * mean_free_path / 3.0  # diffusion coefficient
    Lx, Ly, Lz = room.Lx, room.Ly, room.Lz
    dx = min(Lx, Ly, Lz) / 10
    Nx = max(int(Lx / dx), 3); Ny = max(int(Ly / dx), 3); Nz = max(int(Lz / dx), 3)
    dx = Lx / Nx
    dt = 0.1 * dx**2 / (D + 1e-10)
    Nt = int(duration / dt)
    si = tuple(np.clip([int(s/dx) for s in src], 1, [Nx-2, Ny-2, Nz-2]))
    ri = tuple(np.clip([int(r/dx) for r in rec], 1, [Nx-2, Ny-2, Nz-2]))
    # boundary exchange coefficient h = c*alpha/4
    w = np.zeros((Nx, Ny, Nz))
    w[si] = 1.0
    ir_energy = []
    for n in range(min(Nt, 50000)):
        wp = np.pad(w, 1, mode='edge')
        lap = (wp[2:,1:-1,1:-1] + wp[:-2,1:-1,1:-1] +
               wp[1:-1,2:,1:-1] + wp[1:-1,:-2,1:-1] +
               wp[1:-1,1:-1,2:] + wp[1:-1,1:-1,:-2] -
               6*wp[1:-1,1:-1,1:-1]) / dx**2
        w += dt * D * lap
        # boundary absorption
        for sn, slc in [("left", (0,slice(None),slice(None))),
                         ("right", (-1,slice(None),slice(None))),
                         ("front", (slice(None),0,slice(None))),
                         ("back", (slice(None),-1,slice(None))),
                         ("floor", (slice(None),slice(None),0)),
                         ("ceiling", (slice(None),slice(None),-1))]:
            alpha = room.alpha_at(sn, 500)
            h = c * alpha / 4
            w[slc] -= dt * h / dx * w[slc]
        w = np.maximum(w, 0)
        ir_energy.append(w[ri])
    ir_energy = np.array(ir_energy)
    # convert energy decay to IR: noise * sqrt(energy)
    from scipy.signal import resample
    target_len = int(sr * duration)
    if len(ir_energy) > 0:
        energy_resampled = resample(ir_energy, target_len) if len(ir_energy) != target_len else ir_energy
    else:
        energy_resampled = np.zeros(target_len)
    ir = synth_noise_decay(1.0, sr, duration, seed=104)
    ir *= np.sqrt(np.maximum(energy_resampled, 0) + 1e-30)
    if np.max(np.abs(ir)) > 0: ir /= np.max(np.abs(ir))
    m = all_metrics(ir, sr)
    return Result("Diffusion Eq.", "Energy PDE", ir, sr, m,
                  f"Grid {Nx}x{Ny}x{Nz}, D={D:.2f} m²/s",
                  time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════
# ENGINE REGISTRY
# ═══════════════════════════════════════════════════════════════════
ENGINE_REGISTRY = {
    # --- Statistical ---
    "Sabine":         {"func": run_sabine,         "cat": "Statistical",   "desc": "Sabine diffuse-field RT60 formula"},
    "Eyring":         {"func": run_eyring,         "cat": "Statistical",   "desc": "Eyring-Norris RT60 (high absorption)"},
    "SEA":            {"func": run_sea,             "cat": "Statistical",   "desc": "Statistical Energy Analysis (modal density + CLF)"},
    # --- Modal ---
    "Axial Modes":    {"func": run_axial_modes,     "cat": "Modal",         "desc": "1D standing waves between parallel walls"},
    "Full Modal":     {"func": run_full_modal,      "cat": "Modal",         "desc": "All analytical (n,m,l) room modes"},
    "Modal ROM":      {"func": run_modal_rom,       "cat": "Modal",         "desc": "Reduced-order model (truncated mode basis)"},
    # --- Geometric ---
    "Image Source":   {"func": run_ism,             "cat": "Geometric",     "desc": "Allen & Berkley image source method"},
    "Ray Tracing":    {"func": run_ray_tracing,     "cat": "Geometric",     "desc": "Stochastic Monte Carlo ray tracing"},
    "Cone Tracing":   {"func": run_cone_tracing,    "cat": "Geometric",     "desc": "Ray tracing with angular spread (cones)"},
    "Beam Tracing":   {"func": run_beam_tracing,    "cat": "Geometric",     "desc": "Volumetric beams (exact for shoebox)"},
    "Phonon Tracing": {"func": run_phonon_tracing,  "cat": "Geometric",     "desc": "Energy packets with Lambert scattering"},
    "Path Tracing":   {"func": run_path_tracing,    "cat": "Geometric",     "desc": "Bidirectional Monte Carlo path connections"},
    "Radiosity":      {"func": run_radiosity,       "cat": "Geometric",     "desc": "Diffuse energy exchange between surface patches"},
    # --- Wave (Time Domain) ---
    "FDTD":           {"func": run_fdtd,            "cat": "Wave (TD)",     "desc": "Finite-Difference Time-Domain (2nd order)"},
    "PSTD":           {"func": run_pstd,            "cat": "Wave (TD)",     "desc": "Pseudo-Spectral TD (DCT derivatives)"},
    "TLM":            {"func": run_tlm,             "cat": "Wave (TD)",     "desc": "Transmission Line Matrix method"},
    "ARD":            {"func": run_ard,             "cat": "Wave (TD)",     "desc": "Adaptive Rectangular Decomposition (DCT modes)"},
    "DG (P0)":        {"func": run_dg,              "cat": "Wave (TD)",     "desc": "Discontinuous Galerkin (P0 upwind flux)"},
    "LBM (D3Q19)":    {"func": run_lbm,            "cat": "Wave (TD)",     "desc": "Lattice Boltzmann D3Q19 (acoustic limit)"},
    # --- Wave (Frequency Domain) ---
    "FEM Helmholtz":  {"func": run_fem_helmholtz,   "cat": "Wave (FD)",     "desc": "Finite Element Method (Helmholtz, sparse LU)"},
    "BEM":            {"func": run_bem,             "cat": "Wave (FD)",     "desc": "Boundary Element Method (Kirchhoff-Helmholtz)"},
    # --- Energy PDE ---
    "Diffusion Eq.":  {"func": run_diffusion,       "cat": "Energy PDE",    "desc": "Acoustic energy diffusion equation"},
}

ENGINE_COLORS = [
    "#00e5ff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8",
    "#ff922b", "#20c997", "#339af0", "#f06595", "#94d82d",
    "#845ef7", "#e599f7", "#66d9e8", "#ffe066", "#ff8787",
    "#69db7c", "#748ffc", "#ffa94d", "#63e6be", "#da77f2",
    "#a9e34b", "#4dabf7",
]

def get_engine_color(name):
    names = list(ENGINE_REGISTRY.keys())
    idx = names.index(name) if name in names else 0
    return ENGINE_COLORS[idx % len(ENGINE_COLORS)]
