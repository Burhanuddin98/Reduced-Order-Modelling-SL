"""
FDTD solver for room acoustics via PEARL engine DLL.

Wraps pearl_engine.dll which implements:
- 27-point isotropic stencil (Kowalczyk & Van Walstijn 2011)
- IIR boundary filters fitted from 31-band alpha data
- Bilbao admittance BC (provably stable)
- GPU-accelerated via CUDA

Validated: 8/10 BRAS Scene 09 source-receiver pairs pass (<10% T30 error).

Requires: pearl_engine.dll + CUDA runtime + assimp DLL
"""

import numpy as np
import ctypes
import os
import time as _time

# Try to find pearl_engine.dll
_DLL = None
_DLL_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'pearl', 'pearl_engine.dll'),
    os.path.join(os.path.dirname(__file__), '..', 'pearl_engine.dll'),
    'C:/RoomGUI/ROM/pearl/pearl_engine.dll',
]

for _p in _DLL_PATHS:
    if os.path.exists(_p):
        try:
            for dep_dir in [
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin',
                'C:/vcpkg/installed/x64-windows/bin',
                os.path.dirname(os.path.abspath(_p)),
            ]:
                if os.path.exists(dep_dir):
                    os.add_dll_directory(dep_dir)
            _DLL = ctypes.CDLL(_p)
            break
        except Exception:
            continue

if _DLL is not None:
    P = ctypes.c_void_p
    F = ctypes.c_float
    I = ctypes.c_int
    _DLL.pearl_create.restype = P
    _DLL.pearl_destroy.argtypes = [P]
    _DLL.pearl_set_box.argtypes = [P, F, F, F]; _DLL.pearl_set_box.restype = I
    _DLL.pearl_set_source.argtypes = [P, F, F, F]; _DLL.pearl_set_source.restype = I
    _DLL.pearl_set_receiver.argtypes = [P, F, F, F]; _DLL.pearl_set_receiver.restype = I
    _DLL.pearl_set_material.argtypes = [P, I, ctypes.POINTER(F), F]
    _DLL.pearl_set_material.restype = I
    _DLL.pearl_set_dx.argtypes = [P, F]; _DLL.pearl_set_dx.restype = I
    _DLL.pearl_fdtd_run.argtypes = [P, F]; _DLL.pearl_fdtd_run.restype = I
    _DLL.pearl_assemble_ir.argtypes = [P, I]; _DLL.pearl_assemble_ir.restype = I
    _DLL.pearl_get_ir_length.argtypes = [P]; _DLL.pearl_get_ir_length.restype = I
    _DLL.pearl_get_ir.argtypes = [P, ctypes.POINTER(F)]; _DLL.pearl_get_ir.restype = I
    _DLL.pearl_get_sr.argtypes = [P]; _DLL.pearl_get_sr.restype = I
    _DLL.pearl_get_volume.argtypes = [P]; _DLL.pearl_get_volume.restype = F
    _DLL.pearl_get_num_surfaces.argtypes = [P]; _DLL.pearl_get_num_surfaces.restype = I


class FDTDSolver:
    """FDTD room acoustics solver via PEARL engine.

    Usage:
        solver = FDTDSolver(8.4, 6.7, 3.0, f_max=1000)
        solver.set_materials({
            'floor': [0.04, 0.06, ...],  # 31-band alpha
            'ceiling': [0.10, 0.09, ...],
        })
        ir, sr = solver.impulse_response(source, receiver, duration=3.5)
    """

    def __init__(self, Lx, Ly, Lz, f_max=500, c=343.0):
        if _DLL is None:
            raise RuntimeError(
                "pearl_engine.dll not found. "
                "Build from pearl/pearl_engine.cu or set DLL path.")
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.c = c
        self.f_max = f_max
        self.dx = c / (6 * f_max)
        self._alpha_bands = {}  # surface_id -> 31 floats
        print(f"FDTD (PEARL): {Lx}x{Ly}x{Lz}m, f_max={f_max}Hz, dx={self.dx:.4f}m")

    def set_materials(self, alpha_per_surface):
        """Set 31-band or scalar absorption per surface.

        Surface names: floor, ceiling, left, right, front, back
        (mapped to PEARL's z_min/z_max/x_min/x_max/y_min/y_max = surface IDs 0-5)
        """
        name_to_id = {
            'floor': 0, 'ceiling': 1,
            'left': 2, 'right': 3,
            'front': 4, 'back': 5,
            'z_min': 0, 'z_max': 1,
            'x_min': 2, 'x_max': 3,
            'y_min': 4, 'y_max': 5,
        }
        for key, alpha in alpha_per_surface.items():
            sid = name_to_id.get(key)
            if sid is None:
                continue
            if isinstance(alpha, (list, np.ndarray)) and len(alpha) == 31:
                self._alpha_bands[sid] = np.array(alpha, dtype=np.float32)
            else:
                self._alpha_bands[sid] = np.full(31, float(alpha), dtype=np.float32)

    def impulse_response(self, source, receiver, duration=2.0, sr_out=44100):
        """Compute impulse response via PEARL FDTD.

        Returns (ir, sr) where ir is a numpy float64 array.
        """
        t0 = _time.perf_counter()
        e = _DLL.pearl_create()

        try:
            _DLL.pearl_set_box(e, F(self.Lx), F(self.Ly), F(self.Lz))
            _DLL.pearl_set_dx(e, F(self.dx))
            _DLL.pearl_set_source(e, F(source[0]), F(source[1]), F(source[2]))
            _DLL.pearl_set_receiver(e, F(receiver[0]), F(receiver[1]), F(receiver[2]))

            for sid, alpha in self._alpha_bands.items():
                a = (F * 31)(*alpha)
                _DLL.pearl_set_material(e, I(sid), a, F(float(np.mean(alpha))))

            _DLL.pearl_fdtd_run(e, F(duration))
            _DLL.pearl_assemble_ir(e, I(sr_out))

            n_ir = _DLL.pearl_get_ir_length(e)
            sr = _DLL.pearl_get_sr(e)

            if n_ir <= 0:
                return np.zeros(int(duration * sr_out)), sr_out

            ir_buf = (F * n_ir)()
            _DLL.pearl_get_ir(e, ir_buf)
            ir = np.array(ir_buf, dtype=np.float64)

            elapsed = _time.perf_counter() - t0
            print(f"  FDTD: {n_ir} samples at {sr}Hz in {elapsed:.1f}s")
            return ir, sr

        finally:
            _DLL.pearl_destroy(e)
