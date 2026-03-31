"""
Python wrapper for the CUDA Helmholtz solver.

Loads the compiled shared library via ctypes and provides a numpy-friendly
interface. Falls back to scipy if the library is not compiled.
"""

import numpy as np
import ctypes
import os
import platform

_LIB = None
_LIB_PATH = None


def _find_lib():
    """Find the compiled helmholtz library."""
    global _LIB, _LIB_PATH

    if _LIB is not None:
        return _LIB

    base = os.path.dirname(os.path.abspath(__file__))

    if platform.system() == 'Windows':
        candidates = [
            os.path.join(base, 'helmholtz.dll'),
            os.path.join(base, 'build', 'helmholtz.dll'),
            os.path.join(base, 'build', 'Release', 'helmholtz.dll'),
        ]
    else:
        candidates = [
            os.path.join(base, 'libhelmholtz.so'),
            os.path.join(base, 'build', 'libhelmholtz.so'),
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                _LIB = ctypes.CDLL(path)
                _LIB_PATH = path
                _setup_signatures(_LIB)
                return _LIB
            except OSError as e:
                print(f"Warning: found {path} but failed to load: {e}")

    return None


def _setup_signatures(lib):
    """Set ctypes function signatures."""
    lib.helmholtz_init.argtypes = [
        ctypes.c_int,  # N
        ctypes.c_int,  # nnz
        ctypes.POINTER(ctypes.c_int),    # row_ptr
        ctypes.POINTER(ctypes.c_int),    # col_idx
        ctypes.POINTER(ctypes.c_double), # S_vals
        ctypes.POINTER(ctypes.c_double), # M_diag
        ctypes.c_int,  # use_gpu
    ]
    lib.helmholtz_init.restype = ctypes.c_void_p

    lib.helmholtz_solve.argtypes = [
        ctypes.c_void_p,  # ctx
        ctypes.c_double,  # omega
        ctypes.c_double,  # c
        ctypes.POINTER(ctypes.c_double),  # C_diag
        ctypes.POINTER(ctypes.c_double),  # f_rhs
        ctypes.POINTER(ctypes.c_double),  # x_real
        ctypes.POINTER(ctypes.c_double),  # x_imag
    ]
    lib.helmholtz_solve.restype = ctypes.c_int

    lib.helmholtz_sweep.argtypes = [
        ctypes.c_void_p,  # ctx
        ctypes.c_int,     # n_freqs
        ctypes.POINTER(ctypes.c_double),  # omegas
        ctypes.c_double,  # c
        ctypes.POINTER(ctypes.c_double),  # C_diag
        ctypes.POINTER(ctypes.c_double),  # f_rhs
        ctypes.c_int,     # rec_idx
        ctypes.POINTER(ctypes.c_double),  # H_real
        ctypes.POINTER(ctypes.c_double),  # H_imag
    ]
    lib.helmholtz_sweep.restype = ctypes.c_int

    lib.helmholtz_free.argtypes = [ctypes.c_void_p]
    lib.helmholtz_free.restype = None

    lib.helmholtz_get_N.argtypes = [ctypes.c_void_p]
    lib.helmholtz_get_N.restype = ctypes.c_int

    lib.helmholtz_is_gpu.argtypes = [ctypes.c_void_p]
    lib.helmholtz_is_gpu.restype = ctypes.c_int


def _to_cptr(arr, dtype=ctypes.c_double):
    """Convert numpy array to ctypes pointer."""
    arr = np.ascontiguousarray(arr)
    return arr.ctypes.data_as(ctypes.POINTER(dtype))


class HelmholtzSolver:
    """
    GPU-accelerated Helmholtz solver for room acoustics.

    Usage:
        solver = HelmholtzSolver(ops, mesh, use_gpu=True)
        H = solver.sweep(freqs, src_pos, rec_idx, bc_params)
        ir = solver.to_ir(H, freqs, sr=44100)
    """

    def __init__(self, ops, mesh, use_gpu=True):
        """
        Initialize with assembled SEM operators.

        Parameters
        ----------
        ops : dict with S, M_diag, B_total
        mesh : mesh object with .N_dof, .x, .y, .z
        use_gpu : bool
        """
        self.N = mesh.N_dof
        self.mesh = mesh
        self.ops = ops

        S = ops['S'].tocsr()
        self.S_csr = S
        self.M_diag = np.ascontiguousarray(ops['M_diag'], dtype=np.float64)
        self.B_diag = np.ascontiguousarray(
            np.array(ops['B_total'].diagonal()), dtype=np.float64)

        # Try to load native library
        lib = _find_lib()
        self._lib = lib
        self._ctx = None

        if lib is not None:
            row_ptr = np.ascontiguousarray(S.indptr, dtype=np.int32)
            col_idx = np.ascontiguousarray(S.indices, dtype=np.int32)
            S_vals = np.ascontiguousarray(S.data, dtype=np.float64)

            self._ctx = lib.helmholtz_init(
                self.N, S.nnz,
                _to_cptr(row_ptr, ctypes.c_int),
                _to_cptr(col_idx, ctypes.c_int),
                _to_cptr(S_vals),
                _to_cptr(self.M_diag),
                1 if use_gpu else 0,
            )

            if self._ctx:
                gpu_str = "GPU" if lib.helmholtz_is_gpu(self._ctx) else "CPU"
                print(f"  HelmholtzSolver: N={self.N}, backend={gpu_str}")
            else:
                print("  HelmholtzSolver: native init failed, using scipy fallback")
        else:
            print("  HelmholtzSolver: native library not found, using scipy fallback")

    def sweep(self, freqs, src_pos, rec_idx, bc_params, sigma=0.5, c=343.0,
              rho=1.2):
        """
        Compute transfer function H(f) at all frequencies.

        Returns
        -------
        H : complex array (len(freqs),)
        """
        rc2 = rho * c ** 2

        # Source vector
        if hasattr(self.mesh, '_ensure_coords'):
            self.mesh._ensure_coords()
        r2 = (self.mesh.x - src_pos[0])**2 + (self.mesh.y - src_pos[1])**2
        if hasattr(self.mesh, 'z'):
            r2 += (self.mesh.z - src_pos[2])**2
        f_rhs = np.ascontiguousarray(
            self.M_diag * np.exp(-r2 / sigma**2), dtype=np.float64)

        # Damping vector
        if 'Z_per_node' in bc_params:
            Z = np.asarray(bc_params['Z_per_node'], dtype=np.float64)
        elif 'Z' in bc_params:
            Z = np.full(self.N, bc_params['Z'], dtype=np.float64)
        else:
            Z = np.full(self.N, 1e15, dtype=np.float64)
        C_diag = np.ascontiguousarray(rc2 * self.B_diag / Z, dtype=np.float64)

        freqs = np.asarray(freqs, dtype=np.float64)
        omegas = np.ascontiguousarray(2 * np.pi * freqs, dtype=np.float64)

        if self._ctx and self._lib:
            # Native path
            H_real = np.zeros(len(freqs), dtype=np.float64)
            H_imag = np.zeros(len(freqs), dtype=np.float64)

            ret = self._lib.helmholtz_sweep(
                self._ctx, len(freqs),
                _to_cptr(omegas), c,
                _to_cptr(C_diag), _to_cptr(f_rhs),
                rec_idx,
                _to_cptr(H_real), _to_cptr(H_imag),
            )

            if ret != 0:
                print("  Warning: native sweep failed, falling back to scipy")
                return self._sweep_scipy(freqs, C_diag, f_rhs, rec_idx, c)

            return H_real + 1j * H_imag
        else:
            return self._sweep_scipy(freqs, C_diag, f_rhs, rec_idx, c)

    def _sweep_scipy(self, freqs, C_diag, f_rhs, rec_idx, c):
        """Scipy fallback for when native library is unavailable."""
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve

        S = self.S_csr.tocsc().astype(complex)
        M = self.M_diag
        H = np.zeros(len(freqs), dtype=complex)

        for i, freq in enumerate(freqs):
            omega = 2 * np.pi * freq
            k2 = (omega / c) ** 2
            diag_shift = -k2 * M + 1j * omega * C_diag
            A = S + diags(diag_shift, format='csc')
            try:
                p = spsolve(A, f_rhs)
                H[i] = p[rec_idx]
            except Exception:
                H[i] = 0.0

        return H

    def to_ir(self, H, freqs, sr=44100, T=None):
        """Convert transfer function to impulse response via IFFT."""
        from room_acoustics.freq_domain import transfer_function_to_ir
        return transfer_function_to_ir(H, freqs, sr=sr, T=T)

    def __del__(self):
        if self._ctx and self._lib:
            self._lib.helmholtz_free(self._ctx)
            self._ctx = None
