/**
 * helmholtz.h — GPU-accelerated frequency-domain Helmholtz solver
 *
 * Solves (S - omega^2 * M + i*omega * C) * x = f
 * for many omega values efficiently.
 *
 * S: sparse symmetric stiffness matrix (CSR format)
 * M: diagonal mass matrix
 * C: diagonal damping matrix (from boundary impedance)
 * f: source vector
 *
 * The sparsity pattern of S is analyzed once. For each frequency,
 * only the diagonal shift changes — the numeric factorization
 * reuses the symbolic analysis.
 *
 * GPU path: cuSPARSE + cuSOLVER
 * CPU path: direct LU via custom implementation
 */

#ifndef HELMHOLTZ_H
#define HELMHOLTZ_H

#ifdef _WIN32
  #ifdef HELMHOLTZ_EXPORTS
    #define HELMHOLTZ_API __declspec(dllexport)
  #else
    #define HELMHOLTZ_API __declspec(dllimport)
  #endif
#else
  #define HELMHOLTZ_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque context handle.
 */
typedef struct HelmholtzContext HelmholtzContext;

/**
 * Initialize the solver with the stiffness matrix in CSR format.
 *
 * @param N         Number of DOFs
 * @param nnz       Number of nonzeros in S
 * @param row_ptr   CSR row pointers (length N+1)
 * @param col_idx   CSR column indices (length nnz)
 * @param S_vals    CSR values of S (length nnz, real)
 * @param M_diag    Diagonal mass matrix (length N)
 * @param use_gpu   1 = use GPU, 0 = CPU fallback
 *
 * @return Opaque context pointer, or NULL on failure.
 */
HELMHOLTZ_API HelmholtzContext* helmholtz_init(
    int N, int nnz,
    const int* row_ptr, const int* col_idx, const double* S_vals,
    const double* M_diag,
    int use_gpu
);

/**
 * Solve at a single frequency.
 *
 * Computes x such that (S - (omega/c)^2 * M + i*omega * C) * x = f
 *
 * @param ctx       Context from helmholtz_init
 * @param omega     Angular frequency (2*pi*f)
 * @param c         Speed of sound [m/s]
 * @param C_diag    Diagonal damping vector (length N, real)
 * @param f_rhs     Right-hand side (length N, real)
 * @param x_real    Output: real part of solution (length N)
 * @param x_imag    Output: imaginary part of solution (length N)
 *
 * @return 0 on success, nonzero on error.
 */
HELMHOLTZ_API int helmholtz_solve(
    HelmholtzContext* ctx,
    double omega, double c,
    const double* C_diag,
    const double* f_rhs,
    double* x_real, double* x_imag
);

/**
 * Solve at multiple frequencies, returning only the value at rec_idx.
 * This is the fast path — avoids transferring the full solution.
 *
 * @param ctx       Context from helmholtz_init
 * @param n_freqs   Number of frequencies
 * @param omegas    Angular frequencies (length n_freqs)
 * @param c         Speed of sound
 * @param C_diag    Damping vector (length N) — same for all freqs
 * @param f_rhs     Source vector (length N)
 * @param rec_idx   Receiver node index
 * @param H_real    Output: real part of H(omega) (length n_freqs)
 * @param H_imag    Output: imaginary part of H(omega) (length n_freqs)
 *
 * @return 0 on success.
 */
HELMHOLTZ_API int helmholtz_sweep(
    HelmholtzContext* ctx,
    int n_freqs, const double* omegas, double c,
    const double* C_diag,
    const double* f_rhs,
    int rec_idx,
    double* H_real, double* H_imag
);

/**
 * Free all resources.
 */
HELMHOLTZ_API void helmholtz_free(HelmholtzContext* ctx);

/**
 * Query info.
 */
HELMHOLTZ_API int helmholtz_get_N(const HelmholtzContext* ctx);
HELMHOLTZ_API int helmholtz_is_gpu(const HelmholtzContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* HELMHOLTZ_H */
