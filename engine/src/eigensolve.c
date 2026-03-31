/**
 * eigensolve.c — Shift-invert Lanczos eigenvalue solver
 *
 * Computes the smallest eigenvalues and eigenvectors of the
 * generalized eigenvalue problem: S * v = lambda * M * v
 *
 * Uses shift-invert Lanczos: instead of operating on M^{-1}S,
 * operates on (S - sigma*M)^{-1} * M, which converges to
 * eigenvalues near sigma. With sigma=0, finds smallest eigenvalues.
 *
 * The inner linear solve uses UMFPACK (LU factored once, back-sub per iteration).
 * This is the key speedup over scipy's ARPACK which refactors every iteration.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "umfpack.h"

/* ================================================================
 * Shift-invert Lanczos
 * ================================================================ */

typedef struct {
    int N;
    int nnz;

    /* CSC format of S (or S - sigma*M) */
    int *Ap;
    int *Ai;
    double *Ax;

    /* Diagonal mass */
    double *M_diag;

    /* UMFPACK symbolic + numeric for (S - sigma*M) */
    void *Symbolic;
    void *Numeric;

} EigenContext;


static EigenContext* eigen_init(
    int N, int nnz,
    const int* col_ptr, const int* row_idx, const double* S_vals,
    const double* M_diag, double sigma)
{
    EigenContext *ctx = (EigenContext*)calloc(1, sizeof(EigenContext));
    ctx->N = N;
    ctx->nnz = nnz;

    /* Copy and shift: A = S - sigma * M */
    ctx->Ap = (int*)malloc((N+1) * sizeof(int));
    ctx->Ai = (int*)malloc(nnz * sizeof(int));
    ctx->Ax = (double*)malloc(nnz * sizeof(double));
    ctx->M_diag = (double*)malloc(N * sizeof(double));

    memcpy(ctx->Ap, col_ptr, (N+1) * sizeof(int));
    memcpy(ctx->Ai, row_idx, nnz * sizeof(int));
    memcpy(ctx->Ax, S_vals, nnz * sizeof(double));
    memcpy(ctx->M_diag, M_diag, N * sizeof(double));

    /* Subtract sigma * M from diagonal */
    for (int j = 0; j < N; j++) {
        for (int p = ctx->Ap[j]; p < ctx->Ap[j+1]; p++) {
            if (ctx->Ai[p] == j) {
                ctx->Ax[p] -= sigma * M_diag[j];
                break;
            }
        }
    }

    /* UMFPACK factorization of (S - sigma*M) — done ONCE */
    double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
    umfpack_di_defaults(Control);
    Control[UMFPACK_PRL] = 0;

    int status = umfpack_di_symbolic(N, N, ctx->Ap, ctx->Ai, ctx->Ax,
                                     &ctx->Symbolic, Control, Info);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "UMFPACK symbolic failed: %d\n", status);
        free(ctx->Ap); free(ctx->Ai); free(ctx->Ax); free(ctx->M_diag);
        free(ctx);
        return NULL;
    }

    status = umfpack_di_numeric(ctx->Ap, ctx->Ai, ctx->Ax,
                                ctx->Symbolic, &ctx->Numeric, Control, Info);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "UMFPACK numeric failed: %d\n", status);
        umfpack_di_free_symbolic(&ctx->Symbolic);
        free(ctx->Ap); free(ctx->Ai); free(ctx->Ax); free(ctx->M_diag);
        free(ctx);
        return NULL;
    }

    return ctx;
}


static void eigen_free(EigenContext* ctx) {
    if (!ctx) return;
    if (ctx->Numeric) umfpack_di_free_numeric(&ctx->Numeric);
    if (ctx->Symbolic) umfpack_di_free_symbolic(&ctx->Symbolic);
    free(ctx->Ap); free(ctx->Ai); free(ctx->Ax); free(ctx->M_diag);
    free(ctx);
}


/* Solve (S - sigma*M) x = b using pre-factored UMFPACK */
static int eigen_solve(EigenContext* ctx, const double* rhs, double* sol) {
    double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
    umfpack_di_defaults(Control);
    Control[UMFPACK_PRL] = 0;

    return umfpack_di_solve(UMFPACK_A, ctx->Ap, ctx->Ai, ctx->Ax,
                            sol, rhs, ctx->Numeric, Control, Info);
}


/* M-inner product: <u, v>_M = u^T M v */
static double m_dot(const double* u, const double* v, const double* M, int N) {
    double s = 0;
    for (int i = 0; i < N; i++) s += u[i] * M[i] * v[i];
    return s;
}


/* M-norm */
static double m_norm(const double* v, const double* M, int N) {
    return sqrt(m_dot(v, v, M, N));
}


/**
 * Shift-invert Lanczos eigenvalue solver.
 *
 * Finds the n_eigs smallest eigenvalues of S v = lambda M v.
 *
 * @param N         System size
 * @param nnz       Number of nonzeros in S (CSC)
 * @param col_ptr   CSC column pointers (N+1)
 * @param row_idx   CSC row indices (nnz)
 * @param S_vals    CSC values of S (nnz)
 * @param M_diag    Diagonal mass matrix (N)
 * @param n_eigs    Number of eigenvalues to find
 * @param k_lanczos Lanczos iterations (k >= n_eigs, typically 2*n_eigs)
 * @param sigma     Shift (0.0 for smallest eigenvalues)
 * @param eigenvalues  Output: eigenvalues (n_eigs)
 * @param eigenvectors Output: eigenvectors (N * n_eigs, column-major)
 *
 * @return 0 on success
 */
#ifdef _WIN32
__declspec(dllexport)
#endif
int eigensolve_lanczos(
    int N, int nnz,
    const int* col_ptr, const int* row_idx, const double* S_vals,
    const double* M_diag,
    int n_eigs, int k_lanczos, double sigma,
    double* eigenvalues, double* eigenvectors)
{
    if (k_lanczos < n_eigs) k_lanczos = 2 * n_eigs;
    if (k_lanczos > N) k_lanczos = N;

    fprintf(stderr, "eigensolve: N=%d, n_eigs=%d, k=%d, sigma=%.2f\n",
            N, n_eigs, k_lanczos, sigma);

    /* Initialize UMFPACK factorization */
    EigenContext* ctx = eigen_init(N, nnz, col_ptr, row_idx, S_vals,
                                   M_diag, sigma);
    if (!ctx) return -1;

    /* Allocate Lanczos vectors and tridiagonal entries */
    double* V = (double*)calloc((size_t)N * k_lanczos, sizeof(double));
    double* alpha = (double*)calloc(k_lanczos, sizeof(double));
    double* beta = (double*)calloc(k_lanczos, sizeof(double));
    double* w = (double*)malloc(N * sizeof(double));
    double* Mv = (double*)malloc(N * sizeof(double));
    double* temp = (double*)malloc(N * sizeof(double));

    if (!V || !alpha || !beta || !w || !Mv || !temp) {
        fprintf(stderr, "eigensolve: memory allocation failed\n");
        eigen_free(ctx);
        return -2;
    }

    /* Starting vector: random, M-normalized */
    srand(42);
    double* v0 = V;  /* first column */
    for (int i = 0; i < N; i++) v0[i] = (double)rand() / RAND_MAX - 0.5;
    double nrm = m_norm(v0, M_diag, N);
    for (int i = 0; i < N; i++) v0[i] /= nrm;

    /* Lanczos iteration */
    /* Operating on A_inv = (S - sigma*M)^{-1} * M */
    /* At each step: w = A_inv * v_j = (S-sigma*M)^{-1} * (M * v_j) */

    for (int j = 0; j < k_lanczos; j++) {
        double* vj = V + (size_t)j * N;

        /* Mv = M * v_j */
        for (int i = 0; i < N; i++) Mv[i] = M_diag[i] * vj[i];

        /* w = (S - sigma*M)^{-1} * Mv */
        if (eigen_solve(ctx, Mv, w) != UMFPACK_OK) {
            fprintf(stderr, "eigensolve: UMFPACK solve failed at iter %d\n", j);
            break;
        }

        /* alpha_j = <w, v_j>_M */
        alpha[j] = m_dot(w, vj, M_diag, N);

        /* w = w - alpha_j * v_j */
        for (int i = 0; i < N; i++) w[i] -= alpha[j] * vj[i];

        /* w = w - beta_j * v_{j-1} */
        if (j > 0) {
            double* vjm1 = V + (size_t)(j-1) * N;
            for (int i = 0; i < N; i++) w[i] -= beta[j] * vjm1[i];
        }

        /* Full reorthogonalization */
        for (int jj = 0; jj <= j; jj++) {
            double* vjj = V + (size_t)jj * N;
            double coeff = m_dot(w, vjj, M_diag, N);
            for (int i = 0; i < N; i++) w[i] -= coeff * vjj[i];
        }

        /* beta_{j+1} = ||w||_M */
        if (j < k_lanczos - 1) {
            beta[j+1] = m_norm(w, M_diag, N);

            if (beta[j+1] < 1e-14) {
                fprintf(stderr, "eigensolve: breakdown at iter %d\n", j);
                k_lanczos = j + 1;
                break;
            }

            /* v_{j+1} = w / beta_{j+1} */
            double* vjp1 = V + (size_t)(j+1) * N;
            for (int i = 0; i < N; i++) vjp1[i] = w[i] / beta[j+1];
        }

        if ((j+1) % (k_lanczos/10 > 0 ? k_lanczos/10 : 1) == 0) {
            fprintf(stderr, "  %d/%d\n", j+1, k_lanczos);
        }
    }

    eigen_free(ctx);

    /* Solve tridiagonal eigenvalue problem (small: k x k) */
    /* Use simple QR algorithm or LAPACK dsyev */
    /* For now: build tridiagonal matrix, use LAPACK if available,
       otherwise use a simple bisection + inverse iteration */

    /* Build tridiagonal matrix T */
    double* T = (double*)calloc((size_t)k_lanczos * k_lanczos, sizeof(double));
    for (int i = 0; i < k_lanczos; i++) {
        T[i * k_lanczos + i] = alpha[i];
        if (i > 0) {
            T[i * k_lanczos + (i-1)] = beta[i];
            T[(i-1) * k_lanczos + i] = beta[i];
        }
    }

    /* Simple eigenvalue solver for symmetric tridiagonal matrix */
    /* Use Jacobi rotation method (good enough for k <= 2000) */
    double* eig_vecs_T = (double*)calloc((size_t)k_lanczos * k_lanczos, sizeof(double));
    for (int i = 0; i < k_lanczos; i++) eig_vecs_T[i * k_lanczos + i] = 1.0;

    /* Jacobi eigenvalue algorithm */
    int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off_diag = 0;
        for (int i = 0; i < k_lanczos; i++)
            for (int j = i+1; j < k_lanczos; j++)
                off_diag += T[i*k_lanczos+j] * T[i*k_lanczos+j];

        if (off_diag < 1e-24 * k_lanczos) break;

        for (int p = 0; p < k_lanczos; p++) {
            for (int q = p+1; q < k_lanczos; q++) {
                double Tpq = T[p*k_lanczos+q];
                if (fabs(Tpq) < 1e-15) continue;

                double tau = (T[q*k_lanczos+q] - T[p*k_lanczos+p]) / (2*Tpq);
                double t = (tau >= 0 ? 1 : -1) / (fabs(tau) + sqrt(1 + tau*tau));
                double cos_t = 1.0 / sqrt(1 + t*t);
                double sin_t = t * cos_t;

                /* Apply rotation to T */
                T[p*k_lanczos+p] -= t * Tpq;
                T[q*k_lanczos+q] += t * Tpq;
                T[p*k_lanczos+q] = 0;
                T[q*k_lanczos+p] = 0;

                for (int r = 0; r < k_lanczos; r++) {
                    if (r == p || r == q) continue;
                    double Trp = T[r*k_lanczos+p];
                    double Trq = T[r*k_lanczos+q];
                    T[r*k_lanczos+p] = cos_t*Trp - sin_t*Trq;
                    T[p*k_lanczos+r] = T[r*k_lanczos+p];
                    T[r*k_lanczos+q] = sin_t*Trp + cos_t*Trq;
                    T[q*k_lanczos+r] = T[r*k_lanczos+q];
                }

                /* Apply rotation to eigenvector matrix */
                for (int r = 0; r < k_lanczos; r++) {
                    double vp = eig_vecs_T[r*k_lanczos+p];
                    double vq = eig_vecs_T[r*k_lanczos+q];
                    eig_vecs_T[r*k_lanczos+p] = cos_t*vp - sin_t*vq;
                    eig_vecs_T[r*k_lanczos+q] = sin_t*vp + cos_t*vq;
                }
            }
        }
    }

    /* Extract eigenvalues (diagonal of T) and sort */
    /* The shift-invert eigenvalues are mu = 1/(lambda - sigma) */
    /* So lambda = sigma + 1/mu */
    double* mu = (double*)malloc(k_lanczos * sizeof(double));
    int* idx = (int*)malloc(k_lanczos * sizeof(int));
    for (int i = 0; i < k_lanczos; i++) {
        mu[i] = T[i*k_lanczos+i];
        idx[i] = i;
    }

    /* Convert to original eigenvalues and sort */
    double* lambda = (double*)malloc(k_lanczos * sizeof(double));
    for (int i = 0; i < k_lanczos; i++) {
        if (fabs(mu[i]) > 1e-30)
            lambda[i] = sigma + 1.0 / mu[i];
        else
            lambda[i] = 1e30;
    }

    /* Simple insertion sort by lambda */
    for (int i = 1; i < k_lanczos; i++) {
        double key = lambda[i];
        int key_idx = idx[i];
        int j = i - 1;
        while (j >= 0 && lambda[j] > key) {
            lambda[j+1] = lambda[j];
            idx[j+1] = idx[j];
            j--;
        }
        lambda[j+1] = key;
        idx[j+1] = key_idx;
    }

    /* Copy n_eigs smallest eigenvalues and corresponding eigenvectors */
    int n_out = (n_eigs < k_lanczos) ? n_eigs : k_lanczos;
    for (int i = 0; i < n_out; i++) {
        eigenvalues[i] = lambda[i];

        /* Eigenvector = V * (Ritz vector from tridiagonal) */
        /* y_i = V * z_i where z_i is the i-th column of eig_vecs_T[:, idx[i]] */
        int tri_idx = idx[i];
        double* ev_out = eigenvectors + (size_t)i * N;
        memset(ev_out, 0, N * sizeof(double));
        for (int j = 0; j < k_lanczos; j++) {
            double coeff = eig_vecs_T[j * k_lanczos + tri_idx];
            double* vj = V + (size_t)j * N;
            for (int ii = 0; ii < N; ii++) {
                ev_out[ii] += coeff * vj[ii];
            }
        }
    }

    fprintf(stderr, "eigensolve: done, %d eigenvalues computed\n", n_out);

    free(V); free(alpha); free(beta); free(w); free(Mv); free(temp);
    free(T); free(eig_vecs_T); free(mu); free(idx); free(lambda);

    return 0;
}
