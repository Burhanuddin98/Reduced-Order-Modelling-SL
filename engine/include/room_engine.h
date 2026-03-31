/**
 * room_engine.h — Core C API for room acoustics simulation
 *
 * Single library. No Python dependencies in the hot path.
 * Python calls this via ctypes for the user-facing API.
 *
 * Pipeline:
 *   1. Import geometry (STL/OBJ/SKP/MSH)
 *   2. Mesh with Gmsh (tet or hex)
 *   3. Assemble SEM operators (S, M, B)
 *   4. Compute eigenmodes (shift-invert Lanczos with UMFPACK)
 *   5. Modal synthesis (analytical per-mode, GPU parallel)
 *   6. ISM for high frequencies
 *   7. Crossover blend → full IR
 *   8. Compute metrics (T30, EDT, C80)
 *   9. Write WAV
 */

#ifndef ROOM_ENGINE_H
#define ROOM_ENGINE_H

#include <stdint.h>

#ifdef _WIN32
  #ifdef ROOM_ENGINE_EXPORTS
    #define RE_API __declspec(dllexport)
  #else
    #define RE_API __declspec(dllimport)
  #endif
#else
  #define RE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Opaque handles
 * ================================================================ */

typedef struct RE_Mesh RE_Mesh;
typedef struct RE_Operators RE_Operators;
typedef struct RE_Modes RE_Modes;
typedef struct RE_Room RE_Room;

/* ================================================================
 * Room lifecycle
 * ================================================================ */

/** Create empty room context */
RE_API RE_Room* re_room_create(void);

/** Import geometry from file (STL, OBJ, MSH) */
RE_API int re_room_load_geometry(RE_Room* room, const char* filepath,
                                  double h_target);

/** Set material for a surface label */
RE_API int re_room_set_material(RE_Room* room, const char* surface,
                                 double impedance_Z);

/** Set default material for unlabeled surfaces */
RE_API int re_room_set_default_material(RE_Room* room, double impedance_Z);

/**
 * Build: mesh + assemble operators + compute eigenmodes.
 * This is the expensive one-time step.
 *
 * @param n_modes  Number of eigenmodes to compute
 * @param use_gpu  1 = GPU acceleration, 0 = CPU only
 */
RE_API int re_room_build(RE_Room* room, int n_modes, int use_gpu);

/* ================================================================
 * Impulse Response
 * ================================================================ */

/**
 * Compute impulse response.
 *
 * @param src_x,y,z   Source position [m]
 * @param rec_x,y,z   Receiver position [m]
 * @param duration     IR duration [s]
 * @param sample_rate  Output sample rate [Hz]
 * @param ir_out       Output buffer (allocated by caller, length = duration * sample_rate)
 * @param ir_length    Output: actual IR length written
 */
RE_API int re_room_impulse_response(
    RE_Room* room,
    double src_x, double src_y, double src_z,
    double rec_x, double rec_y, double rec_z,
    double duration, int sample_rate,
    double* ir_out, int* ir_length
);

/* ================================================================
 * Metrics
 * ================================================================ */

typedef struct {
    double T30;      /* Reverberation time [s] */
    double T20;
    double EDT;      /* Early Decay Time [s] */
    double C80;      /* Clarity [dB] */
    double D50;      /* Definition [0-1] */
    double TS;       /* Centre Time [ms] */
    double T30_R2;   /* T30 fit quality */
} RE_Metrics;

/** Compute ISO 3382 metrics from an impulse response */
RE_API int re_metrics_compute(const double* ir, int length,
                               int sample_rate, RE_Metrics* out);

/* ================================================================
 * WAV output
 * ================================================================ */

/** Save impulse response as WAV file */
RE_API int re_wav_write(const char* path, const double* ir, int length,
                         int sample_rate);

/** Convolve IR with audio file, write result */
RE_API int re_auralize(const char* audio_in, const char* audio_out,
                        const double* ir, int ir_length, int sample_rate);

/* ================================================================
 * Query
 * ================================================================ */

RE_API int re_room_get_n_dof(const RE_Room* room);
RE_API int re_room_get_n_modes(const RE_Room* room);
RE_API double re_room_get_volume(const RE_Room* room);
RE_API double re_room_get_f_max(const RE_Room* room);
RE_API int re_room_get_n_surfaces(const RE_Room* room);
RE_API const char* re_room_get_surface_name(const RE_Room* room, int idx);

/** Free room and all resources */
RE_API void re_room_free(RE_Room* room);

#ifdef __cplusplus
}
#endif

#endif /* ROOM_ENGINE_H */
