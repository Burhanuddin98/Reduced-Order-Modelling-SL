"""BRAS Scene 9 — 1-second IR via IFFT reconstruction."""
import os, sys, time, json
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, find_peaks
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from romacoustics.sem import BoxMesh3D, assemble_3d
from romacoustics.solver import (
    C_AIR, RHO_AIR, ifft_frequencies, ifft_to_ir, _alpha_to_Z_internal
)
from romacoustics.ir import ImpulseResponse
from romacoustics.metrics import octave_band_metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_bras_ifft')
os.makedirs(OUT, exist_ok=True)

BRAS = r"C:\Users\bsaka\Downloads\BRAS"
MAT_DIR = os.path.join(BRAS, "surface_data", "3 Surface descriptions", "_csv", "fitted_estimates")
RIR_DIR = os.path.join(BRAS, "scene9_data", "1 Scene descriptions",
                        "09 small room (seminar room)", "RIRs", "wav")

Lx, Ly, Lz = 8.4, 6.7, 3.0
SRC = (2.0, 3.35, 1.5)
REC = (6.0, 1.5, 1.2)
FS = 44100
T_MAX = 1.0
F_MAX = 500
N_FREQ = 500

SMAP = {'z_min':'floor','z_max':'ceiling','y_min':'plaster','y_max':'plaster',
        'x_min':'windows','x_max':'concrete'}
OCTAVE_BANDS = [125, 250, 500]

def load_mat(name):
    lines = open(os.path.join(MAT_DIR, "mat_scene09_%s.csv" % name)).read().strip().split('\n')
    return np.array([float(x) for x in lines[0].split(',')]), np.array([float(x) for x in lines[2].split(',')])

def main():
    print("=" * 60)
    print("  BRAS Scene 9 -- IFFT, 0-%dHz, %.1fs" % (F_MAX, T_MAX))
    print("=" * 60)

    Ne, P = 4, 4
    print("\nMesh Ne=%d P=%d..." % (Ne, P))
    mesh = BoxMesh3D(Lx, Ly, Lz, Ne, Ne, Ne, P)
    ops = assemble_3d(mesh)
    N = mesh.N_dof
    mesh._ensure_coords()
    rec_idx = mesh.nearest_node(*REC)
    print("  N=%d" % N)

    r2 = (mesh.x-SRC[0])**2 + (mesh.y-SRC[1])**2 + (mesh.z-SRC[2])**2
    p0 = np.exp(-r2/0.2**2)
    c2S = (C_AIR**2 * ops['S']).tocsc()
    M = ops['M_diag']
    B_labels = ops['B_labels']

    mat_data = {}
    for face, mname in SMAP.items():
        mat_data[face] = load_mat(mname)

    # IFFT frequencies
    s_vals, freqs, sigma = ifft_frequencies(F_MAX, N_FREQ)
    print("\nSolving (%d frequencies, IFFT path)..." % N_FREQ)

    H = np.zeros(N_FREQ, dtype=complex)
    t0 = time.perf_counter()
    for i, s in enumerate(s_vals):
        f = max(abs(s.imag)/(2*np.pi), 1.0)
        Br_diag = np.zeros(N)
        for face, (f_mat, alpha_mat) in mat_data.items():
            a = np.clip(np.interp(min(f, f_mat[-1]), f_mat, alpha_mat), 0.001, 0.999)
            Z = _alpha_to_Z_internal(a)
            if face in B_labels:
                Br_diag += C_AIR**2 * RHO_AIR * B_labels[face] / Z
        sig, omg = s.real, s.imag
        Kr = c2S + sparse.diags((sig**2-omg**2)*M + sig*Br_diag, format='csc')
        Kc = sparse.diags(2*sig*omg*M + omg*Br_diag, format='csc')
        A = sparse.bmat([[Kr,-Kc],[Kc,Kr]], format='csc')
        rhs = np.concatenate([sig*p0*M, omg*p0*M])
        x = spsolve(A, rhs)
        H[i] = x[rec_idx] + 1j*x[N+rec_idx]
        if (i+1) % 50 == 0:
            el = time.perf_counter()-t0
            print("  %d/%d (%.0fs, ETA %.0fs)" % (i+1, N_FREQ, el, el/(i+1)*(N_FREQ-i-1)),
                  end='', flush=True)
    print(" done (%.0fs)" % (time.perf_counter()-t0))

    # IFFT reconstruction
    ir_sim, t_sim = ifft_to_ir(H, freqs, sigma, T_MAX, FS)
    print("  IR: %.2fs, max=%.4e, NaN=%s" % (t_sim[-1], np.max(np.abs(ir_sim)), np.any(np.isnan(ir_sim))))

    ImpulseResponse(ir_sim, FS).to_wav(os.path.join(OUT, 'sim.wav'))
    np.savez_compressed(os.path.join(OUT, 'sim.npz'), ir=ir_sim, fs=FS, t=t_sim)

    # Measured
    print("\nLoading measured...")
    sr, ir_raw = wavfile.read(os.path.join(RIR_DIR, "scene9_RIR_LS1_MP1_Dodecahedron.wav"))
    ir_raw = ir_raw.astype(np.float64)
    # Lowpass to our bandwidth
    ir_meas = sosfilt(butter(6, F_MAX/(sr/2), btype='low', output='sos'), ir_raw)
    # Trim to T_MAX
    n_meas = min(int(T_MAX*sr), len(ir_meas))
    ir_meas = ir_meas[:n_meas]
    t_meas = np.arange(n_meas)/sr

    # Metrics
    print("\nMetrics (octave bands %s):" % str(OCTAVE_BANDS))
    sim_met = octave_band_metrics(ir_sim, FS, OCTAVE_BANDS)
    meas_met = octave_band_metrics(ir_meas, sr, OCTAVE_BANDS)

    print("        %8s %8s %8s" % ('T30_m', 'T30_s', 'err%'))
    t30_errs = []
    for i, fc in enumerate(OCTAVE_BANDS):
        tm = meas_met['T30'][i]
        ts = sim_met['T30'][i]
        if tm and ts and tm > 0:
            err = abs(ts-tm)/tm*100
            t30_errs.append(err)
            print("  %dHz: %8.3f %8.3f %6.1f%%" % (fc, tm, ts, err))
        else:
            print("  %dHz: %8s %8s    N/A" % (fc, tm, ts))

    if t30_errs:
        print("\n  Mean T30 error: %.1f%%" % np.mean(t30_errs))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ir_sim_n = ir_sim / max(np.max(np.abs(ir_sim)), 1e-30)
    ir_meas_n = ir_meas / max(np.max(np.abs(ir_meas)), 1e-30)

    axes[0,0].plot(t_meas*1e3, ir_meas_n, 'b-', lw=0.3, alpha=0.5, label='Measured')
    axes[0,0].plot(t_sim*1e3, ir_sim_n, 'r-', lw=0.3, alpha=0.5, label='Simulated')
    axes[0,0].set_title('(a) IR waveform (0-%dHz)' % F_MAX)
    axes[0,0].set_xlabel('ms'); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(0, 500)

    sp_s = np.abs(np.fft.rfft(ir_sim)); fr_s = np.fft.rfftfreq(len(ir_sim), 1.0/FS)
    sp_m = np.abs(np.fft.rfft(ir_meas)); fr_m = np.fft.rfftfreq(len(ir_meas), 1.0/sr)
    axes[0,1].semilogy(fr_s, sp_s/max(sp_s.max(),1e-30), 'r-', lw=0.5, label='Sim')
    axes[0,1].semilogy(fr_m, sp_m/max(sp_m.max(),1e-30), 'b-', lw=0.5, alpha=0.5, label='Meas')
    axes[0,1].set_xlim(0, F_MAX); axes[0,1].set_ylim(1e-3, 2)
    axes[0,1].set_title('(b) Spectrum'); axes[0,1].legend(fontsize=8); axes[0,1].grid(True, alpha=0.3)

    x = np.arange(len(OCTAVE_BANDS)); w = 0.35
    t30_m = [v if v else 0 for v in meas_met['T30']]
    t30_s = [v if v else 0 for v in sim_met['T30']]
    axes[1,0].bar(x-w/2, t30_m, w, label='Measured', color='steelblue')
    axes[1,0].bar(x+w/2, t30_s, w, label='Simulated', color='coral')
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels([str(f) for f in OCTAVE_BANDS])
    axes[1,0].set_title('(c) T30'); axes[1,0].set_ylabel('s')
    axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.3)

    edc_s = np.cumsum(ir_sim[::-1]**2)[::-1]
    edc_m = np.cumsum(ir_meas[::-1]**2)[::-1]
    axes[1,1].plot(t_meas*1e3, 10*np.log10(edc_m/max(edc_m[0],1e-30)+1e-30), 'b-', label='Meas')
    axes[1,1].plot(t_sim*1e3, 10*np.log10(edc_s/max(edc_s[0],1e-30)+1e-30), 'r-', label='Sim')
    axes[1,1].set_ylim(-60, 0); axes[1,1].set_xlim(0, 1000)
    axes[1,1].set_title('(d) EDC'); axes[1,1].set_xlabel('ms'); axes[1,1].set_ylabel('dB')
    axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('BRAS Scene 9: IFFT, 0-%dHz, %.1fs, N=%d' % (F_MAX, T_MAX, N),
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: %s" % OUT)

if __name__ == '__main__':
    main()
