"""BRAS Scene 9 — Low-frequency fair validation (0-500Hz, 0.1s)."""
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
from romacoustics.solver import C_AIR, RHO_AIR, weeks_s_values, laplace_to_ir
from romacoustics.ir import ImpulseResponse
from romacoustics.materials import absorption_to_impedance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_bras_v2')
os.makedirs(OUT, exist_ok=True)

BRAS = r"C:\Users\bsaka\Downloads\BRAS"
MAT_DIR = os.path.join(BRAS, "surface_data", "3 Surface descriptions", "_csv", "fitted_estimates")
RIR_DIR = os.path.join(BRAS, "scene9_data", "1 Scene descriptions",
                        "09 small room (seminar room)", "RIRs", "wav")

Lx, Ly, Lz = 8.4, 6.7, 3.0
SRC = (2.0, 3.35, 1.5)
REC = (6.0, 1.5, 1.2)
F_MAX = 500
T_MAX = 0.1
FS = 44100

SMAP = {'z_min':'floor','z_max':'ceiling','y_min':'plaster','y_max':'plaster',
        'x_min':'windows','x_max':'concrete'}

def load_mat(name):
    lines = open(os.path.join(MAT_DIR, "mat_scene09_%s.csv" % name)).read().strip().split('\n')
    return np.array([float(x) for x in lines[0].split(',')]), np.array([float(x) for x in lines[2].split(',')])

def lowpass(sig, fs, fc):
    return sosfilt(butter(6, fc/(fs/2), btype='low', output='sos'), sig)

def main():
    print("=" * 60)
    print("  BRAS Scene 9 -- 0-%dHz, %.1fs" % (F_MAX, T_MAX))
    print("=" * 60)

    Ne, P = 4, 4
    print("\nMesh Ne=%d P=%d..." % (Ne, P))
    mesh = BoxMesh3D(Lx, Ly, Lz, Ne, Ne, Ne, P)
    ops = assemble_3d(mesh)
    N = mesh.N_dof
    mesh._ensure_coords()
    rec_idx = mesh.nearest_node(*REC)
    print("  N=%d, rec=%d" % (N, rec_idx))

    r2 = (mesh.x-SRC[0])**2 + (mesh.y-SRC[1])**2 + (mesh.z-SRC[2])**2
    p0 = np.exp(-r2/0.2**2)
    c2S = (C_AIR**2 * ops['S']).tocsc()
    M = ops['M_diag']
    B_labels = ops['B_labels']

    mat_data = {}
    print("\nMaterials:")
    for face, mname in SMAP.items():
        freqs, alpha = load_mat(mname)
        mat_data[face] = (freqs, alpha)
        print("  %s -> %s: a125=%.3f a500=%.3f" % (
            face, mname, np.interp(125,freqs,alpha), np.interp(500,freqs,alpha)))

    sigma_w, b_w, Ns = 10.0, 500.0, 500
    s_vals, _ = weeks_s_values(sigma_w, b_w, Ns)
    t_eval = np.arange(0, T_MAX, 1.0/FS)

    print("\nSolving (Ns=%d)..." % Ns)
    H = np.zeros(Ns, dtype=complex)
    t0 = time.perf_counter()
    for i, s in enumerate(s_vals):
        f = max(abs(s.imag)/(2*np.pi), 1.0)
        Br = np.zeros(N)
        for face, (freqs, alpha) in mat_data.items():
            a = np.clip(np.interp(min(f, freqs[-1]), freqs, alpha), 0.001, 0.999)
            Br += C_AIR**2 * RHO_AIR * B_labels[face] / absorption_to_impedance(a)
        sig, omg = s.real, s.imag
        Kr = c2S + sparse.diags((sig**2-omg**2)*M + sig*Br, format='csc')
        Kc = sparse.diags(2*sig*omg*M + omg*Br, format='csc')
        A = sparse.bmat([[Kr,-Kc],[Kc,Kr]], format='csc')
        rhs = np.concatenate([sig*p0*M, omg*p0*M])
        x = spsolve(A, rhs)
        H[i] = x[rec_idx] + 1j*x[N+rec_idx]
        if (i+1) % 100 == 0:
            el = time.perf_counter()-t0
            print("  %d/%d (%.0fs, ETA %.0fs)" % (i+1, Ns, el, el/(i+1)*(Ns-i-1)),
                  end='', flush=True)
    print(" done (%.0fs)" % (time.perf_counter()-t0))

    ir_sim = laplace_to_ir(H, sigma_w, b_w, t_eval)
    print("  IR max: %.4e" % np.max(np.abs(ir_sim)))
    ImpulseResponse(ir_sim, FS).to_wav(os.path.join(OUT, 'sim.wav'))
    np.savez_compressed(os.path.join(OUT, 'sim.npz'), ir=ir_sim, fs=FS)

    # Measured
    print("\nLoading measured...")
    sr, ir_raw = wavfile.read(os.path.join(RIR_DIR, "scene9_RIR_LS1_MP1_Dodecahedron.wav"))
    ir_raw = ir_raw.astype(np.float64)
    ir_filt = lowpass(ir_raw, sr, F_MAX)
    n = int(T_MAX * sr)
    ir_meas = ir_filt[:n]
    t_meas = np.arange(n) / sr

    # Normalize
    ir_sim_n = ir_sim / max(np.max(np.abs(ir_sim)), 1e-30)
    ir_meas_n = ir_meas / max(np.max(np.abs(ir_meas)), 1e-30)

    # Eigenfrequencies
    print("\n--- Eigenfrequencies ---")
    f_ana = sorted(set([round(C_AIR/2*np.sqrt((l/Lx)**2+(m/Ly)**2+(n_/Lz)**2), 1)
                        for l in range(20) for m in range(20) for n_ in range(20)
                        if 0 < C_AIR/2*np.sqrt((l/Lx)**2+(m/Ly)**2+(n_/Lz)**2) < F_MAX]))
    print("  Analytical modes: %d" % len(f_ana))

    def peaks(sig, fs, fmax):
        sp = np.abs(np.fft.rfft(sig))
        fr = np.fft.rfftfreq(len(sig), 1.0/fs)
        mask = fr < fmax
        sp_m = sp[mask] / max(sp[mask].max(), 1e-30)
        df = fr[1] if len(fr) > 1 else 1
        pk, pr = find_peaks(sp_m, height=0.05, distance=max(1, int(5/df)))
        order = np.argsort(pr['peak_heights'])[::-1][:20]
        return fr[mask][pk[order]]

    pk_sim = peaks(ir_sim, FS, F_MAX)
    pk_meas = peaks(ir_meas, sr, F_MAX)
    matched = sum(1 for fp in pk_sim[:15] if min(abs(fp-fa) for fa in f_ana) < 10)
    print("  Sim peaks matched to analytical: %d/15 within 10Hz" % matched)

    # D50
    def D50(sig, fs):
        s2 = sig**2; return np.sum(s2[:int(0.05*fs)]) / max(np.sum(s2), 1e-30)
    d50_m, d50_s = D50(ir_meas, sr), D50(ir_sim, FS)
    print("\n--- D50 ---")
    print("  Measured: %.3f, Simulated: %.3f, diff: %.3f" % (d50_m, d50_s, abs(d50_m-d50_s)))

    # 1/3 octave
    third = [50,63,80,100,125,160,200,250,315,400,500]
    spl_s, spl_m = [], []
    for fc in third:
        fl, fh = fc/2**(1/6), fc*2**(1/6)
        if fh > FS/2*0.95: continue
        sos = butter(4, [fl/(FS/2), fh/(FS/2)], btype='band', output='sos')
        spl_s.append(10*np.log10(max(np.sum(sosfilt(sos,ir_sim)**2), 1e-30)))
        spl_m.append(10*np.log10(max(np.sum(sosfilt(sos,ir_meas)**2), 1e-30)))
    spl_s = np.array(spl_s); spl_m = np.array(spl_m)
    spl_s -= spl_s.max(); spl_m -= spl_m.max()
    print("\n--- Spectral shape ---")
    print("  Mean diff: %.1f dB, Max diff: %.1f dB" % (np.mean(np.abs(spl_s-spl_m)), np.max(np.abs(spl_s-spl_m))))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(t_meas*1e3, ir_meas_n, 'b-', lw=0.5, alpha=0.6, label='Measured')
    axes[0,0].plot(t_eval*1e3, ir_sim_n, 'r-', lw=0.5, alpha=0.6, label='Simulated')
    axes[0,0].set_title('(a) IR waveform'); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlabel('ms'); axes[0,0].set_ylabel('Normalized')

    sp_s = np.abs(np.fft.rfft(ir_sim)); fr_s = np.fft.rfftfreq(len(ir_sim), 1.0/FS)
    sp_m = np.abs(np.fft.rfft(ir_meas)); fr_m = np.fft.rfftfreq(len(ir_meas), 1.0/sr)
    axes[0,1].semilogy(fr_s, sp_s/max(sp_s.max(),1e-30), 'r-', lw=0.5, label='Sim')
    axes[0,1].semilogy(fr_m, sp_m/max(sp_m.max(),1e-30), 'b-', lw=0.5, alpha=0.5, label='Meas')
    for fa in f_ana[:30]: axes[0,1].axvline(fa, color='k', alpha=0.1, lw=0.5)
    axes[0,1].set_xlim(0,F_MAX); axes[0,1].set_ylim(1e-3,2)
    axes[0,1].set_title('(b) Spectrum'); axes[0,1].legend(fontsize=8); axes[0,1].grid(True, alpha=0.3)

    x = np.arange(len(third))
    axes[1,0].bar(x-0.2, spl_m, 0.35, label='Meas', color='steelblue')
    axes[1,0].bar(x+0.2, spl_s, 0.35, label='Sim', color='coral')
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels([str(f) for f in third], fontsize=7)
    axes[1,0].set_title('(c) 1/3 octave SPL'); axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.3)

    edc_s = np.cumsum(ir_sim[::-1]**2)[::-1]
    edc_m = np.cumsum(ir_meas[::-1]**2)[::-1]
    axes[1,1].plot(t_meas*1e3, 10*np.log10(edc_m/max(edc_m[0],1e-30)+1e-30), 'b-', label='Meas')
    axes[1,1].plot(t_eval*1e3, 10*np.log10(edc_s/max(edc_s[0],1e-30)+1e-30), 'r-', label='Sim')
    axes[1,1].set_ylim(-30,0); axes[1,1].set_title('(d) EDC')
    axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlabel('ms'); axes[1,1].set_ylabel('dB')

    plt.suptitle('BRAS Scene 9: 0-%dHz, %gs, N=%d' % (F_MAX, T_MAX, N), fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: %s" % OUT)

if __name__ == '__main__':
    main()
