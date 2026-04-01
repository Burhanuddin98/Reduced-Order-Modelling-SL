#!/usr/bin/env python
"""
A/B comparison: hybrid IR with and without axial modes.

Tests on the BRAS CR2 box room (8.4 x 6.7 x 3.0 m) using the Room API.
Compares T30, EDT, C80 with and without the axial mode engine.

Run: python test_axial_improvement.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    from room_acoustics.room import Room

    print("=" * 60)
    print("  A/B Test: Axial Mode Engine Impact")
    print("  BRAS CR2: 8.4 x 6.7 x 3.0 m")
    print("=" * 60)

    # Build room
    # Coarser mesh for quick A/B test (f_target=250 keeps N manageable)
    room = Room.from_box(8.4, 6.7, 3.0, P=4, ppw=4, f_target=250)
    room.set_material('floor', 'tile')
    room.set_material('ceiling', 'acoustic_panel')
    room.set_material('left', 'glass')
    room.set_material('right', 'plaster')
    room.set_material_default('concrete')
    room.build(n_modes=100)

    source = (2.0, 3.35, 1.5)
    receiver = (6.0, 2.0, 1.2)

    # === RUN A: With axial modes (current pipeline) ===
    print("\n--- RUN A: WITH axial modes ---")
    t0 = time.perf_counter()
    ir_with = room.impulse_response(source, receiver, T=2.0, n_rays=3000)
    t_with = time.perf_counter() - t0
    print(f"  Time: {t_with:.1f}s")
    ir_with.summary()

    # === RUN B: Without axial modes (bypass) ===
    print("\n--- RUN B: WITHOUT axial modes ---")
    saved_pairs = room._parallel_pairs
    room._parallel_pairs = []  # disable axial modes
    t0 = time.perf_counter()
    ir_without = room.impulse_response(source, receiver, T=2.0, n_rays=3000)
    t_without = time.perf_counter() - t0
    room._parallel_pairs = saved_pairs  # restore
    print(f"  Time: {t_without:.1f}s")
    ir_without.summary()

    # === Comparison ===
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)

    metrics = ['T30', 'EDT', 'C80', 'D50']
    units = ['s', 's', 'dB', '']

    # Sabine/Eyring reference for this room
    from room_acoustics.acoustics_metrics import sabine_rt60, eyring_rt60, impedance_to_alpha
    from room_acoustics.materials import get_material
    Lx, Ly, Lz = 8.4, 6.7, 3.0
    V = Lx * Ly * Lz
    S_total = 2 * (Lx*Ly + Lx*Lz + Ly*Lz)

    mat_map = {
        'floor': 'tile', 'ceiling': 'acoustic_panel',
        'left': 'glass', 'right': 'plaster',
        'front': 'concrete', 'back': 'concrete',
    }
    areas = {
        'floor': Lx*Ly, 'ceiling': Lx*Ly,
        'left': Ly*Lz, 'right': Ly*Lz,
        'front': Lx*Lz, 'back': Lx*Lz,
    }
    alphas = {}
    for label, mat_name in mat_map.items():
        mat = get_material(mat_name)
        alphas[label] = impedance_to_alpha(mat['Z'])

    rt60_sab = sabine_rt60(V, areas, alphas)
    mean_alpha = sum(areas[k]*alphas[k] for k in areas) / S_total
    rt60_eyr = eyring_rt60(V, S_total, mean_alpha)

    print(f"\n  Reference:  Sabine T60 = {rt60_sab:.3f}s, Eyring T60 = {rt60_eyr:.3f}s")
    print(f"\n  {'Metric':<8s} {'With axial':>12s} {'Without':>12s} {'Delta':>10s}")
    print(f"  {'-'*8:<8s} {'-'*12:>12s} {'-'*12:>12s} {'-'*10:>10s}")

    for m, u in zip(metrics, units):
        v_with = getattr(ir_with, m)
        v_without = getattr(ir_without, m)
        if v_with is not None and v_without is not None:
            delta = v_with - v_without
            sign = '+' if delta >= 0 else ''
            print(f"  {m:<8s} {v_with:>10.3f}{u:>2s} {v_without:>10.3f}{u:>2s} "
                  f"{sign}{delta:>8.3f}{u}")

    # Spectral comparison: energy per octave band
    print(f"\n  Octave-band energy (dB relative to without-axial):")
    sr = ir_with.sr
    from scipy.signal import butter, filtfilt
    nyq = sr / 2

    for fc in [250, 500, 1000, 2000, 4000]:
        fl = fc / np.sqrt(2)
        fh = min(fc * np.sqrt(2), nyq * 0.95)
        b, a = butter(4, [fl / nyq, fh / nyq], btype='band')

        band_with = filtfilt(b, a, ir_with.data)
        band_without = filtfilt(b, a, ir_without.data)

        e_with = np.sum(band_with**2)
        e_without = np.sum(band_without**2)

        if e_without > 0 and e_with > 0:
            delta_dB = 10 * np.log10(e_with / e_without)
            print(f"    {fc:5d} Hz: {delta_dB:+.1f} dB")

    print("\n  Axial mode overhead: {:.3f}s".format(t_with - t_without))
    print("=" * 60)


if __name__ == '__main__':
    main()
