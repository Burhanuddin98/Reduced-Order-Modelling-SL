"""
Image Source Method for shoebox rooms.

Computes early and late reflections by mirroring the source across
each wall. For rectangular rooms, this gives the exact solution —
every reflection path corresponds to a unique image source.

The impulse response is a sum of delayed, attenuated impulses:
    h(t) = sum_i  A_i * delta(t - d_i/c)

where d_i is the distance from image source i to the receiver,
and A_i accounts for geometric spreading and wall absorption.
"""

import numpy as np


def image_sources_shoebox(Lx, Ly, Lz, src, rec, max_order=20,
                          alpha_walls=None, sr=44100, T=2.0,
                          c=343.0):
    """
    Compute impulse response for a shoebox room using image sources.

    Parameters
    ----------
    Lx, Ly, Lz : float — room dimensions [m]
    src : (x, y, z) — source position
    rec : (x, y, z) — receiver position
    max_order : int — maximum reflection order (total across all axes)
    alpha_walls : dict or None
        Absorption coefficients per wall. Keys: 'x0','x1','y0','y1','z0','z1'
        (x0 = wall at x=0, x1 = wall at x=Lx, etc.)
        If None, all walls are rigid (alpha=0).
        Can be a dict of arrays for frequency-dependent absorption,
        keyed by octave-band center frequency.
    sr : int — sample rate [Hz]
    T : float — IR duration [s]
    c : float — speed of sound [m/s]

    Returns
    -------
    ir : ndarray — impulse response at sample rate sr
    reflections : list of dicts with 'time', 'amplitude', 'order', 'distance'
    """
    sx, sy, sz = src
    rx, ry, rz = rec

    # Default: rigid walls
    if alpha_walls is None:
        alpha_walls = {k: 0.0 for k in ['x0','x1','y0','y1','z0','z1']}

    n_samples = int(T * sr)
    ir = np.zeros(n_samples)
    reflections = []

    # Image sources: for each combination of (nx, ny, nz) reflections,
    # the image source position is determined by mirroring.
    #
    # For reflection order n in x: the image x-coordinate is
    #   x_img = 2*nx*Lx + sx  if nx is even (even number of reflections)
    #   x_img = 2*nx*Lx - sx  if nx is odd  (for positive nx)
    # Similarly for negative nx.
    #
    # More precisely:
    #   x_img(nx) = { 2*n*Lx + sx  if n >= 0, even reflections from x=0 wall
    #               { 2*n*Lx - sx  if n >= 0, odd reflections (first hit x=Lx)
    # etc.
    #
    # Standard formulation: image at (nx, ny, nz) has position:
    #   x_img = nx*2*Lx + (-1)^abs(nx) * sx  ... no, simpler:

    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                order = abs(nx) + abs(ny) + abs(nz)
                if order > max_order:
                    continue
                if order == 0:
                    # Direct sound
                    pass

                # Image source position
                if nx >= 0:
                    x_img = 2 * nx * Lx + sx if nx % 2 == 0 else 2 * nx * Lx - sx + 2 * Lx
                else:
                    # Negative nx: mirror in negative direction
                    abs_nx = abs(nx)
                    x_img = -2 * abs_nx * Lx + sx if abs_nx % 2 == 0 else -2 * abs_nx * Lx - sx

                if ny >= 0:
                    y_img = 2 * ny * Ly + sy if ny % 2 == 0 else 2 * ny * Ly - sy + 2 * Ly
                else:
                    abs_ny = abs(ny)
                    y_img = -2 * abs_ny * Ly + sy if abs_ny % 2 == 0 else -2 * abs_ny * Ly - sy

                if nz >= 0:
                    z_img = 2 * nz * Lz + sz if nz % 2 == 0 else 2 * nz * Lz - sz + 2 * Lz
                else:
                    abs_nz = abs(nz)
                    z_img = -2 * abs_nz * Lz + sz if abs_nz % 2 == 0 else -2 * abs_nz * Lz - sz

                # Distance
                dist = np.sqrt((x_img - rx)**2 + (y_img - ry)**2 +
                               (z_img - rz)**2)

                # Travel time
                t_arrive = dist / c
                if t_arrive >= T:
                    continue

                # Amplitude: 1/(4*pi*d) geometric spreading
                amp = 1.0 / (4 * np.pi * max(dist, 0.01))

                # Wall reflections: count how many times each wall is hit
                # x=0 wall: ceil(|nx|/2) times if nx>0, floor(|nx|/2)+1 if nx<0
                # Actually, simpler: for |nx| reflections in x,
                # hits on x=0 wall: ceil(abs(nx)/2) if nx>0, floor(abs(nx)/2)+1...
                #
                # Simpler approach: for nx reflections total in x direction,
                # the number of hits on each wall depends on the parity:
                n_x0 = (abs(nx) + 1) // 2 if nx > 0 else abs(nx) // 2  # hits on far wall
                n_x1 = abs(nx) // 2 if nx > 0 else (abs(nx) + 1) // 2
                if nx < 0:
                    n_x0, n_x1 = n_x1, n_x0
                # Actually this is getting complicated. Use the standard formula:
                # For |nx| total reflections in x, walls at x=0 and x=Lx are hit
                # alternately. The number depends on direction.
                # Simplified: use abs(nx) total bounces, split roughly evenly
                abs_nx = abs(nx)
                n_hits_x0 = abs_nx // 2 + (1 if nx < 0 and abs_nx % 2 == 1 else 0)
                n_hits_x1 = abs_nx - n_hits_x0

                abs_ny = abs(ny)
                n_hits_y0 = abs_ny // 2 + (1 if ny < 0 and abs_ny % 2 == 1 else 0)
                n_hits_y1 = abs_ny - n_hits_y0

                abs_nz = abs(nz)
                n_hits_z0 = abs_nz // 2 + (1 if nz < 0 and abs_nz % 2 == 1 else 0)
                n_hits_z1 = abs_nz - n_hits_z0

                # Reflection coefficient product
                r_total = ((1 - alpha_walls.get('x0', 0))**n_hits_x0 *
                           (1 - alpha_walls.get('x1', 0))**n_hits_x1 *
                           (1 - alpha_walls.get('y0', 0))**n_hits_y0 *
                           (1 - alpha_walls.get('y1', 0))**n_hits_y1 *
                           (1 - alpha_walls.get('z0', 0))**n_hits_z0 *
                           (1 - alpha_walls.get('z1', 0))**n_hits_z1)

                amp *= np.sqrt(r_total)  # pressure reflection = sqrt(energy)

                # Add to IR
                sample = int(round(t_arrive * sr))
                if 0 <= sample < n_samples:
                    ir[sample] += amp

                reflections.append({
                    'time': t_arrive,
                    'amplitude': amp,
                    'order': order,
                    'distance': dist,
                    'nx': nx, 'ny': ny, 'nz': nz,
                })

    return ir, reflections


def image_source_ir_octave_bands(Lx, Ly, Lz, src, rec, alpha_per_band,
                                  max_order=20, sr=44100, T=2.0, c=343.0):
    """
    Compute frequency-dependent IR using per-band image sources.

    Parameters
    ----------
    alpha_per_band : dict
        Maps octave-band center frequency -> dict of wall alphas.
        e.g. {250: {'x0':0.05, 'x1':0.03, ...}, 500: {...}, ...}

    Returns
    -------
    ir : full-bandwidth IR (sum of filtered per-band IRs)
    """
    from scipy.signal import butter, filtfilt

    nyq = sr / 2
    n_samples = int(T * sr)
    ir_total = np.zeros(n_samples)

    bands = sorted(alpha_per_band.keys())

    for fc in bands:
        alpha_w = alpha_per_band[fc]
        ir_band, _ = image_sources_shoebox(Lx, Ly, Lz, src, rec,
                                            max_order=max_order,
                                            alpha_walls=alpha_w,
                                            sr=sr, T=T, c=c)
        # Bandpass filter
        fl = fc / np.sqrt(2)
        fh = fc * np.sqrt(2)
        if fh >= nyq * 0.95:
            fh = nyq * 0.95
        if fl < 10:
            fl = 10

        b, a = butter(4, [fl / nyq, fh / nyq], btype='band')
        ir_filtered = filtfilt(b, a, ir_band)
        ir_total += ir_filtered

    return ir_total
