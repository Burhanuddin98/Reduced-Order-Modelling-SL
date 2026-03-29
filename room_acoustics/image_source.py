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
                          alpha_walls=None, scatter_walls=None,
                          sr=44100, T=2.0, c=343.0):
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
    scatter_walls : dict or None
        Scattering coefficients per wall. Same keys as alpha_walls.
        At each reflection, specular energy is multiplied by (1-s)
        in addition to (1-alpha). Scattered energy is lost from the
        specular path (goes into diffuse field).
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

    # Default: rigid walls, no scattering
    if alpha_walls is None:
        alpha_walls = {k: 0.0 for k in ['x0','x1','y0','y1','z0','z1']}
    if scatter_walls is None:
        scatter_walls = {k: 0.0 for k in ['x0','x1','y0','y1','z0','z1']}

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
                # Each reflection: energy *= (1-alpha) * (1-scatter)
                # (1-alpha) = energy not absorbed
                # (1-scatter) = fraction that stays in specular path
                def _refl(wall, n_hits):
                    a = alpha_walls.get(wall, 0)
                    s = scatter_walls.get(wall, 0)
                    return ((1 - a) * (1 - s)) ** n_hits

                r_total = (_refl('x0', n_hits_x0) * _refl('x1', n_hits_x1) *
                           _refl('y0', n_hits_y0) * _refl('y1', n_hits_y1) *
                           _refl('z0', n_hits_z0) * _refl('z1', n_hits_z1))

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
                                  scatter_per_band=None,
                                  max_order=20, sr=44100, T=2.0, c=343.0):
    """
    Compute frequency-dependent IR using per-band image sources.

    Parameters
    ----------
    alpha_per_band : dict
        Maps octave-band center frequency -> dict of wall alphas.
    scatter_per_band : dict or None
        Maps octave-band center frequency -> dict of wall scattering coeffs.

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
        scatter_w = scatter_per_band[fc] if scatter_per_band else None
        ir_band, _ = image_sources_shoebox(Lx, Ly, Lz, src, rec,
                                            max_order=max_order,
                                            alpha_walls=alpha_w,
                                            scatter_walls=scatter_w,
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


def hybrid_ism_diffuse(Lx, Ly, Lz, src, rec, alpha_walls, scatter_walls,
                       max_order=50, sr=44100, T=3.5, c=343.0):
    """
    ISM with scattered energy feeding a diffuse reverberant tail.

    At each reflection:
    - Specular path keeps (1-alpha)*(1-s) of energy
    - Scattered energy s*(1-alpha) is injected into a diffuse pool
      at the arrival time of that reflection

    The diffuse pool decays exponentially from each injection point
    with the Eyring decay rate, producing the late reverberant tail.

    Returns
    -------
    ir : full impulse response (specular + diffuse)
    ir_specular : specular-only component
    ir_diffuse : diffuse tail component
    """
    n_samples = int(T * sr)
    dt = 1.0 / sr

    # Compute specular ISM with scattering
    ir_specular, reflections = image_sources_shoebox(
        Lx, Ly, Lz, src, rec, max_order=max_order,
        alpha_walls=alpha_walls, scatter_walls=scatter_walls,
        sr=sr, T=T, c=c)

    # Also compute without scattering to get the energy that was scattered
    ir_full, reflections_full = image_sources_shoebox(
        Lx, Ly, Lz, src, rec, max_order=max_order,
        alpha_walls=alpha_walls, scatter_walls=None,
        sr=sr, T=T, c=c)

    # The scattered energy at each sample = full_energy - specular_energy
    # This is the energy removed from specular paths by scattering
    diffuse_injection = ir_full**2 - ir_specular**2
    diffuse_injection = np.maximum(diffuse_injection, 0)

    # Eyring decay rate for the diffuse tail
    V = Lx * Ly * Lz
    S_total = 2 * (Lx*Ly + Lx*Lz + Ly*Lz)

    # Mean absorption (area-weighted)
    areas = {'x0': Ly*Lz, 'x1': Ly*Lz, 'y0': Lx*Lz, 'y1': Lx*Lz,
             'z0': Lx*Ly, 'z1': Lx*Ly}
    mean_alpha = sum(alpha_walls.get(k, 0) * areas[k] for k in areas) / S_total
    mean_scatter = sum(scatter_walls.get(k, 0) * areas[k] for k in areas) / S_total

    # Effective absorption for diffuse field includes scattering effect
    # Scattered energy bounces diffusely, seeing mean_alpha on average
    alpha_eff = mean_alpha + mean_scatter * (1 - mean_alpha) * 0.5
    alpha_eff = min(alpha_eff, 0.99)

    if alpha_eff > 0 and alpha_eff < 1:
        T60_diffuse = 0.161 * V / (-S_total * np.log(1 - alpha_eff))
    else:
        T60_diffuse = 0.161 * V / (S_total * max(alpha_eff, 0.01))

    decay_rate = 6.91 / max(T60_diffuse, 0.01)

    # Build diffuse tail: convolve injection energy with exponential decay
    # For efficiency, use the cumulative approach:
    # At each sample n, the diffuse energy = sum of all previous injections
    # each decayed by exp(-decay_rate * (t_n - t_inject))
    t = np.arange(n_samples) * dt
    ir_diffuse = np.zeros(n_samples)

    # Create the decay kernel (truncate when negligible)
    kernel_len = min(n_samples, int(5 * T60_diffuse * sr))
    kernel = np.exp(-decay_rate * np.arange(kernel_len) * dt)
    kernel /= np.sqrt(np.sum(kernel**2) * dt)  # normalize

    # Modulate with noise for diffuse character
    np.random.seed(42)
    noise = np.random.randn(n_samples)

    # Convolve: at each time where specular energy was scattered,
    # inject decaying noise into the diffuse tail
    # Use a simplified approach: shape the noise envelope by the
    # accumulated scattered energy convolved with exponential decay
    from scipy.signal import fftconvolve

    # Energy envelope of diffuse injection
    envelope = fftconvolve(diffuse_injection, kernel[:kernel_len], mode='full')[:n_samples]
    envelope = np.sqrt(np.maximum(envelope, 0))

    ir_diffuse = noise * envelope

    # Scale: the diffuse tail should carry the scattered energy
    # Total specular energy
    E_spec = np.sum(ir_specular**2)
    E_full = np.sum(ir_full**2)
    E_scattered = max(E_full - E_spec, 0)
    E_diffuse = np.sum(ir_diffuse**2)
    if E_diffuse > 0 and E_scattered > 0:
        ir_diffuse *= np.sqrt(E_scattered / E_diffuse)

    ir = ir_specular + ir_diffuse
    return ir, ir_specular, ir_diffuse
