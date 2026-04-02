"""
Frequency-dependent Image Source Method for shoebox rooms.

Each reflection is filtered by the wall material's spectral absorption.
After N bounces off a carpet wall, high frequencies are heavily attenuated
while low frequencies pass through — this gives the room its tonal character.

Instead of delta impulses attenuated by a scalar, each image source
produces a short filtered pulse whose spectrum reflects the cumulative
wall absorption along its path.

Usage:
    from room_acoustics.ism_spectral import ism_spectral

    ir = ism_spectral(Lx, Ly, Lz, source, receiver, materials,
                      max_order=30, sr=44100, T=3.5)
"""

import numpy as np
from scipy.signal import firwin, fftconvolve


def ism_spectral(Lx, Ly, Lz, src, rec, materials,
                 max_order=30, sr=44100, T=3.5, c=343.0,
                 humidity=50.0, temperature=20.0,
                 n_filt=64):
    """
    Frequency-dependent ISM for shoebox rooms.

    Each image source reflection is convolved with a short FIR filter
    that represents the cumulative spectral absorption of all walls
    hit along its path.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Room dimensions [m].
    src : (x, y, z)
        Source position.
    rec : (x, y, z)
        Receiver position.
    materials : dict
        Maps wall labels to MaterialFunction objects.
        Keys: 'floor','ceiling','left','right','front','back'
        (mapped internally to x0/x1/y0/y1/z0/z1).
    max_order : int
        Maximum reflection order.
    sr : int
        Sample rate [Hz].
    T : float
        IR duration [s].
    c : float
        Speed of sound [m/s].
    humidity, temperature : float
        For air absorption along path.
    n_filt : int
        FIR filter length for spectral shaping (samples).
        Longer = more accurate spectrum, more computation.

    Returns
    -------
    ir : ndarray (n_samples,)
        Impulse response with frequency-dependent reflections.
    """
    from .material_function import MaterialFunction, air_absorption_coefficient

    sx, sy, sz = src
    rx, ry, rz = rec
    n_samples = int(T * sr)
    ir = np.zeros(n_samples, dtype=np.float64)

    # Map surface labels to wall codes
    label_map = {
        'x0': 'left', 'x1': 'right',
        'y0': 'front', 'y1': 'back',
        'z0': 'floor', 'z1': 'ceiling',
    }

    # Resolve materials to MaterialFunction
    wall_mats = {}
    for code, label in label_map.items():
        mat = materials.get(label)
        if mat is None:
            mat = materials.get(code)
        if mat is None or not isinstance(mat, MaterialFunction):
            wall_mats[code] = MaterialFunction.from_scalar(0.05, name=label)
        else:
            wall_mats[code] = mat

    # Frequency vector for spectral filter design
    freqs = np.linspace(0, sr / 2, n_filt // 2 + 1)

    # Precompute wall reflection spectra: R(f) = sqrt(1 - alpha(f))
    # (pressure reflection coefficient)
    wall_R = {}
    wall_scatter = {}
    for code, mat in wall_mats.items():
        alpha = np.clip(mat(freqs), 0.0, 0.999)
        wall_R[code] = np.sqrt(1.0 - alpha)
        # Mean scattering coefficient for this surface
        wall_scatter[code] = float(np.mean(mat.scatter(freqs)))

    # Air absorption spectrum
    m_air = air_absorption_coefficient(freqs, humidity, temperature)

    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                order = abs(nx) + abs(ny) + abs(nz)
                if order > max_order:
                    continue

                # Image source position
                x_img = _mirror(nx, Lx, sx)
                y_img = _mirror(ny, Ly, sy)
                z_img = _mirror(nz, Lz, sz)

                # Distance and travel time
                dist = np.sqrt((x_img - rx)**2 + (y_img - ry)**2 +
                               (z_img - rz)**2)
                t_arrive = dist / c
                if t_arrive >= T:
                    continue

                # Geometric spreading
                amp = 1.0 / (4 * np.pi * max(dist, 0.01))

                # Count wall hits per surface
                hits = _count_hits(nx, ny, nz)

                # Build cumulative spectral reflection filter
                # R_total(f) = product over walls of R_wall(f)^n_hits
                R_spectrum = np.ones(len(freqs), dtype=np.float64)
                for code, n_hits in hits.items():
                    if n_hits > 0:
                        R_spectrum *= wall_R[code] ** n_hits

                # Air absorption along path
                R_spectrum *= np.exp(-m_air * dist)

                # Scale by geometric spreading
                R_spectrum *= amp

                # Skip negligible reflections
                if np.max(R_spectrum) < 1e-8:
                    continue

                # Convert spectrum to time-domain pulse
                H = R_spectrum
                pulse = _spectrum_to_pulse(H, n_filt)

                # Scattering: smear the pulse proportional to total
                # scattering accumulated over all wall hits.
                # Each hit adds temporal spread — higher order reflections
                # become increasingly diffused.
                total_scatter = 0.0
                for code, n_hits in hits.items():
                    total_scatter += wall_scatter.get(code, 0.05) * n_hits

                if total_scatter > 0.01 and order > 0:
                    # Spread width in samples: scatter * order * 0.5ms
                    spread_ms = total_scatter * 0.5
                    spread_n = max(1, int(spread_ms * sr / 1000))
                    # Convolve pulse with a short Gaussian window
                    if spread_n > 1:
                        gauss = np.exp(-0.5 * (np.arange(spread_n * 4) - spread_n * 2) ** 2 / max(spread_n, 1) ** 2)
                        gauss /= gauss.sum()
                        pulse = np.convolve(pulse, gauss)

                # Place in IR at arrival time
                n_arrive = int(np.round(t_arrive * sr))
                n_end = min(n_arrive + len(pulse), n_samples)
                n_write = n_end - n_arrive
                if n_write > 0:
                    ir[n_arrive:n_end] += pulse[:n_write]

    return ir


def _mirror(n, L, s):
    """Compute image source coordinate for n reflections along axis of length L."""
    if n >= 0:
        if n % 2 == 0:
            return 2 * n * L + s
        else:
            return 2 * n * L - s + 2 * L
    else:
        abs_n = abs(n)
        if abs_n % 2 == 0:
            return -2 * abs_n * L + s
        else:
            return -2 * abs_n * L - s


def _count_hits(nx, ny, nz):
    """Count wall hits per surface for image source (nx, ny, nz)."""
    def _split(n):
        """Split |n| reflections between the two walls on this axis."""
        a = abs(n)
        if n > 0:
            hits_0 = a // 2
            hits_1 = a - hits_0
        elif n < 0:
            hits_1 = a // 2
            hits_0 = a - hits_1
        else:
            hits_0 = hits_1 = 0
        return hits_0, hits_1

    hx0, hx1 = _split(nx)
    hy0, hy1 = _split(ny)
    hz0, hz1 = _split(nz)

    return {
        'x0': hx0, 'x1': hx1,
        'y0': hy0, 'y1': hy1,
        'z0': hz0, 'z1': hz1,
    }


def _spectrum_to_pulse(H, n_filt):
    """
    Convert a magnitude spectrum to a minimum-phase FIR pulse.

    Uses the cepstral method: log-magnitude → IFFT → causal window → FFT → IFFT.
    This ensures the pulse is causal (energy after t=0) and has the
    correct magnitude spectrum.
    """
    N = (len(H) - 1) * 2  # full FFT length
    if N < n_filt:
        N = n_filt

    # Ensure positive magnitude
    H_safe = np.maximum(H, 1e-20)

    # Full symmetric spectrum
    H_full = np.zeros(N)
    H_full[:len(H)] = H_safe
    H_full[N - len(H) + 2:] = H_safe[1:-1][::-1]

    # Log magnitude → cepstrum
    log_H = np.log(np.maximum(H_full, 1e-20))
    cepstrum = np.fft.ifft(log_H).real

    # Make minimum phase: double causal part, zero anti-causal
    min_cepstrum = np.zeros(N)
    min_cepstrum[0] = cepstrum[0]
    min_cepstrum[1:N // 2] = 2 * cepstrum[1:N // 2]
    # Nyquist
    if N % 2 == 0:
        min_cepstrum[N // 2] = cepstrum[N // 2]

    # Back to spectrum → minimum phase filter
    min_H = np.exp(np.fft.fft(min_cepstrum))
    pulse = np.fft.ifft(min_H).real

    # Window to n_filt samples
    pulse = pulse[:n_filt]

    # Apply fade-out window to avoid truncation artifacts
    if n_filt > 8:
        fade = min(n_filt // 4, 16)
        pulse[-fade:] *= np.linspace(1, 0, fade)

    return pulse
