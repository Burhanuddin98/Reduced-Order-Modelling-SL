"""
Room acoustics metrics computed from impulse responses.

RT60, EDT, C80, D50, TS — the standard ISO 3382 parameters
that acousticians use to characterize rooms.
"""

import numpy as np


def schroeder_decay(ir, dt):
    """Schroeder backward integration of an impulse response.

    Returns the energy decay curve in dB (normalized to 0 dB at t=0).

    Parameters
    ----------
    ir : 1D array — impulse response (pressure vs time)
    dt : float — time step [s]

    Returns
    -------
    t : time vector [s]
    decay_dB : Schroeder decay curve [dB]
    """
    ir = np.asarray(ir, dtype=float)
    energy = ir ** 2

    # Backward integration: integral from t to infinity of p^2 dt
    # Approximate as cumulative sum from end to start
    backward = np.cumsum(energy[::-1])[::-1]

    # Normalize and convert to dB
    backward = backward / backward[0]
    # Avoid log(0)
    backward = np.maximum(backward, 1e-30)
    decay_dB = 10.0 * np.log10(backward)

    t = np.arange(len(ir)) * dt
    return t, decay_dB


def compute_rt(ir, dt, decay_range=(-5, -35)):
    """Compute reverberation time by linear regression on Schroeder curve.

    Parameters
    ----------
    ir : impulse response
    dt : time step [s]
    decay_range : tuple (start_dB, end_dB)
        Default (-5, -35) gives T30 (standard).
        Use (-5, -25) for T20, (-5, -15) for T10.

    Returns
    -------
    RT : reverberation time [s] (extrapolated to -60 dB)
    r_squared : R^2 of the linear fit (should be > 0.95)
    """
    t, decay = schroeder_decay(ir, dt)

    start_dB, end_dB = decay_range

    # Find the time range where decay is between start_dB and end_dB
    mask = (decay >= end_dB) & (decay <= start_dB)

    if mask.sum() < 3:
        return np.nan, 0.0

    t_fit = t[mask]
    d_fit = decay[mask]

    # Linear regression: decay = slope * t + intercept
    coeffs = np.polyfit(t_fit, d_fit, 1)
    slope = coeffs[0]  # dB/s

    if abs(slope) < 1e-10:
        return np.inf, 0.0

    # Extrapolate to -60 dB
    RT = -60.0 / slope

    # R^2
    d_pred = np.polyval(coeffs, t_fit)
    ss_res = np.sum((d_fit - d_pred) ** 2)
    ss_tot = np.sum((d_fit - np.mean(d_fit)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    return RT, r_squared


def compute_t30(ir, dt):
    """T30: RT from -5 to -35 dB decay, extrapolated to -60 dB."""
    return compute_rt(ir, dt, (-5, -35))


def compute_t20(ir, dt):
    """T20: RT from -5 to -25 dB decay, extrapolated to -60 dB."""
    return compute_rt(ir, dt, (-5, -25))


def compute_edt(ir, dt):
    """Early Decay Time: RT from 0 to -10 dB, extrapolated to -60 dB."""
    return compute_rt(ir, dt, (0, -10))


def compute_c80(ir, dt):
    """Clarity C80: ratio of early (0-80ms) to late (80ms+) energy [dB].

    Measures how clearly musical detail is perceived.
    Typical values: -5 to +5 dB. Higher = more clarity.
    """
    n80 = int(round(0.08 / dt))
    n80 = min(n80, len(ir))

    energy_early = np.sum(ir[:n80] ** 2)
    energy_late = np.sum(ir[n80:] ** 2)

    if energy_late < 1e-30:
        return np.inf

    return 10.0 * np.log10(energy_early / energy_late)


def compute_d50(ir, dt):
    """Definition D50: ratio of early (0-50ms) to total energy.

    Measures speech intelligibility. Range 0-1. Higher = better.
    """
    n50 = int(round(0.05 / dt))
    n50 = min(n50, len(ir))

    energy_early = np.sum(ir[:n50] ** 2)
    energy_total = np.sum(ir ** 2)

    if energy_total < 1e-30:
        return np.nan

    return energy_early / energy_total


def compute_ts(ir, dt):
    """Centre Time TS: energy-weighted mean arrival time [ms].

    Lower TS = better definition. Typical: 60-200 ms.
    """
    t = np.arange(len(ir)) * dt
    energy = ir ** 2
    total = np.sum(energy)

    if total < 1e-30:
        return np.nan

    return np.sum(t * energy) / total * 1000  # convert to ms


def sabine_rt60(volume, surface_areas, alpha_coeffs):
    """Sabine reverberation time.

    Parameters
    ----------
    volume : float [m^3]
    surface_areas : dict or list of float [m^2]
        Area of each surface.
    alpha_coeffs : dict or list of float
        Absorption coefficient (0-1) for each surface.
        Must match the keys/order of surface_areas.

    Returns
    -------
    RT60 : float [s]
    """
    if isinstance(surface_areas, dict):
        A = sum(surface_areas[k] * alpha_coeffs[k] for k in surface_areas)
    else:
        A = sum(s * a for s, a in zip(surface_areas, alpha_coeffs))

    if A < 1e-10:
        return np.inf

    return 0.161 * volume / A


def eyring_rt60(volume, total_surface, mean_alpha):
    """Eyring reverberation time (more accurate for high absorption).

    Parameters
    ----------
    volume : float [m^3]
    total_surface : float [m^2]
    mean_alpha : float — mean absorption coefficient (0-1)

    Returns
    -------
    RT60 : float [s]
    """
    if mean_alpha >= 1.0:
        return 0.0
    if mean_alpha <= 0.0:
        return np.inf

    return 0.161 * volume / (-total_surface * np.log(1 - mean_alpha))


def impedance_to_alpha(Z):
    """Convert normal-incidence impedance Z to absorption coefficient alpha.

    Uses the normal-incidence formula: alpha = 1 - |R|^2
    where R = (Z - rho*c) / (Z + rho*c).
    """
    rho_c = 1.2 * 343.0
    R = (Z - rho_c) / (Z + rho_c)
    return 1.0 - R ** 2


def diffuse_alpha_for_Z(Z, n_angles=200):
    """Compute random-incidence (diffuse-field) absorption for impedance Z.

    Integrates the angle-dependent absorption over all incidence angles
    weighted by sin(2*theta) (Lambert's cosine law for diffuse field):

        alpha_d = integral_0^{pi/2} alpha(theta) * sin(2*theta) d_theta

    where alpha(theta) = 1 - |R(theta)|^2 and
    R(theta) = (Z*cos(theta) - rho*c) / (Z*cos(theta) + rho*c).
    """
    rho_c = 1.2 * 343.0
    theta = np.linspace(0, np.pi/2, n_angles + 1)[:-1]  # exclude pi/2
    dtheta = theta[1] - theta[0]

    cos_t = np.cos(theta)
    Z_eff = Z * cos_t  # effective impedance at angle theta
    R = (Z_eff - rho_c) / (Z_eff + rho_c)
    alpha_t = 1.0 - R**2
    weight = np.sin(2 * theta)

    return np.sum(alpha_t * weight * dtheta)


def alpha_random_to_Z(alpha_random, tol=1e-6, max_iter=100):
    """Convert random-incidence absorption coefficient to impedance Z.

    Inverts the diffuse-field alpha-Z relationship numerically.
    Given alpha_random (what's measured in a room / given in databases),
    find the impedance Z such that diffuse_alpha_for_Z(Z) = alpha_random.

    This is the correct conversion for room acoustics — NOT the
    normal-incidence formula which gives Z that's too low (too absorptive).
    """
    rho_c = 1.2 * 343.0

    if alpha_random <= 0:
        return 1e15
    if alpha_random >= 1:
        return rho_c

    # Bisection search: Z is monotonically related to alpha
    # Higher Z = lower alpha (more reflective)
    Z_lo = rho_c       # alpha = 1.0 (perfect absorber)
    Z_hi = 1e8          # alpha ~ 0

    for _ in range(max_iter):
        Z_mid = np.sqrt(Z_lo * Z_hi)  # geometric mean for log-scale search
        alpha_mid = diffuse_alpha_for_Z(Z_mid)

        if abs(alpha_mid - alpha_random) < tol:
            return Z_mid

        if alpha_mid > alpha_random:
            Z_lo = Z_mid  # need higher Z (less absorption)
        else:
            Z_hi = Z_mid  # need lower Z (more absorption)

    return Z_mid


def all_metrics(ir, dt):
    """Compute all standard room acoustics metrics from an IR.

    Returns dict with T30, T20, EDT, C80, D50, TS.
    """
    t30, t30_r2 = compute_t30(ir, dt)
    t20, t20_r2 = compute_t20(ir, dt)
    edt, edt_r2 = compute_edt(ir, dt)

    return {
        'T30_s': t30, 'T30_R2': t30_r2,
        'T20_s': t20, 'T20_R2': t20_r2,
        'EDT_s': edt, 'EDT_R2': edt_r2,
        'C80_dB': compute_c80(ir, dt),
        'D50': compute_d50(ir, dt),
        'TS_ms': compute_ts(ir, dt),
    }
