"""
Impedance calibration from Sabine absorption coefficients.

Fits Miki model parameters (flow resistivity sigma, thickness d) to
reproduce measured random-incidence absorption coefficients via Paris
integration. This is the correct way to convert material datasheets
(which give Sabine alpha) to impedance Z for wave-based solvers.

Based on:
  Mondet et al. (2019) "From absorption to impedance: Enhancing
  boundary conditions in room acoustic simulations", Applied Acoustics.

  Kuttruff (2009) "Room Acoustics", 5th ed., Chapter 2.
"""

import numpy as np
from scipy.optimize import minimize


# ===================================================================
# Miki impedance model
# ===================================================================

def miki_surface_impedance(f, sigma, d):
    """Surface impedance of porous material on rigid backing (Miki 1990).

    Parameters
    ----------
    f : array of frequencies [Hz]
    sigma : flow resistivity [N*s/m^4]
    d : material thickness [m]

    Returns
    -------
    Zs : complex array — surface impedance [Pa*s/m]
    """
    rho = 1.2
    c = 343.0
    f = np.asarray(f, dtype=float)
    X = f / sigma

    Zc = rho * c * (1.0 + 0.0699 * X**(-0.632)
                     - 1j * 0.1071 * X**(-0.618))
    kc = (2.0 * np.pi * f / c) * (1.0 + 0.1093 * X**(-0.618)
                                    - 1j * 0.1597 * X**(-0.683))
    Zs = -1j * Zc / np.tan(kc * d)
    return Zs


# ===================================================================
# Paris integration (angle-averaged absorption)
# ===================================================================

def paris_alpha(Zs, n_angles=200):
    """Random-incidence absorption from complex surface impedance.

    Computes the Paris formula:
        alpha_rand = integral_0^{pi/2} alpha(theta) * sin(2*theta) d_theta

    where alpha(theta) = 1 - |R(theta)|^2 and
    R(theta) = (Zs*cos(theta) - rho*c) / (Zs*cos(theta) + rho*c)

    Parameters
    ----------
    Zs : complex — surface impedance at a single frequency
    n_angles : int — number of integration points

    Returns
    -------
    alpha_rand : float — random-incidence absorption coefficient
    """
    rho_c = 1.2 * 343.0
    theta = np.linspace(0, np.pi/2, n_angles + 1)[:-1]
    dtheta = theta[1] - theta[0]

    cos_t = np.cos(theta)
    R = (Zs * cos_t - rho_c) / (Zs * cos_t + rho_c)
    alpha_t = 1.0 - np.abs(R)**2
    weight = np.sin(2 * theta)

    return np.sum(alpha_t * weight * dtheta)


def miki_random_alpha(f, sigma, d):
    """Random-incidence absorption from Miki model via Paris integration.

    Parameters
    ----------
    f : array of frequencies [Hz]
    sigma : flow resistivity
    d : material thickness

    Returns
    -------
    alpha_rand : array — random-incidence alpha at each frequency
    """
    Zs = miki_surface_impedance(f, sigma, d)
    return np.array([paris_alpha(z) for z in Zs])


# ===================================================================
# Fitting: Sabine alpha -> Miki parameters
# ===================================================================

def fit_miki_to_sabine(freqs, alpha_sabine, sigma_bounds=(500, 200000),
                       d_bounds=(0.001, 0.5), n_restarts=5):
    """Fit Miki model parameters to match Sabine absorption coefficients.

    Finds (sigma, d) that minimize the squared error between the
    Paris-integrated Miki alpha and the measured Sabine alpha.

    Parameters
    ----------
    freqs : array of frequencies [Hz]
    alpha_sabine : array of Sabine absorption coefficients
    sigma_bounds : (min, max) for flow resistivity search
    d_bounds : (min, max) for thickness search
    n_restarts : number of random restarts for optimization

    Returns
    -------
    sigma_opt : optimal flow resistivity [N*s/m^4]
    d_opt : optimal thickness [m]
    alpha_fit : fitted random-incidence alpha curve
    error : RMS fitting error
    """
    freqs = np.asarray(freqs, dtype=float)
    alpha_sabine = np.asarray(alpha_sabine, dtype=float)

    # Clip alpha to valid range
    alpha_sabine = np.clip(alpha_sabine, 0.001, 0.999)

    def objective(params):
        log_sigma, log_d = params
        sigma = np.exp(log_sigma)
        d = np.exp(log_d)
        try:
            alpha_pred = miki_random_alpha(freqs, sigma, d)
            return np.sum((alpha_pred - alpha_sabine)**2)
        except (ValueError, FloatingPointError):
            return 1e10

    best_cost = 1e10
    best_params = None

    for _ in range(n_restarts):
        # Random initial point in log space
        log_sigma0 = np.random.uniform(np.log(sigma_bounds[0]),
                                        np.log(sigma_bounds[1]))
        log_d0 = np.random.uniform(np.log(d_bounds[0]),
                                    np.log(d_bounds[1]))

        result = minimize(objective, [log_sigma0, log_d0],
                         method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.01,
                                  'fatol': 1e-8})

        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x

    sigma_opt = np.exp(best_params[0])
    d_opt = np.exp(best_params[1])
    alpha_fit = miki_random_alpha(freqs, sigma_opt, d_opt)
    error = np.sqrt(np.mean((alpha_fit - alpha_sabine)**2))

    return sigma_opt, d_opt, alpha_fit, error


def fit_bras_materials(csv_dir, scene_prefix='scene09'):
    """Fit Miki parameters to all BRAS material CSVs for a scene.

    Parameters
    ----------
    csv_dir : path to fitted_estimates CSV directory
    scene_prefix : e.g. 'scene09', 'scene11'

    Returns
    -------
    dict mapping material_name -> {sigma, d, alpha_fit, freqs, alpha_measured}
    """
    import os

    results = {}
    for fname in sorted(os.listdir(csv_dir)):
        if scene_prefix not in fname:
            continue

        mat_name = fname.replace(f'mat_{scene_prefix}_', '').replace('.csv', '')
        path = os.path.join(csv_dir, fname)

        with open(path) as f:
            lines = f.readlines()

        freqs = np.array([float(x) for x in lines[0].strip().split(',')])
        alphas = np.array([float(x) for x in lines[1].strip().split(',')])

        # Use octave-band frequencies for fitting (more stable)
        octave_freqs = [125, 250, 500, 1000, 2000, 4000]
        oct_idx = [np.argmin(np.abs(freqs - f)) for f in octave_freqs]
        f_fit = freqs[oct_idx]
        a_fit = alphas[oct_idx]

        sigma, d, alpha_fitted, err = fit_miki_to_sabine(f_fit, a_fit)

        results[mat_name] = {
            'sigma': sigma,
            'd': d,
            'freqs_fit': f_fit.tolist(),
            'alpha_measured': a_fit.tolist(),
            'alpha_fitted': alpha_fitted.tolist(),
            'rms_error': err,
        }

        print(f"  {mat_name:20s}: sigma={sigma:8.0f}, d={d:.4f}m, "
              f"err={err:.4f}")

    return results
