import numpy as np
from rbergomi import RoughBergomi
from scipy.optimize import least_squares
from scipy import interpolate


def fun(
    x,
    K,
    F,
    impvols,
    T,
    ORDER=3,
    EPS=1e-3,
    N_QUAD=50,
    return_error=True,
    weights=None,
):
    """Objective function to minimize."""
    k = np.log(K / F)

    xi0, H, eta_1, eta_2, lbd = x

    params = {
        "s0": 1.0,
        "xi0": lambda u: xi0 * np.ones_like(u),
        "rho": -0.7,
        "H": H,
        "eta": eta_1,
    }
    rbergomi = RoughBergomi(**params, delta_vix=30 / 365.25)

    F_approx, impvols_approx = rbergomi.implied_vol_vix_approx_mixed(
        lbd=lbd,
        eta_2=eta_2,
        k=k,
        T=T,
        order=ORDER,
        eps=EPS,
        n_quad=N_QUAD,
        return_opt="all",
    )

    if not return_error:
        return F_approx, impvols_approx

    error_F = (F - F_approx) / F
    error_impvols = (impvols - impvols_approx) / impvols

    if weights is not None:
        error_impvols *= weights

    error = np.concatenate((np.atleast_1d(error_F), np.asarray(error_impvols)))
    return error


def optimize_calibration(
    x0,
    K,
    F,
    impvols,
    T,
    ORDER=3,
    EPS=1e-3,
    N_QUAD=50,
    weights=None,
):
    """Optimize the calibration parameters."""

    # H, eta_1, eta_2, lbd
    result = least_squares(
        fun,
        x0=x0,
        args=(K, F, impvols, T, ORDER, EPS, N_QUAD, True, weights),
        bounds=(
            [1e-4, 1e-2, 1e-3, 1e-3, 1e-4],  # Lower bounds
            [1.0, 0.5, 20.0, 20.0, 1 - 1e-4],  # Upper bounds
        ),
        verbose=2,
    )

    return result


#### full calibration


def fun_all(
    x,
    days,
    voldata,
    ORDER=3,
    EPS=1e-3,
    N_QUAD=50,
    return_error=True,
    weights=False,
):
    """Objective function to minimize."""

    # x = b0, b1, b2, tau1, tau2, H, eta_1, eta_2, lbd

    b0, b1, b2, tau1, tau2 = x[:5]

    def xi0_func(u):
        return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)

    H, eta_1, eta_2, lbd = x[5:]
    params = {
        "s0": 1.0,
        "xi0": xi0_func,
        "rho": -0.7,
        "H": H,
        "eta": eta_1,
    }
    rbergomi = RoughBergomi(**params, delta_vix=30 / 365.25)

    Fs_approx = []
    impvols_approx = []

    for day in days:
        F_approx_day, impvols_approx_day = rbergomi.implied_vol_vix_approx_mixed(
            lbd=lbd,
            eta_2=eta_2,
            k=voldata[day]["k"],
            T=day / 365.25,
            order=ORDER,
            eps=EPS,
            n_quad=N_QUAD,
            return_opt="all",
        )
        if weights:
            impvols_approx_day *= voldata[day]["weights"]

        Fs_approx.append(float(F_approx_day))
        impvols_approx.append(np.asarray(impvols_approx_day))

    if not return_error:
        return Fs_approx, impvols_approx

    Fs_approx = np.asarray(Fs_approx)
    impvols_approx = np.concatenate(impvols_approx)

    Fs = np.array([voldata[day]["F"] for day in days])
    impvols = np.concatenate([voldata[day]["Mid"] for day in days])

    error_Fs = 10 * (Fs - Fs_approx) / Fs
    error_impvols = (impvols - impvols_approx) / impvols

    error = np.concatenate((np.atleast_1d(error_Fs), np.asarray(error_impvols)))
    return error


def optimize_calibration_all(
    x0,
    days,
    voldata,
    ORDER=3,
    EPS=1e-3,
    N_QUAD=50,
    weights=False,
):
    """Optimize the calibration parameters."""

    # x = b0, b1, b2, tau1, tau2, H, eta_1, eta_2, lbd

    lower_bounds = [1e-4, -1.0, -1.0, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 0.0]
    upper_bounds = [1.0, 1.0, 1.0, 20.0, 20.0, 0.5, 20.0, 20.0, 1.0]
    bounds = (lower_bounds, upper_bounds)

    result = least_squares(
        fun_all,
        x0=x0,
        args=(days, voldata, ORDER, EPS, N_QUAD, True, weights),
        bounds=bounds,
        verbose=2,
    )

    return result
