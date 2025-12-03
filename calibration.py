import numpy as np
from rbergomi import RoughBergomi
from scipy.optimize import least_squares


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

    def fxi0(u):
        return (
            0.04139184 * (u < 37 / 365.0)
            + 0.04710356 * (u >= 37 / 365.0) * (u < 44 / 365.0)
            + xi0 * (u >= 44 / 365.0)
        )

    params = {
        "s0": 1.0,
        # "xi0": lambda u: xi0 * np.ones_like(u),
        "xi0": fxi0,
        "rho": -0.7,
        "H": H,
        "eta": eta_1,
    }
    rbergomi = RoughBergomi(**params)

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
