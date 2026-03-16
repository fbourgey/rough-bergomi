import numpy as np
from scipy.optimize import least_squares

from rbergomi import RoughBergomi


def fun(
    x,
    K,
    F,
    impvols,
    T,
    ORDER=3,
    N_QUAD=50,
    return_error=True,
    weights=None,
):
    """
    Compute residuals between market and model VIX futures and implied volatilities.

    Parameters
    ----------
    x : array_like
        Model parameters [xi0, H, eta_1, eta_2, lbd].
    K : array_like
        Strike prices.
    F : float
        VIX futures price.
    impvols : array_like
        Market implied volatilities.
    T : float
        Time to maturity (years).
    ORDER : int, optional
        Expansion order (default 3).
    EPS : float, optional
        Integration tolerance (default 1e-3).
    N_QUAD : int, optional
        Number of quadrature points (default 50).
    return_error : bool, optional
        If True, return errors; if False, return model prices. Default is True.
    weights : array_like, optional
        Weights for implied volatility errors. Default is None.

    Returns
    -------
    array_like
        Residual errors [error_F, error_impvols] if return_error=True,
        else (F_approx, impvols_approx).
    """
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
    """
    Calibrate Mixed Rough Bergomi model to single-day VIX smile data.

    Parameters
    ----------
    x0 : array_like
        Initial guess [xi0, H, eta_1, eta_2, lbd].
    K : array_like
        Strike prices.
    F : float
        VIX futures price.
    impvols : array_like
        Market implied volatilities.
    T : float
        Time to maturity (years).
    ORDER : int, optional
        Expansion order (default 3).
    EPS : float, optional
        Integration tolerance (default 1e-3).
    N_QUAD : int, optional
        Number of quadrature points (default 50).
    weights : array_like, optional
        Weights for implied volatility errors (default None).

    Returns
    -------
    OptimizeResult
        Optimization result from scipy.optimize.least_squares.
    """

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


def xi0_nelson_siegel(u, b0, b1, b2, tau1, tau2):
    return b0 + b1 * np.exp(-u / tau1) + b2 * (u / tau2) * np.exp(-u / tau2)


def fun_all(x, days, voldata, order=3, n_quad=50, return_error=True, weights=False):
    """
    Compute residuals across multiple days with parametric forward variance curve.

    Parameters
    ----------
    x : array_like
        Model parameters [b0, b1, b2, tau1, tau2, H, eta_1, eta_2, lbd].
    days : array_like
        Days to maturity (in calendar days).
    voldata : dict
        Volatility data indexed by day, containing 'k', 'F', 'Mid', and 'weights'.
    order : int, optional
        Expansion order (default 3).
    n_quad : int, optional
        Number of quadrature points (default 50).
    return_error : bool, optional
        If True, return errors; if False, return model prices (default True).
    weights : bool, optional
        If True, apply weights from voldata (default False).

    Returns
    -------
    array_like
        Concatenated residual errors across all days if return_error=True,
        else (Fs_approx, impvols_approx).
    """

    b0, b1, b2, tau1, tau2, H, eta_1, eta_2, lbd = x

    rbergomi = RoughBergomi(
        xi0=lambda u: xi0_nelson_siegel(u, b0, b1, b2, tau1, tau2),
        params={"H": H, "eta": eta_1},
        rho=-0.9,
    )

    Fs_approx = []
    impvols_approx = []

    for day in days:
        F_approx_day, impvols_approx_day = rbergomi.implied_vol_vix_approx_mixed(
            volvol_2=eta_2,
            lbd=lbd,
            k=voldata[day]["k"],
            T=day / 365.25,
            order=order,
            n_quad=n_quad,
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
    print(f"Error: {np.linalg.norm(error)}")
    return error


def optimize_calibration_all(
    x0,
    days,
    voldata,
    order=3,
    n_quad=50,
    weights=False,
):
    """
    Calibrate Mixed Rough Bergomi model to multi-day VIX smile data.

    Parameters
    ----------
    x0 : array_like
        Initial guess [b0, b1, b2, tau1, tau2, H, eta_1, eta_2, lbd].
    days : array_like
        Days to maturity (in calendar days).
    voldata : dict
        Volatility data indexed by day.
    order : int, optional
        Expansion order (default 3).
    n_quad : int, optional
        Number of quadrature points (default 50).
    weights : bool, optional
        If True, apply weights from voldata (default False).

    Returns
    -------
    OptimizeResult
        Optimization result from scipy.optimize.least_squares.
    """
    lower_bounds = [1e-4, -1.0, -1.0, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 0.0]
    upper_bounds = [1.0, 1.0, 1.0, 20.0, 20.0, 0.5, 20.0, 20.0, 1.0]
    bounds = (lower_bounds, upper_bounds)
    result = least_squares(
        fun_all,
        x0=x0,
        args=(days, voldata, order, n_quad, True, weights),
        bounds=bounds,
        verbose=2,
    )
    return result
