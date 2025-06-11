import numpy as np
from scipy import optimize, stats


def gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on the interval [a, b].

    Parameters
    ----------
    a : float
        Lower bound of the integration interval.
    b : float
        Upper bound of the integration interval.
    n : int
        Number of quadrature points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 1-D arrays:
        - Quadrature points on [a, b].
        - Quadrature weights on [a, b].
    """
    knots, weights = np.polynomial.legendre.leggauss(n)
    knots_a_b = 0.5 * (b - a) * knots + 0.5 * (b + a)
    weights_a_b = 0.5 * (b - a) * weights
    return knots_a_b, weights_a_b


def cholesky_from_svd(a: np.ndarray) -> np.ndarray:
    """
    Compute the Cholesky decomposition of a matrix using SVD and QR.

    This function works with positive semi-definite matrices.

    Parameters
    ----------
    a : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        The Cholesky decomposition of the input matrix.
    """
    u, s, _ = np.linalg.svd(a)
    b = np.diag(np.sqrt(s)) @ u.T
    _, r = np.linalg.qr(b)
    return r.T


def black_price(K, T, F, vol, opttype=1):
    """
    Calculate the Black option price.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    opttype : int, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        The Black price of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    d2 = d1 - s
    price = opttype * (
        F * stats.norm.cdf(opttype * d1) - K * stats.norm.cdf(opttype * d2)
    )
    return price


def black_delta(K, T, F, vol, opttype=1):
    """
    Calculate the Black delta of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    opttype : int, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        The Black delta of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return opttype * stats.norm.cdf(opttype * d1)


def black_gamma(K, T, F, vol):
    """
    Calculate the Black gamma of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The Black gamma of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return stats.norm.pdf(d1) / (F * s)


def black_speed(K, T, F, vol):
    """
    Calculate the Black speed of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The Black speed of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return -(d1 / s + 1.0) * stats.norm.pdf(d1) / (F**2 * s)


def black_vega(K, T, F, vol):
    """
    Calculate the Black vega of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The Black vega of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return F * stats.norm.pdf(d1) * np.sqrt(T)


@np.vectorize
def black_impvol_brentq(K, T, F, value, opttype=1):
    """
    Calculate the Black implied volatility using the Brent's method.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    value : float
        Observed market price of the option.
    opttype : int, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        Implied volatility corresponding to the input option price. Returns NaN
        if the implied volatility does not converge or if invalid inputs are provided.
    """
    if (K <= 0) or (T <= 0) or (F <= 0) or (value <= 0):
        return np.nan

    try:
        result = optimize.root_scalar(
            f=lambda vol: black_price(K, T, F, vol, opttype) - value,
            bracket=[1e-10, 5.0],
            method="brentq",
        )
        return result.root if result.converged else np.nan
    except ValueError:
        return np.nan


def black_impvol(
    K, T, F, value, opttype: int | np.ndarray = 1, TOL=1e-5, MAX_ITER=1000
):
    """
    Calculate the Black implied volatility using a bisection method.

    Parameters
    ----------
    K : ndarray or float
        Strike price(s) of the option(s).
    T : float
        Time to maturity of the option(s).
    F : float
        Forward price of the underlying asset.
    value : ndarray or float
        Observed market price(s) of the option(s).
    opttype : int or ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.
    TOL : float, optional
        Tolerance for convergence of the implied volatility. Default is 1e-6.
    MAX_ITER : int, optional
        Maximum number of iterations for the bisection method. Default is 1000.

    Returns
    -------
    ndarray or float
        Implied volatility(ies) corresponding to the input option prices. If the
        input arrays are multidimensional, the output will have the same shape.
        Returns NaN if the implied volatility does not converge or if invalid
        inputs are provided.

    Raises
    ------
    ValueError
        If `K` and `value` do not have the same shape.
        If `opttype` is not 1 or -1.
        If the implied volatility does not converge within `MAX_ITER` iterations.
    """
    K = np.atleast_1d(K)
    value = np.atleast_1d(value)
    opttype = np.full_like(K, opttype)

    if K.shape != value.shape:
        raise ValueError("K and value must have the same shape.")

    if np.abs(opttype).any() != 1:
        raise ValueError("opttype must be either 1 or -1.")

    F = float(F)
    T = float(T)

    if T <= 0 or F <= 0:
        return np.full_like(K, np.nan)

    low = np.full_like(K, 1e-10)
    high = np.full_like(K, 5.0)
    mid = 0.5 * (low + high)
    for _ in range(MAX_ITER):
        price = black_price(K, T, F, mid, opttype)
        diff = (price - value) / value

        if np.all(np.abs(diff) < TOL):
            return mid

        mask = diff > 0
        high[mask] = mid[mask]
        low[~mask] = mid[~mask]
        mid = 0.5 * (low + high)

    raise ValueError("Implied volatility did not converge.")


def black_otm_impvol_mc(S, k, T, mc_error=False):
    """
    Calculate Black implied volatility using Monte Carlo simulated stock prices and
    out-of-the-money (OTM) prices.

    Parameters
    ----------
    S : ndarray
        Array of Monte Carlo simulated stock prices.
    k : float or ndarray
        Log-Forward Moneyness `k=log(K/F)` for which the implied volatility is
        calculated.
    T : float
        Time to maturity of the option.
    mc_error : bool, optional
        If True, computes the 95% confidence interval for the implied volatility.

    Returns
    -------
    dict or ndarray
        If `mc_error` is False, returns an ndarray of OTM implied volatilities.
        If `mc_error` is True, returns a dictionary with the following keys:
        - 'otm_impvol': ndarray of OTM implied volatilities.
        - 'otm_impvol_high': ndarray of upper bounds of the 95% confidence interval.
        - 'otm_impvol_low': ndarray of lower bounds of the 95% confidence interval.
        - 'error_95': ndarray of the 95% confidence interval errors for the option
                      prices.
        - 'otm_price': ndarray of the calculated OTM option prices.
    """
    k = np.atleast_1d(np.asarray(k))
    F = np.mean(S)
    K = F * np.exp(k)
    opttype = 2 * (K >= F) - 1
    payoff = np.maximum(opttype[None, :] * (S[:, None] - K[None, :]), 0.0)
    otm_price = np.mean(payoff, axis=0)
    otm_impvol = black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

    if mc_error:
        error_95 = 1.96 * np.std(payoff, axis=0) / S.shape[0] ** 0.5
        otm_impvol_high = black_impvol(
            K=K, T=T, F=F, value=otm_price + error_95, opttype=opttype
        )
        otm_impvol_low = black_impvol(
            K=K, T=T, F=F, value=otm_price - error_95, opttype=opttype
        )
        return {
            "otm_impvol": otm_impvol,
            "otm_impvol_high": otm_impvol_high,
            "otm_impvol_low": otm_impvol_low,
            "error_95": error_95,
            "otm_price": otm_price,
        }

    return otm_impvol


def fourier(n, t):
    """
    Compute the first n Fourier basis functions evaluated at point s.

    The basis consists of:
        - 1 (constant term)
        - sqrt(2) * sin(2 * pi * k * t), sqrt(2) * cos(2 * pi * k * t) for k = 1, 2, ...

    Parameters
    ----------
    n : int
        Number of Fourier basis functions to compute.
    t : float
        Point at which to evaluate the basis functions.

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing the values of the first n Fourier basis
        functions at s.
    """
    tab = np.zeros(n)
    tab[0] = 1.0
    for i in range(1, n):
        tab[i] = (
            np.cos(2.0 * np.pi * i * t) if i % 2 == 0 else np.sin(2.0 * np.pi * i * t)
        )
        tab[i] *= np.sqrt(2.0)
    return tab
