import numpy as np
from scipy import optimize, stats, integrate, special
import mpmath as mpm


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
    w = T * vol**2
    d1 = np.log(F / K) / w**0.5 + 0.5 * w**0.5
    d2 = d1 - w**0.5
    price = opttype * (
        F * stats.norm.cdf(opttype * d1) - K * stats.norm.cdf(opttype * d2)
    )
    return price


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
    w = T * vol**2
    d1 = np.log(F / K) / w**0.5 + 0.5 * w**0.5
    return (F * stats.norm.pdf(d1) * np.sqrt(T)) / (K * vol)


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


def black_impvol(K, T, F, value, opttype=1, TOL=1e-6, MAX_ITER=1000):
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

    low = 1e-10 * np.ones_like(K)
    high = 5.0 * np.ones_like(K)
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


def kernel_rbergomi(t, s, H):
    """Compute the kernel function in the rough Bergomi model."""
    return np.sqrt(2.0 * H) * (t - s) ** (H - 0.5)


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


def fourier_hat(n, t, H, n_quad=20):
    r"""
    Compute the integrals of the rough Bergomi kernel times the Fourier basis functions.

    For each k in 0,...,n-1, computes:
        \int_0^t sqrt(2H) * (t-s)^{H-1/2} * e_k(s) ds,
    where e_k(s) is the k-th Fourier basis function.

    Parameters
    ----------
    n : int
        Number of Fourier basis functions.
    t : float
        Upper limit of integration.
    H : float
        Hurst parameter.

    Returns
    -------
    np.ndarray
        Array of shape (n,) with the value of each integral for k = 0,...,n-1.
    """
    # we can use a gauss-jacobi quadrature rule for the integral

    nodes, weights = special.roots_jacobi(n_quad, H - 0.5, 0)

    tab = np.zeros(n)
    tab[0] = np.sqrt(2 * H) * t ** (H + 0.5) / (H + 0.5)
    for i in range(1, n):
        # if i % 2 == 0:
        #     # \int_0^t sqrt(2H) * (t-s)^{H-1/2} * sqrt(2) * cos(i*pi*s) ds
        #     # tab[i] = integrate.quad(
        #     #     lambda s: kernel_rbergomi(t, s, H)
        #     #     * np.sqrt(2)
        #     #     * np.cos(2 * np.pi * i * s),
        #     #     0.0,
        #     #     t,
        #     # )[0]
        #     # tab[i] = mpm.hyp1f2(
        #     #     1, 0.75 + 0.5 * H, 1.25 + 0.5 * H, -((0.5 * i * np.pi * t) ** 2)
        #     # )
        #     # tab[i] *= (4.0 * np.sqrt(H) * t ** (H + 0.5)) / (1.0 + 2.0 * H)
        #     func = lambda s: np.sqrt(2 * np.pi) * np.sqrt(2) * np.cos(2 * np.pi * i * s)
        # else:
        #     # \int_0^t sqrt(2H) * (t-s)^{H-1/2} * sqrt(2) * sin(i*pi*s) ds
        #     # tab[i] = integrate.quad(
        #     #     lambda s: kernel_rbergomi(t, s, H)
        #     #     * np.sqrt(2)
        #     #     * np.sin(2 * np.pi * i * s),
        #     #     0.0,
        #     #     t,
        #     # )[0]
        #     # tab[i] = mpm.hyp1f2(
        #     #     1, 1.25 + 0.5 * H, 1.75 + 0.5 * H, -((0.5 * i * np.pi * t) ** 2)
        #     # )
        #     # tab[i] *= (16.0 * np.sqrt(H) * 0.5 * i * np.pi * t ** (1.5 + H)) / (
        #     #     3 + 8 * H + 4 * H**2
        #     # )
        #     func = lambda s: np.sqrt(2 * np.pi) * np.sqrt(2) * np.sin(2 * np.pi * i * s)
        # tab[i] = (0.5 * t) ** (H + 0.5) * np.dot(weights, func(0.5 * t * (1 + nodes)))
        alpha = H + 0.5
        tab[i] = (
            np.real(
                np.exp(1j * 2 * np.pi * i * t)
                * special.hyp1f1(alpha, alpha + 1, 1j * 2 * np.pi * i * t)
            )
            * t**alpha
            / alpha
        )
        tab[i] *= np.sqrt(2 * np.pi) * np.sqrt(2)
    return tab


def objective_function(a, y, sigma_0, eta, H, rho):
    # TODO: check again
    """
    Compute the rate function objective for the large deviations principle in
    rough volatility models.

    Parameters
    ----------
    a : np.ndarray
        Array of Fourier coefficients (shape: n,).
    y : float
        Target value (e.g., log-moneyness).
    sigma_0 : float
        Initial volatility.
    eta : float
        Volatility of volatility parameter.
    H : float
        Hurst parameter.
    rho : float
        Correlation parameter.

    Returns
    -------
    float
        Value of the objective function.
    """
    n = np.shape(a)[0]

    def h_fourier(t):
        r"""
        Compute the Fourier series expansion
        h_fourier(a, t) = \sum_{k=0}^{n-1} a_k * e_k(t)
        """
        return np.dot(a, fourier(n=n, t=t))

    def h_hat(t):
        r"""
        Compute \int_0^t kernel_rb(t, s, H) * h_prime(a, s) ds
        """
        return np.dot(a, fourier_hat(n=n, t=t, H=H))

    def sigma(x):
        return sigma_0 * np.exp(0.5 * eta * x)

    a = np.atleast_1d(a)
    norm_h_fourier_squared = np.sum(a**2)
    F = integrate.quad(lambda x: sigma(h_hat(x)) ** 2, 0, 1)[0]
    G = integrate.quad(lambda x: sigma(h_hat(x)) * h_fourier(x), 0, 1)[0]

    return (y - rho * G) ** 2 / (
        2.0 * (1.0 - rho**2) * F
    ) + 0.5 * norm_h_fourier_squared
