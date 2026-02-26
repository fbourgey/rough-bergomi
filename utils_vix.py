import numpy as np
from scipy import optimize, special

from utils import gauss_hermite


def _inner_mixed_func(x, lbd, mu_2, volvol_1, volvol_2, fvix2):
    """
    Compute the inner function for mixed proxy payoff calculations.

    Parameters
    ----------
    x : float or np.ndarray
        Gaussian random variable.
    lbd : float
        Mixing weight in [0, 1].
    mu_2 : float
        Location parameter of the second component.
    volvol_1, volvol_2 : float
        Scale parameters of the two volatility-of-volatility components.
    fvix2 : float
        VIX squared futures price.

    Returns
    -------
    float or np.ndarray
        Value of the inner function.
    """
    log_fvix2 = np.log(fvix2)
    term_x = (volvol_1 - volvol_2) * (mu_2 - log_fvix2) / volvol_2 + (
        volvol_2 / volvol_1
    ) * x
    return fvix2 * (lbd * np.exp(x) + (1 - lbd) * np.exp(term_x))


def _inverse_x_inner_mixed_func(z, mu_2, lbd, volvol_1, volvol_2, fvix2):
    """
    Compute the inverse of the inner function for mixed proxy payoff calculations.

    Parameters
    ----------
    z : float
        Target value.
    mu_2 : float
        Location parameter of the second component.
    lbd : float
        Mixing weight in [0, 1].
    volvol_1, volvol_2 : float
        Scale parameters of the two volatility-of-volatility components.
    fvix2 : float
        VIX squared futures price.

    Returns
    -------
    float
        Solution x to the inner function equation.
    """

    def func(x):
        return _inner_mixed_func(x, lbd, mu_2, volvol_1, volvol_2, fvix2) - z

    return optimize.root_scalar(func, bracket=[-100, 100]).root


def _vix_payoff(opt_payoff, K=0.0):
    """
    Create VIX payoff function based on option type.

    Parameters
    ----------
    opt_payoff : str
        Option type: 'fut' for futures, 'call' for call options, 'put' for put options.
    K : float, optional
        Strike price (only used for calls and puts). Default is 0.0.

    Returns
    -------
    callable
        Payoff function that takes VIX squared values and returns payoffs.
    """
    if opt_payoff not in ["fut", "call", "put"]:
        raise ValueError("opt_payoff must be one of 'fut', 'call', or 'put'.")

    # Define payoff function
    if opt_payoff == "fut":
        payoff = lambda x: np.sqrt(x)
    elif opt_payoff == "call":
        payoff = lambda x: np.maximum(np.sqrt(x) - K, 0.0)
    else:  # "put"
        payoff = lambda x: np.maximum(K - np.sqrt(x), 0.0)

    return payoff


def _deriv_vix_payoff_mixed(opt_payoff, K=0.0):
    """
    Create derivative of VIX payoff function for mixed proxy calculations.

    Parameters
    ----------
    opt_payoff : str
        Option type: 'fut' for futures, 'call' for call options, 'put' for put options.
    K : float, optional
        Strike price (only used for calls and puts). Default is 0.0.

    Returns
    -------
    callable
        Derivative of payoff function used in mixed proxy calculations.
    """
    if opt_payoff not in ["fut", "call", "put"]:
        raise ValueError("opt_payoff must be one of 'fut', 'call', or 'put'.")

    def dpayoff_mixed_dy(x, lbd, mu_2, volvol_1, volvol_2, fvix2):
        sqrt_inner = _inner_mixed_func(x, lbd, mu_2, volvol_1, volvol_2, fvix2) ** 0.5
        base_derivative = fvix2 * lbd * np.exp(x) / (2.0 * sqrt_inner)

        if opt_payoff == "fut":
            return base_derivative
        elif opt_payoff == "call":
            return base_derivative * (sqrt_inner > K)
        else:  # "put"
            return -base_derivative * (sqrt_inner < K)

    return dpayoff_mixed_dy


def _inverse_mixture_lognormal(y, lbd, mu_1, mu_2, sig_1, sig_2):
    """
    Solve for x in the mixture of lognormals equation.

    Finds x such that:
        lbd * exp(mu_1 + sig_1 * x) + (1 - lbd) * exp(mu_2 + sig_2 * x) = y

    Parameters
    ----------
    y : float
        Target value.
    lbd : float
        Mixing weight in [0, 1].
    mu_1, mu_2 : float
        Location parameters of the two lognormal components.
    sig_1, sig_2 : float
        Scale parameters of the two lognormal components.

    Returns
    -------
    float
        Solution x to the mixture equation.
    """
    return optimize.root_scalar(
        lambda x: (
            lbd * np.exp(mu_1 + sig_1 * x) + (1 - lbd) * np.exp(mu_2 + sig_2 * x) - y
        ),
        bracket=[-100, 100],
    ).root


def _hermite_polynomial_weights(n_trunc, b, c, n_quad):
    """
    Compute weighted sum of normalized Hermite polynomials using Gauss-Hermite
    quadrature.

    Parameters
    ----------
    n_trunc : int
        Number of Hermite polynomials to sum over.
    b, c : float
        Parameters for the weight function g(y) = sqrt(1 + b * exp(c * y)).
    n_quad : int
        Number of Gauss-Hermite quadrature points.

    Returns
    -------
    np.ndarray
        Array of weighted sums of normalized Hermite polynomials for orders 0 to n_trunc.
    """
    x_herm, w_herm = gauss_hermite(n_quad)

    def g(y):
        return (1 + b * np.exp(c * y)) ** 0.5

    def integrand(n, y):
        return g(y) * special.eval_hermitenorm(n, y) / special.factorial(n)

    weights = np.array(
        [np.sum(w_herm * integrand(n, x_herm)) for n in range(n_trunc + 1)]
    )
    return weights
