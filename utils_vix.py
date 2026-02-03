import numpy as np
from scipy import optimize, special
from utils import gauss_hermite


def _inner_mixed_func(x, lbd, mu_2, eta_1, eta_2, fvix2):
    """Inner function for mixed proxy payoff calculations."""
    log_fvix2 = np.log(fvix2)
    term_x = 0.5 * eta_2 * (eta_1 - eta_2) * (mu_2 - log_fvix2) + (eta_2 / eta_1) * x
    return fvix2 * (lbd * np.exp(x) + (1 - lbd) * np.exp(term_x))


def _inverse_x_inner_mixed_func(z, mu_2, lbd, eta_1, eta_2, fvix2):
    """Inverse of the inner function for mixed proxy payoff calculations."""

    def func(x):
        return _inner_mixed_func(x, lbd, mu_2, eta_1, eta_2, fvix2) - z

    return optimize.root_scalar(func, bracket=[-100, 100]).root


def _vix_payoff(opt_payoff, K=0.0):
    """Create vix payoff functions based on option type."""
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
    """Create derivative of vix payoff functions for mixed proxy."""
    if opt_payoff not in ["fut", "call", "put"]:
        raise ValueError("opt_payoff must be one of 'fut', 'call', or 'put'.")

    def dpayoff_mixed_dy(x, lbd, mu_2, eta_1, eta_2, fvix2):
        sqrt_inner = _inner_mixed_func(x, lbd, mu_2, eta_1, eta_2, fvix2) ** 0.5
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
    float
        Weighted sum of normalized Hermite polynomials.
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
