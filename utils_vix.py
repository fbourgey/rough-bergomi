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
    scaled_x = (volvol_2 / volvol_1) * x
    adjustment = (volvol_1 / volvol_2 - 1.0) * (np.log(fvix2) - mu_2)
    return fvix2 * (lbd * np.exp(x) + (1.0 - lbd) * np.exp(adjustment + scaled_x))


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
            return base_derivative * (sqrt_inner >= K)
        else:  # "put"
            return -base_derivative * (sqrt_inner <= K)

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
    Compute weighted sums of normalized Hermite polynomials using Gauss-Hermite
    quadrature for g(y) = sqrt(1 + b * exp(c * y)) and its derivatives.

    Parameters
    ----------
    n_trunc : int
        Maximum order of Hermite polynomials.
    b, c : float
        Parameters for the weight function g(y) = sqrt(1 + b * exp(c * y)).
    n_quad : int
        Number of Gauss-Hermite quadrature points.

    Returns
    -------
    np.ndarray
        Shape (4, n_trunc + 1) array where weights[k, n] corresponds to the k-th
        derivative order (k=0,1,2,3) and n-th Hermite polynomial order.
    """
    x_herm, w_herm = gauss_hermite(n_quad)

    def _g(y):
        return (1.0 + b * np.exp(c * y)) ** 0.5

    def _g_deriv(y, order):
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be 0, 1, 2, or 3")
        if order == 0:
            return _g(y)
        elif order == 1:
            return 0.5 * (_g(y) - _g(y) ** (-1))
        elif order == 2:
            return 0.25 * (_g(y) - _g(y) ** (-3))
        else:
            return 0.125 * (_g(y) + 2.0 * _g(y) ** (-3) - 3.0 * _g(y) ** (-5))

    def integrand(n, y, order):
        return (
            _g_deriv(y, order) * special.eval_hermitenorm(n, y) / special.factorial(n)
        )

    weights = np.array(
        [
            [np.sum(w_herm * integrand(n, x_herm, order)) for n in range(n_trunc + 1)]
            for order in range(4)
        ]
    )
    return weights


def _compute_coeff_mixed_case(params):
    """Compute coefficients c0, c1, c2, c3 based on the mixed case formulas."""

    lbd = params["lbd"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    gamma_1 = params["gamma_1"]
    gamma_2 = params["gamma_2"]
    gamma_3 = params["gamma_3"]

    a = (1 - lbd) ** 0.5 * np.exp(meanp_2 / 2 + sigp_2**2 / 8)

    c0 = (1 + 0.5 * gamma_1[1] + 0.25 * gamma_2[1] + 0.125 * gamma_3[1]) * a

    c1 = (
        gamma_1[0]
        - gamma_1[1]
        + gamma_2[0] * sigp_2 / (2 * sigp_1)
        - gamma_2[1] * (1 - sigp_1 / (2 * sigp_2))
        + gamma_3[0] * sigp_2**2 / (4 * sigp_1**2)
        - gamma_3[1] * (0.75 - sigp_1 / (2 * sigp_2))
    ) * a

    c2 = (
        gamma_2[0] * (1 - sigp_2 / sigp_1)
        + gamma_2[1] * (1 - sigp_1 / sigp_2)
        + gamma_3[0] * (sigp_2 / sigp_1 - sigp_2**2 / sigp_1**2)
        + gamma_3[1] * (1.5 - 2 * sigp_1 / sigp_2 + sigp_1**2 / (2 * sigp_2**2))
    ) * a

    c3 = (
        gamma_3[0] * (1 - 2 * sigp_2 / sigp_1 + sigp_2**2 / sigp_1**2)
        - gamma_3[1] * (1 - 2 * sigp_1 / sigp_2 + sigp_1**2 / sigp_2**2)
    ) * a

    c_vec = np.array([c0, c1, c2, c3])

    return c_vec
