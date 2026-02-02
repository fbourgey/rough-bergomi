import numpy as np
from collections.abc import Callable

from utils import gauss_hermite, gauss_legendre


class OneFactorBergomi:
    """
    Implementation of the one-factor Bergomi model.

    Parameters
    ----------
    s0 : float
        Initial spot price (must be positive).
    xi0 : callable
        Forward variance curve function xi0(t), must return positive values for t >= 0.
    k : float
        Speed of mean reversion (must be >=0).
    w : float
        Volatility of volatility parameter (must be positive).
    rho : float
        Correlation between the spot and the volatility processes (must be in [-1, 1]).
    """

    def __init__(
        self,
        xi0: Callable[[np.ndarray], np.ndarray],
        k: float,
        w: float,
        rho: float,
        s0: float = 1.0,
        delta_vix: float = 1.0 / 12.0,
    ) -> None:
        """
        Initialize the one-factor Bergomi model.
        See class docstring for parameter definitions.
        """
        if k <= 0.0:
            raise ValueError("Speed of mean reversion k must be positive.")
        if s0 <= 0.0:
            raise ValueError("Initial spot price s0 must be positive.")
        if w <= 0.0:
            raise ValueError("Volatility of volatility w must be positive.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("Correlation rho must be in [-1, 1].")
        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")
        # Check positivity for a range of t >= 0
        t_test = np.linspace(1e-10, 10, 1000)
        if not np.all(xi0(t_test) > np.array([0.0])):
            raise ValueError("xi0 must be positive for all t >= 0.")

        self.s0 = s0
        self.xi0 = xi0
        self.xi0_0 = self.xi0(np.zeros(1))[0]
        self.xi0_flat = self._is_xi0_flat()
        self.k = k
        self.w = w
        self.rho = rho
        self.delta_vix = delta_vix

    def _is_xi0_flat(self) -> bool:
        """Check if the forward variance curve xi0 is flat."""
        t_test = np.linspace(1e-10, 10, 1000)
        return np.allclose(self.xi0(t_test), self.xi0_0)

    def var_x(self, t):
        if self.k == 0:
            return t
        else:
            return (1 - np.exp(-2 * self.k * t)) / (2 * self.k)

    def _f_xi(self, t, u, x):
        """
        Helper function to compute the forward variance at time t for maturity u
        given the OU factor X_t = x.
            xi_t(u) = xi_0(u) * f^u(t, x)
        """
        _exp = np.exp(-self.k * (u - t))
        return np.exp(self.w * _exp * x - 0.5 * self.w**2 * _exp**2 * self.var_x(t))

    def price_vix_fut(self, T, n_quad, lbd=None, eta_2=None):
        """
        Estimate the price of a VIX futures contract at maturity T using Gauss
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int
            Number of quadrature points for numerical integration.
        Returns
        -------
        float
            Estimated VIX futures price at maturity T.
        """

        x_herm, w_herm = gauss_hermite(n_quad)
        x_leg, w_leg = gauss_legendre(T, T + self.delta_vix, n_quad)
        std_x = self.var_x(T) ** 0.5
        xi0_leg = self.xi0(x_leg)
        vix2_herm = np.array(
            [np.sum(w_leg * xi0_leg * self._f_xi(T, x_leg, x / std_x)) for x in x_herm]
        )
        return np.sum(w_herm * vix2_herm**0.5)

    def price_vix(self, T, n_quad, opt_payoff, K=0.0, lbd=None, eta_2=None):
        """
        Estimate the price of a VIX option at maturity T using Gauss
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        n_quad : int
            Number of quadrature points for numerical integration.
        opt_payoff : str
            Type of option payoff ('call' or 'put').
        K : float, optional
            Strike price of the option (default is 0.0).
        Returns
        -------
        float
            Estimated VIX option price at maturity T.
        """

        x_herm, w_herm = gauss_hermite(n_quad)
        x_leg, w_leg = gauss_legendre(T, T + self.delta_vix, n_quad)
        std_x = self.var_x(T) ** 0.5
        xi0_leg = self.xi0(x_leg)
        vix2_herm = np.array(
            [np.sum(w_leg * xi0_leg * self._f_xi(T, x_leg, x / std_x)) for x in x_herm]
        )
        payoff = _vix_payoff(opt_payoff, K=K)
        return np.sum(w_herm * payoff(vix2_herm))


def _vix_payoff(opt_payoff, K=0.0):
    """Create payoff and derivative functions for mixed proxy approximation."""
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
