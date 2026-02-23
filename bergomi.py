from collections.abc import Callable
import numpy as np
from scipy import stats
from model import ForwardVarianceModel

from utils import black_impvol, gauss_hermite, gauss_legendre
from utils_vix import _vix_payoff


class OneFactorBergomi(ForwardVarianceModel):
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
        super().__init__(xi0=xi0, rho=rho, s0=s0, delta_vix=delta_vix)
        if k <= 0.0:
            raise ValueError("Speed of mean reversion k must be positive.")
        if w <= 0.0:
            raise ValueError("Volatility of volatility w must be positive.")
        self.k = k
        self.w = w

    def kernel(self, u, t):
        """
        Compute the one-factor Bergomi kernel function w * exp(-k * (u - t)).

        Parameters
        ----------
        u : float or np.ndarray
            Upper time(s) (must satisfy u > t).
        t : float or np.ndarray
            Lower time(s).

        Returns
        -------
        float or np.ndarray
            Value(s) of the kernel.
        """
        return self.w * np.exp(-self.k * (u - t))

    def var_x(self, t):
        if self.k == 0:
            return t
        else:
            return (1 - np.exp(-2 * self.k * t)) / (2 * self.k)

    def _f_xi(self, t, u, x):
        """
        Helper function to compute the forward variance at time t for maturity u
        given the OU factor X_t = x where xi_t(u) = xi_0(u) * f^u(t, x).
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
            VIX futures price at maturity T.
        """
        return self.price_vix(T=T, n_quad=n_quad, opt_payoff="fut")

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
        v_leg, w_leg = gauss_legendre(0, 1, n_quad)
        std_x = self.var_x(T) ** 0.5
        x_norm = stats.norm.ppf(v_leg)
        xi0_leg = self.xi0(T + v_leg * self.delta_vix)
        vix2_norm = np.array(
            [
                np.sum(
                    w_leg
                    * xi0_leg
                    * self._f_xi(t=T, u=v_leg * self.delta_vix + T, x=std_x * x)
                )
                for x in x_norm
            ]
        )
        return np.sum(w_leg * _vix_payoff(opt_payoff, K=K)(vix2_norm))

    def implied_vol_vix(self, k, T, n_quad, lbd=None, eta_2=None) -> np.ndarray:
        """
        Compute the implied volatility of a VIX option at a given log-moneyness
        using Monte Carlo simulation.

        Parameters
        ----------
        k : float or np.ndarray
            Log-moneyness of the VIX option (typically 0 for ATM). Can be a scalar
            or array.
        T : float
            Maturity of the VIX option.
        n_quad : int
            Number of quadrature points for numerical integration.

        Returns
        -------
        np.ndarray
            Implied volatility values for the VIX option(s) at the specified
            log-moneyness.
        """
        F = self.price_vix(T=T, n_quad=n_quad, opt_payoff="fut")
        k = np.atleast_1d(np.asarray(k))
        K = F * np.exp(k)
        opttype = 2 * (K >= F) - 1
        otm_price = np.array(
            [
                self.price_vix(
                    T=T,
                    n_quad=n_quad,
                    opt_payoff="put" if opttype[i] == -1 else "call",
                    K=K_i,
                )
                for i, K_i in enumerate(K)
            ]
        )
        return black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)
