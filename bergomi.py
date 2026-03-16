from collections.abc import Callable

import numpy as np
from scipy import stats

from model import ForwardVarianceModel
from utils import black_impvol, gauss_legendre
from utils_vix import _vix_payoff


class OneFactorBergomi(ForwardVarianceModel):
    """
    Implementation of the one-factor Bergomi model.

    Parameters
    ----------
    xi0 : callable
        Forward variance curve function xi0(t), must return positive values for t >= 0.
    rho : float
        Correlation between the spot and the volatility processes (must be in [-1, 1]).
    params : dict
        Dictionary containing model parameters: 'k' (speed of mean reversion,
        must be positive)
        and 'w' (volatility of volatility, must be positive).
    s0 : float, optional
        Initial spot price (must be positive). Default is 1.0.
    name : str, optional
        Name identifier for the model. Default is 'one_factor_bergomi'.
    """

    def __init__(
        self,
        xi0: Callable[[np.ndarray], np.ndarray],
        rho: float,
        params: dict,
        s0: float = 1.0,
        name: str = "one_factor_bergomi",
    ) -> None:
        """
        Initialize the one-factor Bergomi model.
        See class docstring for parameter definitions.
        """
        super().__init__(xi0=xi0, rho=rho, s0=s0, params=params, name=name)
        self.k = params["k"]
        self.w = params["w"]
        # check parameter validity
        if self.k <= 0.0:
            raise ValueError("Speed of mean reversion k must be positive.")
        if self.w <= 0.0:
            raise ValueError("Volatility of volatility w must be positive.")

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
        """
        Compute the variance of the OU factor X_t.

        Parameters
        ----------
        t : float or np.ndarray
            Time point(s).

        Returns
        -------
        float or np.ndarray
            Variance of X_t at time t.
        """
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

    def price_vix_fut(self, T, n_quad, lbd=None, w_2=None):
        """
        Estimate the price of a VIX futures contract at maturity T using Gauss
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int
            Number of quadrature points for numerical integration.
        lbd : float or None, optional
            If provided, use a mixed model with two different volatility-of-volatility
            values. lbd is the weight for the first w value. Default is None.
        w_2 : float or None, optional
            If provided, use a mixed model with two different w values.
            This is the second w value. Must be provided if `lbd` is not None.
        Returns
        -------
        float
            VIX futures price at maturity T.
        """
        return self.price_vix(T=T, n_quad=n_quad, opt_payoff="fut", lbd=lbd, w_2=w_2)

    def price_vix(self, T, n_quad, opt_payoff, K=0.0, lbd=None, w_2=None):
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
            Type of option payoff ('call', 'put', or 'fut').
        K : float, optional
            Strike price of the option. Default is 0.0.
        lbd : float or None, optional
            If provided, use a mixed model with two different volatility-of-volatility values.
            lbd is the weight for the first w value. Default is None.
        w_2 : float or None, optional
            If provided, use a mixed model with two different w values.
            This is the second w value. Must be provided if `lbd` is not None.
        Returns
        -------
        float
            Estimated VIX option price at maturity T.
        """
        v_leg, w_leg = gauss_legendre(0, 1, n_quad)
        std_x = self.var_x(T) ** 0.5
        x_norm = stats.norm.ppf(v_leg)
        u_leg = T + v_leg * self.delta_vix
        xi0_leg = self.xi0(u_leg)
        if lbd is not None and w_2 is not None:
            # mixed case
            # create a second Bergomi model with vol-of-vol w_2
            onebergomi_2 = self._clone_with_params(w=w_2)
            func = lambda y: (
                lbd * self._f_xi(t=T, u=u_leg, x=std_x * y)
                + (1 - lbd) * onebergomi_2._f_xi(t=T, u=u_leg, x=std_x * y)
            )
        else:
            func = lambda y: self._f_xi(t=T, u=u_leg, x=std_x * y)

        vix2_norm = np.array([np.sum(w_leg * xi0_leg * func(x)) for x in x_norm])
        return np.sum(w_leg * _vix_payoff(opt_payoff, K=K)(vix2_norm))

    def implied_vol_vix(self, k, T, n_quad, lbd=None, w_2=None) -> np.ndarray:
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
        lbd : float or None, optional
            If provided, use a mixed model with two different eta values.
            lbd is the weight for the first eta value.
        w_2 : float or None, optional
            If provided, use a mixed model with two different w values.
            This is the second w value. Must be provided if `lbd` is not None.

        Returns
        -------
        np.ndarray
            Implied volatility values for the VIX option(s) at the specified
            log-moneyness.
        """
        F = self.price_vix(T=T, n_quad=n_quad, opt_payoff="fut", lbd=lbd, w_2=w_2)
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
                    lbd=lbd,
                    w_2=w_2,
                )
                for i, K_i in enumerate(K)
            ]
        )
        return black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)
