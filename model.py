from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy import integrate, stats
from scipy.special import eval_hermitenorm

import utils
from utils_vix import (
    _compute_coeff_mixed_case,
    _deriv_vix_payoff_mixed,
    _hermite_polynomial_weights,
    _inverse_mixture_lognormal,
    _inverse_x_inner_mixed_func,
    _vix_payoff,
)


class ForwardVarianceModel(ABC):
    def __init__(
        self,
        xi0: Callable[[np.ndarray], np.ndarray],
        rho: float,
        params: dict,
        name: str,
        s0: float = 1.0,
    ) -> None:
        """
        Initialize ForwardVarianceModel.

        Parameters
        ----------
        xi0 : callable
            Forward variance curve function xi0(t), must return positive values for
            t >= 0.
        rho : float
            Correlation between the spot and the volatility processes
            (must be in [-1, 1]).
        params : dict
            Model-specific parameters.
        name : str
            Name of the model.
        s0 : float
            Initial spot price (must be positive).
        """
        if s0 <= 0.0:
            raise ValueError("Initial spot price s0 must be positive.")

        if not (-1.0 <= rho <= 1.0):
            raise ValueError("Correlation rho must be in [-1, 1].")

        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")

        self.xi0 = xi0
        self._is_xi0_positive()
        self.xi0_0 = self.xi0(np.zeros(1))[0]
        self.xi0_flat = self._is_xi0_flat()
        self.rho = rho
        self.params = params
        self.name = name
        self.s0 = s0
        self.delta_vix = 1.0 / 12.0

    @abstractmethod
    def kernel(self, u, t) -> float | np.ndarray:
        """
        Compute the model-specific kernel function.

        Parameters
        ----------
        u : float or np.ndarray
            Upper time(s) (must satisfy u >= t).
        t : float or np.ndarray
            Lower time(s).

        Returns
        -------
        float or np.ndarray
            Value(s) of the kernel function evaluated at (u, t).
        """
        pass

    def _clone_with_params(self, **updates):
        """Return a new model instance with updated params."""
        params = self.params.copy()
        params.update(updates)
        return self.__class__(
            s0=self.s0, xi0=self.xi0, rho=self.rho, name=self.name, params=params
        )

    def _is_xi0_flat(self) -> bool:
        """Check if the forward variance curve xi0 is flat."""
        t_test = np.linspace(1e-10, 10, 1000)
        return np.allclose(self.xi0(t_test), self.xi0_0)

    def _is_xi0_positive(self):
        """Check if the forward variance curve xi0 is positive for t >= 0."""
        t_test = np.linspace(1e-10, 10, 1000)
        if not np.all(self.xi0(t_test) > np.array([0.0])):
            raise ValueError("xi0 must be positive for all t >= 0.")

    def fut_vix2(self, T: float) -> float:
        r"""
        Compute the fair value of a VIX squared futures contract at maturity T. It
        corresponds to:

            E[VIX_T^2] = 1/delta \int_{T}^{T+delta} \xi_0^u du

        where delta is the VIX window (30/365 years by default) and xi0(u) is the
        forward variance curve.

        Parameters
        ----------
        T : float
            Maturity of the VIX future (must be non-negative).

        Returns
        -------
        float
            Fair value of the VIX squared futures contract at time T.

        Raises
        ------
        ValueError
            If T is negative.

        Notes
        -----
        This is a model-free quantity, depending only on the forward variance curve.
        """
        if T < 0:
            raise ValueError("Maturity T must be non-negative.")

        integral, _ = integrate.quad(lambda u: self.xi0(u), T, T + self.delta_vix)
        return integral / self.delta_vix

    ####################################################################################
    # Weak approximation methods for VIX pricing
    ####################################################################################

    def mean_proxy(self, T, n_quad=30, quad_scipy=True):
        r"""
        Compute the mean of the VIX proxy (log-variance process) at maturity T.

        The mean is defined as:
        -1/2 * \int_0^T {
            1/delta * \int_{T}^{T+delta} \xi_0(u) * kernel(u, t)^2 du / F_{VIX^2}
        } dt

        where delta is the VIX window (30/365), xi0(u) is the forward variance curve,
        kernel(u, t) is the rough Bergomi kernel, and F_{VIX^2} is the VIX^2 futures
        price.

        Parameters
        ----------
        T : float
            Maturity.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 40.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. Default is True.

        Returns
        -------
        float
            Mean of the VIX proxy at maturity T.
        """
        if quad_scipy:

            def integrand(t):
                integral = integrate.quad(
                    lambda u: self.xi0(u) * self.kernel(u, t) ** 2,
                    T,
                    T + self.delta_vix,
                )
                return integral[0] / self.delta_vix

            return -0.5 * integrate.quad(integrand, 0, T)[0] / self.fut_vix2(T)

        else:
            v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            mean = -0.5 * T / self.fut_vix2(T)
            mean *= np.sum(
                v_weights[:, None]
                * v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                )
                ** 2
            )
            return mean

    def var_proxy(self, T, n_quad=30, quad_scipy: bool = True):
        r"""
        Compute the variance of the VIX proxy at maturity T.

        The variance is defined as:
        \int_0^T {
            1/delta * \int_{T}^{T+delta} \xi_0(u) * kernel(u, t)^2 du / F_{VIX^2}
        }^2 dt

        where delta is the VIX window (30/365), xi0(u) is the forward variance curve,
        kernel(u, t) is the rough Bergomi kernel, and F_{VIX^2} is the VIX^2 futures
        price.

        Parameters
        ----------
        T : float
            Maturity.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 20.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. Default is True.

        Returns
        -------
        float
            Variance of the VIX proxy at maturity T.
        """
        if quad_scipy:
            fvix2 = self.fut_vix2(T)

            def integrand(t):
                integral = (
                    integrate.quad(
                        lambda u: self.xi0(u) * self.kernel(u, t),
                        T,
                        T + self.delta_vix,
                    )[0]
                    / fvix2
                )
                return (integral / self.delta_vix) ** 2

            return integrate.quad(integrand, 0, T)[0]
        else:
            v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
            inner = np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                )
                ** 2,
                axis=1,
            ) / self.fut_vix2(T)
            return T * np.sum(v_weights * inner**2)

    def integral_kernel(self, t, T):
        r"""
        Compute the normalized integral of the kernel over the VIX proxy time interval.

        Specifically, computes:
            (1/delta) * \int_{T}^{T+delta} xi0(u) * kernel(u, t) du / F_{VIX^2}

        This is used in the calculation of gamma coefficients for the VIX proxy
        expansion.

        Parameters
        ----------
        t : float
            Lower time of the kernel.
        T : float
            Start of the VIX window.

        Returns
        -------
        float
            Value of the normalized kernel integral.
        """
        return (
            integrate.quad(
                lambda u: self.xi0(u) * self.kernel(u, t), T, T + self.delta_vix
            )[0]
            / self.delta_vix
        ) / self.fut_vix2(T)

    def integral_kernel_squared(self, t, T):
        r"""
        Compute the normalized integral of the squared kernel over the VIX proxy time
        interval.

        Specifically, computes:
            (1/delta) * \int_{T}^{T+delta} xi0(u) * kernel(u, t)^2 du / F_{VIX^2}

        This is used in the calculation of gamma coefficients for the VIX proxy
        expansion.

        Parameters
        ----------
        t : float
            Lower time of the kernel.
        T : float
            Start of the VIX window.

        Returns
        -------
        float
            Value of the normalized squared kernel integral.
        """
        return (
            integrate.quad(
                lambda u: self.xi0(u) * self.kernel(u, t) ** 2, T, T + self.delta_vix
            )[0]
            / self.delta_vix
        ) / self.fut_vix2(T)

    def gamma_1_proxy(self, T, n_quad=30):
        """
        Compute the first-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 20.

        Returns
        -------
        float
            First-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")
        v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
        fvix2 = self.fut_vix2(T=T)
        int_kernel_squared = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                )
                ** 2,
                axis=1,
            )
            / fvix2
        )
        int_kernel = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                ),
                axis=1,
            )
            / fvix2
        )
        integrand = (
            0.125
            * T**2
            * np.sum(
                v_weights[None, :]
                * (
                    self.kernel(
                        u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                    )
                    ** 2
                    - int_kernel_squared[None, :]
                ),
                axis=1,
            )
            ** 2
        )
        integrand += (
            0.5
            * T
            * np.sum(
                v_weights[None, :]
                * (
                    self.kernel(
                        u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                    )
                    - int_kernel[None, :]
                )
                ** 2,
                axis=1,
            )
        )
        gamma_1 = (
            np.sum(v_weights * self.xi0(T + self.delta_vix * v_nodes) * integrand)
            / fvix2
        )

        return gamma_1

    def gamma_2_proxy(self, T, n_quad=30):
        """
        Compute the second-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 20.

        Returns
        -------
        float
            Second-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
        fvix2 = self.fut_vix2(T=T)
        int_kernel_squared = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                )
                ** 2,
                axis=1,
            )
            / fvix2
        )
        int_kernel = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                ),
                axis=1,
            )
            / fvix2
        )
        integrand = np.sum(
            v_weights[None, :]
            * int_kernel[None, :]
            * (
                self.kernel(
                    u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                )
                - int_kernel[None, :]
            ),
            axis=1,
        )
        integrand *= np.sum(
            v_weights[None, :]
            * (
                self.kernel(
                    u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                )
                ** 2
                - int_kernel_squared[None, :]
            ),
            axis=1,
        )
        gamma_2 = (
            -0.5
            * T**2
            * np.sum(v_weights * self.xi0(T + self.delta_vix * v_nodes) * integrand)
            / fvix2
        )

        return gamma_2

    def gamma_3_proxy(self, T, n_quad=30):
        """
        Compute the third-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 20.

        Returns
        -------
        float
            Third-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
        fvix2 = self.fut_vix2(T=T)
        int_kernel = (
            np.sum(
                v_weights[None, :]
                * self.xi0(T + self.delta_vix * v_nodes[None, :])
                * self.kernel(
                    u=T + self.delta_vix * v_nodes[None, :], t=T * v_nodes[:, None]
                ),
                axis=1,
            )
            / fvix2
        )
        integrand = (
            np.sum(
                v_weights[None, :]
                * int_kernel[None, :]
                * (
                    self.kernel(
                        u=T + self.delta_vix * v_nodes[:, None], t=T * v_nodes[None, :]
                    )
                    - int_kernel[None, :]
                ),
                axis=1,
            )
            ** 2
        )
        gamma_3 = (
            0.5
            * T**2
            * np.sum(v_weights * self.xi0(T + self.delta_vix * v_nodes) * integrand)
            / fvix2
        )
        return gamma_3

    def price_vix_approx(
        self, k, T, opttype=1, order=3, return_fut=False, meanp=None, tot_varp=None
    ) -> float:
        """
        Approximate the price of a VIX option using a proxy expansion.

        Parameters
        ----------
        k : float
            Log-moneyness (typically 0 for ATM).
        T : float
            Maturity of the VIX option.
        opttype : int, optional
            Option type: 1 for call, -1 for put. Default is 1 (call).
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.
        return_fut : bool, optional
            If True, return the proxy for the VIX future instead of the option price.
            Default is False.
        meanp : float, optional
            Mean parameter for the proxy expansion. If None, it will be computed
            internally.
        tot_varp : float, optional
            Total variance parameter for the proxy expansion. If None, it will be
            computed internally.

        Returns
        -------
        float
            Approximated VIX option price (or VIX future if return_fut is True).

        Raises
        ------
        ValueError
            If order or opttype is invalid, or if T <= 0.
        """
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")

        if opttype not in [-1, 1]:
            raise ValueError("opttype must be either -1 (put) or 1 (call).")

        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        if order >= 1:
            gamma_1 = self.gamma_1_proxy(T=T)
        if order >= 2:
            gamma_2 = self.gamma_2_proxy(T=T)
        if order == 3:
            gamma_3 = self.gamma_3_proxy(T=T)

        if meanp is None:
            meanp = self.mean_proxy(T) + np.log(self.fut_vix2(T))
        if tot_varp is None:
            tot_varp = self.var_proxy(T)

        volp = np.sqrt(tot_varp / T)
        S = np.exp(0.5 * meanp + 0.125 * tot_varp)

        if order >= 0:
            F = S
        if order >= 1:
            F += gamma_1 * 0.5 * S
        if order >= 2:
            F += gamma_2 * 0.25 * S
        if order == 3:
            F += gamma_3 * 0.125 * S
        if return_fut:
            return F

        K = F * np.exp(k)
        # order 0
        price_0 = utils.black_price(K=K, T=T, F=S, vol=0.5 * volp, opttype=opttype)
        if order == 0:
            return price_0
        # order 1
        price_1 = (
            0.5 * S * utils.black_delta(K=K, T=T, F=S, vol=0.5 * volp, opttype=opttype)
        )
        if order == 1:
            return price_0 + gamma_1 * price_1
        # order 2
        price_2 = 0.5 * price_1
        price_2 += 0.25 * S**2 * utils.black_gamma(K=K, T=T, F=S, vol=0.5 * volp)
        if order == 2:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2
        # order 3
        price_3 = -0.5 * price_1 + 1.5 * price_2
        price_3 += 0.125 * S**3 * utils.black_speed(K=K, T=T, F=S, vol=0.5 * volp)
        if order == 3:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2 + gamma_3 * price_3

        raise ValueError("Invalid order specified for VIX option price approximation.")

    def price_vix_fut_approx(self, T, order=3, meanp=None, tot_varp=None) -> float:
        """
        Approximate the price of a VIX futures contract at maturity T using a proxy
        expansion.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.

        Returns
        -------
        float
            Approximated VIX futures price.

        Raises
        ------
        ValueError
            If order is invalid or T <= 0.
        """
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")

        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if meanp is None:
            meanp = self.mean_proxy(T) + np.log(self.fut_vix2(T))
        if tot_varp is None:
            tot_varp = self.var_proxy(T)
        S = np.exp(meanp / 2.0 + tot_varp / 8.0)
        # order 0
        price_0 = S
        if order == 0:
            return price_0
        # order 1
        gamma_1 = self.gamma_1_proxy(T=T)
        price_1 = 0.5 * S
        if order == 1:
            return price_0 + gamma_1 * price_1
        # order 2
        gamma_2 = self.gamma_2_proxy(T=T)
        price_2 = 0.25 * S
        if order == 2:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2
        # order 3
        gamma_3 = self.gamma_3_proxy(T=T)
        price_3 = 0.125 * S
        if order == 3:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2 + gamma_3 * price_3

        raise ValueError("Invalid order specified for VIX futures price approximation.")

    def implied_vol_vix_approx(self, T, k, order=3, meanp=None, tot_varp=None):
        """
        Approximate the implied volatility of a VIX option at a given log-moneyness
        using the proxy expansion.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        k : float or np.ndarray
            Log-moneyness (typically 0 for ATM). Can be a scalar or array.
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.
        mean_p : float or None, optional
            Precomputed mean proxy value. If None, it will be computed internally.
        tot_varp : float or None, optional
            Precomputed total variance proxy value. If None, it will be computed
            internally.

        Returns
        -------
        np.ndarray
            Approximated implied volatility for each log-moneyness value.

        Raises
        ------
        ValueError
            If T <= 0.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        k = np.atleast_1d(np.asarray(k))
        F = self.price_vix_fut_approx(T=T, order=order, meanp=meanp, tot_varp=tot_varp)
        K = F * np.exp(k)
        opttype = 2 * (K >= F) - 1
        otm_price = np.array(
            [
                self.price_vix_approx(
                    T=T,
                    k=k_i,
                    opttype=opttype_i,
                    order=order,
                    meanp=meanp,
                    tot_varp=tot_varp,
                )
                for k_i, opttype_i in zip(k, opttype, strict=True)
            ]
        )
        return utils.black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

    def implied_vol_vix_expansion(self, k, T, order: int = 0):
        """
        Compute VIX implied volatility expansion.

        Parameters
        ----------
        k : float or array_like
            Log-moneyness (k = log(K/F)).
        T : float
            Time to maturity (T > 0).
        order : {0, 1, 2}, optional
            Expansion order (default 0).

        Returns
        -------
        float or ndarray
            Approximated implied volatility, same shape as `k`.

        Raises
        ------
        ValueError
            If `order` not in {0,1,2} or if `T <= 0`.
        """
        if order not in [0, 1, 2]:
            raise ValueError("order must be one of 0, 1, or 2.")
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        tot_var_proxy = self.var_proxy(T)
        vol_proxy = np.sqrt(tot_var_proxy / T)
        gamma_2 = self.gamma_2_proxy(T)
        gamma_3 = self.gamma_3_proxy(T)
        xp = 0.5 * self.mean_proxy(T) + tot_var_proxy / 8

        if order == 0:
            return 0.5 * vol_proxy + 0.0 * k
        else:
            return (
                0.5 * vol_proxy
                + gamma_2 / (2 * vol_proxy * T)
                + 3 * gamma_3 / (8 * vol_proxy * T)
                - gamma_3 * (xp - k) / (vol_proxy**3 * T**2)
            )

    ####################################################################################
    # Mixed case approximation methods for VIX pricing
    ####################################################################################

    def _validate_mixed_params(self, T, lbd, volvol_2, opt_payoff, order, n_quad):
        """Validate parameters for price_vix_approx_mixed."""
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if not 0.0 <= lbd <= 1.0:
            raise ValueError("lbd must be in the interval [0, 1].")
        if volvol_2 <= 0.0:
            raise ValueError("volvol_2 must be positive.")
        if opt_payoff not in ["fut", "call", "put"]:
            raise ValueError("opt_payoff must be one of 'fut', 'call', or 'put'.")
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")
        if n_quad is not None and n_quad <= 0:
            raise ValueError("n_quad must be a positive integer or None.")

    def _get_params_mixed(self, T, lbd, volvol_2, order):
        """Compute parameters for price_vix_approx_mixed."""
        # Create model with vol-of-vol=volvol_2
        if self.name == "rough_bergomi":
            model_2 = self._clone_with_params(eta=volvol_2)
        elif self.name == "one_factor_bergomi":
            model_2 = self._clone_with_params(w=volvol_2)
        else:
            raise ValueError(
                "model name not one of 'rough_bergomi' or 'one_factor_bergomi'."
            )

        # Parameters common to both regimes
        fvix2 = self.fut_vix2(T)
        log_fvix2 = np.log(fvix2)

        # Mean proxy and variance proxy for both regimes
        mean_1 = log_fvix2 + self.mean_proxy(T)
        mean_2 = log_fvix2 + model_2.mean_proxy(T)
        sig_1 = self.var_proxy(T) ** 0.5
        sig_2 = model_2.var_proxy(T) ** 0.5

        volvol_1 = (
            self.params["eta"] if self.name == "rough_bergomi" else self.params["w"]
        )

        # Initialize params dictionary
        params = {
            "fvix2": fvix2,
            "volvol_1": volvol_1,
            "volvol_2": volvol_2,
            "lbd": lbd,
            "meanp_1": mean_1,
            "meanp_2": mean_2,
            "sigp_1": sig_1,
            "sigp_2": sig_2,
            "gamma_1": (self.gamma_1_proxy(T=T), model_2.gamma_1_proxy(T=T)),
            "gamma_2": (self.gamma_2_proxy(T=T), model_2.gamma_2_proxy(T=T)),
            "gamma_3": (self.gamma_3_proxy(T=T), model_2.gamma_3_proxy(T=T)),
        }

        return params

    def price_vix_approx_mixed(
        self,
        T: float,
        lbd: float,
        volvol_2: float,
        opt_payoff: str,
        order: int,
        n_quad: int | None = 50,
        K: float = 0.0,
        n_trunc_herm: int = 0,
    ):
        """
        Price a VIX option in the mixed case using the weak approximation.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        lbd : float
            Mixing parameter between the two regimes.
        volvol_2 : float
            Volatility of volatility parameter for the second regime.
        opt_payoff : str
            Payoff function of the option, e.g., "call" for a call option. Use "put"
            for a put option, or "fut" for a future payoff.
        order : int
            Order of the approximation expansion.
        n_quad : int | None = 50
            Number of quadrature points for numerical integration. If n_quad is None,
            scipy integrate.quad is used.
        K : float, optional (default is 0.0)
            Strike of the VIX option.
        n_trunc_herm : int optional (default is 0)
            If greater than 0, use Hermite series expansion with truncation order
            n_trunc_herm for the order 0 approximation.

        Returns
        -------
        float
            Approximated price of the VIX option using the mixed method.
        """
        # Validation
        self._validate_mixed_params(T, lbd, volvol_2, opt_payoff, order, n_quad)

        # Use Hermite series expansion for order 0 if specified
        if n_trunc_herm > 0:
            return self.price_vix_approx_mixed_series(
                T=T,
                lbd=lbd,
                volvol_2=volvol_2,
                opt_payoff=opt_payoff,
                order=order,
                n_quad=n_quad,
                n_trunc_herm=n_trunc_herm,
                K=K,
            )

        # Get parameters once
        params = self._get_params_mixed(T, lbd, volvol_2, order)

        # Accumulate price contributions up to requested order
        prices = {}
        total_price = 0.0
        for current_order in range(order + 1):
            prices[current_order] = _compute_price_mixed(
                n_quad, K, opt_payoff, params, current_order
            )
            total_price += prices[current_order]

        return total_price

    def price_vix_approx_mixed_series(
        self, T, lbd, volvol_2, opt_payoff, order, n_quad, n_trunc_herm, K=0.0
    ):
        """
        Approximate the price a VIX option in the mixed case using Hermite
        series expansion.
        """
        params = self._get_params_mixed(T, lbd, volvol_2, order)
        meanp_1 = params["meanp_1"]
        meanp_2 = params["meanp_2"]
        sigp_1 = params["sigp_1"]
        sigp_2 = params["sigp_2"]

        if order not in [0, 3]:
            raise ValueError(
                "Hermite series expansion only implemented for order 0 and 3."
            )

        a = (1 - lbd) ** 0.5 * np.exp(meanp_2 / 2 + sigp_2**2 / 8)
        b = (lbd / (1 - lbd)) * np.exp(
            meanp_1 - meanp_2 + (sigp_1 - sigp_2) * sigp_2 / 2
        )
        # get Hermite polynomial weights
        weights_herm = _hermite_polynomial_weights(
            n_trunc_herm, b, sigp_1 - sigp_2, n_quad
        )
        c_vec = _compute_coeff_mixed_case(params)

        if opt_payoff == "fut":
            if order == 0:
                return a * weights_herm[0, 0]
            else:
                return np.dot(c_vec, weights_herm[:, 0])

        A = _inverse_mixture_lognormal(K**2, lbd, meanp_1, meanp_2, sigp_1, sigp_2)
        B = A - sigp_2 / 2

        I_N_vec = np.array(
            [
                weights_herm[_order, 0] * stats.norm.cdf(-B)
                + np.sum(
                    weights_herm[_order, 1:]
                    * eval_hermitenorm(np.arange(n_trunc_herm), B)
                    * stats.norm.pdf(B)
                )
                for _order in [0, 1, 2, 3]
            ]
        )

        if order == 0:
            return (
                a * I_N_vec[0] - K * stats.norm.cdf(-A)
                if opt_payoff == "call"
                else K * stats.norm.cdf(A) - a * (weights_herm[0, 0] - I_N_vec[0])
            )

        # order 3

        I_N_tot = np.dot(c_vec, I_N_vec)

        return (
            I_N_tot - K * stats.norm.cdf(-A)
            if opt_payoff == "call"
            else K * stats.norm.cdf(A) - (a * weights_herm[0, 0] - I_N_tot)
        )

    def implied_vol_vix_approx_mixed(
        self,
        T: float,
        k: float | np.ndarray,
        order: int,
        lbd: float,
        volvol_2: float,
        n_quad: int | None = 50,
        return_opt="impvol",
        n_trunc_herm: int = 0,
    ):
        """
        Compute the implied volatility of a VIX option using a mixed
        approximation method.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        k : float | np.ndarray
            Log-moneyness of the VIX option.
        order : int
            Order of the approximation expansion.
        lbd : float
            Mixing parameter between the two regimes.
        volvol_2 : float
            Volatility of volatility parameter for the second regime.
        n_quad : int | None
            Number of quadrature points for numerical integration.
        return_opt : str, optional
            If 'impvol', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        n_trunc_herm : int, optional (default is 0)
            If greater than 0, use Hermite series expansion with truncation order
            n_trunc_herm for the order 0 approximation.

        Returns
        -------
        float or tuple
            Approximated Black-Scholes implied volatility for the VIX option, or a tuple
            containing the futures price and the implied volatility if return_opt is
            'all'.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        if lbd < 0 or lbd > 1:
            raise ValueError("lbd must be in the interval [0, 1].")

        if volvol_2 < 0:
            raise ValueError("volvol_2 must be non-negative.")

        if n_quad is not None and n_quad <= 0:
            raise ValueError("n_quad must be a positive integer or None.")

        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")

        if return_opt not in ["impvol", "all"]:
            raise ValueError("return_opt must be either 'impvol' or 'all'.")

        F = self.price_vix_approx_mixed(
            T=T,
            lbd=lbd,
            volvol_2=volvol_2,
            opt_payoff="fut",
            order=order,
            n_quad=n_quad,
            n_trunc_herm=n_trunc_herm,
        )
        k = np.atleast_1d(np.asarray(k))
        K = F * np.exp(k)
        opttype = 2 * (K >= F) - 1
        otm_price = np.array(
            [
                self.price_vix_approx_mixed(
                    T=T,
                    K=K_i,
                    lbd=lbd,
                    volvol_2=volvol_2,
                    opt_payoff="call" if opttype_i == 1 else "put",
                    order=order,
                    n_quad=n_quad,
                    n_trunc_herm=n_trunc_herm,
                )
                for K_i, opttype_i in zip(K, opttype, strict=True)
            ]
        )
        impvol_approx = utils.black_impvol(
            K=K, T=T, F=F, value=otm_price, opttype=opttype
        )
        if return_opt == "all":
            return F, impvol_approx
        else:
            return impvol_approx

    def implied_vol_vix_lognorm_approx_mixed(
        self, T: float, k: float | np.ndarray, order: int, lbd: float, volvol_2: float
    ):
        """
        Compute the implied volatility of a VIX option approximating the sum of two
        lognormal distributions with a single lognormal distribution in the mixed case.
        """
        if order != 0:
            raise NotImplementedError(
                "Lognormal approximation is only implemented for order=0."
            )
        params = self._get_params_mixed(T, lbd, volvol_2, order)
        lbd = params["lbd"]
        meanp_1 = params["meanp_1"]
        meanp_2 = params["meanp_2"]
        sigp_1 = params["sigp_1"]
        sigp_2 = params["sigp_2"]
        params_lognorm = utils.sum_lognorm_single_lognorm_approx(
            lbd=lbd,
            mu_1=meanp_1,
            mu_2=meanp_2,
            sig_1=sigp_1,
            sig_2=sigp_2,
        )
        meanp = params_lognorm["mu_y"]
        tot_varp = params_lognorm["sig_y"] ** 2
        return self.implied_vol_vix_approx(
            T=T,
            k=k,
            order=order,
            meanp=meanp,
            tot_varp=tot_varp,
        )

    def implied_vol_vix_shifted_lognorm_approx_mixed(
        self, T: float, k: float | np.ndarray, order: int, lbd: float, volvol_2: float
    ):
        """
        Compute the implied volatility of a VIX option approximating the sum of two
        lognormal distributions with a single shifted lognormal distribution in the
        mixed case.
        """
        # TODO: most likely wrong as of now, need to check and clean up code
        raise NotImplementedError(
            "Shifted lognormal approximation in the mixed case is not fully implemented yet."
        )
        if order != 0:
            raise NotImplementedError(
                "Shifted lognormal approximation is only implemented for order=0."
            )
        params = self._get_params_mixed(T, lbd, volvol_2, order)
        F = self.price_vix_approx_mixed(
            T=T, lbd=lbd, volvol_2=volvol_2, opt_payoff="fut", order=order
        )
        K = F * np.exp(k)
        # shifted lognormal parameters
        params_sl = utils.sqrt_sum_lognorm_shifted_lognorm_approx(
            lbd=params["lbd"],
            mu_1=params["meanp_1"],
            mu_2=params["meanp_2"],
            sig_1=params["sigp_1"],
            sig_2=params["sigp_2"],
        )
        meanp = 2 * params_sl["mu_y"]
        tot_varp = (2 * params_sl["sig_y"]) ** 2
        F = self.price_vix_fut_approx(T=T, order=order, meanp=meanp, tot_varp=tot_varp)
        volp = np.sqrt(tot_varp / T)
        c_y = params_sl["c_y"]
        price_0 = utils.black_price(K=K - c_y, T=T, F=F, vol=0.5 * volp, opttype=1)
        return utils.black_impvol(K=K - c_y, T=T, F=F, value=price_0, opttype=1)

        # k = np.log((K - c_y) / F)

        # return self.implied_vol_vix_approx(
        #     T=T,
        #     k=k,
        #     order=order,
        #     meanp=meanp,
        #     tot_varp=tot_varp,
        # )

    def implied_vol_vix_expansion_mixed(
        self,
        k,
        T,
        lbd,
        volvol_2,
        opt=1,
        order=3,
        n_quad=30,
        n_trunc_herm=10,
    ):
        """
        Compute VIX implied volatility expansion in the mixed case.

        Parameters
        ----------
        k : float or array_like
            Log strike price.
        T : float
            Time to maturity (T > 0).
        lbd : float
            Mixing weight in [0, 1].
        volvol_2 : float
            Volatility of volatility parameter for the second regime.
        order : {0, 1, 2}
            Expansion order.
        opt : {1, 2, 3}
            Type of approximation method.

        Returns
        -------
        float or ndarray
            Approximated implied volatility, same shape as `k`.

        Raises
        ------
        ValueError
            If `order` not in {0,1,2} or if `T <= 0`.
        """
        # TODO: finish and check implementation and clean up code
        # This is probably wrong as of now
        if opt not in [0, 1, 2, 3]:
            raise ValueError("opt 0, 1, 2, or 3 must be specified.")

        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        params = self._get_params_mixed(T, lbd, volvol_2, order)
        meanp_1 = params["meanp_1"]
        meanp_2 = params["meanp_2"]
        sigp_1 = params["sigp_1"]
        sigp_2 = params["sigp_2"]

        a = (1 - lbd) ** 0.5 * np.exp(meanp_2 / 2 + sigp_2**2 / 8)
        b = (lbd / (1 - lbd)) * np.exp(
            meanp_1 - meanp_2 + (sigp_1 - sigp_2) * sigp_2 / 2
        )
        weights_herm = _hermite_polynomial_weights(
            n_trunc_herm, b, sigp_1 - sigp_2, n_quad
        )

        F = a * weights_herm[0, 0]
        K = F * np.exp(k)
        A = _inverse_mixture_lognormal(K**2, lbd, meanp_1, meanp_2, sigp_1, sigp_2)
        B = A - sigp_2 / 2
        sig0 = sigp_2 / 2

        if opt == 0:
            sig1 = (
                np.sum(
                    weights_herm[0, 1:] * eval_hermitenorm(np.arange(n_trunc_herm), B)
                )
                / weights_herm[0, 0]
            )
            # sig2 = -A * B * sig1**2 / sigp_2
            # sig3 = (
            #     (2.0 / 3.0)
            #     * (2.0 * (A * B) ** 2 + (A**2 + B**2 + A * B))
            #     * sig1**3
            #     / (sigp_2**2)
            # )
            return (sig0 + sig1) / np.sqrt(T)
            # return (sig0 + sig1 + sig2) / np.sqrt(T)
            # return (sig0 + sig1 + sig2 + sig3) / np.sqrt(T)

        # formulas from Ying
        x1 = np.exp(K) - A * sigp_2 / 2 + sigp_2**2 / 8
        x2 = np.log(float(F))
        x3 = 0.5 * (x1 + x2)
        sig_tilde = sigp_2 / np.sqrt(T)
        c_vec = _compute_coeff_mixed_case(params)

        if opt == 1:
            # new log spot / unchanged log-strike
            x_opt = x1
        elif opt == 2:
            # unchanged log spot / new log-strike
            x_opt = x2
        elif opt == 3:
            # new log spot / new log-strike
            x_opt = x3

        impvol = 0.5 * sig_tilde
        impvol += np.sum(
            c_vec[:, None]
            * weights_herm[:, 1:]
            * eval_hermitenorm(np.arange(n_trunc_herm), B)
        ) / (np.exp(x_opt) * np.sqrt(T))

        return impvol


def _compute_price_0_mixed(n_quad, K, opt_payoff, params):
    """Compute order 0 vix option price in the mixed case."""
    lbd = params["lbd"]
    meanp_1 = params["meanp_1"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    payoff = _vix_payoff(opt_payoff, K)

    if n_quad is None:
        price_0 = integrate.quad(
            lambda x: payoff(
                lbd * np.exp(meanp_1 + sigp_1 * stats.norm.ppf(x))
                + (1.0 - lbd) * np.exp(meanp_2 + sigp_2 * stats.norm.ppf(x))
            ),
            0,
            1,
        )[0]
    else:
        nodes, weights = _get_nodes_weights(n_quad, K, opt_payoff, params, order=0)
        # order 0
        price_0 = np.sum(
            weights
            * payoff(
                lbd * np.exp(meanp_1 + sigp_1 * nodes)
                + (1.0 - lbd) * np.exp(meanp_2 + sigp_2 * nodes)
            )
        )

    return price_0


def _compute_price_mixed(n_quad, K, opt_payoff, params, order):
    """Compute order 1, 2, or 3 price contribution for vix option in the mixed case."""
    if order not in [0, 1, 2, 3]:
        raise ValueError("order must be one of 0, 1, 2, or 3.")

    if order == 0:
        return _compute_price_0_mixed(n_quad, K, opt_payoff, params)

    # Unpack parameters
    lbd = params["lbd"]
    meanp_1 = params["meanp_1"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    volvol_1 = params["volvol_1"]
    volvol_2 = params["volvol_2"]
    fvix2 = params["fvix2"]
    # Gamma values for this order
    gammas = params[f"gamma_{order}"]
    # Derivative of payoff function
    dpayoff_mixed_dy = _deriv_vix_payoff_mixed(opt_payoff, K)

    def _func_psi(x, idx=1):
        """Compute psi function for given index (1 or 2)."""
        sig_idx = sigp_1 if idx == 1 else sigp_2
        mean_idx = meanp_1 - np.log(fvix2) if idx == 1 else meanp_2 - np.log(fvix2)
        return (
            dpayoff_mixed_dy(
                x=mean_idx + sig_idx * x,
                lbd=lbd if idx == 1 else 1 - lbd,
                volvol_1=volvol_1 if idx == 1 else volvol_2,
                volvol_2=volvol_2 if idx == 1 else volvol_1,
                fvix2=fvix2,
                mu_2=meanp_2 if idx == 1 else meanp_1,
            )
            * sig_idx
        )

    def _get_order_weight(x, order):
        """Get the order-dependent weight function."""
        if order == 1:
            return 1.0
        elif order == 2:
            return x
        else:  # order == 3
            return x**2 - 1

    psi_1 = lambda x: _func_psi(x, idx=1)
    psi_2 = lambda x: _func_psi(x, idx=2)
    order_weight = lambda x: _get_order_weight(x, order)

    if n_quad is None:
        # Use scipy.quad for continuous integration
        def integrand_1(u):
            x = stats.norm.ppf(u)
            return order_weight(x) * gammas[0] * psi_1(x)

        def integrand_2(u):
            x = stats.norm.ppf(u)
            return order_weight(x) * gammas[1] * psi_2(x)

        left_1, right_1 = _get_nodes_weights(
            n_quad, K, opt_payoff, params, order, idx=1
        )
        left_2, right_2 = _get_nodes_weights(
            n_quad, K, opt_payoff, params, order, idx=2
        )

        price_1 = integrate.quad(integrand_1, left_1, right_1)[0]
        price_2 = integrate.quad(integrand_2, left_2, right_2)[0]
    else:
        # Use Gauss quadrature for discrete approximation
        nodes_1, weights_1 = _get_nodes_weights(
            n_quad, K, opt_payoff, params, order, idx=1
        )
        nodes_2, weights_2 = _get_nodes_weights(
            n_quad, K, opt_payoff, params, order, idx=2
        )

        # Compute integrands with order-dependent weights
        weight_fn = order_weight(nodes_1)
        integrand_1 = weight_fn * psi_1(nodes_1)

        weight_fn = order_weight(nodes_2)
        integrand_2 = weight_fn * psi_2(nodes_2)

        price_1 = gammas[0] * np.sum(weights_1 * np.asarray(integrand_1))
        price_2 = gammas[1] * np.sum(weights_2 * np.asarray(integrand_2))

    return price_1 + price_2


def _get_nodes_weights(n_quad, K, opt_payoff, params, order, idx=1):
    """
    Get quadrature nodes and weights for mixed VIX pricing.

    Parameters
    ----------
    n_quad : int
        Quadrature points.
    K : float
        Strike price.
    opt_payoff : str
        Option payoff type.
    params : dict
        Model parameters.
    order : int
        Approximation order.
    idx : int, optional
        Index for psi function, by default 1

    Returns
    -------
    tuple
        Quadrature nodes, weights, left, right.
    """
    lbd = params["lbd"]
    meanp_1 = params["meanp_1"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    volvol_1 = params["volvol_1"]
    volvol_2 = params["volvol_2"]
    fvix2 = params["fvix2"]
    # log_fvix2 = np.log(fvix2)

    if opt_payoff == "fut" and n_quad is None:
        left = 0.0
        right = 1.0

    if opt_payoff in ["call", "put"]:
        # TODO: check why order >= 1 is not working

        # A = _inverse_mixture_lognormal(K**2, lbd, meanp_1, meanp_2, sigp_1, sigp_2)
        # endpoint = A - sigp_2 / 2
        # print("endpoint:", endpoint)

        # endpoint1 = _inverse_x_inner_mixed_func(
        #     z=K**2,
        #     mu_2=meanp_2 if idx == 1 else meanp_1,
        #     lbd=lbd if idx == 1 else 1 - lbd,
        #     volvol_1=volvol_1 if idx == 1 else volvol_2,
        #     volvol_2=volvol_2 if idx == 1 else volvol_1,
        #     fvix2=fvix2,
        # )
        # print("endpoint1:", endpoint1)

        if order == 0:
            endpoint = _inverse_mixture_lognormal(
                K**2, lbd, meanp_1, meanp_2, sigp_1, sigp_2
            )
            # endpoint = A - sigp_2 / 2
        else:
            endpoint = _inverse_x_inner_mixed_func(
                z=K**2,
                mu_2=meanp_2 if idx == 1 else meanp_1,
                lbd=lbd if idx == 1 else 1 - lbd,
                volvol_1=volvol_1 if idx == 1 else volvol_2,
                volvol_2=volvol_2 if idx == 1 else volvol_1,
                fvix2=fvix2,
            )

        left = stats.norm.cdf(endpoint) if opt_payoff == "call" else 0.0
        right = 1.0 if opt_payoff == "call" else stats.norm.cdf(endpoint)

    if n_quad is None:
        return left, right

    if opt_payoff == "fut":
        # Gauss-Hermite quadrature for future payoff
        nodes, weights = utils.gauss_hermite(n_quad)
    else:
        # Gauss-Legendre quadrature for call/put payoff
        nodes, weights = utils.gauss_legendre(float(left), float(right), n_quad)
        nodes = stats.norm.ppf(nodes)

    return nodes, weights
