from collections.abc import Callable
from abc import ABC, abstractmethod

import numpy as np
from scipy import integrate

import utils


class ForwardVarianceModel(ABC):
    def __init__(
        self,
        xi0: Callable[[np.ndarray], np.ndarray],
        rho: float,
        s0: float = 1.0,
        delta_vix: float = 1.0 / 12.0,
    ) -> None:
        """
        Initialize ForwardVarianceModel.

        Parameters
        ----------
        s0 : float
            Initial spot price (must be positive).
        xi0 : callable
            Forward variance curve function xi0(t), must return positive values for
            t >= 0.
        rho : float
            Correlation between the spot and the volatility processes
            (must be in [-1, 1]).
        """
        if s0 <= 0.0:
            raise ValueError("Initial spot price s0 must be positive.")
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
        self.rho = rho
        self.delta_vix = delta_vix

    @abstractmethod
    def kernel(self, u, t) -> float | np.ndarray:
        """Compute the model-specific kernel function."""
        pass

    def _is_xi0_flat(self) -> bool:
        """Check if the forward variance curve xi0 is flat."""
        t_test = np.linspace(1e-10, 10, 1000)
        return np.allclose(self.xi0(t_test), self.xi0_0)

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

    ####################################################################################
    # VIX implied volatility expansions
    ####################################################################################

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
