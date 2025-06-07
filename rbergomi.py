import numpy as np
from scipy import optimize, special, stats
import utils

# add vix (rect/trap scheme) and weak approx
# add tests


class RoughBergomi:
    """
    Implementation of the rough Bergomi model.

    Parameters
    ----------
    s0 : float
        Initial spot price (must be positive).
    xi0 : callable
        Forward variance curve function xi0(t), must return positive values for t >= 0.
    H : float
        Hurst parameter of the fractional Brownian motion (must be positive).
    eta : float
        Volatility of volatility parameter (must be positive).
    rho : float
        Correlation between the spot and the volatility processes (must be in [-1, 1]).

    Notes
    -----
    The model is described in:
    Bayer, C., Friz, P., & Gatheral, J. (2016). "Pricing under rough volatility".
    Quantitative Finance, 16(6), 887-904.
    """

    def __init__(self, s0, xi0, H, eta, rho):
        """
        Initialize the rough Bergomi model.
        See class docstring for parameter definitions.
        """
        if self.H <= 0.0:
            raise ValueError("Hurst parameter H must be positive.")
        if self.s0 <= 0.0:
            raise ValueError("Initial spot price s0 must be positive.")
        if self.eta <= 0.0:
            raise ValueError("Volatility of volatility eta must be positive.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("Correlation rho must be in [-1, 1].")
        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")
        if not np.all(xi0(np.linspace(1e-10, 10, 1000)) > 0):
            raise ValueError("xi0 must be positive for all t >= 0.")

        self.s0 = s0
        self.xi0 = xi0
        self.xi0_0 = self.xi0(0.0)
        self.H = H
        self.eta = eta
        self.rho = rho

    def cov_levy_fbm(self, u, v):
        r"""
        Compute the covariance matrix of Levy's fractional Brownian motion.

        The process is defined as:
            W_u^H = \int_0^u (u-s)^{H-1/2} dW_s

        Parameters
        ----------
        u : np.ndarray or float
            First set of time points.
        v : np.ndarray or float
            Second set of time points.

        Returns
        -------
        np.ndarray
            Covariance matrix evaluated at (u, v).
        """
        u_max_v = np.maximum(u, v)
        u_min_v = np.minimum(u, v)
        cov = (
            u_min_v ** (self.H + 0.5)
            * u_max_v ** (self.H - 0.5)
            * special.hyp2f1(1.0, 0.5 - self.H, 1.5 + self.H, u_min_v / u_max_v)
        )
        return cov / (self.H + 0.5)

    def cholesky_cov_matrix(
        self, tab_t, return_cov: bool = False, conditioning: bool = False
    ):
        r"""
        Compute the lower-triangular Cholesky factor
        of the covariance matrix of the Gaussian vector (Y_{t_i}, W_{t_i})
        for 1 <= i <= n, where t_i are the timesteps in tab_t.

        Here W a standard Brownion motion and
        Y_t = \sqrt{2H} \int_0^t (t-s)^{H-1/2} dW_s.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        return_cov : bool, optional
            If True, return the full covariance matrix instead of its Cholesky factor.
            Default is False.
        conditioning : bool, optional
            If True, compute the conditional covariance (see Bergomi's book, Chapter
            8, Appendix A).
            Default is False.

        Returns
        -------
        np.ndarray
            Lower-triangular Cholesky factor of the covariance matrix, or the covariance
            matrix itself if `return_cov` is True.
        """
        n_disc = tab_t.shape[0] - 1
        # repeat tab_t[1:] n_disc times as columns (shape: n_disc x n_disc)
        u = np.tile(tab_t[1:], (n_disc, 1)).T
        cov_y = 2.0 * self.H * self.cov_levy_fbm(u, u.T)
        cov_w = np.minimum(u, u.T)
        cov_yw = u ** (self.H + 0.5) - (u - cov_w) ** (self.H + 0.5)
        cov_yw *= np.sqrt(2.0 * self.H) / (self.H + 0.5)
        if not conditioning:
            cov_yw *= self.rho
        cov = np.block(
            [
                [cov_y, cov_yw],
                [cov_yw.T, cov_w],
            ]
        )
        if return_cov:
            return cov
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = utils.cholesky_from_svd(cov)
        except Exception as e:
            print(f"Error in Cholesky decomposition: {e}")
            raise

        return chol

    def simulate(
        self, tab_t, n_mc: int, n_loop: int = 1, seed=None, conditioning: bool = False
    ):
        """
        Simulate sample paths of the Rough Bergomi model.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        n_mc : int
            Total number of Monte Carlo paths to simulate.
        n_loop : int, optional
            Number of loops to split the simulation into (for memory efficiency).
            Must divide n_mc exactly. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        conditioning : bool, optional
            If True, simulate under the conditional law given the Brownian motion
            driving the spot. See Bergomi's book, Chapter 8, Appendix A.

        Returns
        -------
        dict
            Dictionary with the following keys:
                - 'v': np.ndarray, simulated variance paths
                    (shape: n_steps + 1, n_mc_loop)
                - 'y': np.ndarray, simulated log-volatility factor
                    (shape: n_steps + 1, n_mc_loop)
                - 'w': np.ndarray, simulated Brownian motion paths
                    (shape: n_steps + 1, n_mc_loop)
                - 'int_v_dt': np.ndarray, time-integrated variance for each path
                    (shape: n_mc,)
                - 'int_sqrt_v_dw': np.ndarray, stochastic integral for each path
                    (shape: n_mc,)

        Notes
        -----
        For H=0.5, the model reduces to SABR. For H<0.5, the full rough Bergomi model is
        simulated.
        """
        if seed is not None:
            np.random.seed(seed)

        n_mc_loop, remainder = divmod(n_mc, n_loop)
        if remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop")

        n_disc = tab_t.shape[0] - 1
        dt = tab_t[1] - tab_t[0]

        v = np.zeros((n_disc + 1, n_mc))
        int_v_dt = np.zeros(n_mc)
        int_sqrt_v_dw = np.zeros(n_mc)

        for i in range(n_loop):
            # print("iteration", i + 1, "of", n_loop)
            normal = np.random.randn(2 * n_disc, n_mc_loop)

            if self.H == 0.5:
                # The rough Bergomi model reduces to a SABR model.
                # The spot variance simply writes as
                # v_t = xi_0^t * exp(eta * W_t - 0.5 * eta^2 * t)
                y = dt**0.5 * np.cumsum(normal[:n_disc, :], axis=0)
                y = np.insert(y, 0, 0.0, axis=0)
                if conditioning:
                    w = y
                else:
                    w_perp = dt**0.5 * np.cumsum(normal[n_disc:, :], axis=0)
                    w_perp = np.insert(w_perp, 0, 0.0, axis=0)
                    w = self.rho * y + np.sqrt(1.0 - self.rho**2) * w_perp
                    w_perp = None
            else:
                normal = np.matmul(
                    self.cholesky_cov_matrix(tab_t, conditioning=conditioning), normal
                )
                y = normal[:n_disc, :]
                y = np.insert(y, 0, 0.0, axis=0)
                w = normal[n_disc:, :]
                w = np.insert(w, 0, 0.0, axis=0)

            normal = None  # free memory

            # spot variance process
            v_loop = self.xi0(tab_t[:, None]) * np.exp(
                self.eta * y - 0.5 * self.eta**2 * tab_t[:, None] ** (2.0 * self.H)
            )
            int_sqrt_v_dw[i * n_mc_loop : (i + 1) * n_mc_loop] = np.sum(
                np.sqrt(v_loop[:-1, :]) * (w[1:, :] - w[:-1, :]), axis=0
            )
            # trapezoidal rule for the integral
            int_v_dt[i * n_mc_loop : (i + 1) * n_mc_loop] = (
                0.5 * dt * np.sum(v_loop[:-1, :] + v_loop[1:, :], axis=0)
            )
            v[:, i * n_mc_loop : (i + 1) * n_mc_loop] = v_loop

        return {
            "v": v,
            "y": y,
            "w": w,
            "int_v_dt": int_v_dt,
            "int_sqrt_v_dw": int_sqrt_v_dw,
        }

    def local_vol_from_paths(
        self, k, v_t, int_v_dt, int_sqrt_v_dw, return_skew: bool = False
    ):
        r"""
        Estimate the local volatility and (optionally) its skew using Monte Carlo
        simulation.

        This method computes the local volatility at a given log-moneyness and, if
        requested, the slope (skew) of the local volatility surface, based on a ratio of
        expectations as described in Bourgey et al., "Local volatility under rough
        volatility".

        Parameters
        ----------
        k : float or np.ndarray
            The log-moneyness (typically 0 for ATM).
        v_t : np.ndarray
            Array of instantaneous variances at time `t` (shape: n_samples,).
        int_v_dt : np.ndarray
            Array of time-integrated variances (shape: n_samples,).
        int_sqrt_v_dw : np.ndarray
            Array of stochastic integrals ∫ sqrt(v) dW (shape: n_samples,).
        return_skew : bool, optional
            If True, also return the estimated local volatility skew. Default is False.

        Returns
        -------
        float or tuple
            Estimated local volatility. If `return_skew` is True, returns a tuple
            (local_vol, local_vol_skew).

        References
        ----------
        Bourgey, F., et al. "Local volatility under rough volatility", Equations (25)
        and (26).
        """
        k = np.atleast_1d(k)
        k = k.reshape(-1, 1)
        int_v_dt = int_v_dt.reshape(1, -1)
        int_sqrt_v_dw = int_sqrt_v_dw.reshape(1, -1)
        v_t = v_t.reshape(1, -1)

        u = k + 0.5 * int_v_dt - self.rho * int_sqrt_v_dw
        weight = np.exp(-(u**2) / (2.0 * (1.0 - self.rho**2) * int_v_dt))
        weight /= int_v_dt**0.5
        exp_w = np.mean(weight, axis=1)
        exp_vw = np.mean(v_t * weight, axis=1)
        lv = np.sqrt(exp_vw / exp_w)

        if return_skew:
            lv_skew = exp_vw * np.mean(u * weight / int_v_dt, axis=1)
            lv_skew -= exp_w * np.mean(u * weight * v_t / int_v_dt, axis=1)
            lv_skew /= 2.0 * (1.0 - self.rho**2) * exp_vw**0.5 * exp_w**1.5

            return lv, lv_skew

        return lv

    def local_vol(
        self,
        k,
        T,
        n_mc: int,
        n_disc: int,
        n_loop: int = 1,
        seed=None,
        return_skew: bool = False,
    ):
        """
        Estimate the local volatility (and optionally its skew) at a given log-moneyness
        and maturity using Monte Carlo simulation.

        Parameters
        ----------
        k : float or np.ndarray
            Log-moneyness (typically 0 for ATM).
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        return_skew : bool, optional
            If True, also return the estimated local volatility skew. Default is False.

        Returns
        -------
        float or tuple
            Estimated local volatility. If `return_skew` is True, returns a tuple
            (local_vol, local_vol_skew).
        """
        tab_t = np.linspace(0.0, T, n_disc + 1)
        paths = self.simulate(
            tab_t=tab_t, n_mc=n_mc, n_loop=n_loop, seed=seed, conditioning=True
        )
        return self.local_vol_from_paths(
            k=k,
            v_t=paths["v"][-1, :],
            int_v_dt=paths["int_v_dt"],
            int_sqrt_v_dw=paths["int_sqrt_v_dw"],
            return_skew=return_skew,
        )

    def implied_vol(
        self,
        k,
        T,
        n_mc: int,
        n_disc: int,
        n_loop: int = 1,
        seed=None,
        conditioning: bool = False,
        return_skew: bool = False,
    ):
        """
        Estimate the implied volatility (and optionally its skew) at a given
        log-moneyness and maturity using Monte Carlo simulation.

        Parameters
        ----------
        k : float or np.ndarray
            Log-moneyness (typically 0 for ATM).
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        conditioning : bool, optional
            If True, simulate under the conditional law given the Brownian motion
            driving the spot. Default is False.
        return_skew : bool, optional
            If True, also return the estimated implied volatility skew. Default is
            False.

        Returns
        -------
        float or tuple
            Estimated implied volatility. If `return_skew` is True, returns a tuple
            (implied_vol, implied_vol_skew).
        """
        tab_t = np.linspace(0.0, T, n_disc + 1)
        paths = self.simulate(
            tab_t=tab_t, n_mc=n_mc, n_loop=n_loop, seed=seed, conditioning=conditioning
        )
        return self.implied_vol_from_paths(
            k=k,
            T=T,
            int_v_dt=paths["int_v_dt"],
            int_sqrt_v_dw=paths["int_sqrt_v_dw"],
            conditioning=conditioning,
            return_skew=return_skew,
        )

    def implied_vol_from_paths(
        self,
        T,
        int_v_dt,
        int_sqrt_v_dw,
        k=0.0,
        conditioning: bool = False,
        return_skew: bool = False,
    ):
        r"""
        Estimate the implied volatility (and optionally its skew) from simulated paths.

        Parameters
        ----------
        T : float
            Maturity.
        int_v_dt : np.ndarray
            Array of time-integrated variances (shape: n_samples,).
        int_sqrt_v_dw : np.ndarray
            Array of stochastic integrals \int sqrt(v) dW (shape: n_samples,).
        k : float or np.ndarray, optional
            Log-moneyness (default is 0.0 for ATM).
        conditioning : bool, optional
            If True, use the conditional law given the Brownian motion driving the
            spot. Default is False.
        return_skew : bool, optional
            If True, also return the estimated implied volatility skew.
            Default is False.

        Returns
        -------
        float or tuple
            Estimated implied volatility. If `return_skew` is True, returns a
            tuple (implied_vol, implied_vol_skew).
        """
        int_v_dt = int_v_dt.flatten()
        int_sqrt_v_dw = int_sqrt_v_dw.flatten()
        k = np.atleast_1d(k)
        S = self.s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
        F = S.mean()
        K = F * np.exp(k)

        if conditioning:
            s0_cond = self.s0 * np.exp(
                -0.5 * self.rho**2 * int_v_dt + self.rho * int_sqrt_v_dw
            )
            vol_cond = np.sqrt((1.0 - self.rho**2) * int_v_dt / T)
            F_cond = np.mean(s0_cond)
            opttype = 2 * (self.s0 <= K) - 1
            impvol = np.zeros_like(K)
            if return_skew:
                digit = np.zeros_like(K)
            for i, K_i in enumerate(K):
                price_cond_i = utils.black_price(
                    K=K_i, T=T, F=s0_cond, vol=vol_cond, opttype=opttype[i]
                ).mean()
                impvol[i] = utils.black_impvol(
                    K=K_i, T=T, F=F_cond, value=price_cond_i, opttype=opttype[i]
                )
                if return_skew:
                    w_cond = vol_cond * T**0.5
                    d2_cond_i = np.log(s0_cond / K_i) / w_cond - 0.5 * w_cond
                    digit[i] = stats.norm.cdf(d2_cond_i).mean()
        else:
            impvol = utils.black_otm_impvol_mc(S=S, k=k, T=T)
            if return_skew:
                digit = np.mean((S >= K) * 1.0)

        if return_skew:
            w = impvol * T**0.5
            d2 = -k / w - 0.5 * w
            skew = stats.norm.cdf(d2) - digit
            skew /= stats.norm.pdf(d2) * T**0.5
            return impvol, skew

        return impvol

    def atm_implied_vol_local_vol_skew(
        self, tab_t, n_mc, n_loop=1, seed=None, n_disc_split=50
    ):
        """
        Compute the ATM implied volatility skew and local volatility skew
        as a function of maturity.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        n_mc : int
            Number of Monte Carlo paths.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        n_disc_split : int, optional
            Number of time discretization in each split. Must divide the total number of
            time steps (n_steps) exactly. Default is 50.

        Returns
        -------
        tuple
            (tab_t_split, (atm_impvol, atm_impvol_skew), (atm_lv, atm_lv_skew)), where
            each is a np.ndarray of length n_split.
        """
        paths = self.simulate(
            tab_t=tab_t, n_mc=n_mc, n_loop=n_loop, seed=seed, conditioning=True
        )
        v = paths["v"]
        w = paths["w"]
        paths = None
        dt = tab_t[1] - tab_t[0]
        n_disc = tab_t.shape[0] - 1

        int_sqrt_v_dw_cumsum = np.cumsum(
            np.sqrt(v[:-1, :]) * np.diff(w, axis=0), axis=0
        )
        int_v_dt_cumsum = np.cumsum(v[:-1, :] * dt, axis=0)
        int_sqrt_v_dw_cumsum = np.insert(int_sqrt_v_dw_cumsum, 0, 0.0, axis=0)
        int_v_dt_cumsum = np.insert(int_v_dt_cumsum, 0, 0.0, axis=0)

        n_split, remainder = divmod(n_disc, n_disc_split)
        if remainder != 0:
            raise ValueError("n_disc must be divisible by n_disc_split for splitting.")

        int_sqrt_v_dw_split = np.array(
            [t[-1] for t in np.split(int_sqrt_v_dw_cumsum[1:], n_split, axis=0)]
        )
        int_v_dt_split = np.array(
            [t[-1] for t in np.split(int_v_dt_cumsum[1:], n_split, axis=0)]
        )
        v_split = np.array([t[-1] for t in np.split(v[1:, :], n_split, axis=0)])
        tab_t_split = np.array([t[-1] for t in np.split(tab_t[1:], n_split)])

        atm_impvol = np.zeros(n_split)
        atm_impvol_skew = np.zeros(n_split)
        atm_lv = np.zeros(n_split)
        atm_lv_skew = np.zeros(n_split)

        for i in range(n_split):
            atm_impvol[i], atm_impvol_skew[i] = self.implied_vol_from_paths(
                T=tab_t_split[i],
                int_v_dt=int_v_dt_split[i, :],
                int_sqrt_v_dw=int_sqrt_v_dw_split[i, :],
                conditioning=True,
                return_skew=True,
            )
            atm_lv[i], atm_lv_skew[i] = self.local_vol_from_paths(
                k=0.0,
                v_t=v_split[i, :],
                int_v_dt=int_v_dt_split[i, :],
                int_sqrt_v_dw=int_sqrt_v_dw_split[i, :],
                return_skew=True,
            )
        return tab_t_split, (atm_impvol, atm_impvol_skew), (atm_lv, atm_lv_skew)

    def rate_function(self, tab_y, N):
        """
        Compute the rate function minimizing path coefficients for the
        large deviations principle.

        Parameters
        ----------
        tab_y : np.ndarray
            Array of y values (log-moneyness or displacement).
        N : int
            Number of Fourier basis functions for the Ritz projection.

        Returns
        -------
        tuple
            tab_a : np.ndarray
                Array of optimal coefficients (shape: len(tab_y), N).
            tab_rate : np.ndarray
                Array of rate function values (shape: len(tab_y),).
        """

        tab_rate = np.zeros_like(tab_y)
        tab_a = np.zeros((np.shape(tab_y)[0], N))
        sigma_0 = self.xi0_0**0.5
        for i, y in enumerate(tab_y):
            a_guess = np.zeros(N)
            a_guess[0] = (
                y * self.rho / sigma_0
                + y** 2
                / 2
                * self.eta
                / sigma_0** 2
                * np.sqrt(2 * self.H)
                / (self.H + 0.5)
                * ((self.rho**2 + 1) * 0.65 - 3.0 * self.rho**2 / (self.H + 1.5))
            )
            optim = optimize.minimize(
                lambda a: utils.objective_function(
                    a=a, y=y, sigma_0=sigma_0, eta=self.eta, H=self.H, rho=self.rho
                ),
                x0=a_guess,
            )
            tab_a[i, :] = optim.x
            tab_rate[i] = optim.fun

        return tab_a, tab_rate
