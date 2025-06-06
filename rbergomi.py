import numpy as np
from scipy import optimize, special, stats
import utils

# add control variate option


class RoughBergomi:
    """
    Rough Bergomi model.
    """

    def __init__(self, s0, xi0, H, eta, rho):
        self.s0 = s0
        self.xi0 = xi0
        self.H = H
        self.eta = eta
        self.rho = rho

    def cov_levy_fbm(self, u, v):
        r"""
        Covariance matrix of Levy's fractional Brownian motion
        `W_u^H = \int_0^u (u-s)^{H-1/2} dW_s`.
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
        Covariance matrix of the Gaussian vector `(Y_ti, W_ti)_{1<=i<=n}` with `ti` the
        timesteps of `tab_t`, `W` a standard Brownion motion and
        `Y_t = \sqrt{2H} \int_0^t (t-s)^{H-1/2} dW_s`.

        NB: for now, we do not use any control variate.
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
        Simulate the Rough Bergomi model.

        If conditioning is True, the simulation is conditioned on the
        Brownian motion path independent of that of the spot variance,
        otherwise it is not. See Bergomi's book, Chapter 8, Appendix A.
        """
        if seed is not None:
            np.random.seed(seed)

        n_mc_loop, remainder = divmod(n_mc, n_loop)
        if remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop")

        n_disc = tab_t.shape[0] - 1
        dt = tab_t[1] - tab_t[0]

        int_v_dt = np.zeros(n_mc)
        int_sqrt_v_dw = np.zeros(n_mc)

        for i in range(n_loop):
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
            v = self.xi0(tab_t[:, None]) * np.exp(
                self.eta * y - 0.5 * self.eta**2 * tab_t[:, None] ** (2.0 * self.H)
            )

            int_sqrt_v_dw[i * n_mc_loop : (i + 1) * n_mc_loop] = np.sum(
                np.sqrt(v[:-1, :]) * (w[1:, :] - w[:-1, :]), axis=0
            )
            # trapezoidal rule for the integral
            int_v_dt[i * n_mc_loop : (i + 1) * n_mc_loop] = (
                0.5 * dt * np.sum(v[:-1, :] + v[1:, :], axis=0)
            )

        return {
            "v": v,
            "y": y,
            "w": w,
            "int_v_dt": int_v_dt,
            "int_sqrt_v_dw": int_sqrt_v_dw,
        }

    def atm_local_vol_skew(self, v_t, int_v_dt, int_sqrt_v_dw):
        r"""
        Estimate the at-the-money (ATM) local volatility skew using Monte Carlo
        simulation.

        This method computes the slope of the local volatility surface at the money,
        based on a ratio of expectations.

        Parameters
        ----------
        x : float or np.ndarray
            The log-moneyness (typically 0 for ATM).
        v_t : np.ndarray
            Array of instantaneous variances at time `t` (shape: n_samples,).
        int_v_dt : np.ndarray
            Array of time-integrated variances (shape: n_samples,).
        int_sqrt_v_dw : np.ndarray
            Array of stochastic integrals \int sqrt(v) dW (shape: n_samples,).

        Returns
        -------
        float
            Estimated ATM local volatility skew.

        References
        ----------
        Bourgey, F., et al. "Local volatility under rough volatility", Equation (26).
        """
        u = 0.5 * int_v_dt - self.rho * int_sqrt_v_dw
        weight = np.exp(-(u**2) / (2.0 * (1.0 - self.rho**2) * int_v_dt))
        weight /= int_v_dt**0.5
        exp_w = np.mean(weight)
        exp_vw = np.mean(v_t * weight)
        lv_skew = exp_vw * np.mean(u * weight / int_v_dt)
        lv_skew -= exp_w * np.mean(u * weight * v_t / int_v_dt)
        lv_skew /= 2.0 * (1.0 - self.rho**2) * exp_vw**0.5 * exp_w**1.5

        return lv_skew

    def implied_vol(
        self,
        T,
        int_v_dt,
        int_sqrt_v_dw,
        k=0.0,
        conditioning: bool = False,
        return_skew: bool = False,
    ):
        """
        Estimate the implied volatility using Monte Carlo simulation.
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

    # def atm_digit_call(self, T, K, int_v_dt, int_sqrt_v_dw, conditioning: bool = False):
    #     """
    #     Estimate P(S > K).
    #     """
    #     if conditioning:
    #         s0_cond = self.s0 * np.exp(
    #             -0.5 * self.rho**2 * int_v_dt + self.rho * int_sqrt_v_dw
    #         )
    #         vol_cond = np.sqrt((1.0 - self.rho**2) * int_v_dt / T)
    #         d2_cond = np.log(s0_cond / K) - 0.5 * vol_cond**2 * T
    #         digit = stats.norm.cdf(d2_cond).mean()
    #     else:
    #         S = self.s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
    #         digit = np.mean((S >= K) * 1.0)

    #     return digit

    # def atm_implied_vol_skew(self, T, int_v_dt, int_sqrt_v_dw):
    #     r"""
    #     Calculate the at-the-money (ATM) implied volatility skew.

    #     This method estimates the slope of the implied volatility smile ATM
    #     for a given forward price, maturity, and ATM implied volatility.

    #     Parameters
    #     ----------
    #     T : float
    #         Time to maturity (in years).
    #     int_v_dt : np.ndarray
    #         Array of time-integrated variances (shape: n_samples,).
    #     int_sqrt_v_dw : np.ndarray
    #         Array of stochastic integrals \int sqrt(v) dW (shape: n_samples,).

    #     Returns
    #     -------
    #     float
    #         The estimated ATM implied volatility skew.

    #     References
    #     ----------
    #     Bourgey, F., et al. "Local volatility under rough volatility." Section 4.1.
    #     """
    #     int_v_dt = int_v_dt.flatten()
    #     int_sqrt_v_dw = int_sqrt_v_dw.flatten()
    #     S = self.s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
    #     F = S.mean()
    #     atm_impvol = self.implied_vol(
    #         T=T,
    #         K=F,
    #         int_v_dt=int_v_dt,
    #         int_sqrt_v_dw=int_sqrt_v_dw,
    #         conditioning=True,
    #     )
    #     atm_impvol = np.asarray(atm_impvol)[0]
    #     skew = stats.norm.cdf(-0.5 * atm_impvol * T**0.5) - np.mean((S >= F) * 1.0)
    #     skew /= utils.black_vega(K=F, T=T, F=F, vol=atm_impvol)

    #     return F * skew

    def rate_function(self, tab_x, N):
        tab_rate = np.zeros_like(tab_x)  # minimum values for rate function
        tab_a = np.zeros(
            (np.shape(tab_x)[0], N)
        )  # minimum values for Fourier coefficients
        sigma_0 = np.sqrt(self.xi0)
        for i, x in enumerate(tab_x):
            a_guess = np.zeros(N)
            a_guess[0] = x * self.rho / sigma_0 + pow(x, 2) / 2 * self.eta / pow(
                sigma_0, 2
            ) * np.sqrt(2 * self.H) / (self.H + 0.5) * (
                (self.rho**2 + 1) * 0.65 - 3 * pow(self.rho, 2) / (self.H + 1.5)
            )
            optim = optimize.minimize(
                objective_function,
                x0=a_guess,
                args=(x, sigma_0, self.eta, self.H, self.rho),
            )
            tab_a[i, :] = optim.x
            tab_rate[i] = optim.fun
        return tab_a, tab_rate
