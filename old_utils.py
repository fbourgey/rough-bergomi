import numpy as np
from scipy import special, stats, optimize, integrate


def error(x_ref, x):
    """
    To compute relative error in percentage
    """
    return 100 * (x_ref - x) / x_ref


def cov_fbm(u, v, H):
    """
    Covariance of fBm with Hurst parameter 0 < H < 1
    """
    return 0.5 * (
        np.absolute(u) ** (2 * H)
        + np.absolute(v) ** (2 * H)
        - np.absolute(u - v) ** (2 * H)
    )


def brownian(T, n):
    """
    Returns Brownian motion on [0,T] with n discretization points
    """
    delta = T / n
    bm = np.sqrt(delta) * np.random.randn(n)
    return np.cumsum(bm)


def brownian_bridge_chol(t, T, a, b, n):
    """
    Returns Brownian bridge on [t,T] with B_t = a and B_T = b and
    n discretization points.
    """
    tab_t = np.linspace(t, T, n + 1)
    mean = a + ((b - a) / (T - t)) * (tab_t[1:] - t)
    # covariance matrix
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                cov[i, j] = ((T - tab_t[i + 1]) * (tab_t[j + 1] - t)) / (T - t)
            cov[j, i] = cov[i, j]
    bridge = np.random.multivariate_normal(mean, cov)
    return np.insert(bridge, 0, a)


def fbrownian_chol(T, H, n):
    """
    Returns fractional brownian motion on [0,T] with n discretization points
    via Cholesky decomposition.
    """
    # timestep
    delta = T / n
    # covariance matrix
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                cov[i, j] = cov_fbm((i + 1) * delta, (j + 1) * delta, H)
            cov[j, i] = cov[i, j]
    fbm = np.random.multivariate_normal(np.zeros(n), cov)
    return np.insert(fbm, 0, 0)


def cov_rl(u, v, H):
    """
    Computes integral \int_0^{min(u,v)} ((u-x)(v-x))^{H-1/2} dx
    """
    return np.where(
        u == v,
        pow(u, 2 * H) / (2 * H),
        (1 / (H + 1 / 2))
        * np.where(
            u - v < 0,
            pow(u, H + 1 / 2)
            * pow(v, H - 1 / 2)
            * special.hyp2f1(1, 1 / 2 - H, 3 / 2 + H, u / v),
            pow(v, H + 1 / 2)
            * pow(u, H - 1 / 2)
            * special.hyp2f1(1, 1 / 2 - H, 3 / 2 + H, v / u),
        ),
    )


def f_cov(tab_t, H, option="rl"):
    """
    Returns covariance on tab_t with n (size of tab_t)
    """
    n = np.shape(tab_t)[0]
    cov = np.zeros((n, n))  # covariance matrix of size (n,n)
    for i, u in enumerate(tab_t):
        for j, v in enumerate(tab_t):
            if i <= j:
                if option == "rl":
                    cov[i, j] = cov_rl(u, v, H)
                if option == "bm":
                    cov[i, j] = u
                if option == "fbm":
                    cov[i, j] = cov_fbm(u, v, H)
            cov[j, i] = cov[i, j]
    return cov


def sabr_beta0(T, S0, xi0, rho, eta, n, M, optionAllTraj=False):
    """
    Returns simulation of SABR model with beta=0
    """
    deltaT = T / n
    tmp = np.sqrt(deltaT) * np.random.randn(n * M).reshape(M, n)
    W = np.insert(np.cumsum(tmp, axis=1), 0, 0, axis=1)
    tmp_ = np.sqrt(deltaT) * np.random.randn(n * M).reshape(M, n)
    Wortho = np.insert(np.cumsum(tmp_, axis=1), 0, 0, axis=1)
    V = np.zeros((M, n + 1))  # variance process
    S = np.zeros((M, n + 1))  # spot process
    V[:, 0] = xi0
    S[:, 0] = S0
    for i in range(n):
        V[:, i + 1] = V[:, i] * np.exp(
            eta * (W[:, i + 1] - W[:, i]) - 0.5 * pow(eta, 2) * deltaT
        )
        S[:, i + 1] = S[:, i] * np.exp(
            -0.5 * V[:, i] * deltaT
            + np.sqrt(V[:, i])
            * (
                rho * (W[:, i + 1] - W[:, i])
                + np.sqrt(1 - rho**2) * (Wortho[:, i + 1] - Wortho[:, i])
            )
        )
    if optionAllTraj:
        return S, V, W
    else:
        return S[:, -1], V[:, -1]


def call_bs(x, y, sigma):
    """
    Returns call price in the Black-Scholes model
    """
    # if((sigma==0) and (x>y)): return(x-y)
    # if((sigma==0) and (x<y)): return(0)
    tmp0 = x * stats.norm.cdf(np.log(x / y) / sigma + sigma / 2)
    tmp1 = y * stats.norm.cdf(np.log(x / y) / sigma - sigma / 2)
    return tmp0 - tmp1


def put_bs(x, y, sigma):
    """
    returns put price in the Black-Scholes model
    """
    return call_bs(x, y, sigma) - (x - y)


def vega_bs(x, y, sigma, T):
    d1 = np.log(x / y) / (sigma * np.sqrt(T)) + sigma * np.sqrt(T) / 2
    return x * stats.norm.pdf(d1) * np.sqrt(T)


def find_vol_brent(target, S, K, T, option="call"):
    """
    function to find implied volatility
    """
    foo = (
        lambda x: call_bs(S, K, x * np.sqrt(T)) - target
        if (option == "call")
        else put_bs(S, K, x * np.sqrt(T)) - target
    )
    return optimize.brentq(foo, 0.05, 1.0, maxiter=100)


def cross_cov(tab_t, H, rho):
    n = np.shape(tab_t)[0]
    cov = np.zeros((n, n))  # covariance matrix of size (n,n)
    for i, u in enumerate(tab_t):
        for j, v in enumerate(tab_t):
            cov[i, j] = (
                rho
                * (np.sqrt(2 * H) / (H + 1 / 2))
                * (pow(u, H + 1 / 2) - pow(u - np.minimum(v, u), H + 1 / 2))
            )
    return cov


def sigloc_phl(x, T, V, W, rho):
    """
    Returns local vol using ratio of expectations
    """
    n = np.shape(V)[1] - 1
    M = np.shape(V)[0]
    intV = 0.5 * (T / n) * np.sum(V[:, :-1] + V[:, 1:], axis=1)  # trapezoidal rule
    U = (
        x
        + 0.5 * intV
        - rho * np.sum(np.sqrt(V[:, :-1]) * (W[:, 1:] - W[:, :-1]), axis=1)
    )
    weight = np.exp(-pow(U, 2) / (2 * (1 - rho**2) * intV)) / np.sqrt(intV)
    ci1 = 1.96 * np.std(V[:, -1] * weight) / np.sqrt(M)
    ci2 = 1.96 * np.std(weight) / np.sqrt(M)
    estim1 = np.mean(V[:, -1] * weight)
    estim2 = np.mean(weight)
    return (
        np.sqrt(estim1 / estim2),
        np.sqrt((estim1 + ci1) / (estim2 - ci2)),
        np.sqrt((estim1 - ci1) / (estim2 + ci2)),
    )


def sigloc_skew_phl(x, T, V, W, rho):
    """
    Returns local vol skew using ratio of expectations
    """
    n = np.shape(V)[1] - 1
    M = np.shape(V)[0]
    # trapezoidal rule
    intV = 0.5 * (T / n) * np.sum(V[:, :-1] + V[:, 1:], axis=1)
    U = (
        x
        + 0.5 * intV
        - rho * np.sum(np.sqrt(V[:, :-1]) * (W[:, 1:] - W[:, :-1]), axis=1)
    )
    weight = np.exp(-pow(U, 2) / (2 * (1 - rho**2) * intV)) / np.sqrt(intV)
    exp_w = np.mean(weight)
    exp_Vw = np.mean(V[:, -1] * weight)
    exp1 = np.mean(U * weight / intV)
    exp2 = np.mean(U * weight * V[:, -1] / intV)

    return (exp_Vw * exp1 - exp_w * exp2) / (
        2 * (1 - rho**2) * pow(exp_Vw, 1 / 2) * pow(exp_w, 3 / 2)
    )


def fourier(n, s):
    tab = np.zeros(n)
    tab[0] = 1
    for i in range(1, n):
        if i % 2 == 0:
            tab[i] = np.sqrt(2) * np.cos(2 * np.pi * i * s)
        else:
            tab[i] = np.sqrt(2) * np.sin(2 * np.pi * i * s)
    return tab


def fourier_hat(N, t, H):
    tab = np.zeros(N)
    tab[0] = (np.sqrt(2 * H) / (H + 0.5)) * pow(t, H + 0.5)
    for i in range(1, N):
        if i % 2 == 0:
            # computes \int_0^t \sqrt(2H) * (t-s)^{H-1/2} * sqrt(2) * cos(i*pi*s) ds
            # tab[i] = integrate.quad(lambda s: kernel_rb(t, s, H) * np.sqrt(2) * np.cos(i * np.pi * s), 0.01, t)[0]
            tab[i] = mpm.hyp1f2(
                1, 0.75 + 0.5 * H, 1.25 + 0.5 * H, -pow(0.5 * i * np.pi * t, 2)
            )
            tab[i] *= (4 * np.sqrt(H) * pow(t, H + 0.5)) / (1 + 2 * H)
            # print("cos:", tab[i] - tmp)
        else:
            # computes \int_0^t \sqrt(2H) * (t-s)^{H-1/2} * sqrt(2) * sin(i*pi*s) ds
            # tab[i] = integrate.quad(lambda s: kernel_rb(t, s, H) * np.sqrt(2) * np.sin(i * np.pi * s), 0.01, t)[0]
            tab[i] = mpm.hyp1f2(
                1, 1.25 + 0.5 * H, 1.75 + 0.5 * H, -pow(0.5 * i * np.pi * t, 2)
            )
            tab[i] *= (16 * np.sqrt(H) * 0.5 * i * np.pi * pow(t, 1.5 + H)) / (
                3 + 8 * H + 4 * pow(H, 2)
            )
            # print("sin:", tab[i]-tmp)
    return tab


def f_prime(a, s):
    return np.dot(a, fourier(np.shape(a)[0], s))


def f_hat(a, t, H):
    """
    returns integrate.quad(lambda s: kernel_rb(t, s, H) * f_prime(a, s), 0, t)[0]
    """
    return np.dot(a, fourier_hat(np.shape(a)[0], t, H))


def fsigma(x, sigma_0, eta):
    return sigma_0 * np.exp(0.5 * eta * x)


def G(a, sigma_0, H):
    return integrate.quad(
        lambda t: fsigma(f_hat(a, t, H), sigma_0, H) * f_prime(a, t), 0, 1
    )[0]


def E(a):
    return integrate.quad(lambda t: f_prime(a, t) ** 2, 0, 1)[0]


def FF(a, sigma_0, eta, H):
    return integrate.quad(lambda t: fsigma(f_hat(a, t, H), sigma_0, eta) ** 2, 0, 1)[0]


def objective_function(a, x, sigma_0, eta, H, rho):
    res = pow(x - rho * G(a, sigma_0, H), 2) / (
        2 * (1 - rho**2) * FF(a, sigma_0, eta, H)
    ) + 0.5 * E(a)
    return res
