import pytest
from rbergomi import RoughBergomi
import numpy as np
from scipy import integrate


@pytest.fixture(
    params=[
        {
            "s0": 1.0,
            "xi0": lambda t: np.ones_like(t) * 0.2**2,
            "H": 0.3,
            "eta": 1.4,
            "rho": -0.7,
        },
        {
            "s0": 2.0,
            "xi0": lambda t: np.ones_like(t) * 0.1**2,
            "H": 0.1,
            "eta": 0.8,
            "rho": 0.5,
        },
    ]
)
def rbergomi(request):
    return RoughBergomi(**request.param)


class TestRoughBergomi:
    def test_initialization(self, rbergomi):
        assert rbergomi.s0 > 0.0
        assert rbergomi.H > 0.0
        assert rbergomi.eta > 0.0
        assert callable(rbergomi.xi0)

    @pytest.mark.parametrize(
        "u, v",
        [
            (np.linspace(1e-2, 3.0, 200), np.linspace(1e-8, 2.0, 200)),
            (np.linspace(0.2, 1.0, 300), np.linspace(2.0, 10.0, 300)),
        ],
    )
    def test_cov_fbm_levy(self, u, v, rbergomi):
        min_uv = np.minimum(u, v)
        H = rbergomi.H

        def integrand(s, ui, vi):
            return ((ui - s) * (vi - s)) ** (H - 0.5)

        integral = np.array(
            [
                integrate.quad(integrand, 0.0, min_uv[i], args=(u[i], v[i]))[0]
                for i in range(u.shape[0])
            ]
        )

        assert np.allclose(integral, rbergomi.cov_levy_fbm(u, v), rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize(
        "tab_u",
        [
            np.linspace(0.5, 2.0, 201),
        ],
    )
    def test_cov_matrix_vix(self, tab_u, rbergomi):
        H = rbergomi.H
        T = tab_u[0]
        n_disc = tab_u.shape[0] - 1
        u = np.tile(tab_u, (n_disc + 1, 1)).T
        v = u.T

        cov_quad = np.zeros_like(u)

        def integrand(s, ui, vi):
            return 2.0 * H * ((ui - s) * (vi - s)) ** (H - 0.5)

        for i in range(n_disc + 1):
            for j in range(n_disc + 1):
                cov_quad[i, j], _ = integrate.quad(
                    integrand, 0.0, T, args=(tab_u[i], tab_u[j])
                )

        assert np.allclose(cov_quad, rbergomi.covariance_levy_fbm_vix(T=T, u=u, v=v))

    @pytest.mark.parametrize(
        "T",
        [1e-3, 0.3, 0.5, 1.0, 2.0, 5.0],
    )
    def test_params_proxy(self, T, rbergomi):
        """
        Check that the proxy parameters are consistent when xi0 is flat.
        """
        N_QUAD = 80
        A_TOL = 1e-4
        # mean_proxy
        assert np.allclose(
            rbergomi.mean_proxy(T, quad_scipy=True), rbergomi.mean_proxy_flat(T)
        )
        # var_proxy
        assert np.allclose(
            rbergomi.var_proxy(T, quad_scipy=True), rbergomi.var_proxy_flat(T)
        )
        # gamma_1
        gam1_proxy = rbergomi.gamma_1_proxy(T, n_quad=N_QUAD)
        gam1_proxy_flat = rbergomi.gamma_1_proxy_flat(T)
        assert np.allclose(gam1_proxy, gam1_proxy_flat, atol=A_TOL)
        # gamma_2
        gam2_proxy = rbergomi.gamma_2_proxy(T, n_quad=N_QUAD)
        gam2_proxy_flat = rbergomi.gamma_2_proxy_flat(
            T, n_quad=N_QUAD, quad_scipy=False
        )
        gam2_proxy_flat_sp = rbergomi.gamma_2_proxy_flat(T, quad_scipy=True)
        assert np.allclose(gam2_proxy, gam2_proxy_flat, atol=A_TOL)
        assert np.allclose(gam2_proxy, gam2_proxy_flat_sp, atol=A_TOL)
        assert np.allclose(gam2_proxy_flat, gam2_proxy_flat_sp, atol=A_TOL)
        # gamma_3
        gam3_proxy = rbergomi.gamma_3_proxy(T, n_quad=N_QUAD)
        gam3_proxy_flat = rbergomi.gamma_3_proxy_flat(
            T, n_quad=N_QUAD, quad_scipy=False
        )
        gam3_proxy_flat_sp = rbergomi.gamma_3_proxy_flat(T, quad_scipy=True)
        assert np.allclose(gam3_proxy, gam3_proxy_flat, atol=A_TOL)
        assert np.allclose(gam3_proxy, gam3_proxy_flat_sp, atol=A_TOL)
        assert np.allclose(gam3_proxy_flat, gam3_proxy_flat_sp, atol=A_TOL)
