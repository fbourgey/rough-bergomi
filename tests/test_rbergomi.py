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
        for i in range(n_disc + 1):
            for j in range(n_disc + 1):
                integral = integrate.quad(
                    lambda s: 2.0 * H * ((tab_u[i] - s) * (tab_u[j] - s)) ** (H - 0.5),
                    0.0,
                    T,
                )[0]
                cov_quad[i, j] = integral

        assert np.allclose(
            cov_quad,
            rbergomi.covariance_levy_fbm_vix(T=T, u=u, v=v),
            rtol=1e-6,
            atol=1e-3,
        )
