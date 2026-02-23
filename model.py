from collections.abc import Callable
from abc import ABC, abstractmethod

import numpy as np
from scipy import integrate


class ForwardVarianceModel(ABC):
    def __init__(
        self,
        xi0: Callable[[np.ndarray], np.ndarray],
        rho: float,
        s0: float = 1.0,
        delta_vix: float = 1.0 / 12.0,
    ) -> None:
        """Initialize ForwardVarianceModel."""
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
    def kernel(self, u, t):
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
