# -*- coding: utf-8 -*-

"""Implementation of a Nelson-Siegel interest rate curve model.
See `NelsonSiegelCurve` class for details.
"""

from numbers import Real
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from numpy import exp

EPS = np.finfo(float).eps


@dataclass
class NelsonSiegelCurve:
    """Implementation of a Nelson-Siegel interest rate curve model.
    This curve can be interpreted as a factor model with three
    factors (including a constant).
    """

    beta1: float
    beta2: float
    beta3: float
    lambd: float
    
    @staticmethod
    def _as_tau(T: Union[float, np.ndarray], t: Union[float, np.ndarray]
        ) -> Union[float, np.ndarray]:
        """Compute τ = T - t with broadcasting; default t may be 0."""
        if isinstance(T, np.ndarray) or isinstance(t, np.ndarray):
            return np.asarray(T) - np.asarray(t)
        else:
            return float(T) - float(t)

    def factors(self, T: Union[float, np.ndarray],t: Union[float, np.ndarray] = 0.0
        ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """Factor loadings for time(s) (t, T), excluding constant."""
        lambd = self.lambd
        if not np.isfinite(lambd) or lambd <= 0.0:
            tau = self._as_tau(T, t)
            if isinstance(tau, Real):
                return 1.0, 0.0
            n = np.asarray(tau).size
            return np.ones(n, dtype=float), np.zeros(n, dtype=float)
        
        tau = self._as_tau(T, t)
        if isinstance(tau, Real) and tau <= 0:
            return 1.0, 0.0
        elif isinstance(tau, np.ndarray):
            zero_idx = tau <= 0
            tau[zero_idx] = EPS  # avoid warnings in calculations
        exp_tt0 = exp(-tau / lambd)
        factor1 = (1.0 - exp_tt0) / (tau / lambd)
        factor2 = factor1 - exp_tt0
        if isinstance(tau, np.ndarray):
            tau[zero_idx] = 0.0
            factor1[zero_idx] = 1.0
            factor2[zero_idx] = 0.0
        return factor1, factor2

    def factor_matrix(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0
        ) -> Union[float, np.ndarray]:
        """Factor loadings for time(s) (t, T) as matrix columns,
        including constant column (=1.0).
        """
        factor1, factor2 = self.factors(T, t)
        if isinstance(factor1, np.ndarray):
            constant: np.ndarray = np.ones(factor1.size)
            return np.stack([constant, factor1, factor2]).transpose()
        else:
            return np.array([[1.0, factor1, factor2]])
        
    def zero(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0
        ) -> Union[float, np.ndarray]:
        """Zero rate(s) this curve at time(s) (t, T)."""
        factor1, factor2 = self.factors(T, t)
        return self.beta1 + self.beta2 * factor1 + self.beta3 * factor2

    def __call__(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0,
        ) -> Union[float, np.ndarray]:
        """Alias of zero(T, t): zero rate(s) y(t,T)."""
        return self.zero(T, t)
    
    def forward(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0,
        ) -> Union[float, np.ndarray]:
        """Instantaneous forward rate f(t,T) for the Nelson–Siegel interest rate curve.
        """
        tau = self._as_tau(T, t)
        if isinstance(tau, Real):
            if tau < 0:
                tau = 0.0
            exp_tt0 = exp(-tau / self.lambd)
            return self.beta1 + self.beta2 * exp_tt0 + self.beta3 * exp_tt0 * (tau / self.lambd)
        else:
            tau = np.asarray(tau, dtype=float)
            tau_clip = np.maximum(tau, 0.0)
            exp_tt0 = exp(-tau_clip / self.lambd)
            return self.beta1 + self.beta2 * exp_tt0 + self.beta3 * exp_tt0 * (tau_clip / self.lambd)