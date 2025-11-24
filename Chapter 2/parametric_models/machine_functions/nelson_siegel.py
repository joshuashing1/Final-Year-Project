from numbers import Real
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from numpy import exp

EPS = np.finfo(float).eps

@dataclass
class NelsonSiegelCurve:
    """Implementation of a Nelson-Siegel interest rate curve model.
    This curve can be interpreted as a factor model with three factors
    with constant included.
    """

    beta1: float
    beta2: float
    beta3: float
    lambd: float
    
    @staticmethod
    def _tau(T: Union[float, np.ndarray], t: Union[float, np.ndarray]
        ) -> Union[float, np.ndarray]:
        """
        Compute τ = T - t.
        """
        if isinstance(T, np.ndarray) or isinstance(t, np.ndarray):
            return np.asarray(T) - np.asarray(t)
        else:
            return float(T) - float(t)

    def factors(self, T: Union[float, np.ndarray],t: Union[float, np.ndarray] = 0.0
        ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Computes the indvidual factors of the Nelson-Siegel interest rate curve model.
        """
        lambd = self.lambd
        if not np.isfinite(lambd) or lambd <= 0.0:
            tau = self._tau(T, t)
            if isinstance(tau, Real):
                return 1.0, 0.0
            n = np.asarray(tau).size
            return np.ones(n, dtype=float), np.zeros(n, dtype=float)
        
        tau = self._tau(T, t)
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
        """
        Expresses the factors of the Nelson-Siegel interest rate curve as a matrix for computation.
        """
        factor1, factor2 = self.factors(T, t)
        if isinstance(factor1, np.ndarray):
            constant: np.ndarray = np.ones(factor1.size)
            return np.stack([constant, factor1, factor2]).transpose()
        else:
            return np.array([[1.0, factor1, factor2]])
        
    def _yield(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0
        ) -> Union[float, np.ndarray]:
        """
        Nelson-Siegel interest rate curve representation of the zero coupon yield.
        """
        factor1, factor2 = self.factors(T, t)
        return self.beta1 + self.beta2 * factor1 + self.beta3 * factor2

    def __call__(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0,
        ) -> Union[float, np.ndarray]:
        """
        Callable yield curve object.
        """
        return self._yield(T, t)
    
    def forward(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0,
        ) -> Union[float, np.ndarray]:
        """
        Nelson-Siegel interest rate curve representation of the instantaneous forward rate.
        """
        tau = self.tau(T, t)
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