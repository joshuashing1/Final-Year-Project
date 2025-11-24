from numbers import Real
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from numpy import exp

EPS = np.finfo(float).eps

@dataclass
class SvenssonCurve:
    """Implementation of a Svensson interest rate curve model.
    This curve can be interpreted as a factor model with four factors 
    with constant included.
    """

    beta1: float 
    beta2: float  
    beta3: float  
    beta4: float  
    lambd1: float   
    lambd2: float   

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

    def factors(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0
    ) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Computes the indvidual factors of the Svensson interest rate curve model.
        """
        lambd1 = self.lambd1
        lambd2 = self.lambd2
        tau = self._tau(T, t)

        if not np.isfinite(lambd1) or lambd1 <= 0.0:
            if isinstance(tau, Real):
                return 1.0, 0.0, 0.0
            n = np.asarray(tau).size
            ones = np.ones(n, dtype=float)
            zeros = np.zeros(n, dtype=float)
            return ones, zeros, zeros

        if isinstance(tau, Real) and tau <= 0:
            return 1.0, 0.0, 0.0
        elif isinstance(tau, np.ndarray):
            zero_idx = tau <= 0
            tau = tau.astype(float, copy=True)
            tau[zero_idx] = EPS   # avoid warnings in calculations
            
        exp_tt0_1 = exp(-tau / lambd1)
        factor1 = (1.0 - exp_tt0_1) / (tau / lambd1)
        factor2 = factor1 - exp_tt0_1

        if not np.isfinite(lambd2) or lambd2 <= 0.0:
            if isinstance(tau, Real):
                factor3 = 0.0
            else:
                factor3 = np.zeros_like(factor1)
        else:
            exp_tt0_2 = exp(-tau / lambd2)
            factor3 = (1.0 - exp_tt0_2) / (tau / lambd2) - exp_tt0_2

        if isinstance(tau, np.ndarray):
            tau[zero_idx] = 0.0
            factor1[zero_idx] = 1.0
            factor2[zero_idx] = 0.0
            if isinstance(factor3, np.ndarray):
                factor3[zero_idx] = 0.0

        return factor1, factor2, factor3

    def factor_matrix(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0
    ) -> Union[float, np.ndarray]:
        """
        Expresses the factors of the Svensson interest rate curve as a matrix for computation.
        """
        factor1, factor2, factor3 = self.factors(T, t)
        if isinstance(factor1, np.ndarray):
            constant: np.ndarray = np.ones(factor1.size)
            return np.stack([constant, factor1, factor2, factor3]).transpose()
        else:
            return np.array([[1.0, factor1, factor2, factor3]])

    def _yield(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0
    ) -> Union[float, np.ndarray]:
        """
        Svensson interest rate curve representation of the zero coupon yield.
        """
        factor1, factor2, factor3 = self.factors(T, t)
        return self.beta1 + self.beta2 * factor1 + self.beta3 * factor2 + self.beta4 * factor3

    def __call__(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0
    ) -> Union[float, np.ndarray]:
        """
        Callable yield curve object.
        """
        return self._yield(T, t)

    def forward(self, T: Union[float, np.ndarray], t: Union[float, np.ndarray] = 0.0
    ) -> Union[float, np.ndarray]:
        """
        Svensson interest rate curve representation of the instantaneous forward rate.
        """
        tau = self._tau(T, t)
        if isinstance(tau, Real):
            if tau < 0:
                tau = 0.0
            exp_tt0_1 = exp(-tau / self.lambd1)
            exp_tt0_2 = exp(-tau / self.lambd2)
            return self.beta1 + self.beta2 * exp_tt0_1 + self.beta3 * exp_tt0_1 * (tau / self.lambd1) + self.beta4 * exp_tt0_2 * (tau / self.lambd2)
        else:
            tau = np.asarray(tau, dtype=float)
            tau_clip = np.maximum(tau, 0.0)
            exp_tt0_1 = exp(-tau_clip / self.lambd1)
            exp_tt0_2 = exp(-tau_clip / self.lambd2)
            return self.beta1 + self.beta2 * exp_tt0_1 + self.beta3 * exp_tt0_1 * (tau_clip / self.lambd1) + self.beta4 * exp_tt0_2 * (tau_clip / self.lambd2)