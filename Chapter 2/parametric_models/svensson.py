# -*- coding: utf-8 -*-

"""Implementation of a Svensson interest rate curve model.
See `SvenssonCurve` class for details.
"""

from numbers import Real
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from numpy import exp

EPS = np.finfo(float).eps


@dataclass
class SvenssonCurve:
    """Implementation of a Svensson interest rate curve model.
    This curve can be interpreted as a factor model with four
    factors (including a constant).
    """

    beta1: float
    beta2: float
    beta3: float
    beta4: float
    lambd1: float
    lambd2: float

    def factors(
        self, T: Union[float, np.ndarray]
    ) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Factor loadings for time(s) T, excluding constant."""
        lambd1 = self.lambd1
        lambd2 = self.lambd2
        if isinstance(T, Real) and T <= 0:
            return 1, 0, 0
        elif isinstance(T, np.ndarray):
            zero_idx = T <= 0
            T[zero_idx] = EPS  # avoid warnings in calculations
        exp_tt1 = exp(-T / lambd1)
        exp_tt2 = exp(-T / lambd2)
        factor1 = (1 - exp_tt1) / (T / lambd1)
        factor2 = factor1 - exp_tt1
        factor3 = (1 - exp_tt2) / (T / lambd2) - exp_tt2
        if isinstance(T, np.ndarray):
            T[zero_idx] = 0
            factor1[zero_idx] = 1
            factor2[zero_idx] = 0
            factor3[zero_idx] = 0
        return factor1, factor2, factor3

    def factor_matrix(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Factor loadings for time(s) T as matrix columns,
        including constant column (=1.0).
        """
        factor1, factor2, factor3 = self.factors(T)
        constant: Union[float, np.ndarray] = (
            np.ones(T.size) if isinstance(T, np.ndarray) else 1.0
        )
        return np.stack([constant, factor1, factor2, factor3]).transpose()

    def zero(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Zero rate(s) of this curve at time(s) T."""
        beta1 = self.beta1
        beta2 = self.beta2
        beta3 = self.beta3
        beta4 = self.beta4
        factor1, factor2, factor3 = self.factors(T)
        res = beta1 + beta2 * factor1 + beta3 * factor2 + beta4 * factor3
        return res

    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Zero rate(s) of this curve at time(s) T."""
        return self.zero(T)

    def forward(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Instantaneous forward rate(s) of this curve at time(s) T."""
        exp_tt0 = exp(-T / self.lambd1)
        exp_tt1 = exp(-T / self.lambd2)
        return (
            self.beta1
            + self.beta2 * exp_tt0
            + self.beta3 * exp_tt0 * T / self.lambd1
            + self.beta4 * exp_tt1 * T / self.lambd2
        )