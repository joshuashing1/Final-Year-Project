import numpy as np
from typing import Sequence, List

class PolynomialInterpolator:
    """Tiny wrapper around np.polyfit/np.polyval (coeffs in descending powers)."""
    def __init__(self, coeffs: np.ndarray):
        self.coeffs = np.asarray(coeffs, dtype=float)

    @classmethod
    def fit(cls, x, y, degree: int) -> "PolynomialInterpolator":
        coeffs = np.polyfit(np.asarray(x, float), np.asarray(y, float), degree)
        return cls(coeffs)

    def calc(self, x: float) -> float:
        return float(np.polyval(self.coeffs, x))

    def __call__(self, x):
        """Vectorized evaluation."""
        return np.polyval(self.coeffs, np.asarray(x, float))


class VolatilityFitter:
    """
    Fit one polynomial per PCA volatility factor and evaluate on any tenor grid.

    vols shape: [n_tenor, k]  (columns = factors)
    tenors: same length as n_tenor, in years (float)
    """
    def __init__(self, tenors: np.ndarray, vols: np.ndarray):
        self.tenors = np.asarray(tenors, float)
        self.vols = np.asarray(vols, float)
        if self.vols.ndim == 1:
            self.vols = self.vols[:, None]
        self.k = self.vols.shape[1]
        self.models: List[PolynomialInterpolator] = []

    @staticmethod
    def from_pca(princ_eigval: np.ndarray, princ_comp: np.ndarray) -> np.ndarray:
        """
        Discretized volatility functions from PCA:
            vols[:, i] = sqrt(lambda_i) * PC_i(tenor)
        Returns vols with shape [n_tenor, k].
        """
        sqrt_eig = np.sqrt(np.asarray(princ_eigval, float))
        return np.asarray(princ_comp, float) * sqrt_eig[np.newaxis, :]

    def fit(self, degrees: Sequence[int]) -> List[PolynomialInterpolator]:
        """
        degrees: length-k list (or single int broadcast to all factors).
        """
        if isinstance(degrees, int):
            degrees = [degrees] * self.k
        elif len(degrees) == 1:
            degrees = [degrees[0]] * self.k

        self.models = [
            PolynomialInterpolator.fit(self.tenors, self.vols[:, i], deg)
            for i, deg in enumerate(degrees)
        ]
        return self.models

    def predict(self, grid: np.ndarray) -> np.ndarray:
        """Evaluate fitted vol curves on 'grid' (years). Returns array [len(grid), k]."""
        if not self.models:
            raise RuntimeError("Call fit() before predict().")
        grid = np.asarray(grid, float)
        return np.column_stack([m(grid) for m in self.models])
