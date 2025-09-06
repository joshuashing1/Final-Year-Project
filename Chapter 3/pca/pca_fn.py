import numpy as np
from typing import Tuple

class PCAFactors:
    """Compact PCA on rate increments -> covariance -> eig -> top-k."""
    def __init__(self, k: int, annualize: float | None = 252.0):
        self.k, self.annualize = k, annualize
        self.diff_rates_ = self.sigma_ = self.eigvals_ = self.eigvecs_ = None
        self.princ_eigval_ = self.princ_comp_ = self.order_ = None

    @staticmethod
    def difference(hist_rates: np.ndarray, axis: int = 0) -> np.ndarray:
        if hist_rates.ndim != 2: raise ValueError("hist_rates must be 2D [n_time, n_tenor].")
        return np.diff(hist_rates, axis=axis)

    def covariance(self, diff_rates: np.ndarray) -> np.ndarray:
        if diff_rates.ndim != 2: raise ValueError("diff_rates must be 2D.")
        sigma = np.cov(diff_rates.T)
        return sigma if self.annualize is None else sigma * float(self.annualize)

    def _eig(self, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals, vecs = np.linalg.eig(sigma)
        return np.real_if_close(vals), np.real_if_close(vecs)

    def eigenvalues(self, sigma: np.ndarray) -> np.ndarray:
        return self._eig(sigma)[0]

    def eigenvectors(self, sigma: np.ndarray) -> np.ndarray:
        return self._eig(sigma)[1]

    def pca(self, sigma: np.ndarray, k: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = self.k if k is None else k
        vals, vecs = self._eig(sigma)
        order = np.argsort(vals)[::-1]
        if k > vals.size: raise ValueError(f"k={k} exceeds dimension {vals.size}")
        top = order[:k]
        return vals[top], vecs[:, top], order

    # def fit(self, hist_rates: np.ndarray, axis: int = 0):
    #     self.diff_rates_ = self.difference(hist_rates, axis)
    #     self.sigma_ = self.covariance(self.diff_rates_)
    #     self.princ_eigval_, self.princ_comp_, self.order_ = self.pca(self.sigma_, self.k)
    #     self.eigvals_, self.eigvecs_ = self._eig(self.sigma_)
    #     return self
