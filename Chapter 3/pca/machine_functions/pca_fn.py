import numpy as np
from typing import Tuple

class PCAFactors:
    """
    Computation of the principal eigenpairs (top-p) using PCA methodology.
    """
    def __init__(self, p: int, annualize: float | None = 252.0):
        self.p, self.annualize = p, annualize
        self.diff_rates_ = self.sigma_ = self.eigvals_ = self.eigvecs_ = None
        self.princ_eigval_ = self.princ_comp_ = self.order_ = None

    @staticmethod
    def difference(hist_rates: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Computes the first differencing on the historical forward rates.
        """
        if hist_rates.ndim != 2: raise ValueError("hist_rates must be 2D [n_time, n_tenor].")
        return np.diff(hist_rates, axis=axis)

    def covariance(self, diff_rates: np.ndarray) -> np.ndarray:
        """
        Computes the covariance matrix of the first differenced rates.
        """
        sigma = np.cov(diff_rates.T)
        return sigma if self.annualize is None else sigma * float(self.annualize)

    def _eig(self, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes eigendecomposition of the covariance matrix.
        """
        vals, vecs = np.linalg.eig(sigma)
        return np.real_if_close(vals), np.real_if_close(vecs)

    def eigenvalues(self, sigma: np.ndarray) -> np.ndarray:
        """
        Get eigenvalues.
        """
        return self._eig(sigma)[0]

    def eigenvectors(self, sigma: np.ndarray) -> np.ndarray:
        """
        Get eigenvectors.
        """
        return self._eig(sigma)[1]

    def pca(self, sigma: np.ndarray, p: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns principal eigenpairs.
        """
        p = self.p if p is None else p
        vals, vecs = self._eig(sigma)
        order = np.argsort(vals)[::-1]
        if p > vals.size: raise ValueError(f"p={p} exceeds dimension {vals.size}")
        top = order[:p]
        return vals[top], vecs[:, top], order