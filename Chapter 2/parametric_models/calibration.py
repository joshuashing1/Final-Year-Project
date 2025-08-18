# -*- coding: utf-8 -*-

"""Calibration methods for Nelson-Siegel and Svensson Models.
See `calibrate_ns_ols` and `calibrate_svn_ols` for ordinary least squares
(OLS) based methods.
"""

from typing import Tuple, Any

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize

from nelson_siegel import NelsonSiegelCurve
from svensson import SvenssonCurve

def _assert_same_shape(t: np.ndarray, y: np.ndarray) -> None:
    assert t.shape == y.shape, "Mismatching shapes of time and values"


def betas_ns_ols(
    lambd: float, t: np.ndarray, y: np.ndarray
) -> Tuple[NelsonSiegelCurve, Any]:
    """Calculate the best-fitting beta-values given lambd
    for time-value pairs t and y and return a corresponding
    Nelson-Siegel curve instance.
    """
    _assert_same_shape(t, y)
    curve = NelsonSiegelCurve(0, 0, 0, lambd)
    factors = curve.factor_matrix(t)
    lstsq_res = lstsq(factors, y, rcond=None)
    beta = lstsq_res[0]
    return NelsonSiegelCurve(beta[0], beta[1], beta[2], lambd), lstsq_res


def errorfn_ns_ols(lambd: float, t: np.ndarray, y: np.ndarray) -> float:
    """Sum of squares error function for a Nelson-Siegel model and
    time-value pairs t and y. All betas are obtained by ordinary
    least squares given lambd.
    """
    _assert_same_shape(t, y)
    curve, lstsq_res = betas_ns_ols(lambd, t, y)
    return np.sum((curve(t) - y) ** 2)


def calibrate_ns_ptwise(
    t: np.ndarray, y: np.ndarray, lambd0: float
) -> Tuple[NelsonSiegelCurve, Any]:
    """Pointwise calibration of a Nelson-Siegel curve to time-value pairs
    t and y, by optimizing lambd and chosing all betas
    using ordinary least squares.
    """
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_ns_ols, x0=lambd0, args=(t, y), method = "BFGS")
    curve, lstsq_res = betas_ns_ols(opt_res.x[0], t, y)
    return curve, opt_res

def calibrate_ns_grid(
    t: np.ndarray, y: np.ndarray, lambd_lo: float, lambd_upp: float, n_grid: int
) -> Tuple[NelsonSiegelCurve, Any]:
    """Grid search calibration of a Nelson-Siegel curve to time-value pairs
    t and y, by iterating various lambds and chosing all betas
    using ordinary least squares.
    """
    _assert_same_shape(t, y)
    assert lambd_lo > 0 and lambd_upp > lambd_lo and n_grid >= 2
    lambds = np.linspace(lambd_lo, lambd_upp, n_grid)
    res_list = []
    for lambd in lambds:
        res = errorfn_ns_ols(lambd, t, y)
        res_list.append(float(res))
        
    res_arr = np.asarray(res_list, dtype=float)
    opt_idx = int(np.argmin(res_arr))
    opt_lambd = float(lambds[opt_idx])
    opt_curve, lstsq_res = betas_ns_ols(opt_lambd, t, y)
    opt_res = float(res_arr[opt_idx])
    
    return opt_curve, opt_res, opt_lambd


def betas_svn_ols(
    lambd: Tuple[float, float], t: np.ndarray, y: np.ndarray
) -> Tuple[SvenssonCurve, Any]:
    """Calculate the best-fitting beta-values given lambd (= array of lambd1
    and lambd2) for time-value pairs t and y and return a corresponding
    Svensson curve instance.
    """
    _assert_same_shape(t, y)
    curve = SvenssonCurve(0, 0, 0, 0, lambd[0], lambd[1])
    factors = curve.factor_matrix(t)
    lstsq_res = lstsq(factors, y, rcond=None)
    beta = lstsq_res[0]
    return SvenssonCurve(beta[0], beta[1], beta[2], beta[3], lambd[0], lambd[1]), lstsq_res


def errorfn_svn_ols(lambd: Tuple[float, float], t: np.ndarray, y: np.ndarray) -> float:
    """Sum of squares error function for a Svensson
    model and time-value pairs t and y. All betas are obtained
    by ordinary least squares given lambd (= array of lambd1
    and lambd2).
    """
    _assert_same_shape(t, y)
    curve, lstsq_res = betas_svn_ols(lambd, t, y)
    return np.sum((curve(t) - y) ** 2)


def calibrate_svn_ptwise(
    t: np.ndarray, y: np.ndarray, lambd0: Tuple[float, float] = (2.0, 5.0)
) -> Tuple[SvenssonCurve, Any]:
    """Pointwise calibration of a Svensson curve to time-value
    pairs t and y, by optimizing lambd1 and lambd2 and chosing
    all betas using ordinary least squares. This method does
    not work well regarding the recovery of true parameters.
    """
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_svn_ols, x0=np.array(lambd0), args=(t, y), method = "BFGS")
    curve, lstsq_res = betas_svn_ols(opt_res.x, t, y)
    return curve, opt_res


# def calibrate_svn_grid(
#     t: np.ndarray, y: np.ndarray, lambd1_lo: float, lambd1_upp: float, lambd2_lo: float, lambd2_upp: float, n_grid1: int, n_grid2: int,
# ) -> Tuple[SvenssonCurve, Any]:
#     """Grid search calibration of a Svensson curve to time-value pairs t and y.
#     Searches over (λ1, λ2) on a rectangular grid, refitting betas by OLS at each pair.
#     """
#     _assert_same_shape(t, y)
#     assert lambd1_lo > 0 and lambd1_upp > lambd1_lo and n_grid1 >= 2
#     assert lambd2_lo > 0 and lambd2_upp > lambd2_lo and n_grid2 >= 2

#     lambd1s = np.linspace(lambd1_lo, lambd1_upp, n_grid1)
#     lambd2s = np.linspace(lambd2_lo, lambd2_upp, n_grid2)

#     res_list = []
#     for l1 in lambd1s:
#         for l2 in lambd2s:
#             res = errorfn_svn_ols((float(l1), float(l2)), t, y)
#             res_list.append(float(res))

#     res_arr = np.asarray(res_list, dtype=float).reshape(n_grid1, n_grid2)
#     flat_idx = int(np.argmin(res_arr))
#     i1, i2 = np.unravel_index(flat_idx, res_arr.shape)

#     opt_lambd1 = float(lambd1s[i1])
#     opt_lambd2 = float(lambd2s[i2])
#     opt_curve, _ = betas_svn_ols((opt_lambd1, opt_lambd2), t, y)
#     opt_res = float(res_arr[i1, i2])

#     return opt_curve, opt_res, (opt_lambd1, opt_lambd2)

def calibrate_svn_grid(
    t: np.ndarray,
    y: np.ndarray,
    lambd1_lo: float, lambd1_upp: float,
    lambd2_lo: float, lambd2_upp: float,
    n_grid1: int, n_grid2: int,
):
    """
    GPU-accelerated (if CuPy available) batched grid search for Svensson.
    Computes OLS betas for every (λ1, λ2) in one batched pass and returns the best.
    """
    _assert_same_shape(t, y)
    assert lambd1_lo > 0 and lambd1_upp > lambd1_lo and n_grid1 >= 2
    assert lambd2_lo > 0 and lambd2_upp > lambd2_lo and n_grid2 >= 2

    # Try CuPy; fall back to NumPy
    try:
        import cupy as cp
        xp = cp  # GPU
    except Exception:
        xp = np  # CPU fallback

    # Move data
    t_x = xp.asarray(t, dtype=xp.float64)
    y_x = xp.asarray(y, dtype=xp.float64)

    # Build grid
    l1s = xp.linspace(lambd1_lo, lambd1_upp, n_grid1, dtype=xp.float64)
    l2s = xp.linspace(lambd2_lo, lambd2_upp, n_grid2, dtype=xp.float64)
    L1, L2 = xp.meshgrid(l1s, l2s, indexing="ij")        # shape (n_grid1, n_grid2)
    G = n_grid1 * n_grid2
    L1g = L1.reshape(G, 1)  # (G,1)
    L2g = L2.reshape(G, 1)  # (G,1)

    # τ = t (if your Svensson uses τ = T)
    tau = t_x.reshape(1, -1)  # (1, n)
    # Avoid division by zero at tau=0
    eps = xp.finfo(t_x.dtype).eps
    tau_safe = xp.where(tau <= 0, eps, tau)

    # Svensson basis per grid-point (broadcast to (G, n))
    # f1(λ1) = (1 - e^{-τ/λ1}) / (τ/λ1)
    # f2(λ1) = f1(λ1) - e^{-τ/λ1}
    # f3(λ2) = (1 - e^{-τ/λ2}) / (τ/λ2) - e^{-τ/λ2}
    a1 = tau_safe / L1g                    # (G, n)
    e1 = xp.exp(-a1)
    f1 = (1 - e1) / a1
    f2 = f1 - e1

    a2 = tau_safe / L2g
    e2 = xp.exp(-a2)
    f3 = (1 - e2) / a2 - e2

    # Design matrices X for each grid point: columns [1, f1, f2, f3]
    # We won’t materialize a 3D tensor X; we’ll build XtX and XtY directly.
    ones = xp.ones_like(f1)
    # XtX = [ [Σ1], [Σf1], [Σf2], [Σf3] ]^T cross-products for each grid point → (G,4,4)
    def colsum(Z):  # sum over n
        return Z.sum(axis=1)  # (G,)

    s1  = colsum(ones)     # Σ 1
    s11 = colsum(f1)       # Σ f1
    s12 = colsum(f2)       # Σ f2
    s13 = colsum(f3)       # Σ f3

    s22 = colsum(f1*f1)    # Σ f1^2
    s23 = colsum(f1*f2)    # Σ f1 f2
    s24 = colsum(f1*f3)    # Σ f1 f3
    s33 = colsum(f2*f2)    # Σ f2^2
    s34 = colsum(f2*f3)    # Σ f2 f3
    s44 = colsum(f3*f3)    # Σ f3^2

    # Build symmetric XtX
    XtX = xp.stack([
        xp.stack([s1,  s11, s12, s13], axis=-1),
        xp.stack([s11, s22, s23, s24], axis=-1),
        xp.stack([s12, s23, s33, s34], axis=-1),
        xp.stack([s13, s24, s34, s44], axis=-1),
    ], axis=1)  # (G,4,4)

    # XtY
    ycol = y_x.reshape(1, -1)  # (1,n)
    xTy0 = colsum(ones * ycol)
    xTy1 = colsum(f1   * ycol)
    xTy2 = colsum(f2   * ycol)
    xTy3 = colsum(f3   * ycol)
    XtY = xp.stack([xTy0, xTy1, xTy2, xTy3], axis=-1).reshape(G, 4)  # (G,4)

    # Solve (XtX) beta = XtY in batch
    # Add tiny ridge for stability if needed
    ridge = 1e-12
    I4 = xp.eye(4, dtype=xp.float64)
    XtX_reg = XtX + ridge * I4.reshape(1, 4, 4)

    # batched solve
    betas = xp.linalg.solve(XtX_reg, XtY[..., None]).squeeze(-1)  # (G,4)

    # SSE = ||X beta - y||^2, compute via y^T y - 2 beta^T XtY + beta^T XtX beta
    yTy = float(np.sum(y * y))  # small scalar; OK on CPU
    # move XtY, betas to xp; already are
    term1 = yTy
    term2 = 2.0 * xp.sum(betas * XtY, axis=1)
    # beta^T XtX beta (batch)
    bXtX = xp.einsum('gij,gj->gi', XtX, betas)            # (G,4)
    term3 = xp.sum(betas * bXtX, axis=1)                  # (G,)

    sse = term3 - term2 + term1                           # (G,)
    min_idx = int(xp.argmin(sse))
    i1, i2 = divmod(min_idx, n_grid2)
    opt_l1 = float(l1s[i1].get() if xp is not np else l1s[i1])
    opt_l2 = float(l2s[i2].get() if xp is not np else l2s[i2])

    # Bring betas back and build CPU curve object
    b = betas[min_idx]
    if xp is not np:
        b = b.get()
    beta1, beta2, beta3, beta4 = map(float, b.tolist())

    curve = SvenssonCurve(beta1, beta2, beta3, beta4, opt_l1, opt_l2)
    opt_res = float(sse[min_idx].get() if xp is not np else sse[min_idx])
    return curve, opt_res, (opt_l1, opt_l2)
