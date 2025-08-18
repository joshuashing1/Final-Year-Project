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


def calibrate_svn_grid(
    t: np.ndarray, y: np.ndarray, lambd1_lo: float, lambd1_upp: float, lambd2_lo: float, lambd2_upp: float, n_grid1: int, n_grid2: int,
) -> Tuple[SvenssonCurve, Any]:
    """Grid search calibration of a Svensson curve to time-value pairs t and y.
    Searches over (λ1, λ2) on a rectangular grid, refitting betas by OLS at each pair.
    """
    _assert_same_shape(t, y)
    assert lambd1_lo > 0 and lambd1_upp > lambd1_lo and n_grid1 >= 2
    assert lambd2_lo > 0 and lambd2_upp > lambd2_lo and n_grid2 >= 2

    lambd1s = np.linspace(lambd1_lo, lambd1_upp, n_grid1)
    lambd2s = np.linspace(lambd2_lo, lambd2_upp, n_grid2)

    res_list = []
    for l1 in lambd1s:
        for l2 in lambd2s:
            res = errorfn_svn_ols((float(l1), float(l2)), t, y)
            res_list.append(float(res))

    res_arr = np.asarray(res_list, dtype=float).reshape(n_grid1, n_grid2)
    flat_idx = int(np.argmin(res_arr))
    i1, i2 = np.unravel_index(flat_idx, res_arr.shape)

    opt_lambd1 = float(lambd1s[i1])
    opt_lambd2 = float(lambd2s[i2])
    opt_curve, _ = betas_svn_ols((opt_lambd1, opt_lambd2), t, y)
    opt_res = float(res_arr[i1, i2])

    return opt_curve, opt_res, (opt_lambd1, opt_lambd2)
