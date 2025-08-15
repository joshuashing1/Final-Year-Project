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


# def calibrate_ns_grid(
#     t: np.ndarray, y: np.ndarray, lambd_lo: float = 0.05, lambd_hi: float = 5.0, n_grid: int = 100
# ) -> Tuple[NelsonSiegelCurve, dict]:
#     """
#     Nelson–Siegel calibration by λ grid search (Nelson & Siegel, 1987):
#       - Try n_grid equally spaced λ in [lambd_lo, lambd_hi]
#       - For each λ, estimate betas by OLS
#       - Pick λ that minimizes SSE
#     Returns: (best_curve, info_dict)
#     """
#     _assert_same_shape(t, y)
#     lambdas = np.linspace(lambd_lo, lambd_hi, n_grid)

#     best_sse = np.inf
#     best_curve = None
#     sse_list = []

#     for L in lambdas:
#         curve, _ = betas_ns_ols(L, t, y)
#         resid = curve(t) - y
#         sse = float(np.dot(resid, resid))
#         sse_list.append(sse)
#         if sse < best_sse:
#             best_sse = sse
#             best_curve = curve

#     info = {
#         "lambdas": lambdas,
#         "sse": np.array(sse_list, dtype=float),
#         "best_sse": best_sse,
#         "best_lambda": float(best_curve.lambd) if best_curve else np.nan,
#     }
#     return best_curve, info


def errorfn_ns_ols(lambd: float, t: np.ndarray, y: np.ndarray) -> float:
    """Sum of squares error function for a Nelson-Siegel model and
    time-value pairs t and y. All betas are obtained by ordinary
    least squares given lambd.
    """
    _assert_same_shape(t, y)
    curve, lstsq_res = betas_ns_ols(lambd, t, y)
    return np.sum((curve(t) - y) ** 2)


def calibrate_ns_ols(
    t: np.ndarray, y: np.ndarray, lambd0: float = 1.0
) -> Tuple[NelsonSiegelCurve, Any]:
    """Calibrate a Nelson-Siegel curve to time-value pairs
    t and y, by optimizing lambd and chosing all betas
    using ordinary least squares.
    """
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_ns_ols, x0=lambd0, args=(t, y))
    curve, lstsq_res = betas_ns_ols(opt_res.x[0], t, y)
    return curve, opt_res

# I think can delete this. 
# def empirical_factors(
#     y_3m: float, y_2y: float, y_10y: float
# ) -> Tuple[float, float, float]:
#     """Calculate the empirical factors according to
#     Diebold and Li (2006)."""
#     return y_10y, y_10y - y_3m, 2 * y_2y - y_3m - y_10y


# def betas_svn_ols(
#     tau: Tuple[float, float], t: np.ndarray, y: np.ndarray
# ) -> Tuple[SvenssonCurve, Any]:
#     """Calculate the best-fitting beta-values given tau (= array of tau1
#     and tau2) for time-value pairs t and y and return a corresponding
#     Svensson curve instance.
#     """
#     _assert_same_shape(t, y)
#     curve = SvenssonCurve(0, 0, 0, 0, tau[0], tau[1])
#     factors = curve.factor_matrix(t)
#     lstsq_res = lstsq(factors, y, rcond=None)
#     beta = lstsq_res[0]
#     return (
#         SvenssonCurve(beta[0], beta[1], beta[2], beta[3], tau[0], tau[1]),
#         lstsq_res,
#     )


# def errorfn_svn_ols(tau: Tuple[float, float], t: np.ndarray, y: np.ndarray) -> float:
#     """Sum of squares error function for a Svensson
#     model and time-value pairs t and y. All betas are obtained
#     by ordinary least squares given tau (= array of tau1
#     and tau2).
#     """
#     _assert_same_shape(t, y)
#     curve, lstsq_res = betas_svn_ols(tau, t, y)
#     return np.sum((curve(t) - y) ** 2)


# def calibrate_svn_ols(
#     t: np.ndarray, y: np.ndarray, tau0: Tuple[float, float] = (2.0, 5.0)
# ) -> Tuple[SvenssonCurve, Any]:
#     """Calibrate a Svensson curve to time-value
#     pairs t and y, by optimizing tau1 and tau2 and chosing
#     all betas using ordinary least squares. This method does
#     not work well regarding the recovery of true parameters.
#     """
#     _assert_same_shape(t, y)
#     opt_res = minimize(errorfn_svn_ols, x0=np.array(tau0), args=(t, y))
#     curve, lstsq_res = betas_svn_ols(opt_res.x, t, y)
#     return curve, opt_res