"""Grid search calibration of a Svensson curve using GPU-accelerated computation with CuPy.
For details of the logic, see function 'calibrate_svn_grid' from 'calibration'.
Install package cupy-cuda12x (-based off relevant CuDA version).
"""
import numpy as np

from svensson import SvenssonCurve
from calibration import _assert_same_shape

def calibrate_svn_grid_CuPy(t: np.ndarray, y: np.ndarray, lambd1_lo: float, lambd1_upp: float, lambd2_lo: float, lambd2_upp: float, n_grid1: int, n_grid2: int):
    """
    GPU-accelerated batched grid search for Svensson on CuDA v12.6.
    Computes OLS betas for every (λ1, λ2).
    """
    _assert_same_shape(t, y)
    assert lambd1_lo > 0 and lambd1_upp > lambd1_lo and n_grid1 >= 2
    assert lambd2_lo > 0 and lambd2_upp > lambd2_lo and n_grid2 >= 2

    try:
        import cupy as cp
        xp = cp  # GPU
    except Exception:
        xp = np  # CPU backup

    t_x = xp.asarray(t, dtype=xp.float64)
    y_x = xp.asarray(y, dtype=xp.float64)

    l1s = xp.linspace(lambd1_lo, lambd1_upp, n_grid1, dtype=xp.float64)
    l2s = xp.linspace(lambd2_lo, lambd2_upp, n_grid2, dtype=xp.float64)
    L1, L2 = xp.meshgrid(l1s, l2s, indexing="ij")        
    G = n_grid1 * n_grid2
    L1g = L1.reshape(G, 1)  
    L2g = L2.reshape(G, 1)  

    tau = t_x.reshape(1, -1)  
    eps = xp.finfo(t_x.dtype).eps
    tau_safe = xp.where(tau <= 0, eps, tau)

    a1 = tau_safe / L1g # Svensson basis per grid-point mapped to (G, n)
    e1 = xp.exp(-a1)
    f1 = (1 - e1) / a1
    f2 = f1 - e1

    a2 = tau_safe / L2g
    e2 = xp.exp(-a2)
    f3 = (1 - e2) / a2 - e2

    # Design matrices X for each grid point: columns [1, f1, f2, f3]
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