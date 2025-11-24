"""
This Python script pertains to the grid search calibration of the Svensson interest rate curve
using GPU-accelerated computation with CuPy library.
"""
import numpy as np

from machine_functions.svensson import SvenssonCurve
from calibration import _assert_same_shape

def calibrate_svn_grid_CuPy(t: np.ndarray, y: np.ndarray, lambd1_lo: float, lambd1_upp: float, lambd2_lo: float, lambd2_upp: float,
    n_grid1: int, n_grid2: int, return_surface: bool = False):
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
        on_gpu = True
    except Exception:
        xp = np  # CPU backup
        on_gpu = False

    """
    store maturity and yield arrays into GPU memory.
    """
    t_x = xp.asarray(t, dtype=xp.float64) 
    y_x = xp.asarray(y, dtype=xp.float64)

    """
    create grid.
    """
    l1s = xp.linspace(lambd1_lo, lambd1_upp, n_grid1, dtype=xp.float64)
    l2s = xp.linspace(lambd2_lo, lambd2_upp, n_grid2, dtype=xp.float64)
    L1, L2 = xp.meshgrid(l1s, l2s, indexing="ij")        
    G = n_grid1 * n_grid2 # total grid points
    L1g = L1.reshape(G, 1) # λ1 grid
    L2g = L2.reshape(G, 1) # λ2 grid 

    """
    defining tenor array structure
    """
    tau = t_x.reshape(1, -1) # reshape tenor array to shape (1, n)
    eps = xp.finfo(t_x.dtype).eps # machine epsilon for float64
    tau_safe = xp.where(tau <= 0, eps, tau) # returns eps if tau is ≤ 0

    """
    Svensson basis formed by λ1.
    """
    a1 = tau_safe / L1g
    e1 = xp.exp(-a1)
    f1 = (1 - e1) / a1
    f2 = f1 - e1
    
    """
    Svensson basis formed by λ2.
    """
    a2 = tau_safe / L2g
    e2 = xp.exp(-a2)
    f3 = (1 - e2) / a2 - e2

    """
    builds XtX matrix with cross-product computed using colsum().
    """
    ones = xp.ones_like(f1)
    def colsum(Z):  
        return Z.sum(axis=1)  

    s1  = colsum(ones)     
    s11 = colsum(f1)       
    s12 = colsum(f2)       
    s13 = colsum(f3)       

    s22 = colsum(f1*f1)    
    s23 = colsum(f1*f2)    
    s24 = colsum(f1*f3)    
    s33 = colsum(f2*f2)    
    s34 = colsum(f2*f3)    
    s44 = colsum(f3*f3)    

    XtX = xp.stack([
        xp.stack([s1,  s11, s12, s13], axis=-1),
        xp.stack([s11, s22, s23, s24], axis=-1),
        xp.stack([s12, s23, s33, s34], axis=-1),
        xp.stack([s13, s24, s34, s44], axis=-1),
    ], axis=1)

    """
    builds XtY matrix with cross-product computed using colsum().
    """
    ycol = y_x.reshape(1, -1)  
    xTy0 = colsum(ones * ycol)
    xTy1 = colsum(f1   * ycol)
    xTy2 = colsum(f2   * ycol)
    xTy3 = colsum(f3   * ycol)
    XtY = xp.stack([xTy0, xTy1, xTy2, xTy3], axis=-1).reshape(G, 4)  

    """
    We solve for beta using the form (XtX) beta = XtY for each grid point.
    We also introduce some noise terms to evoke numerical stability and reduce collinearity of 
    design matrix.
    """
    ridge = 1e-12
    I4 = xp.eye(4, dtype=xp.float64)
    XtX_reg = XtX + ridge * I4.reshape(1, 4, 4)

    betas = xp.linalg.solve(XtX_reg, XtY[..., None]).squeeze(-1)  

    yTy = float(np.sum(y * y))  
    term1 = yTy
    term2 = 2.0 * xp.sum(betas * XtY, axis=1)
    bXtX = xp.einsum('gij,gj->gi', XtX, betas)            
    term3 = xp.sum(betas * bXtX, axis=1)                 

    sse = term3 - term2 + term1                           
    min_idx = int(xp.argmin(sse)) # returns index of the smallest sse
    i1, i2 = divmod(min_idx, n_grid2) # row, col indices of that grid point
    opt_l1 = float(l1s[i1].get() if xp is not np else l1s[i1]) # optimal λ1 corr to min sse
    opt_l2 = float(l2s[i2].get() if xp is not np else l2s[i2]) # optimal λ2 corr to min sse

    """
    returns the optimal beta vector using .get() and build CPU curve object.
    """
    b = betas[min_idx]
    if on_gpu: b = b.get()
    beta1, beta2, beta3, beta4 = map(float, b.tolist())
    curve = SvenssonCurve(beta1, beta2, beta3, beta4, opt_l1, opt_l2)
    opt_res = float(sse[min_idx].get() if on_gpu else sse[min_idx])

    if return_surface:
        sse_grid = sse.reshape(n_grid1, n_grid2)
        l1s_np   = l1s.get() if on_gpu else l1s
        l2s_np   = l2s.get() if on_gpu else l2s
        sse_np   = sse_grid.get() if on_gpu else sse_grid
        return curve, opt_res, (opt_l1, opt_l2), (sse_np, np.asarray(l1s_np), np.asarray(l2s_np))

    return curve, opt_res, (opt_l1, opt_l2)