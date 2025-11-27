import numpy as np
from scipy.integrate import quad

def optimal_SVHT_coef(beta, sigma_known=True):
    """
    Compute optimal hard threshold coefficient for singular value hard thresholding.
    Based on Gavish & Donoho (2014).
    
    Parameters
    ----------
    beta : float or array_like
        Aspect ratio m/n of the matrix to be denoised, 0 < beta <= 1.
    sigma_known : bool
        If True, noise level known; if False, unknown.
    
    Returns
    -------
    coef : float or ndarray
        Optimal threshold coefficient.
    """
    beta = np.atleast_1d(beta).astype(float)
    if sigma_known:
        coef = _optimal_SVHT_coef_sigma_known(beta)
    else:
        coef = _optimal_SVHT_coef_sigma_unknown(beta)
    return coef if coef.size > 1 else coef.item()


def _optimal_SVHT_coef_sigma_known(beta):
    if np.any(beta <= 0) or np.any(beta > 1):
        raise ValueError("beta must satisfy 0 < beta <= 1")
    w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
    return np.sqrt(2 * (beta + 1) + w)


def _optimal_SVHT_coef_sigma_unknown(beta):
    coef = _optimal_SVHT_coef_sigma_known(beta)
    MPmedian = np.array([_MedianMarcenkoPastur(b) for b in beta])
    return coef / np.sqrt(MPmedian)


def _MarcenkoPasturIntegral(x, beta):
    if not (0 < beta <= 1):
        raise ValueError("beta must satisfy 0 < beta <= 1")
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2
    if not (lobnd <= x <= hibnd):
        raise ValueError("x out of bounds for given beta")
    
    def dens(t):
        return np.sqrt((hibnd - t) * (t - lobnd)) / (2 * np.pi * beta * t)
    
    I, _ = quad(dens, lobnd, x)
    return I


def _MedianMarcenkoPastur(beta):
    def MarPas(x):
        return 1 - _incMarPas(x, beta, 0)
    
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2
    
    while hibnd - lobnd > 1e-3:
        xs = np.linspace(lobnd, hibnd, 5)
        ys = [MarPas(x) for x in xs]
        ys = np.array(ys)
        if np.any(ys < 0.5):
            lobnd = np.max(xs[ys < 0.5])
        if np.any(ys > 0.5):
            hibnd = np.min(xs[ys > 0.5])
    
    return 0.5 * (hibnd + lobnd)


def _incMarPas(x0, beta, gamma):
    if beta > 1:
        raise ValueError("beta must be <= 1")
    topSpec = (1 + np.sqrt(beta))**2
    botSpec = (1 - np.sqrt(beta))**2
    
    def MarPas(x):
        if (topSpec - x) * (x - botSpec) > 0:
            return np.sqrt((topSpec - x) * (x - botSpec)) / (beta * x) / (2 * np.pi)
        else:
            return 0.0
    
    if gamma != 0:
        fun = lambda x: (x**gamma) * MarPas(x)
    else:
        fun = MarPas
    
    I, _ = quad(fun, x0, topSpec)
    return I
