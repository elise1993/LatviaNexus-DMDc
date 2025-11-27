import numpy as np
from numpy.linalg import svd
from scipy.linalg import hankel
from scipy.special import huber
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def hankel_matrix(x, stackmax):
    # Construct Hankel matrix with 'stackmax delayed copies of x
    N = len(x)
    H = np.zeros((stackmax, N - stackmax))
    for i in range(stackmax):
        H[i, :] = x[i:N - stackmax + i]
    return H

def hankel_svd(H):
    U, s, Vt = svd(H, full_matrices=False)
    S = np.diag(s)
    return U, S, Vt.T

def average_hankel(H, num_vars=1, q=1, method='diagonal'):
    m = H.shape[0]
    X = np.zeros((m+q-1, num_vars))
    
    if num_vars==1:
        match method:
            case 'edges': # less accurate (especially on noisy/wavy data), but faster
                X = np.concatenate((H[0,:] , H[1:,-1]))
                
            case 'diagonal':
                H_flip = np.fliplr(H)
                for k, j in enumerate(range(q-1, -m, -1)):
                    diag_values = np.diag(H_flip, j)
                    X[k] = np.mean(diag_values)
                    
    else: # for multi-variable Hankel arrays (where all vars have the same q)
        for i in range(num_vars):
            H_block = H[:, q*i : q*(i+1)]
            X[:, i] = average_hankel(H_block, num_vars=1, q=q, method=method).flatten()
    
    return X

# def average_hankel(H, num_vars=1, q=1, method='diagonal'):
#     m, n = H.shape
#     X = np.zeros((m, num_vars))
    
#     if num_vars == 1:
#         if method == 'edges': # less accurate (especially on noisy/wavy data), but faster
#             X = np.concatenate((H[0, :q], H[q:, -1]))
            
#         elif method == 'diagonal':
#             H_flip = np.fliplr(H)
#             for j, k in enumerate(range(0, -m, -1)):
#                 diag_values = np.diag(H_flip, k)
#                 X[j] = np.mean(diag_values)
                
#     else: # for multi-variable Hankel arrays (where all vars have the same q)
#         for i in range(num_vars):
#             H_block = H[:, q*i : q*(i+1)]
#             X[:, i] = average_hankel(H_block, num_vars=1, q=q, method=method).flatten()
    
#     return X


def nash_sutcliffe_efficiency(true, pred): # same as r2_score in sklearn
    return 1 - np.sum((pred - true)**2, axis=0) / np.sum((true - np.mean(true))**2, axis=0)

def normalized_mean_squared_error(true, pred):
    return np.sum((true - pred)**2)/np.sum((true - np.mean(true))**2)

def choose_loss(loss_func):
    match loss_func:
        case 'NSE': loss_fnc_ = nash_sutcliffe_efficiency
        case 'NMSE': loss_fnc_ = normalized_mean_squared_error
        case 'MAE': loss_fnc_ = mean_absolute_error
        case 'MAPE': loss_fnc_ = mean_absolute_percentage_error
        case 'RMSE': loss_fnc_ = root_mean_squared_error
        case 'R2': loss_fnc_ = r2_score

    if isinstance(loss_fnc_, (type(nash_sutcliffe_efficiency), type(r2_score))):
        direction = "maximize"
    else:
        direction = "minimize"
    
    loss_fnc = lambda x_true, x_pred: np.mean(loss_fnc_(x_true, x_pred))
    
    return loss_fnc, direction