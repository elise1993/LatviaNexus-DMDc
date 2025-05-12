import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import (root_mean_squared_error as RMSE,
                             mean_absolute_error as MAE,
                             mean_absolute_percentage_error as MAPE)



def apply_filter(use_filter, k_best, data_train_val, control_names,
                 important_names, geo_vars):
    
    if use_filter:

        # filter out variables with low mutual information with control variables
        selector = SelectKBest(mutual_info_regression, k=k_best)
        feature_names_filter = set()
        for name in control_names:
            selector.fit(data_train_val, data_train_val[name])
            add_features = set(selector.get_feature_names_out())
            feature_names_filter.update(add_features)
        
        # add back important/control variables if filtered out
        keep_names = important_names + control_names + geo_vars
        feature_names_filter.update(keep_names)
        feature_names_filter = list(feature_names_filter)
        
    else:
        k_best = "N/A"
        feature_names_filter = data_train_val.columns.tolist()
    
    return feature_names_filter, k_best



def NSE(true, pred): # Nash-Sutcliffe Efficiency
    return 1 - np.sum((pred - true)**2, axis=0) / np.sum((true - np.mean(true, axis=0))**2, axis=0)


def loss_func(pred, true, score_func, weights=None):
    '''
    Parameters
    ----------
    pred : pandas or numpy array
        predicted values
    true : pandas or numpy array
        true/benchmark values
    weights : TYPE, optional
        weighted errors to promote accuracy in certain regions. The default is
        None.

    Returns
    -------
    loss : numpy float
        loss function output
    '''
    
    # pred, true = check_X_y(pred, true, multi_output=True)
    pred, true = np.array(pred), np.array(true) # if nan values are present
    call = {'NSE': NSE, 'RMSE': RMSE, 'MAE': MAE, 'MAPE': MAPE}
    
    if weights is None:
        
        error = call[score_func](true, pred)
        loss = np.mean(error[np.isfinite(error)])
        
        # maximize NSE, R2 or minimize RMSE, MAE
        if score_func not in ['NSE', 'R2']:
            loss = -loss
        
    else: # scale loss function with control (improve accuracy during control changes)
        # USES NSE, DOES NOT SUPPORT OTHER SCORE FUNCS IN ITS CURRENT STATE
        
        m, n = pred.shape
        error = np.empty((m, n))
        for k in range(m):
            error[k,:] = (pred[k,:] - true[k,:])**2
        
        error = error * weights.sum(axis=1).reshape(-1,1)
        error = np.sum(error, axis=0)
        norm = np.sum((true - np.mean(true, axis=0))**2, axis=0)
        loss = np.mean(1 - error / norm)
        
    return loss