from numpy.random import rand
from numpy import std
import pandas as pd
from datetime import datetime

def data_loader(train='P0', test='P19', interpolate=False):
    
    interpolate_method = 'linear'
    # interpolate_method = 'spline'
    order = 1
    
    start = datetime(2000,1,1)
    end = datetime(2051,1,1)
    t = pd.date_range(start, end, freq='ME').to_numpy()
    
    X_train_val = pd.read_csv(f"./data/latvia_sdm_policy{train[1:]}.csv")\
        .set_index(t).resample('YE').first()
        
    X_test = pd.read_csv(f"./data/latvia_sdm_policy{test[1:]}.csv")\
        .set_index(t).resample('YE').first()
        
    # interpolate
    if interpolate:
        X_train_val = X_train_val.resample('ME').interpolate(method=interpolate_method, order=order)
        X_test = X_test.resample('ME').interpolate(method=interpolate_method, order=order)
    
    t = X_train_val.index
    
    # add noise (can sometimes stabilize model)
    noise_factor = 0
    m, n = X_train_val.shape
    mag_train = noise_factor * std(X_train_val, axis=0)
    mag_test = noise_factor * std(X_test, axis=0)
    X_train_val = X_train_val + rand(m, n)*mag_train.to_numpy()
    X_test_val = X_train_val + rand(m, n)*mag_test.to_numpy()
    
    return X_train_val, X_test, t