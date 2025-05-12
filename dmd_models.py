import numpy as np
import pandas as pd
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pydmd import DMD as dmd, DMDc as dmdc


def check_pandas(X, I=None):
    '''
    Helper function for outputting as pandas dataframe.

    Parameters
    ----------
    X : pandas dataframe/series
    I : pandas dataframe/series, optional

    Returns
    -------
    is_pandas : boolean
    feature_names : list
        Original feature names of X.
    dates : numpy datetime array
        Original dates of X.

    '''

    panda_objects = [pd.core.frame.DataFrame, pd.core.series.Series]
    if type(X) in panda_objects: # use isinstance instead?
        is_pandas = True
        try:
            feature_names = X.columns.to_list();
            dates = X.index
        except:
            feature_names = X.index.to_list()
            if I is None:
                dates = None
            else:
                dates = I.index
    else:
        is_pandas = False
    return is_pandas, feature_names, dates


class DMD():
    
    def __init__(self, svd_rank=0, reduced_matrices=False):
        '''

        Parameters
        ----------
        svd_rank : positive integer, optional
            Rank truncation of singular value decomposition X'. The default is
            0, which uses the optimized SVD truncation by Gavish & Donoho 
            (2016).

        '''
        
        self.svd_rank = svd_rank
        self.reduced_matrices = reduced_matrices
        
    def fit(self, X):
        '''
        
        Parameters
        ----------
        X : numpy array or pandas dataframe
            Rows as time snapshots and columns as state variables.
            
        '''

        X = check_array(X).T
        model = dmd(svd_rank=self.svd_rank)
        model.fit(X)
        
        self.modes = model.modes
        self.eigs = model.eigs
        self.basis = model._svd_modes
        self.reconstructed_data = model.reconstructed_data.real.T
        self.Atilde = model.operator.as_numpy_array
        self.A = model._svd_modes @ self.Atilde @ model._svd_modes.conj().T
        self.is_fitted_ = True
        
        return self
        
    def predict(self, X, m_predict=1):
        '''

        Parameters
        ----------
        X : numpy array or pandas dataframe
            Rows as time snapshots and columns as state variables.
        m_predict : integer
            How many time steps to predict. Default is 1.

        Returns
        -------
        X_pred : numpy array or pandas dataframe
            Rows as time snapshots and columns as input variables.

        '''
        
        check_is_fitted(self)
        is_pandas, feature_names, dates = check_pandas(X)
        X = check_array(X, ensure_2d=False).T
        
        if self.reduced_matrices:
            X = self.basis.conj().T @ X
            A = self.Atilde
        else:
            A = self.A
        
        try: n, m = X.shape
        except: n, m = len(X), 1
        
        # if only an initial condition for the state is given, perform multi-step prediction
        if m == 1:
            
            X_pred = np.empty((n, m_predict))
            X_pred[:,0] = X
            
            for k in range(m_predict-1):
                X_pred[:,k+1] = (A @ X_pred[:,k]).real
            
        else: # perform single-step prediction for each state snapshot
            X_pred = (A @ X).real
        
        if self.reduced_matrices:
            X_pred = (self.basis @ X_pred).T
        else:
            X_pred = X_pred.T
        
        if is_pandas:
            X_pred = pd.DataFrame(X_pred, columns=feature_names, index=dates)
        
        return X_pred


class DMDc():
    
    def __init__(self, svd_rank=0, svd_rank_omega=0, reduced_matrices=False):
        '''

        Parameters
        ----------
        svd_rank : positive integer, optional
            Rank truncation of singular value decomposition X'. The default is
            0, which uses the optimized SVD truncation by Gavish & Donoho 
            (2016).
        svd_rank_omega : positive integer, optional
            Rank truncation of singular value decomposition of concatenated
            data matrix Omega. The default is 0.

        '''
        
        self.svd_rank = svd_rank
        self.svd_rank_omega = svd_rank_omega
        self.reduced_matrices = reduced_matrices
        
    def fit(self, X, I):
        '''
        
        Parameters
        ----------
        X : numpy array or pandas dataframe
            Rows as time snapshots and columns as state variables.
        I : numpy array or pandas dataframe
            Rows as time snapshots and columns as input variables.
            
        '''
        
        X = check_array(X).T
        I = check_array(I).T
        model = dmdc(svd_rank=self.svd_rank, svd_rank_omega=self.svd_rank_omega)
        model.fit(X, I[:,:-1])

        self.modes = model.modes
        self.eigs = model.eigs
        self.basis = model.basis
        self.reconstructed_data = model.reconstructed_data
        self.Atilde = model.operator.as_numpy_array
        self.A = model.basis @ self.Atilde @ model.basis.conj().T
        self.B = model.B
        self.Btilde = np.linalg.pinv(self.basis) @ self.B
        self.is_fitted_ = True
        
        return self
        
    def predict(self, X, I):
        '''

        Parameters
        ----------
        X : numpy array or pandas dataframe
            Rows as time snapshots and columns as state variables.
        I : numpy array or pandas dataframe
            Rows as time snapshots and columns as input variables.

        Returns
        -------
        X_pred : numpy array or pandas dataframe
            Rows as time snapshots and columns as state variables.

        '''
        
        check_is_fitted(self)
        is_pandas, feature_names, dates = check_pandas(X, I=I)
        X = check_array(X, ensure_2d=False).T
        I = check_array(I).T
        
        if self.reduced_matrices:
            X = self.basis.conj().T @ X
            A = self.Atilde
            B = self.Btilde
        else:
            A = self.A
            B = self.B
        
        try: n, m_state = X.shape
        except: n, m_state = len(X), 1
        m_input = I.shape[1]
        
        # if only an initial condition for the state is given, perform multi-step prediction
        if m_state == 1:
            
            X_pred = np.empty((n, m_input))
            X_pred[:,0] = X#.squeeze()
            
            for k in range(m_input-1):
                X_pred[:,k+1] = (A @ X_pred[:,k] + B @ I[:,k]).real
            
        else: # perform single-step prediction for each state snapshot
            X_pred = (A @ X + B @ I).real
        
        if self.reduced_matrices:
            X_pred = (self.basis @ X_pred).T
        else:
            X_pred = X_pred.T
        
        if is_pandas:
            X_pred = pd.DataFrame(X_pred, columns=feature_names, index=dates)
            
        return X_pred



