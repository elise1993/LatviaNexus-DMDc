# %% -------------------------- HYPERPARAMETERS -------------------------- %% #
from initialize import *

# policy dataset to train/validate and test on
# dataset options: P0, P3/4, P5, P11/12, P19, test
policy = "P4"


# options (True/False)
use_filter = True
optimize_feature_selection = True
descending_feature_selection = True
reduced_dataset = False
weighted_accuracy = False
reduced_matrices = False


# hyperparameters
n_ensemble = 5
n_folds = 3
k_best = 100
penalty = 0
n_trials = 5
interpolate = None # 'D' (daily), 'ME' (monthly), None (annual)
interpolate_method = 'linear' # 'linear', 'cubic', 'akima', etc.
score_func = 'NSE' # 'NSE', 'RMSE', 'MAE', 'MAPE'
nc = 15 # 5 or 15 or 30 (for P19 only)


# where to split training and testing data
date_split = '2019-12-31'

 
# DMDc svd-ranks to try out
# (r_min has to be 1 if descending_feature_selection=False) (rtilde should >= r)
r_min, r_max = 5,15
rtilde_min, rtilde_max = r_min, r_max


# %% ------------------------- DATA-PREPROCESSING ------------------------ %% #


# import data
if policy == "test": file = "test.csv"
else: file = "latvia_sdm_policy" + policy[1:] + ".csv"
data = pd.read_csv("./data/"+file, header=0)
all_feature_names = list(data.columns)
m, n_total = data.shape


# reverse data
# data = data.iloc[::-1,:]


# the control variables for each policy are specified in a separate file:
control_names, important_names, geo_vars, all_feature_names = specify_control_vars(
    policy, reduced_dataset, all_feature_names, nc)


# duplicate variables in dataset can cause issues
if set([name for name in all_feature_names if all_feature_names.count(name) > 1]):
    raise ValueError('Duplicates found in feature name list')


# add date ranges to dataset
if policy=="test":
    from dateutil.relativedelta import relativedelta
    start_date = datetime(2000,1,1)
    end_date = start_date + relativedelta(years=m)
    dates = pd.date_range(start_date, end_date, freq='YE')
else:
    start_date= datetime(2000,1,1)
    end_date = datetime(2051,1,1)
    dates = pd.date_range(start_date, end_date, freq='ME')
data.insert(0, "date", dates)


data = data.drop(columns="Months")
data = data.set_index('date')



# SDM outputs data at monthly resolution constant-interpolated to monthy. So
# here we resample it back to annual data
if policy != "test":
    data = data.resample('YE').first()


# interpolate data
if interpolate:
    data = data.resample(interpolate).interpolate(interpolate_method)


# split into training and testing periods
dates = data.index
split_index = np.searchsorted(dates, date_split, side="left")
data_train_val = data.iloc[:split_index,:]
data_test = data.iloc[split_index-1:,:]

# if testing entire period:
# data_test = data

dates_train = data_train_val.index
dates_test = data_test.index


# normalize data
scaler = MinMaxScaler().set_output(transform='pandas')
data_train_val_scaled = scaler.fit_transform(data_train_val)
data_test_scaled = scaler.transform(data_test)

dates_train = data_train_val.index
dates_test = data_test.index


# data sizes (number of snapshots (m) and features (n))
m_train, n = data_train_val.shape
m_test = data_test.shape[0]
ni = len(important_names)
nc = len(control_names)


# training, validation, and test splits (k-fold sampling)
kf = KFold(n_splits=n_folds, shuffle=False)
folds = list(kf.split(data_train_val_scaled))


# %% ---------- FEATURE SELECTION & HYPERPARAMETER OPTIMIZATION ---------- %% #


# apply filter (filter out variables with low mutual info with control variables)
feature_names_filter, k_best = apply_filter(use_filter,
                                            k_best,
                                            data_train_val,
                                            control_names,
                                            important_names,
                                            geo_vars)
n = len(feature_names_filter)



def objective_function(trial):
    # this function has a lot of global dependencies (fix?)
    '''
    Objective function to optimize (mininize) using Optuna. The objective is to
    maximize the model accuracy (NSE) and minimize the number of features used
    using the cross-validation folds defined above.

    Parameters
    ----------
    trial : n/a (Optuna-specific)

    Returns
    -------
    loss : float
        Value of loss function.

    '''
    
    
    # optimize feature selection
    if optimize_feature_selection:
    
        selected_features = []
        for name in list(feature_names_filter):
            if name in control_names + important_names + geo_vars:
                selected = [True]
            else:
                selected = [True, False]
                
            selected_features.append(trial.suggest_categorical(name, selected))
            
        selected_feature_names = [name for name, selected in
                                  zip(feature_names_filter, selected_features)
                                  if selected and not(name in control_names)]
    else:
        selected_feature_names = feature_names_filter
    
    
    # penalize large number of features
    num_features = len(selected_feature_names)
    total_penalty = penalty * num_features / n
    loss = 0
    
    if r_max==0 and rtilde_max==0:
        r = 0
        rtilde = 0
    else:
        # optimize SVD truncation for DMD model
        rmin = max(r_min, 1)
        rmax = min(r_max, num_features)
        rtildemin = max(rtilde_min, 1)
        rtildemax = min(rtilde_max, num_features)
        
        r = trial.suggest_int('r', rmin, rmax)
        rtilde = trial.suggest_int('rtilde', r, rtildemax) # force rtilde to be larger than r
    
    
    # train and validate model with the selected features and hyperparameters
    # on training/validation folds
    for i,(train_idx, val_idx) in enumerate(folds):
        
        # state and input data
        X_train = data_train_val_scaled[selected_feature_names].iloc[train_idx,:]
        X_val = data_train_val_scaled[selected_feature_names].iloc[val_idx,:]
        X_val0 = X_val.iloc[0,:]
        
        if nc > 0:
            I_train = data_train_val_scaled[control_names].iloc[train_idx,:]
            I_val = data_train_val_scaled[control_names].iloc[val_idx,:]
        
            # define model, fit to data, and predict
            model = DMDc(svd_rank=r, svd_rank_omega=rtilde, reduced_matrices=reduced_matrices)
            model.fit(X_train, I_train)
            X_val_pred = model.predict(X_val0, I_val)
        else:
            model = DMD(svd_rank=r, reduced_matrices=reduced_matrices)
            model.fit(X_train)
            X_val_pred = model.predict(X_val0, m_predict=m)
        
        
        # compute loss
        if weighted_accuracy:
            weights = np.insert(np.abs(np.diff(I_val, axis=0)), 0, np.ones((1, I_val.shape[1])), axis=0)
        else:
            weights = None
        
        loss -= loss_func(X_val_pred, X_val, score_func, weights=weights)
    
    
    loss /= n_folds
    loss += total_penalty
    
    return loss


# arrays containing ensemble forecasts of important variables and parameters
pred_train = np.zeros((m_train, ni, n_ensemble))
pred_test = np.zeros((m_test, ni, n_ensemble))
pred_bench = np.zeros((m_test, ni, n_ensemble))
r = np.empty(n_ensemble, dtype=int)
rtilde = np.empty(n_ensemble, dtype=int)
n_opt = np.empty(n_ensemble, dtype=int)
perf_train_important = np.zeros((ni, n_ensemble))
perf_test_important = np.zeros((ni, n_ensemble))
perf_train_mean = np.zeros(n_ensemble)
perf_test_mean = np.zeros(n_ensemble)
eigvec_important = np.empty((ni, n_ensemble))
feature_names_opt = []


# run optimization multiple times for ensemble
for ensemble in range(n_ensemble):
    print(f"Ensemble {ensemble+1}/{n_ensemble}")
    
    # sampler
    sampler = TPESampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    # starting features
    if not optimize_feature_selection or not descending_feature_selection:
        default_features = {ft: False for ft in feature_names_filter}
        for name in important_names + control_names: default_features[name] = True
    else:
        default_features = {ft: True for ft in feature_names_filter}
    
    # optimize
    study.enqueue_trial(default_features)
    study.optimize(objective_function, n_trials=n_trials,
                    show_progress_bar=True, n_jobs=-1)
    
    
    # best features and hyperparameters
    if r_max==0 and rtilde_max==0:
        r[ensemble] = 0
        rtilde[ensemble] = 0
        included_list = list(study.best_params.values())[:]
    else:
        r[ensemble] = study.best_params['r']
        rtilde[ensemble] = study.best_params['rtilde']
        included_list = list(study.best_params.values())[:-2]
    
    if not included_list: included_list = [True for i in range(n)]
    
    i_optim = [i for i, included in enumerate(included_list) if included]
    feature_names_opt.append([feature_names_filter[i] for i in i_optim])
    n_opt[ensemble] = len(feature_names_opt[ensemble])
    
# %% ---------------------- TRAIN MODELS & PREDICT ----------------------- %% #
     
    
    # update scaler and split into state and control variables
    X_scaler = MinMaxScaler().set_output(transform='pandas')
    I_scaler = MinMaxScaler().set_output(transform='pandas')
    
    X_train = X_scaler.fit_transform(data_train_val[feature_names_opt[ensemble]].drop(columns=control_names))
    X_test = X_scaler.transform(data_test[feature_names_opt[ensemble]].drop(columns=control_names))
    X_train0 = X_train.iloc[0,:]
    X_test0 = X_test.iloc[0,:]
    
    
    if nc > 0:
        
        I_train = I_scaler.fit_transform(data_train_val[control_names])
        I_test = I_scaler.transform(data_test[control_names])
        
        # train DMDc and DMD models on best features/hyperparameters
        model = DMDc(svd_rank=r[ensemble].item(),
                                                      svd_rank_omega=rtilde[ensemble].item(),
                                                      reduced_matrices=reduced_matrices)
        model = model.fit(X_train, I_train)
        
        # train/test predictions 
        X_train_pred = model.predict(X_train0, I_train)
        X_test_pred = model.predict(X_test0, I_test)
        
    else:
        
        model = DMD(svd_rank=r[ensemble].item(), reduced_matrices=reduced_matrices)
        model = model.fit(X_train)
        X_train_pred = model.predict(X_train0, m_predict=m_train)
        # X_train_pred = model.reconstructed_data
        X_test_pred = model.predict(X_test0, m_predict=m_test)
        
    
    # benchmark model
    model_benchmark = DMD(svd_rank=r[ensemble].item(), reduced_matrices=reduced_matrices)
    model_benchmark.fit(X_test)
    X_test_bench = model_benchmark.predict(X_test0, m_predict=m_test)
    
    
    # re-scale
    col_names = [name for name in feature_names_opt[ensemble] if name not in control_names]
    X_train_pred = pd.DataFrame(X_scaler.inverse_transform(X_train_pred), columns=col_names)
    X_test_pred = pd.DataFrame(X_scaler.inverse_transform(X_test_pred), columns=col_names)
    X_test_bench = pd.DataFrame(X_scaler.inverse_transform(X_test_bench), columns=col_names)
    X_train = pd.DataFrame(X_scaler.inverse_transform(X_train), columns=col_names)
    X_test = pd.DataFrame(X_scaler.inverse_transform(X_test), columns=col_names)
    
    
    if nc > 0:
        I_train = pd.DataFrame(I_scaler.inverse_transform(I_train), columns=control_names)
        I_test = pd.DataFrame(I_scaler.inverse_transform(I_test), columns=control_names)
    
    
    # performance
    perf_train = NSE(true=X_train, pred=X_train_pred)
    perf_test = NSE(true=X_test, pred=X_test_pred)
    
    
    # ensemble forecasts of important variables
    pred_train[:,:,ensemble] = X_train_pred[important_names]
    pred_test[:,:,ensemble] = X_test_pred[important_names]
    pred_bench[:,:,ensemble] = X_test_bench[important_names]
    
    perf_train_important[:,ensemble] = perf_train[important_names]
    perf_test_important[:,ensemble] = perf_test[important_names]
    
    # mean
    # perf_train_mean[ensemble] = np.ma.masked_invalid(perf_train).mean()
    # perf_test_mean[ensemble] = np.ma.masked_invalid(perf_test).mean()
    
    # median
    perf_train_mean[ensemble] = np.median(np.ma.masked_invalid(perf_train).filled(0))
    perf_test_mean[ensemble] =  np.median(np.ma.masked_invalid(perf_test).filled(0))
    

# %% ----------------------------- PLOTTING ------------------------------ %% #

visualize_predictions('P0', policy,
                      interpolate, interpolate_method, 
                      nc, control_names, important_names,
                      dates_train, dates_test, 
                      I_train, I_test, X_test, X_train, 
                      pred_test,pred_train, m_train, m_test,
                      perf_train_important, perf_test_important, 
                      n_ensemble, k_best, penalty, n_trials, n_folds, 
                      score_func, r, rtilde, n_opt,
                      perf_train_mean, perf_test_mean)


# visualize_eigs('P0', policy, 
#                perf_test_mean, r, rtilde, 
#                reduced_matrices, 
#                X_train, I_train, X_test, I_test)


# visualize_matrices(model_benchmark, model)


# plot_maps(model, geo_vars, X_train, feature_names_opt)











