# data processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from utils.data_loader import data_loader
from utils.names_dict import names_dict
from utils.specify_policies import specify_policies

# models
from pydmd import DMDc
from utils.helper_functions import *
from eigenshuffle import eigenshuffle_eig

# parameter optimization
import optuna
from optuna.samplers import TPESampler, GridSampler
# from utils.optimal_SVHT_coef import optimal_SVHT_coef

# visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from string import ascii_lowercase
from matplotlib.colors import TwoSlopeNorm

# misc
import re
import warnings
warnings.filterwarnings("ignore")
# optuna.logging.set_verbosity(optuna.logging.ERROR)

# define variable names/units (see /utils/names_dict.py)
D, d = names_dict()
locals().update(d)
policy_dict = {'P0':'Policy 0', 'P3': 'Policy A',
               'P12': 'Policy B', 'P19': 'Policy C'}

#%% hyperparameters

# policies to train and test on (policies: 0,3,4,11,12,14,18,19)
train = 'P0'
test = 'P19'

# define policy states/inputs (see /utils/specify_policies.py)
nc = 15                          # no. of control-input variables
input_names, state_names = specify_policies(train, test, nc,
                                            large_data=True)

# load data and initial model parameters from data_loader.py
# X = (m samples x n variables)
interpolate = True
X_train_val, X_test, t = data_loader(train=train, test=test, interpolate=interpolate)
train_crop = slice(0, -1, 1)   # crop train/test data
test_crop = slice(0, -1, 1)

# parameters
n = len(state_names)                    # no. of state variables (n)
stackmin, stackmax = 1, 70              # no. of stack-shifted copies to test
rmin, rmax = 1, 20                      # input svd-ranks to test (rmax ~ n*stackmin)
rtildemin, rtildemax = rmin, rmax + 5   # output svd-ranks to test (rtildemin not used)

# optimization/validation
n_folds = 2                     # no. of cross-val folds (decrease if stackmax is too high)
n_trials = 1000                  # no. of trials with Optuna
n_cpu = -1                      # no. of CPUs to use. default: -1 (use all available cores)
random_seed = False             # note that running Optuna in parallel mode (n_cpu>1) is non-deterministic
loss_func = 'R2'                # NSE, NMSE, RMSE, R2, MAE, MAPE (note that Sklearn's R2_score and the NSE are identical)
shuffle = False                 # shuffle pairs of snapshots for cross-validation (does not support forecast validation) [deprecated]
sampler = 'grid'                # grid or tpes. grid is fine for small parameter sets or large n_trials.
                                # use tpes (Tree-Structured Parzen Estimator) for large parameter sets or small n_trials.

# sensitivity analysis
n_best = 10                     # ensemble size (model sensitivity)
std_factor = 0                  # systematic error in control input (fraction of std) after 2020
n_ensemble = 1                  # ensemble size (input sensitivity)

# plotting
visualize_forecasts = True
visualize_S = False
visualize_U = False
visualize_US = False
visualize_AB = False
visualize_eigs = False

fontsize_small = 7
fontsize_large = 9

# plot colors
if train != 'P0':
    red, blue = "#ff5d5d", "#4e95d9"
else:
    blue, red = "#ff5d5d", "#4e95d9"

# manually test parameters
# stackmax = 18
# stackmin = stackmax
# rmax = 6
# rtildemax = rmax
# rtildemin = rtildemax
# rmin = rmax
# n_trials = 1


#%% pre-process data

# data sizes
X_train_val = X_train_val.iloc[train_crop, :].copy()
X_test = X_test.iloc[test_crop, :].copy()
t_train_val = t[train_crop]
t_test = t[test_crop]

m_train_val = len(X_train_val)
m_test = len(X_test)
n = len(state_names)


# k-fold sampling: training and validation
if n_folds>1:
    
    if shuffle: # shuffle pairs of snapshots and cross-validate
        kf = KFold(n_splits=n_folds, shuffle=True)
        try: idx = np.arange(0, m_train_val).reshape(-1, 2)
        except: idx = np.arange(0, m_train_val-1).reshape(-1, 2)
        folds = []
        for fold_train, fold_val in kf.split(idx):
            folds.append((idx[fold_train].reshape(-1), idx[fold_val].reshape(-1)))
            
    else: # non-shuffled cross-validation
        kf = KFold(n_splits=n_folds, shuffle=False)
        folds = list(kf.split(X_train_val))
        
else: # no cross-validation: split train-val data 50-50
    folds = [train_test_split(np.arange(0, len(X_train_val)),
                              test_size=0.5, shuffle=False)]


# split data into state (X) and control-input (I) datasets
I_train_val = X_train_val[input_names].copy()
X_train_val = X_train_val[state_names].copy()

I_test = X_test[input_names].copy()
X_test = X_test[state_names].copy()


# normalize/scale data
state_scaler = MinMaxScaler().set_output(transform='pandas')
input_scaler = MinMaxScaler().set_output(transform='pandas')

X_train_val = state_scaler.fit_transform(X_train_val)
X_test = state_scaler.transform(X_test)

I_train_val = input_scaler.fit_transform(I_train_val)
I_test = input_scaler.transform(I_test)


# add systematic error to input variable after 2020 (sensitivity test)
if interpolate: k_2020 = 239 # index of 2020/01 (default: 239)
else: k_2020 = 20            # index of 2020 (default: 20)

std_range = np.linspace(-std_factor, std_factor, n_ensemble)
I_sensitivity = np.tile(I_test, (n_ensemble, 1, 1))
I_sensitivity[:, k_2020:, :] += std_range.reshape((n_ensemble, 1, 1)) * \
    I_test.iloc[k_2020:, :].std().to_numpy().reshape(1, 1, -1)


#%% model functions

# convert from regular coordinates (X) to delay-coordinates (V)
def delay_embedding(X, q, n):
    
    # construct Hankel matrix
    H = np.zeros((q*n, len(X)-q))
    for i in range(n):
        x = X.iloc[:,i]
        H[q*i : q*(i+1), :] = hankel_matrix(x, q)
    
    # convert to delay coordinates
    U, S, V = hankel_svd(H)
    
    return H, U, S, V

def truncate_svd(U, S, r):
    Ur = U[:, :r]
    Sr = S[:r, :r]
    
    US = Ur @ Sr
    invS_invU = np.linalg.inv(S) @ np.linalg.pinv(U)
    
    return US, invS_invU

def construct_model(V, I, q, r, rtilde):
    if r > n*q: raise ValueError("r is too large")
    
    model = DMDc(svd_rank=r, svd_rank_omega=rtilde, opt=True, tlsq_rank=0)
    model.fit(V[:, :r].T, I.iloc[:-q-1, :].T)
    
    A = model.basis @ model.operator.as_numpy_array @ model.basis.conj().T
    B = model.B
    
    return A, B


def forecast(A, B, US, r, q, v0, u0, m, u=None, show_progress=False, multi_step=True):
    
    if u is None:
        u = np.full((m-q-1, nc), np.nan)
        u[0] = u0
    
    v = np.zeros((r, m-q))
    v[:, 0] = v0
    
    if multi_step:
        
        k=0
        while k < m-q-1:
            
            v0 = v[:, k]
            h = US @ v0
            u0 = u[k]
    
            v[:, k+1] = A @ v0 + B @ u0
            
            u[k] = u0
    
            k += 1
            
            if show_progress and (k%100==0 or k==m):
                print(f"step: {k}/{m}")
            
    else:
        v = A @ V_val[:, :r].T + B @ I_val.to_numpy().T[:, :-q]

    # convert back from delay coordinates
    h = US @ v
    X_forecast = average_hankel(h.T, num_vars=n, q=q, method='diagonal')

    return X_forecast, u


#%% train/validate model

# find optimal model parameters (q, r, rtilde) using optuna
def objective_function(trial):
    
    q = trial.suggest_int('q', stackmin, stackmax)
    r = trial.suggest_int('r', rmin, rmax)
    rtilde = trial.suggest_int('rtilde', r, rtildemax) # force rtilde to be larger than r
    
    # cross-validate
    loss = 0
    for train_idx, val_idx in folds:
        
        # only validate on training data (disable validation)
        # train_idx = np.arange(0, len(train_idx) + len(val_idx) - 2)
        # val_idx = train_idx
        
        # training and validation folds
        X_train = X_train_val.iloc[train_idx, :]
        I_train = I_train_val.iloc[train_idx, :]
        
        X_val = X_train_val.iloc[val_idx, :]
        I_val = I_train_val.iloc[val_idx, :]
        
        m_train = len(I_train)
        m_val = len(I_val)
        
        # transform data and truncate to rank-r
        _, U, S, V_train = delay_embedding(X_train, q, n)
        H_val, _, _, _ = delay_embedding(X_val, q, n)
        
        US, invS_invU = truncate_svd(U, S, r)
        V_val = (invS_invU @ H_val).T
        
        # don't include accuracy of transformation (can help for small validation data)
        # _, U, S, V_train_val = delay_embedding(X_train_val, q, n)
        # V_train = V_train_val[train_idx[:-q], :]
        # V_val = V_train_val[val_idx[:-q], :]
        
        # initial conditions
        v0_val = V_val[0, :r]
        u0_val = I_val.iloc[0, :].to_numpy()
        
        # train model on training set and forecast on validation set
        try:
            A, B = construct_model(V_train, I_train, q, r, rtilde)
        
            X_val_forecast, _ = forecast(A, B, US, r, q,
                                         v0_val, u0_val,
                                         m_val, u=I_val.to_numpy(),
                                         multi_step=not shuffle)
            
            # performance
            loss += loss_fnc(X_val.iloc[:-1], X_val_forecast)
            
        except: loss += np.nan
    
    loss /= n_folds
    return loss


# loss function, sampler, and optimizer 
loss_fnc, direction = choose_loss(loss_func)

if sampler == 'grid':
    sampler = GridSampler(search_space={"r": range(rmin, rmax+1),
                                        "rtilde": range(rtildemin, rtildemax+1),
                                        "q": range(stackmin, stackmax+1)})
elif sampler == 'tpes':
    sampler = TPESampler(seed=random_seed)


study = optuna.create_study(direction=direction, sampler=sampler)
study.optimize(objective_function, n_trials=n_trials,
                show_progress_bar=True, n_jobs=n_cpu)


# to do multi-objective Optuna study:
# def objective_function(trial): ... return metric1, metric2, ...
# optuna.create_study(directions=["maximize", "minimize", ...])


# identify top n_best trials
loss, params = [], []
for trial in study.trials:
    l = trial.values
    loss += [l if isinstance(l, list) else []] # sometimes loss is "None"
    params += [trial.params]



reverse = True if direction=='maximize' else False
best_params = [param for _, param in 
               sorted(zip(loss, params),
                      key=lambda x: x[0], reverse=reverse)][0:n_best]


q_best, r_best, rtilde_best = [], [], []
for i in range(n_best):
    q_best += [best_params[i]['q']]
    r_best += [best_params[i]['r']]
    rtilde_best += [best_params[i]['rtilde']]


# pick best trial
# locals().update(study.best_params)
# print(f" optimal params: {study.best_params}\n",
#       f"Average validation {loss_func} performance for best params: {study.best_value:.02f}")

#%% testing

# ensemble arrays
X_train_val_forecast = np.full((n_best, m_train_val-1, n), np.nan)
X_test_forecast = np.full((n_best, m_train_val-1, n), np.nan)
X_test_forecast_neg = np.full((n_best, m_train_val-1, n), np.nan)
X_test_forecast_pos = np.full((n_best, m_train_val-1, n), np.nan)


for j, (q, r, rtilde) in enumerate(zip(q_best, r_best, rtilde_best)):
    
    try:
    
        # train DMDc model on entire train_val dataset with the optimal parameters
        H_train, U, S, V_train_val = delay_embedding(X_train_val, q, n)
        H_test, _, _, _ = delay_embedding(X_test, q, n)
        US, invS_invU = truncate_svd(U, S, r)
        A, B = construct_model(V_train_val, I_train_val, q, r, rtilde)
        V_test   = (invS_invU @ H_test).T
        
        # initial conditions
        v0_train_val = V_train_val[0, :r]
        u0_train_val = I_train_val.iloc[0, :].to_numpy()
        v0_test = V_test[k_2020, :r]
        u0_test = I_test.iloc[k_2020, :].to_numpy()
        v0_test = v0_train_val
        
        
        # forecast
        x_train_val_forecast, _ = forecast(A, B, US, r, q, v0_train_val, u0_train_val, m_train_val, u=I_train_val.to_numpy())
        x_test_forecast, _ = forecast(A, B, US, r, q, v0_test, u0_test, m_test, u=I_test.to_numpy())
    
    
        # # sensitivity forecast on test data
        # X_sensitivity_forecast = np.tile(X_test_forecast, (n_ensemble, 1, 1))
        # for i in range(n_ensemble):
        #     X_sensitivity_forecast[i, :, :], _ = forecast(A, B, US, r, q, v0_test, u0_test, m_test, u=I_sensitivity[i, :, :])
        
        
        # x_test_forecast = X_sensitivity_forecast.mean(axis=0)
        # x_test_forecast_neg = x_test_forecast.copy()
        # x_test_forecast_pos = x_test_forecast.copy()
        # STD = X_sensitivity_forecast[:, k_2020:, :].std(axis=0)
        # x_test_forecast_neg[k_2020:, :] = x_test_forecast[k_2020:, :] - STD
        # x_test_forecast_pos[k_2020:, :] = x_test_forecast[k_2020:, :] + STD
        
        
        # j:th ensemble
        X_train_val_forecast[j, :, :] = x_train_val_forecast
        X_test_forecast[j, :, :] = x_test_forecast
        # X_test_forecast_neg[j, :, :] = x_test_forecast_neg
        # X_test_forecast_pos[j, :, :] = x_test_forecast_pos
        
    except: pass
        


# ensemble average, upper, and lower STD
STD_train_val = np.nanstd(X_train_val_forecast, axis=0)
STD_test = np.nanstd(X_test_forecast, axis=0)

X_test_forecast = np.nanmean(X_test_forecast, axis=0)
X_train_val_forecast = np.nanmean(X_train_val_forecast, axis=0)

X_test_forecast_neg = X_test_forecast.copy() - STD_test
X_test_forecast_pos = X_test_forecast.copy() + STD_test


# re-scale
# X_train_val_forecast = state_scaler.inverse_transform(X_train_val_forecast)
# X_test_forecast = state_scaler.inverse_transform(X_test_forecast)
# X_test_forecast_neg = state_scaler.inverse_transform(X_test_forecast_neg)
# X_test_forecast_pos = state_scaler.inverse_transform(X_test_forecast_pos)
# I_train_val = input_scaler.inverse_transform(I_train_val)
# X_train_val = state_scaler.inverse_transform(X_train_val)
# I_test = input_scaler.inverse_transform(I_test)
# X_test = state_scaler.inverse_transform(X_test)


# convert to pandas dataframe
X_train_val_forecast = pd.DataFrame(X_train_val_forecast, columns=state_names, index=t_train_val[1:])
X_test_forecast = pd.DataFrame(X_test_forecast, columns=state_names, index=t_test[1:])
X_test_forecast_neg = pd.DataFrame(X_test_forecast_neg, columns=state_names, index=t_test[1:])
X_test_forecast_pos = pd.DataFrame(X_test_forecast_pos, columns=state_names, index=t_test[1:])

I_train_val = pd.DataFrame(I_train_val, columns=input_names, index=t_train_val)
X_train_val = pd.DataFrame(X_train_val, columns=state_names, index=t_train_val)

I_test = pd.DataFrame(I_test, columns=input_names, index=t_test)
X_test = pd.DataFrame(X_test, columns=state_names, index=t_test)


# accuracy
nse_train = nash_sutcliffe_efficiency(X_train_val[1:], X_train_val_forecast)
nse_test = nash_sutcliffe_efficiency(X_test[1:], X_test_forecast)


# best params
q, r, rtilde = q_best[0], r_best[0], rtilde_best[0]


#%% plot forecasts

if visualize_forecasts:
    symbols = r"\[([A-Za-z0-9\/_\s-]+)\]"
    j, row, n_cols, linewidth = 1, 0, 5, 0.6
    max_rows = min(n+nc, 6*5) # display a maximum of 6 rows
    fig, axes = plt.subplots(max_rows//n_cols, n_cols, sharex='col', sharey='row', figsize=(8, 10), dpi=250)
    axes = axes.flatten() 
    for i, ax in enumerate(axes):
        if i<nc:
            ax.plot(I_train_val.iloc[:, i], '-.', color=red, linewidth=linewidth, label="Train/Val (R1-R5)")
            ax.plot(I_test.iloc[:, i], '--', color=blue, linewidth=linewidth,  label="Test (R1-R5)")
            
            # uncertainty in input after 2020
            lower_bound = I_test.iloc[:, i].copy(deep=True).to_numpy()
            upper_bound = lower_bound.copy()
            STD = lower_bound[k_2020:].std()*std_factor
            lower_bound[k_2020:] -= STD
            upper_bound[k_2020:] += STD
            # ax.fill_between(I_test.index, lower_bound, upper_bound, alpha=0.2, color=blue, edgecolor='none')
            
            if i<n_cols:
                title = f"$\\bf{{R{j}}}$ \n"
            else:
                title = ""
            if j==3:
                title = f"$\\bf{{{ascii_lowercase[row]})}}$ " + D[input_names[i]].replace('(R3) ','')
                row += 1
                if i<n_cols:
                    title = f"$\\bf{{R{j}}}$" + "\n" + title
            elif j==1:
                unit = re.search(symbols, D[input_names[i]]).group(0)
                ax.set_ylabel(unit, fontsize=fontsize_small)
            title = re.sub("([\\[]).*?([\\]])", "", title)
            ax.set_title(title, fontsize=fontsize_large)
            
        else:
            ax.plot(X_test.iloc[:, i-nc], '--', color=blue, linewidth=linewidth, label="True")
            ax.plot(X_test_forecast.iloc[:, i-nc], 's-', color=blue, linewidth=linewidth, label="DMDc", markevery=m_test//8, markersize=1.5)
            ax.fill_between(X_test_forecast.index, X_test_forecast_neg.iloc[:, i-nc], X_test_forecast_pos.iloc[:, i-nc], alpha=0.2, color=blue,
                            edgecolor='none', label=f'model sensitivity \n ({n_best} best param sets)')
            # ax.plot(X_test_forecast.iloc[:, i-nc], 's-', color=blue, linewidth=linewidth, label="DMDc", markevery=m_test//8, markersize=1.5)
            
            ax.plot(X_train_val.iloc[:, i-nc], '-.', color=red, linewidth=linewidth, label="True")
            ax.plot(X_train_val_forecast.iloc[:, i-nc], 's-', color=red, linewidth=linewidth, label="DMDc", markevery=m_test//8, markersize=1.5)
            # title = f"R{j}"
            if j==3:
                title = f"$\\bf{{{ascii_lowercase[row]})}}$ " + D[state_names[i-nc]].replace('(R3) ','') #+ "\n" + title
                title = re.sub("([\\[]).*?([\\]])", "", title)
                # if i==len(axes)-3:
                #     title = f"$\\bf{{{ascii_lowercase[row]})}}$ " + "Totals"
                ax.set_title(title, fontsize=fontsize_large)
                row += 1
            elif j==1:
                unit = re.search(symbols, D[state_names[i-nc]]).group(0)
                ax.set_ylabel(unit, fontsize=fontsize_small)
        if j%n_cols==0:
            j = 1
        else:
            j += 1
        
        ax.tick_params(labelsize=fontsize_small, rotation=60)
        ax.tick_params(axis="y", direction="in", pad=2, rotation=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.get_offset_text().set_size(fontsize_small)
        # ax.set_yscale('log')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_box_aspect(aspect=1)
        
    
    
    ax.legend(bbox_to_anchor=(-1.2, -0.42), ncols=2, fancybox=True,
                    title=f"Test ({policy_dict[test]})    Train/Val ({policy_dict[train]})",
                    title_fontproperties={'weight':'bold', 'size':fontsize_small}, fontsize=fontsize_large)
    
    border_text = (f"$\\bf{{Hyperparameters}}$: n-trials: {n_trials}, k-folds: {n_folds}\n$n={int(n)}$, $n_c={nc}$, "
                   f"q$\\in[{stackmin}$, {stackmax}], $r\\in[{rmin}, {rmax}]$, $\\tilde r\\in[{rmin}, {rtildemax}]$\n"
                   f"$\\bf{{Optimal \\ parameters}}$: $r={int(r)}$,  $\\tilde r={int(rtilde)}$, q={stackmax}\n"
                   f"$\\bf{{Mean \\ performance}}$: NSE$_{{train}}={nse_train.mean():.02f}$, NSE$_{{test}}={nse_test.mean():0.2f}$")
    
    ax.text(-5,-1, border_text,
                bbox=dict(edgecolor="#e4e4e4", facecolor='none'),
                transform=axes[-1].transAxes, fontsize=fontsize_large)
    
    fig.subplots_adjust(wspace=0.25, hspace=0.4)
    plt.savefig('forecast.svg', bbox_inches='tight')
    plt.show()


#%% visualize singular values

if visualize_S:
    fig, ax = plt.subplots(figsize=(3,4))
    s = np.diag(S)
    ax.plot(np.arange(1, len(s)+1), s, color=red, linewidth=3)
    ax.scatter(r, s[r], color=blue, linewidth=2, marker='x', zorder=2, label=f"$r={r}$")
    ax.scatter(rtilde, s[rtilde], color=blue, linewidth=1, marker='o', zorder=3, label=f"$\\tilde r={rtilde}$")
    plt.yscale('linear')
    plt.xscale('linear')
    ax.set_xlabel("number of terms", fontsize=fontsize_large)
    ax.set_title('Singular values in $\\bf{S}$', fontsize=fontsize_large)
    ax.grid(True)
    # ax.set_xticklabels(range(1, 10))
    plt.xticks(range(1, 21, 2), range(1, 21, 2), fontsize=fontsize_small)
    plt.xlim([0, 21])
    plt.legend()
    plt.savefig('singular_values.svg', bbox_inches='tight')
    plt.show()


#%% visualize U coordinates

if visualize_U:
    vmin = U.min()
    vmax = U.max()
    
    fig = plt.figure(dpi=300)
    gs = GridSpec(1, 2, width_ratios=[2, 0.8])
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    im = ax0.imshow(U, aspect='equal', cmap=plt.get_cmap('RdBu'), vmin=vmin, vmax=vmax)
    ax0.set_title('Left singular vectors $\\bf{U}$', fontsize=fontsize_large)
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=n//5))
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=n//5))
    ax0.tick_params(labelsize=fontsize_small)
    
    labels = [item.get_text() for item in ax0.get_yticklabels()]
    for i, name in enumerate(state_names[0:-1:5]):
        labels[i+1] = re.sub("[\\(\\[].*?[\\)\\]]", "", D[name])[:-2]
    ax0.set_yticklabels(labels)
    ax0.tick_params(axis='y', rotation=90)
    ax0.set_xlabel('eigenstates')
    
    fig.colorbar(im, ax=ax1, pad=0)
    ax1.set_axis_off()
    
    fig.savefig('visualize_U.svg', bbox_inches='tight')


    fig, axes = plt.subplots(5, 1, figsize=(3,4))
    fig.suptitle("Column vectors of $\\bf{U}$")
    j = [0, 5, 20, 100, 300]
    for i, ax in enumerate(axes):
        ax.plot(U[:, j[i]], color=red)
        ax.set_ylabel("$\\bf{U}$" + f"$_{{{j[i]}}}$")
    fig.savefig('visualize_U2.svg', bbox_inches='tight')
    
    fig, axes = plt.subplots(5, 1)
    fig.suptitle("Column vectors of $\\bf{V}$")
    for i, ax in enumerate(axes):
        ax.plot(V_test[:, i*q], color=red)
        ax.set_ylabel("$\\bf{v}$" + f"$_{{{i*q}}}$")
    fig.savefig('visualize_V.svg', bbox_inches='tight')

#%% visualize ~US (truncated delay coordinates transformation)

if visualize_US:
    vmin = US.min()
    vmax = US.max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # norm = PowerNorm(vmin=vmin, vmax=vmax, gamma=0.1)
    
    fig = plt.figure(dpi=300)
    gs = GridSpec(1, 2, width_ratios=[2, 0.8])
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    im = ax0.imshow(US, aspect='auto', cmap=plt.get_cmap('RdBu_r'), norm=norm, interpolation='none')
    ax0.set_title('Truncated delay coordinates $\\bf{ \\tilde U \\tilde S}$', fontsize=fontsize_large)
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=n//5))
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=n//5))
    ax0.tick_params(labelsize=fontsize_large)
    ax0.set_xticks(list(range(r)))
    
    labels_y = [item.get_text() for item in ax0.get_yticklabels()]
    for i, name in enumerate(state_names[0:-1:5]):
        labels_y[i+1] = re.sub("[\\(\\[].*?[\\)\\]]", "", D[name])[:-2]
    ax0.set_yticklabels(labels_y)
    
    labels_x = [item.get_text() for item in ax0.get_xticklabels()]
    for i in range(r):
        labels_x[i] = f"$r_{i+1}$"
    ax0.set_xticklabels(labels_x)
    ax0.tick_params(axis='y', rotation=90)
    ax0.tick_params(axis='x', rotation=0)
    
    fig.colorbar(im, ax=ax1, pad=0, ticks=[vmin, 0, vmax])
    ax1.set_axis_off()
    
    fig.savefig('visualize_US.svg', bbox_inches='tight')


#%% visualize A, B matrices

if visualize_AB:
    vmin = min(A.min(), B.min())
    vmax = min(A.max(), B.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    fig = plt.figure()
    # gs = GridSpec(1, 3, width_ratios=[2, 0.8, 0.67])
    gs = GridSpec(1, 3, width_ratios=[1, 3, 0.34])
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    
    im = ax0.imshow(A, aspect='equal', cmap=plt.get_cmap('RdBu'), norm=norm)
    ax0.set_title('$\\bf\\tilde{A}$', fontsize=fontsize_large) # State transition matrix 
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.tick_params(labelsize=fontsize_small)
    
    im = ax1.imshow(B, aspect='equal', cmap=plt.get_cmap('RdBu'), norm=norm)
    ax1.set_title('$\\bf{B}$', fontsize=fontsize_large) # Control-input input matrix 
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(labelsize=fontsize_small)
    
    labels = [item.get_text() for item in ax0.get_yticklabels()]
    for i in range(r):
        labels[i+1] = f"$r_{i+1}$"
    ax0.set_yticklabels(labels)
    ax0.set_xticklabels(labels)
    ax0.tick_params(axis='both', rotation=0)
    
    ax1.set_yticklabels([''])
    
    fig.colorbar(im, ax=ax2, pad=0, extend='neither', ticks=[-.2, -.1, 0, .1], norm=norm)
    ax2.set_axis_off()
    
    fig.savefig('visualize_matrix.svg', bbox_inches='tight')


#%% visualize eigenvalues

if visualize_eigs:
    
    # construct a benchmark model on the test set (for comparison)
    H_train, U, S, V_train_val = delay_embedding(X_train_val, q, n)
    H_test, _, _, _ = delay_embedding(X_test, q, n)
    US, invS_invU = truncate_svd(U, S, r)
    A, B = construct_model(V_train_val, I_train_val, q, r, rtilde)
    V_test   = (invS_invU @ H_test).T
    A_test, B_test = construct_model(V_test, I_test, q, r, rtilde)
    
    # numpy function does not produce sorted eigenvalues
    # eigs, vecs = np.linalg.eig(A)
    # eigs_test, vecs_test = np.linalg.eig(A_test)
    
    # use eigenshuffle to get sorted eigenvalues
    eigs, vecs = eigenshuffle_eig(A.reshape((1, r, r)))
    eigs_test, vecs_tru = eigenshuffle_eig(A_test.reshape((1, r, r))) # "true" eigenvalues for rank r, rtilde
    
    eigs = eigs[0]
    eigs_test = eigs_test[0]
    
    xlims = [-.1, 1.1]
    ylims = [-.1, .1]
    
    eig_names = [f"$\\lambda_{{{i+1}}}$" for i in range(r)]
    
    plt.figure(figsize=(5, 5))
    plt.grid("on")
    
    h2 = plt.scatter(eigs.real, eigs.imag, 70, c=red, marker='x')
    h1 = plt.scatter(eigs_test.real, eigs_test.imag, 70, c=blue, marker='o')
    
    unit_circle = np.exp(1j*np.linspace(0,2*np.pi,500))
    plt.plot(unit_circle.real, unit_circle.imag,'k--', dashes=(3, 5), linewidth=1)
    
    h1.AlphaData = np.zeros((r, 1))
    h2.AlphaData = np.zeros((r, 1))
    h1.MarkerFaceAlpha = 'flat'
    h2.MarkerEdgeAlpha = 'flat'
    
    maxeigs = r
    for i in range(maxeigs):
        plt.text(eigs.real[i]+0.01, eigs.imag[i]+0.003, eig_names[i], c=red)
        plt.text(eigs_test.real[i]+0.01, eigs_test.imag[i]-0.0045, eig_names[i], c=blue)
    
    fontsize=14
    
    plt.rc('axes', axisbelow=True)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel("real($\\lambda$)", fontsize=fontsize_large)
    plt.ylabel("imag($\\lambda$)", fontsize=fontsize_large)
    plt.legend([f"DMDc on {policy_dict[train]}",
                f"DMDc on {policy_dict[test]} (Baseline)", 
                "Unit Circle"], bbox_to_anchor=(0,-0.09), 
               loc="upper left",
               fontsize=fontsize_large)
    plt.title(f"Discrete Eigenvalue Spectrum of $\\bf\\tilde A$ for [$r$, $\\tilde r$]=[{r}, {rtilde}]", 
              fontsize=fontsize_large)
    
    plt.savefig("eigenvalues.svg", bbox_inches='tight')
    
    plt.show()


#%% correlation plots

# i = 1
# input_name, state_name = input_names[i], state_names[i]

# x = I_train_val[input_name].iloc[::12]
# y = X_train_val[state_name].iloc[::12]
# corr = np.corrcoef(x, y)[0, 1]

# plt.plot(x, '.')
# plt.plot(y, 'x')
# plt.show()

# plt.scatter(x, y)
# plt.xlabel(f'{D[input_name]}')
# plt.ylabel(f'{D[state_name]}')
# plt.text(.5, 0, f"Pearson corr: {corr:.2f}")
# plt.axis('equal')
# plt.show()

 