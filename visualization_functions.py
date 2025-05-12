import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
import scipy
from dmd_models import DMD, DMDc
from eigenshuffle import eigenshuffle_eig
import control as ct
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
from pydmd.plotter import plot_summary, plot_eigs
redcolor, bluecolor = "#ff5d5d", "#4e95d9"



# to identify variables that change with policy:
def visualize_policy_changes(data_train_val, data_test, name):
    dat = np.sum(data_train_val - data_test)
    plt.plot(data_test[name], linestyle='-', color=redcolor, label='without policy')
    plt.plot(data_train_val[name], linestyle='--', color=bluecolor, label='with policy')
    plt.legend()
    plt.title(name)
    plt.xlabel('time index')
    plt.show()
    pass



# visualize training-validation folds
def visualize_folds(folds, 
                    n_folds, 
                    dates, 
                    data_train_val, 
                    data_test, 
                    all_feature_names, 
                    viz):
    
    fig, axes = plt.subplots(nrows=len(viz), ncols=n_folds, figsize=(9, 5), sharex=True, sharey=True)
    for i,col in enumerate(viz):
        
        for j,(train_idx, val_idx) in enumerate(folds):
            
            fill_indices = dates[train_idx]
            # fill_indices = list(range(train_idx.min(),train_idx.max()+1))
            
            data_train_val.iloc[train_idx,col].reindex(
                fill_indices, fill_value=None).plot(
                    ax=axes[i,j], label='Training', color=bluecolor, linestyle='-', linewidth=1)
            
            fill_indices = dates[val_idx]
            # fill_indices = list(range(val_idx.min(),val_idx.max()+1))
            
            data_train_val.iloc[val_idx,col].reindex(
                fill_indices, fill_value=None).plot(
                    ax=axes[i,j], label='Validation', color=redcolor, linestyle='--', linewidth=1)
            
            axes[i,j].set_xlabel('')
            
        # axes[i,0].set_ylabel('bla')
        axes[i,n_folds//2].set_title(all_feature_names[col])
    
    axes[-1,n_folds//2].legend(bbox_to_anchor=(1.05, -0.3), ncols=2)
    fig.tight_layout()
    fig.autofmt_xdate(rotation=0)
    # plt.savefig("figure x - train_val_folds.svg")
    plt.show()
    pass



def visualize_predictions(policy_train, policy_test,
                          interpolate, interpolate_method, 
                          nc, control_names, important_names,
                          dates_train, dates_test, 
                          I_train, I_test, X_test, X_train, 
                          pred_test,pred_train, m_train, m_test,
                          perf_train_important, perf_test_important, 
                          n_ensemble, k_best, penalty, n_trials, n_folds, 
                          score_func, r, rtilde, n_opt,
                          perf_train_mean, perf_test_mean):
    
    num_folds_control = 5 # how to split control variable charts
    fontsize = 11
    
    if interpolate:
        interpolate_method2='('+interpolate_method+')'
    else:
        interpolate_method2=''
        
    if nc==5:
        rows=2
        cols=2
        ctr_panels=[0]
        figsize=(8,7)
    elif nc==15:
        rows=2
        cols=3
        ctr_panels=[0,1,2]
        figsize=(12,7)
    elif nc==30:
        rows=3
        cols=3
        ctr_panels=[0,1,2,3,4,5]
        figsize=(12,11)

    fig, axes = plt.subplots(rows,cols, figsize=figsize, sharex=True)
    axes = axes.reshape(-1)
        
    k = 0
    for ax in axes[ctr_panels]:
        for name in control_names[k:k+num_folds_control]:
            ax.plot(dates_train, I_train[name], "--", color=bluecolor, linewidth=1, label=f"Train/Val (R1-R5)")#, markersize=5, markevery=50+i*5, label=f"R{i+1}")
            ax.plot(dates_test, I_test[name], "-.", color=redcolor, linewidth=1,  label=f"Test (R1-R5)")#, markersize=5, markevery=50+i*5, label=f"R{i+1}")
        k += num_folds_control

    for i, ax in enumerate(axes[ctr_panels[-1]+1:]):
        
        # test data
        ax.plot(dates_test, X_test[important_names[i]],'-.', color=redcolor,label="True",linewidth=1)
        
        # dmdc on test data
        mean = pred_test[:,i,:].mean(axis=1)
        lower_bound = mean - pred_test[:,i,:].std(axis=1)
        upper_bound = mean + pred_test[:,i,:].std(axis=1)
        ax.fill_between(dates_test, lower_bound, upper_bound, alpha=0.2, color=redcolor, edgecolor='none')
        ax.plot(dates_test, mean,'-s', color=redcolor, label="DMDc", linewidth=1, markevery=m_test//6, markersize=4)
        
        # train data
        ax.plot(dates_train, X_train[important_names[i]],'--', color=bluecolor, label="True",linewidth=1)
        
        # dmdc train
        mean = pred_train[:,i,:].mean(axis=1)
        lower_bound = mean - pred_train[:,i,:].std(axis=1)
        upper_bound = mean + pred_train[:,i,:].std(axis=1)
        ax.fill_between(dates_train, lower_bound, upper_bound, alpha=0.4, color=bluecolor, edgecolor='none')
        ax.plot(dates_train, mean,'-v', color=bluecolor,label="DMDc", linewidth=1, markevery=m_train//5, markersize=4.5)
        
        # performance text
        ax.text(0.07, 0.72, f" NSE$_{{train}}={perf_train_important[i,:].mean():0.2f}$ \n NSE$_{{test}}={perf_test_important[i,:].mean():0.2f}$", transform=ax.transAxes, fontsize=fontsize)
        ax.set_title(important_names[i])
        
        dat = np.concatenate((X_test[important_names[i]], X_train[important_names[i]]))
        std = dat.std()
        ymin = dat.min() - 0.3*std
        ymax = dat.max() + 1*std
        ax.set_ylim([ymin, ymax])
    
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.4, hspace=0.2)
    # fig.autofmt_xdate(rotation=45, bottom=0.3, ha='center', which='major')
    
    if interpolate is None:
        interp = 'none'
    elif interpolate=='ME':
        interp = 'monthly'
    elif interpolate=='D':
        interp = 'daily'
    
    border_text = (f" $\\bf{{Hyperparameters}}$: ensemble size: {n_ensemble}, "
                   f"k-best: {k_best}, \n penalty: ${penalty:0.2f}$, trials: "
                   f"${n_trials}$, cross-val folds: {n_folds}, \n "
                   f"score function: {score_func}, interpolation: {interp} {interpolate_method2} \n $\\bf{{Model"
                   f" \ Parameters / Performance}}$ (ensemble mean): \n $r="
                   f"{int(r.mean())}$,  $\\tilde r={int(rtilde.mean())}$, $n="
                   f"{int(n_opt.mean())}$, $n_c={nc}$, NSE$_{{train}}="
                   f"{perf_train_mean.mean():.02f}$, NSE$_{{test}}="
                   f"{perf_test_mean.mean():0.2f}$ ")
    
    axes[-1].text(-2.5,-0.5, border_text,
                bbox=dict(edgecolor="#e4e4e4", facecolor='none'),
                transform=axes[-1].transAxes, fontsize=fontsize)
    
    
    axes[-1].legend(bbox_to_anchor=(0, -0.19), ncols=2, fancybox=True,
                    title=f"Test ({policy_test})    Train/Val ({policy_train})",
                    title_fontproperties={'weight':'bold'}, fontsize=fontsize)
    
    if policy_train=='P0':
        policy_train=policy_test
    
    match policy_train:
        
        case "P3" | "P4":
            
            axes[0].set_ylabel("Fertilizer nitrogen \n load (R1-R5) [-]", fontsize=fontsize)
            axes[0].set_title("$\\bf{{a)}}$", fontsize=fontsize)
            
            if nc==5:
                
                axes[1].set_ylabel("Agricultural nitrogen loss (R1) [-]", fontsize=fontsize)
                axes[1].set_title("$\\bf{{b)}}$", fontsize=fontsize)
                
                axes[2].set_ylabel("Perennial grassland nitrogen \n loss (R1) [-]", fontsize=fontsize)
                axes[2].set_title("$\\bf{{c)}}$", fontsize=fontsize)
                
                axes[3].set_ylabel("Cereal land use emissions [ktons CO2e]", fontsize=fontsize)
                axes[3].set_title("$\\bf{{d)}}$", fontsize=fontsize)
                
            elif nc==15:
                
                axes[1].set_ylabel("Cereal production (R1-R5) [-]", fontsize=fontsize)
                axes[1].set_title("$\\bf{{b)}}$", fontsize=fontsize)
                
                axes[2].set_ylabel("Utilized agricultural area \n (R1-R5) [ha]", fontsize=fontsize)
                axes[2].set_title("$\\bf{{c)}}$", fontsize=fontsize)
                
                axes[3].set_ylabel("Agricultural nitrogen loss (R1) [-]", fontsize=fontsize)
                axes[3].set_title("$\\bf{{d)}}$", fontsize=fontsize)
                
                axes[4].set_ylabel("Perennial grassland nitrogen \n loss (R1) [-]", fontsize=fontsize)
                axes[4].set_title("$\\bf{{e)}}$", fontsize=fontsize)
                
                axes[5].set_ylabel("Cereal land use emissions \n [ktons CO2e]", fontsize=fontsize)
                axes[5].set_title("$\\bf{{f)}}$", fontsize=fontsize)
            
        case "P11" | "P12":
            
            axes[0].set_ylabel("Road transport oil fuel \n demand (R1-R5) [-]", fontsize=fontsize)
            axes[0].set_title("$\\bf{{a)}}$", fontsize=fontsize)
            
            if nc==5:
            
                axes[1].set_ylabel("Total emissions [ktons $CO_2e$]", fontsize=fontsize)
                axes[1].set_title("$\\bf{{b)}}$", fontsize=fontsize)
                
                axes[2].set_ylabel("Net cereals [-]", fontsize=fontsize)
                axes[2].set_title("$\\bf{{c)}}$", fontsize=fontsize)
                
                axes[3].set_ylabel("Total food production [tons]", fontsize=fontsize)
                axes[3].set_title("$\\bf{{d)}}$", fontsize=fontsize)
                
            elif nc==15:
                
                axes[1].set_ylabel("Total energy demand (R1-R5) [TJ]", fontsize=fontsize)
                axes[1].set_title("$\\bf{{b)}}$", fontsize=fontsize)
                
                axes[2].set_ylabel("Cereal land use (R1-R5) [ha]", fontsize=fontsize)
                axes[2].set_title("$\\bf{{c)}}$", fontsize=fontsize)
                
                axes[3].set_ylabel("Total emissions [ktons $CO_2e$]", fontsize=fontsize)
                axes[3].set_title("$\\bf{{d)}}$", fontsize=fontsize)
                
                axes[4].set_ylabel("Net cereals [-]", fontsize=fontsize)
                axes[4].set_title("$\\bf{{e)}}$", fontsize=fontsize)
                
                axes[5].set_ylabel("Total food production [tons]", fontsize=fontsize)
                axes[5].set_title("$\\bf{{f)}}$", fontsize=fontsize)
            
        case "P19":
            
            axes[0].set_ylabel("Perennial grassland area \n (R1-R5) [ha]", fontsize=fontsize)
            axes[0].set_title("$\\bf{{a)}}$", fontsize=fontsize)
            
            if nc==5:
                
                axes[1].set_ylabel("Cereal nitrogen loss (R1) [tons]", fontsize=fontsize)
                axes[1].set_title("$\\bf{{b)}}$", fontsize=fontsize)
                
                axes[2].set_ylabel("Total income cereal [EUR]", fontsize=fontsize)
                axes[2].set_title("$\\bf{{c)}}$", fontsize=fontsize)
                
                axes[3].set_ylabel("Total grassland emissions \n [ktons CO2e]", fontsize=fontsize)
                axes[3].set_title("$\\bf{{d)}}$", fontsize=fontsize)
            
            elif nc==15:
                
                axes[1].set_ylabel("Cereal agricultural area (R1-R5) [ha]", fontsize=fontsize)
                axes[1].set_title("$\\bf{{b)}}$", fontsize=fontsize)
                
                axes[2].set_ylabel("Livestock count (R1-R5) [-]", fontsize=fontsize)
                axes[2].set_title("$\\bf{{c)}}$", fontsize=fontsize)
                
                axes[3].set_ylabel("Cereal nitrogen loss (R1) [tons]", fontsize=fontsize)
                axes[3].set_title("$\\bf{{d)}}$", fontsize=fontsize)
                
                axes[4].set_ylabel("Total income cereal [EUR]", fontsize=fontsize)
                axes[4].set_title("$\\bf{{e)}}$", fontsize=fontsize)
                
                axes[5].set_ylabel("Total grassland emissions \n [ktons CO2e]", fontsize=fontsize)
                axes[5].set_title("$\\bf{{f)}}$", fontsize=fontsize)
                
            elif nc==30:
                
                axes[1].set_ylabel("Cereal agricultural area (R1-R5) [ha]", fontsize=fontsize)
                axes[1].set_title("$\\bf{{b)}}$", fontsize=fontsize)
                
                axes[2].set_ylabel("Livestock count (R1-R5) [-]", fontsize=fontsize)
                axes[2].set_title("$\\bf{{c)}}$", fontsize=fontsize)
                
                axes[3].set_ylabel("GHG emissions (R1-R5) \n [tons CO2e]", fontsize=fontsize)
                axes[3].set_title("$\\bf{{d)}}$", fontsize=fontsize)
                
                axes[4].set_ylabel("Water N losses (R1-R5) [-]", fontsize=fontsize)
                axes[4].set_title("$\\bf{{e)}}$", fontsize=fontsize)
                
                axes[5].set_ylabel("Utlized Agr. Land Income \n (R1-R5) [EUR]", fontsize=fontsize)
                axes[5].set_title("$\\bf{{f)}}$", fontsize=fontsize)
                
                axes[6].set_ylabel("Cereal nitrogen loss (R1) [tons]", fontsize=fontsize)
                axes[6].set_title("$\\bf{{g)}}$", fontsize=fontsize)
                
                axes[7].set_ylabel("Total income cereal [EUR]", fontsize=fontsize)
                axes[7].set_title("$\\bf{{h)}}$", fontsize=fontsize)
                
                axes[8].set_ylabel("Total grassland emissions \n [ktons CO2e]", fontsize=fontsize)
                axes[8].set_title("$\\bf{{i)}}$", fontsize=fontsize)
        
    plt.show()
    fig.savefig("figure x.svg", bbox_inches = 'tight')



def visualize_eigs(policy_train, policy_test, 
                   perf_test_mean, r, rtilde, 
                   reduced_matrices, 
                   X_train, I_train, X_test, I_test):

    # select best model
    i_best = np.where(perf_test_mean == np.max(perf_test_mean))[0][0]
    r_best = r[i_best]
    rtilde_best = rtilde[i_best]

    model = DMDc(svd_rank=r_best.item(), svd_rank_omega=rtilde_best.item(), reduced_matrices=reduced_matrices)
    model = model.fit(X_train, I_train)
    # model_benchmark = DMD(svd_rank=r_best.item(), reduced_matrices=reduced_matrices)
    model_benchmark = DMDc(svd_rank=r_best.item(), svd_rank_omega=rtilde_best.item(), reduced_matrices=reduced_matrices)
    model_benchmark.fit(X_test, I_test)

    # dynamic matrices
    Atilde_dmdc = model.Atilde
    Atilde_bench = model_benchmark.Atilde

    # sorted eigenvalues
    eigsc_sort,vecsc_sort = eigenshuffle_eig(Atilde_dmdc.reshape((1,r_best,r_best)))
    eigs_sort,vecs_sort = eigenshuffle_eig(Atilde_bench.reshape((1,r_best,r_best)))

    eigs_sort = eigs_sort.T.squeeze()
    eigsc_sort = eigsc_sort.T.squeeze()

    eigs = np.concatenate([model.eigs,model_benchmark.eigs])
    minx = np.min(eigs.real)
    maxx = np.max(eigs.real)
    miny = np.min(eigs.imag)
    maxy = np.max(eigs.imag)
    stdx = np.std(eigs.real)
    stdy = np.std(eigs.imag)

    xlims = [minx-stdx*.4,maxx+stdx*.4]
    ylims = [miny-stdy*.4,maxy+stdy*.4]

    eig_names = [f"$\\lambda_{{{i+1}}}$" for i in range(r_best)]

    plt.figure(figsize=(6, 6))
    plt.grid("on")

    h2 = plt.scatter(eigsc_sort.real, eigsc_sort.imag, 70, c=bluecolor, marker='x')
    h1 = plt.scatter(eigs_sort.real, eigs_sort.imag, 70, c=redcolor, marker='o')

    unit_circle = np.exp(1j*np.linspace(0,2*np.pi,500))
    plt.plot(unit_circle.real, unit_circle.imag,'k--', dashes=(3, 5), linewidth=1)

    h1.AlphaData = np.zeros((r_best,1))
    h2.AlphaData = np.zeros((r_best,1))
    h1.MarkerFaceAlpha = 'flat'
    h2.MarkerEdgeAlpha = 'flat'

    maxeigs = min(r_best, 7)
    for i in range(maxeigs):
        plt.text(eigs_sort.real[i]+0.000, eigs_sort.imag[i]+0.001, eig_names[i], c=redcolor)
        plt.text(eigsc_sort.real[i]+0.000, eigsc_sort.imag[i]+0.001, eig_names[i], c=bluecolor)

    fontsize=14

    plt.rc('axes', axisbelow=True)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel("real($\\lambda$)", fontsize=fontsize)
    plt.ylabel("imag($\\lambda$)", fontsize=fontsize)
    plt.legend([f"DMDc on {policy_train}",
                f"DMDc on {policy_test} (Baseline)", 
                "Unit Circle"], bbox_to_anchor=(0,-0.09), 
               loc="upper left",
               fontsize=fontsize-1)
    plt.title(f"Discrete Eigenvalue Spectrum of $\\bf\\tilde A$ for [$r$,$\\tilde r$]=[{r_best},{rtilde_best}]", 
              fontsize=fontsize)


    plt.savefig("figure x - eigenvalues.svg", bbox_inches='tight')

    plt.show()



def visualize_matrices(model_benchmark, model):
    plt.pcolor(model_benchmark.A,clim=[-0.1,0.1]);plt.show()
    plt.pcolor(model.A,clim=[-0.1,0.1]);plt.show()

    plt.pcolor(model_benchmark.Atilde,clim=[-0.1,0.1]);plt.show()

    plt.subplot(1,2,1)
    plt.pcolor(model.Atilde,clim=[-0.1,0.1]);plt.show()
    plt.subplot(1,2,2)
    plt.pcolor(model.Btilde,clim=[-0.1,0.1]);plt.show()



def plot_maps(model, geo_vars, X_train, feature_names):
    # X_train = data_train_val_scaled[feature_names_opt].drop(columns=control_names)
    # I_train = data_train_val_scaled[feature_names_opt][control_names]
    # X_train = X_scaler.fit_transform(X_train).to_numpy().T
    # I_train = I_scaler.fit_transform(I_train).to_numpy().T

    # get dynamic modes
    modes = model.modes

    # get specific regional variables and corresponding modes
    n_regions = len(geo_vars)
    i_geo_vars = [X_train.columns.to_list().index(name) for name in geo_vars]
    geo_modes = modes[i_geo_vars,:]

    # phase of oscillation and magnitude
    all_modes_magn = np.abs(modes)
    geo_modes_magn = np.abs(geo_modes)
    geo_modes_angle = np.arctan(geo_modes.imag / geo_modes.real) * 360 / (2*np.pi)

    # mapping
    region_names = ["R1", "R2", "R3", "R4", "R5"] #+ f"\n R{j+1}"
    # x_pos = [0.5, 0.7, 0.21, 0.4, 0.84]
    # y_pos = [0.45, 0.55, 0.58, 0.31, 0.2]
    regions_temp = geopandas.read_file("./data/latvia regions/latvia_regions.shp")

    # reorder regions
    regions = regions_temp.reindex([2,3,0,4,1]).reset_index()

    x_pos, y_pos = [], []
    for i in range(n_regions):
        x_pos.append(regions.geometry[i].centroid.x + 0.1)
        y_pos.append(regions.geometry[i].centroid.y - 0.1)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12,4))
    # for i in range(3):
    #     divider = make_axes_locatable(ax[0,i])
    #     cax = divider.append_axes("bottom", size="10%", pad=0.1)

    for mode in range(3):
        
        if mode==0:
            leg_on=True
        else:
            leg_on=False
        
        regions.plot(
            column=geo_modes_magn[:,mode],
            ax=ax[0,mode],
            edgecolor='black',
            linewidth=0.5,
            legend=True,
            # cax=cax,
            # legend_kwds={"label": "Relative Energy in $\lambda_{i}$",
            #              "orientation": "horizontal",
            #              "location": "bottom",
            #              "fmt": "{:.2f}",
            #              },
            # cmap='Reds'
            cmap="bwr",
            )
        
        for j in range(n_regions):
            ax[0,mode].text(x_pos[j], y_pos[j], f"{region_names[j]}", fontsize=8, horizontalalignment='center') #, transform=ax[0,mode].transAxes
        
        # ax.set_axis_off()
        ax[0,mode].set_xticks([])
        ax[0,mode].set_yticks([])
        ax[0,mode].set_title(f"Mode {mode+1}")



    for mode in range(3):
        
        if mode==0:
            leg_on=True
        else:
            leg_on=False
        
        regions.plot(
            column=geo_modes_angle[:,mode],
            ax=ax[1,mode],
            edgecolor='black',
            linewidth=0.5,
            legend=True,
            # cax=cax,
            # legend_kwds={"label": "Relative Energy in $\lambda_{i}$",
            #              "orientation": "horizontal",
            #              "location": "bottom",
            #              "fmt": "{:.2f}",
            #              },
            # cmap='Reds'
            cmap="bwr",
            )
        
        for j in range(n_regions):
            ax[1,mode].text(x_pos[j], y_pos[j], f"{region_names[j]}", fontsize=8, horizontalalignment='center') #, transform=ax[0,mode].transAxes
        
        # ax.set_axis_off()
        ax[1,mode].set_xticks([])
        ax[1,mode].set_yticks([])
        ax[1,mode].set_title(f"Mode {mode+1}")


    for i in range(3):
        mode = modes[:,i].T
        
        # highlight important feature names
        i_important = np.where(np.abs(mode) > 0.8*np.std(mode))[0]
        imp_names = [name for name in feature_names if feature_names.index(name) in i_important]
        
        ax[2,i].plot(mode.real, '.r', markersize=2)
        for k in range(len(i_important)):
            ax[2,i].text(i_important[k], mode[i_important[k]].real, imp_names[k])
        ax[2,i].set_xlabel("state variables")
        

    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    plt.savefig("figure - map.svg", bbox_inches='tight')
    plt.show()



def visualize_controllability(): # WIP
    # # computing ctrb gramian for unstable systems (for a finite time)
    # # for a stable system, the below converges to the ct.gram for long time periods
    # def ctrGram(A, B, t_end):
    #     f = lambda tau: scipy.linalg.expm(A*tau) @ B @ B.conj().T @ scipy.linalg.expm(A.conj().T * tau)
    #     W = scipy.integrate.quad_vec(f, 0, t_end)
    #     return W[0]

    # W = ctrGram(model.A, model.B, 1)


    # # eigendecomposition to find most controllable state variables
    # eigs, eigvecs = scipy.linalg.eig(W)


    # # sorted eigenvectors by dominant modes
    # # i_sorted = np.argsort(eigs.real)[::-1]
    # # col_names_sorted = [name for _,name in zip(i_sorted, col_names)]
    # # eigvecs_sorted = pd.DataFrame(eigvecs[:,i_sorted], columns=col_names_sorted)


    # # average controllability of important variables normalized by total energy in eigenvectors
    # eigvecs_norm = abs(eigvecs.sum(axis=1) / eigvecs.sum(axis=None)).reshape(1,-1)
    # eigvecs_norm = pd.DataFrame(eigvecs_norm, columns=col_names)


    # # controllability of important variables
    # # eigvec_important[:,ensemble] = eigvecs_sorted[important_names].iloc[0,:].to_numpy().real
    # eigvec_important[:,ensemble] = eigvecs_norm[important_names]
    pass


