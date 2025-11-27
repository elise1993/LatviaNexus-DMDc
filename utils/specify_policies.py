from utils.names_dict import names_dict
from types import SimpleNamespace

def specify_policies(train, test, nc, large_data=False):
    
    # variable names are defined in /utils/names_dict.py
    D, d = names_dict()
    ns = SimpleNamespace(**d)
    input_names, state_names = [], []

    # POLICY 3/4 - 
    if any(x in ['P3','P4'] for x in [train, test]):
        if nc>=5:
            input_names += ns.emissions_cereals
        if nc>=10:
            input_names += ns.income_util_agr_area
        if nc>=15:
            raise ValueError('nc=15 policies not specified for P3/4')
    
    # POLICY 11/12 - 
    if any(x in ['P11','P12'] for x in [train, test]):
        if nc>=5:
            input_names += ns.demand_road_oil
        if nc>=10:
            input_names += ns.demand_tot
        if nc>=15:
            # input_names += ns.demand_tot
            raise ValueError('nc=15 policies not specified for P11/12')
    
    # POLICY 14 - 
    if 'P14' in [train, test]:
        raise ValueError('P14 policies/states not defined')
    
    # POLICY 18/19 - 
    if any(x in ['P18','P19'] for x in [train, test]):
        if nc>=5:
            input_names += ns.area_perennial
        if nc>=10:
            input_names += ns.area_cereal
        if nc>=15:
            input_names += ns.livestock_count
        
    if large_data:
        state_names += ns.n_losses_tot + ns.emissions_tot + ns.food_prod_cereal \
            + ns.income_util_agr_area + ns.area_perennial + ns.area_cereal \
            + ns.livestock_count + ns.n_losses_agr + ns.n_losses_cereal \
            + ns.n_losses_per_area_agr + ns.food_prod_tot + ns.area_util_agr \
            + ns.area_nonutil_agr + ns.area_change_agr + ns.income_arableland \
            + ns.income_cereals + ns.income_perennial + ns.emissions_road \
            + ns.emissions_balance + ns.emissions_grassland + ns.emissions_cereals \
            + ns.emissions_food_prod_tot
            
        state_names = [name for name in state_names if name not in input_names]
    else:
        state_names += ns.n_losses_tot
        state_names += ns.emissions_tot
        state_names += ns.food_prod_cereal
    
    return input_names, state_names

# how to find variables based on keywords
# names = X_train_val.columns
# [name for name in names if all(
#     [key in name.lower() for key in ['x', 'y']]) and not
#     any([key in name.lower() for key in ['y', 'z']])]

# how to find variables that change with policy
# Names = [name for name in names if any(abs(X_train_val[name] - X_test[name])>0.1)]
# for nam in Names:
#     X_train_val[nam].plot(); 
#     plt.legend()
#     X_test[nam].plot()
#     plt.show()