from types import SimpleNamespace

def names_dict():    
    #%% Variable names in data
    d = dict()
    
    # --- inputs --- 
    # policy 3/4
    d['income_util_agr_area'] = ['Income Latvia TOT.Income LU utilized agricultural area TOT R1 (EUR)','Income Latvia TOT.Income LU utilized agricultural area TOT R2 (EUR)','Income Latvia TOT.Income LU utilized agricultural area TOT R3 (EUR)','Income Latvia TOT.Income LU utilized agricultural area TOT R4 (EUR)','Income Latvia TOT.Income LU utilized agricultural area TOT R5 (EUR)']
    
    # policy 11/12
    d['demand_road_oil'] = ['Ed R1 Transport.R1 transport road oil', 'Ed R2 Transport 2.R2 transport road oil', 'Ed R3 Transport.R3 transport road oil', 'Ed R4 Transport.R4 transport road oil', 'Ed R5 Transport.R5 transport road oil']
    d['demand_tot'] = ['Ed tot R1.ed sectors R1 tot', 'Ed R2.ed sectors R2 tot', 'Ed R3.ed sectors R3 tot', 'Ed R4.ed sectors R4 tot', 'Ed R5.ed sectors R5 tot']
    
    # policy 18/19
    d['area_perennial'] = ['LU R1 Livestock.LU R1 Perennial Grasslands Policies', 'LU R2 Livestock.LU R2 Perennial grassland Policies', 'LU R3 Livestock.LU R3 Perennial grassland Policies', 'LU R4 Livestock.LU R4 Perennial grassland Policies', 'LU R5 Livestock.LU R5 Perennial grassland Policies']
    d['area_cereal'] = ['LU R1 Agriculture.LU R1 Cereals Policies', 'LU R2 Agriculture.LU R2 Cereals Policies', 'LU R3 Agriculture.LU R3 Cereals Policies', 'LU R4 Agriculture.LU R4 Cereals Policies', 'LU R5 Agriculture.LU R5 Cereals Policies']
    d['livestock_count'] = ['LU R1 Livestock.LU R1 livistock tot', 'LU R2 Livestock.LU R2 livestock tot', 'LU R3 Livestock.LU R3 Livestock tot', 'LU R4 Livestock.LU R4 Livestock tot', 'LU R5 Livestock.LU R5 Livestock tot']
    
    #  --- states --- 
    d['n_losses_tot'] = ['W R1 Pieriga.W N losses R1', 'W R2 Vidzeme.W N losses R2', 'W R3 Kurzeme.W N losses R3', 'W R4 Zemgale.W N losses R4', 'W R5 Latgale.W N losses R5']
    d['emissions_tot'] = ['C R1 Pieriga."GHG emissions Reg.1"','C R2 Vidzeme."GHG emissions Reg.2"', 'C R3 Kurzeme.GHG emissions Reg 3','C R4 Zemgale.GHG emiss Reg 4','C R5 Latgale.GHG emiss Reg 5']
    d['food_prod_cereal'] = [ 'Fp Crop Food R1.Fp Cereal R1', 'Fp Crop Food R2.Fp Cereal R2','Fp Crop Food R3.Fp Cereal R3', 'Fp Crop Food R4.Fp Cereal R4','Fp Crop Food R5.Fp Cereal R5']
    
    #  --- other --- 
    # water
    d['n_losses_agr'] = ['W R1 Fert N load Agri.W R1 Agriculture Nitrogen loss TOT', 'W R2 Fert N load Agri.W R2 Agriculture Nitrogen loss TOT', 'W R3 Fert N load Agri.W R3 Agriculture Nitrogen loss TOT', 'W R4 Fert N load Agri.W R4 Agriculture Nitrogen loss TOT', 'W R5 Fert N load Agri.W R5 Agriculture Nitrogen loss TOT',]
    d['n_losses_cereal'] = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen loss', 'W R2 Fert N load Agri.W R2 Cereals Nitrogen loss', 'W R3 Fert N load Agri.W R3 Cereals Nitrogen loss', 'W R4 Fert N load Agri.W R4 Cereals Nitrogen loss', 'W R5 Fert N load Agri.W R5 Cereals Nitrogen loss']
    # misconfigured in Stella SDM: d['n_losses_perennial'] = ['W R1 Fert N load Agri.W R1 Perenial grassland Nitrogen loss', 'W R2 Fert N load Agri.W R2 Perenial grassland Nitrogen loss', 'W R3 Fert N load Agri.W R3 Perennial grassland Nitrogen loss', 'W R4 Fert N load Agri.W R4 Agriculture Nitrogen loss', 'W R5 Fert N load Agri.W R5 Perennial grassland Nitrogen loss',]
    d['n_losses_per_area_agr'] = ['W R1 Fert N load Agri.W R1 N Losses AGRI tonha', 'W R2 Fert N load Agri.W R2 N Losses AGRI tonha', 'W R3 Fert N load Agri.W R3 N Losses AGRI tonha', 'W R4 Fert N load Agri.W R4 N Losses AGRI tonha', 'W R5 Fert N load Agri.W R5 N Losses AGRI tonha',]
    
    # energy
    
    # food
    d['food_prod_tot'] = ['Fp R1 Pieriga.Fp R1 Tot', 'Fp R2 Vidzeme.Fp R2 Tot', 'Fp R3 Kurzeme.Fp R3 Tot', 'Fp R4 Zemgale.Fp R4 Tot', 'Fp R5 Latgale.Fp R5 Tot']
    d['cereal_prod'] = ['Fp Crop Food R1.Fp Cereal R1', 'Fp Crop Food R2.Fp Cereal R2', 'Fp Crop Food R3.Fp Cereal R3', 'Fp Crop Food R4.Fp Cereal R4', 'Fp Crop Food R5.Fp Cereal R5']
    
    # land
    d['area_util_agr'] = ['LU R1 Agriculture.LU R1 Utilized agricultural area2', 'LU R2 Agriculture.LU R2 Utilized agricultural area', 'LU R3 Agriculture.LU R3 Utilized agricultural area', 'LU R4 Agriculture.LU R4 Utilized agricultural area', 'LU R5 Agriculture.LU R5 Utilized agricultural area']                          
    d['area_nonutil_agr'] = ['LU R1 Agriculture.LU R1 Agriculture Non utilized area','LU R2 Agriculture.LU R2 Agriculture Non utilized area','LU R3 Agriculture.LU R3 Agriculture Non utilized area', 'LU R4 Agriculture.LU R4 Agriculture Non utilized area','LU R5 Agriculture.LU R5 Agriculture Non utilized area']
    d['area_change_agr'] = ['LU R1 Agriculture.AreaChangeR1','LU R2 Agriculture.AreaChangeR2','LU R3 Agriculture.AreaChangeR3','LU R4 Agriculture.AreaChangeR4','LU R5 Agriculture.AreaChangeR5']
    d['income_arableland'] = ['Income Latvia TOT.Income LU Arable land TOT R1 (EUR)','Income Latvia TOT.Income LU Arable land TOT R2 (EUR)','Income Latvia TOT.Income LU Arable land TOT R3 (EUR)','Income Latvia TOT.Income LU Arable land TOT R4 (EUR)','Income Latvia TOT.Income LU Arable land TOT R5 (EUR)',]
    d['income_cereals'] = ['Income Latvia TOT.Income LU Cereals TOT R1 (EUR)','Income Latvia TOT.Income LU Cereals TOT R2 (EUR)','Income Latvia TOT.Income LU Cereals TOT R3 (EUR)','Income Latvia TOT.Income LU Cereals TOT R4 (EUR)','Income Latvia TOT.Income LU Cereals TOT R5 (EUR)']
    d['income_perennial'] = ['Income Latvia TOT.Income LU perennial grassland TOT R1 (EUR)', 'Income Latvia TOT.Income LU perennial grassland TOT R2 (EUR)', 'Income Latvia TOT.Income LU perennial grassland TOT R3 (EUR)', 'Income Latvia TOT.Income LU perennial grassland TOT R4 (EUR)', 'Income Latvia TOT.Income LU perennial grassland TOT R5 (EUR)',]
    
    # climate
    d['emissions_road'] = ['e R1 Transport.tot e transport CO2 eq','e R2 Transport.tot e transport CO2 eq R2','e R3 Transport.tot e transport CO2 eq R3','e R4 Transport.tot e transport CO2 eq R4','e R5 Transport.tot e transport CO2 eq R5']
    d['emissions_balance'] = ['C R1 Pieriga.GHG balance R1','C R2 Vidzeme.GHG balance R2','C R3 Kurzeme.GHG balance R3','C R4 Zemgale.GHG balance R4','C R5 Latgale.GHG balance R5']
    d['emissions_grassland'] = ['e R1 grassland emissions.e grassland R1','e R2 grassland emissions.e grassland R2','e R3 grassland emissions.e grassland R3','e R4 grassland emissions.e grassland R4','e R5 grassland emissions.e grassland R5']
    d['emissions_cereals'] = ['e R1 Food production.CO2eq emissions from cropsCEREAL R1','e R2 Food production.CO2eq emissions from cropsCEREAL R2','e R3 Food production.CO2eq emissions from cropsCEREAL R3','e R4 Food production.CO2eq emissions from cropsCEREAL R4','e R5 Food production.CO2eq emissions from cropsCEREAL R5']
    d['emissions_food_prod_tot'] = ['e R1 Food production.e tot food R1','e R2 Food production.e tot food R2','e R3 Food production.e tot food R3','e R4 Food production.e tot food R4','e R5 Food production.e tot food R5']
    
    ns = SimpleNamespace(**d)
    # for key, value in d.items():
    #     exec(f'global {key}; {key}=list({value})')
    
    #%% Dictionary
    D = dict()
    
    D.update({f"Cereal Production (R{i+1}) [tons]": ns.cereal_prod[i] for i in range(5)})
    D.update({f"Utilized Agricultural Area (R{i+1}) [ha]": ns.area_util_agr[i] for i in range(5)})
    D.update({f"Road oil energy demand (R{i+1}) [TJ]": ns.demand_road_oil[i] for i in range(5)})
    D.update({f"Energy demand (R{i+1}) [TJ]": ns.demand_tot[i] for i in range(5)})
    D.update({f"Emissions road transport (R{i+1}) [ktons CO2e]": ns.emissions_road[i] for i in range(5)})
    D.update({f"Total emissions (R{i+1}) [ktons CO2e]": ns.emissions_tot[i] for i in range(5)})
    D.update({f"Emissions balance (R{i+1}) [ktons CO2e]": ns.emissions_balance[i] for i in range(5)})
    D.update({f"Perennial grassland area (R{i+1}) [ha]": ns.area_perennial[i] for i in range(5)})
    D.update({f"Cereal agricultural area (R{i+1}) [ha]": ns.area_cereal[i] for i in range(5)})
    D.update({f"Livestock count (R{i+1}) [-]": ns.livestock_count[i] for i in range(5)})
    D.update({f"Utilized agricultural area (R{i+1}) [ha]": ns.area_util_agr[i] for i in range(5)})
    D.update({f"Non-utilized agricultural area (R{i+1}) [ha]": ns.area_nonutil_agr[i] for i in range(5)})
    D.update({f"Agricultural area change (R{i+1}) [ha/year]": ns.area_change_agr[i] for i in range(5)})
    D.update({f"Water N losses (R{i+1}) [tons]": ns.n_losses_tot[i] for i in range(5)})
    D.update({f"Agricultural N losses (R{i+1}) [tons]": ns.n_losses_agr[i] for i in range(5)})
    D.update({f"Agricultural N losses (R{i+1}) [tons/ha]": ns.n_losses_per_area_agr[i] for i in range(5)})
    #D.update({f"Perennial N losses (R{i+1}) [tons]": ns.n_losses_perennial[i] for i in range(5)}) # MISCONFIGURED IN STELLA
    D.update({f"Cereal N losses (R{i+1}) [tons]": ns.n_losses_cereal[i] for i in range(5)})
    D.update({f"Food production (R{i+1}) [tons]": ns.food_prod_tot[i] for i in range(5)})
    D.update({f"Cereal production (R{i+1}) [tons]": ns.food_prod_cereal[i] for i in range(5)})
    D.update({f"Emissions grassland (R{i+1}) [tons CO2e]": ns.emissions_grassland[i] for i in range(5)})
    D.update({f"Emissions cereals (R{i+1}) [tons CO2e]": ns.emissions_cereals[i] for i in range(5)})
    D.update({f"Emissions food production (R{i+1}) [tons CO2e]": ns.emissions_food_prod_tot[i] for i in range(5)})
    D.update({f"Income perennial grassland R{i+1} [EUR]": ns.income_perennial[i] for i in range(5)})
    D.update({f"Income cereals R{i+1} [EUR]": ns.income_cereals[i] for i in range(5)})
    D.update({f"Income arable land R{i+1} [EUR]": ns.income_arableland[i] for i in range(5)})
    D.update({f"Income utilized agricultural area R{i+1} [EUR]": ns.income_util_agr_area[i] for i in range(5)})
    
    # flip dictionary
    D = dict((value, key) for key, value in D.items())
    
    return D, d


