import random

def specify_control_vars(policy_train, reduced_dataset, all_feature_names, nc):

    # specify control variables, important variables (to visualize)
    match policy_train:





        case "P3" | "P4":
            
            '''
            -------------------------------------------------------------------
            Policy 3/4 (P3/4): Reduce N emissions by 10/50% by improving
            management on 50% of cereal land
            -------------------------------------------------------------------
            '''

        
            if nc==5:
                control_names = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                 'W R2 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                 'W R3 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                 'W R4 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                 'W R5 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3']
                
            elif nc==15:
                control_names = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                  'W R2 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                  'W R3 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                  'W R4 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                  'W R5 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
                                  'Fp Crop Food R1.Fp Cereal R1',
                                  'Fp Crop Food R2.Fp Cereal R2',
                                  'Fp Crop Food R3.Fp Cereal R3',
                                  'Fp Crop Food R4.Fp Cereal R4',
                                  'Fp Crop Food R5.Fp Cereal R5',
                                  'LU R1 Agriculture.LU R1 Utilized agricultural area2',
                                  'LU R2 Agriculture.LU R2 Utilized agricultural area',
                                  'LU R3 Agriculture.LU R3 Utilized agricultural area',
                                  'LU R4 Agriculture.LU R4 Utilized agricultural area',
                                  'LU R5 Agriculture.LU R5 Utilized agricultural area'
                                  ]
            
            important_names = ['W R1 Fert N load Agri.W R1 Agriculture Nitrogen loss TOT',
                               'W R1 Fert N load Agri.W R1 Perenial grassland Nitrogen loss',
                               'Cereals LU emissions.LV CO2 emissions cereals'
                               ]
            
            geo_vars = ['W R1 Fert N load Agri.W R1 Perenial grassland Nitrogen loss',
                        'W R2 Fert N load Agri.W R2 Perenial grassland Nitrogen loss',
                        'W R3 Fert N load Agri.W R3 Perennial grassland Nitrogen loss',
                        'W R4 Fert N load Agri.W R4 Agriculture Nitrogen loss',
                        'W R5 Fert N load Agri.W R5 Perennial grassland Nitrogen loss']





        case "P5":
            
            '''
            -------------------------------------------------------------------
            Policy 5 (P5): Reduce industrial heat demand by 20% by improving
            insulation
            -------------------------------------------------------------------
            '''
        
            if nc==5:
                control_names = ['Ed R1 Industry.Ed R1 Industry Heat',
                                 'Ed R2 Industry.Ed R2 Industry Heat',
                                 'Ed R3 Industry.Ed R3 Industry Heat',
                                 'Ed R4 Industry.Ed R4 Industry Heat',
                                 'Ed R5 Industry.Ed R5 Industry Heat']
            elif nc==15 or nc==30:
                ValueError('nc=15/30 not available')
            
            important_names = ['Tot ed sectors/sources Latvia.ed industry heat Latvia',
                               'Ed R1 Industry.Ed R1 Industry TOT',
                               'Ed R2 Industry.Ed R2 Industry TOT',
                               'Ed R3 Industry.Ed R3 Industry TOT',
                               'Ed R4 Industry.Ed R4 Industry TOT',
                               'Ed R5 Industry.Ed R5 Industry TOT']
            
            important_names = [name for i,name in enumerate(important_names) if i in [0,3,4]]





        case "P11" | "P12":
            
            '''
            -------------------------------------------------------------------
            Policy 11/12 (P11/12): Reduce road transport oil fuel demand by
            10/18%
            -------------------------------------------------------------------
            '''
            
            if nc==5:
                control_names = ['Ed R1 Transport.R1 transport road oil',
                                  'Ed R2 Transport 2.R2 transport road oil',
                                  'Ed R3 Transport.R3 transport road oil',
                                  'Ed R4 Transport.R4 transport road oil',
                                  'Ed R5 Transport.R5 transport road oil']
                
            elif nc==15:
                control_names = ['Ed R1 Transport.R1 transport road oil',
                                  'Ed R2 Transport 2.R2 transport road oil',
                                  'Ed R3 Transport.R3 transport road oil',
                                  'Ed R4 Transport.R4 transport road oil',
                                  'Ed R5 Transport.R5 transport road oil',
                                  'Ed tot R1.ed sectors R1 tot',
                                  'Ed R2.ed sectors R2 tot',
                                  'Ed R3.ed sectors R3 tot',
                                  'Ed R4.ed sectors R4 tot',
                                  'Ed R5.ed sectors R5 tot',
                                  'LU R1 Agriculture.LU R1 Cereals Policies',
                                  'LU R2 Agriculture.LU R2 Cereals Policies',
                                  'LU R3 Agriculture.LU R3 Cereals Policies',
                                  'LU R4 Agriculture.LU R4 Cereals Policies',
                                  'LU R5 Agriculture.LU R5 Cereals Policies'
                                  ]
                
            important_names = ['Climate.LV tot GHG emissions ktCO2e',
                               'Food.Fp Net Cereals Latvia',
                               'Food.Af Latvia Tot'
                               ]
            
            geo_vars = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3',
                        'W R2 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3',
                        'W R3 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3',
                        'W R4 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3',
                        'W R5 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3']
            
            if reduced_dataset:
                
                # random feature selection
                n_keep = 500
                keep_vars = random.sample(all_feature_names, n_keep-len(important_names)-1)
                
                # manual feature selection
                
                
                # key-word feature selection
                # keep_vars = [name for name in all_feature_names if any(
                #     [var in name.lower() for var in ["road","transport","emissions", "oil", "months"]])]
                
                for name in control_names + important_names:
                    if name not in keep_vars:
                        keep_vars.append(name)
                
                if 'Months' not in keep_vars:
                    keep_vars.append('Months')





        case "P19":
            
            '''
            -------------------------------------------------------------------
            Policy 19 (P19): 50% increase in perennial grassland use, loss in
            cereal land
            -------------------------------------------------------------------
            '''
            
            if nc==5:
                control_names = ['LU R1 Livestock.LU R1 Perennial Grasslands Policies',
                                  'LU R2 Livestock.LU R2 Perennial grassland Policies',
                                  'LU R3 Livestock.LU R3 Perennial grassland Policies',
                                  'LU R4 Livestock.LU R4 Perennial grassland Policies',
                                  'LU R5 Livestock.LU R5 Perennial grassland Policies']
                
            elif nc==15:
                control_names = ['LU R1 Livestock.LU R1 Perennial Grasslands Policies',
                                  'LU R2 Livestock.LU R2 Perennial grassland Policies',
                                  'LU R3 Livestock.LU R3 Perennial grassland Policies',
                                  'LU R4 Livestock.LU R4 Perennial grassland Policies',
                                  'LU R5 Livestock.LU R5 Perennial grassland Policies',
                                  'LU R1 Agriculture.LU R1 Cereals Policies',
                                  'LU R2 Agriculture.LU R2 Cereals Policies',
                                  'LU R3 Agriculture.LU R3 Cereals Policies',
                                  'LU R4 Agriculture.LU R4 Cereals Policies',
                                  'LU R5 Agriculture.LU R5 Cereals Policies',
                                  'LU R1 Livestock.LU R1 livistock tot',
                                  'LU R2 Livestock.LU R2 livestock tot',
                                  'LU R3 Livestock.LU R3 Livestock tot',
                                  'LU R4 Livestock.LU R4 Livestock tot',
                                  'LU R5 Livestock.LU R5 Livestock tot'
                                  ]
            elif nc==30:
                control_names = ['LU R1 Livestock.LU R1 Perennial Grasslands Policies',
                                  'LU R2 Livestock.LU R2 Perennial grassland Policies',
                                  'LU R3 Livestock.LU R3 Perennial grassland Policies',
                                  'LU R4 Livestock.LU R4 Perennial grassland Policies',
                                  'LU R5 Livestock.LU R5 Perennial grassland Policies',
                                  'LU R1 Agriculture.LU R1 Cereals Policies',
                                  'LU R2 Agriculture.LU R2 Cereals Policies',
                                  'LU R3 Agriculture.LU R3 Cereals Policies',
                                  'LU R4 Agriculture.LU R4 Cereals Policies',
                                  'LU R5 Agriculture.LU R5 Cereals Policies',
                                  'LU R1 Livestock.LU R1 livistock tot',
                                  'LU R2 Livestock.LU R2 livestock tot',
                                  'LU R3 Livestock.LU R3 Livestock tot',
                                  'LU R4 Livestock.LU R4 Livestock tot',
                                  'LU R5 Livestock.LU R5 Livestock tot',
                                  'C R1 Pieriga.GHG balance R1',
                                  'C R2 Vidzeme.GHG balance R2',
                                  'C R3 Kurzeme.GHG balance R3',
                                  'C R4 Zemgale.GHG balance R4',
                                  'C R5 Latgale.GHG balance R5',
                                  'W R1 Pieriga.W N losses R1',
                                  'W R2 Vidzeme.W N losses R2',
                                  'W R3 Kurzeme.W N losses R3',
                                  'W R4 Zemgale.W N losses R4',
                                  'W R5 Latgale.W N losses R5',
                                  'Income Latvia TOT.Income LU utilized agricultural area TOT R1 (EUR)',
                                  'Income Latvia TOT.Income LU utilized agricultural area TOT R2 (EUR)',
                                  'Income Latvia TOT.Income LU utilized agricultural area TOT R3 (EUR)',
                                  'Income Latvia TOT.Income LU utilized agricultural area TOT R4 (EUR)',
                                  'Income Latvia TOT.Income LU utilized agricultural area TOT R5 (EUR)'
                                  ]
            
            important_names = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen loss',
                               'Income Latvia TOT.Income LU Cereals TOT Latvia (EUR)',
                               'Climate.LV grassland emissions'
                               ]
            
            geo_vars = []
            
            if reduced_dataset:
                # manual selection
                keep_vars = ['LU R1 Livestock.LU R1 Perennial grasslands', 'LU R2 Livestock.LU R2 Perennial grasslands', 'LU R3 Livestock.LU R3 Perennial grasslands', 'LU R4 Livestock.LU R4 Perennial grasslands', 'LU R5 Livestock.LU R5 Perennial grasslands',
                             'W R1 Pieriga.W N losses R1', 'W R2 Vidzeme.W N losses R2', 'W R3 Kurzeme.W N losses R3', 'W R4 Zemgale.W N losses R4', 'W R5 Latgale.W N losses R5',
                             'W R1 Fert N load Agri.W R1 Perenial grassland Nitrogen loss',  'W R2 Fert N load Agri.W R2 Perenial grassland Nitrogen loss', 'W R3 Fert N load Agri.W R3 Perennial grassland Nitrogen loss', 'W R4 Fert N load Agri.W R4 Agriculture Nitrogen loss', 'W R5 Fert N load Agri.W R5 Perennial grassland Nitrogen loss',
                             'W R1 Fert N load Agri.W R1 Agriculture Nitrogen loss TOT', 'W R2 Fert N load Agri.W R2 Agriculture Nitrogen loss TOT', 'W R3 Fert N load Agri.W R3 Agriculture Nitrogen loss TOT', 'W R4 Fert N load Agri.W R4 Agriculture Nitrogen loss TOT', 'W R5 Fert N load Agri.W R5 Agriculture Nitrogen loss TOT',
                             'LU R1 Agriculture.LU R1 Cereals 1', 'LU R2 Agriculture.LU R2 Cereals', 'LU R3 Agriculture.LU R3 Cereals', 'LU R4 Agriculture.LU R4 Cereals', 'LU R5 Agriculture.LU R5 Cereals',
                             'W R1 Fert N load Agri.W R1 N Losses AGRI tonha', 'W R2 Fert N load Agri.W R2 N Losses AGRI tonha', 'W R3 Fert N load Agri.W R3 N Losses AGRI tonha', 'W R4 Fert N load Agri.W R4 N Losses AGRI tonha', 'W R5 Fert N load Agri.W R5 N Losses AGRI tonha',
                             'W R1 Fert N load Agri.W R1 Cereals Nitrogen loss', 'W R2 Fert N load Agri.W R2 Cereals Nitrogen loss', 'W R3 Fert N load Agri.W R3 Cereals Nitrogen loss', 'W R4 Fert N load Agri.W R4 Cereals Nitrogen loss', 'W R5 Fert N load Agri.W R5 Cereals Nitrogen loss'
                             ]
                
                # keep_vars = [name for name in all_feature_names if any(
                #     [var in name.lower() for var in ["nitrogen","perennial","perenial", "losses", "cereals", "load"]])]
                
                for name in control_names + important_names:
                    if name not in keep_vars:
                        keep_vars.append(name)





        case _:
            raise ValueError('Incorrect policy specified')





    if reduced_dataset:
        all_feature_names = keep_vars





    return control_names, important_names, geo_vars, all_feature_names

















# # specify control variables, important variables (to visualize) and filter variable
# match policy:
#     case "P4" | "P3": # reduce N emissions by 50,10% by improving management on 50% of cereal land
        
#         # nc = 5
#         control_names = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                          'W R2 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                          'W R3 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                          'W R4 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                          'W R5 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3']
        
#         # nc = 15
#         control_names = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                           'W R2 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                           'W R3 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                           'W R4 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                           'W R5 Fert N load Agri.W R1 Cereals Nitrogen loss Policy3',
#                           'Fp Crop Food R1.Fp Cereal R1',
#                           'Fp Crop Food R2.Fp Cereal R2',
#                           'Fp Crop Food R3.Fp Cereal R3',
#                           'Fp Crop Food R4.Fp Cereal R4',
#                           'Fp Crop Food R5.Fp Cereal R5',
#                           'LU R1 Agriculture.LU R1 Utilized agricultural area2',
#                           'LU R2 Agriculture.LU R2 Utilized agricultural area',
#                           'LU R3 Agriculture.LU R3 Utilized agricultural area',
#                           'LU R4 Agriculture.LU R4 Utilized agricultural area',
#                           'LU R5 Agriculture.LU R5 Utilized agricultural area'
#                           ]
        
#         important_names = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen loss',
#                            'Cereals LU emissions.LV CO2 emissions cereals',
#                            'W R1 Fert N load Agri.W R1 Agriculture Nitrogen loss TOT']
        
#         filter_var = important_names[0]
        
#     case "P5": # reduce industrial heat demand by 20% by improving insulation
    
#         control_names = ['Ed R1 Industry.Ed R1 Industry Heat',
#                          'Ed R2 Industry.Ed R2 Industry Heat',
#                          'Ed R3 Industry.Ed R3 Industry Heat',
#                          'Ed R4 Industry.Ed R4 Industry Heat',
#                          'Ed R5 Industry.Ed R5 Industry Heat']
        
#         important_names = ['Tot ed sectors/sources Latvia.ed industry heat Latvia',
#                            'Ed R1 Industry.Ed R1 Industry TOT',
#                            'Ed R2 Industry.Ed R2 Industry TOT',
#                            'Ed R3 Industry.Ed R3 Industry TOT',
#                            'Ed R4 Industry.Ed R4 Industry TOT',
#                            'Ed R5 Industry.Ed R5 Industry TOT']
        
#         important_names = [name for i,name in enumerate(important_names) if i in [0,3,4]]
#         filter_var = control_names[0]
        
#     case "P11" | "P12": # Reduce road transport oil fuel demand by 10/18%
        
#         # nc = 5
#         control_names = ['Ed R1 Transport.R1 transport road oil',
#                          'Ed R2 Transport 2.R2 transport road oil',
#                          'Ed R3 Transport.R3 transport road oil',
#                          'Ed R4 Transport.R4 transport road oil',
#                          'Ed R5 Transport.R5 transport road oil']
        
#         # nc = 15
#         # control_names = ['Ed R1 Transport.R1 transport road oil',
#         #                   'Ed R2 Transport 2.R2 transport road oil',
#         #                   'Ed R3 Transport.R3 transport road oil',
#         #                   'Ed R4 Transport.R4 transport road oil',
#         #                   'Ed R5 Transport.R5 transport road oil',
#         #                   'e R1 Food production.CO2eq emissions from cropsCEREAL R1',
#         #                   'e R2 Food production.CO2eq emissions from cropsCEREAL R2',
#         #                   'e R3 Food production.CO2eq emissions from cropsCEREAL R3',
#         #                   'e R4 Food production.CO2eq emissions from cropsCEREAL R4',
#         #                   'e R5 Food production.CO2eq emissions from cropsCEREAL R5',
#         #                   'Ed tot R1.ed sectors R1 tot',
#         #                   'Ed R2.ed sectors R2 tot',
#         #                   'Ed R3.ed sectors R3 tot',
#         #                   'Ed R4.ed sectors R4 tot',
#         #                   'Ed R5.ed sectors R5 tot']
        
#         important_names = ['Climate.LV tot road emissions ktCO2e',
#                            'Climate.LV tot GHG emissions ktCO2e',
#                            'e R3 Food production.e tot food R3'
#                            ]
        
#         # important_names = ['Climate.LV tot road emissions ktCO2e',
#         #                    'Climate.LV tot GHG emissions ktCO2e',
#         #                    'Ed R2.ed sectors R2 tot']
        
#         filter_var = important_names[1]
        
        
        
        
#         # control_names = [name for name in all_feature_names if (any(
#         #     [var in name.lower() for var in ["road","transport", "oil"]])
#         #     and name not in important_names)]
        
#         # control_names = [name for name in all_feature_names if rand()>0.9]
        
#         # filter_var = control_names[0]
        
        
#         # geo_vars = ['Ed tot R1.ed sectors R1 tot',
#         #               'Ed R2.ed sectors R2 tot',
#         #               'Ed R3.ed sectors R3 tot',
#         #               'Ed R4.ed sectors R4 tot',
#         #               'Ed R5.ed sectors R5 tot']
        
#         # geo_vars = ['e R1 Transport.tot e transport CO2 eq',
#         #               'e R2 Transport.tot e transport CO2 eq R2',
#         #               'e R3 Transport.tot e transport CO2 eq R3',
#         #               'e R4 Transport.tot e transport CO2 eq R4',
#         #               'e R5 Transport.tot e transport CO2 eq R5']
        
#         # geo_vars = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3',
#         #               'W R2 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3',
#         #               'W R3 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3',
#         #               'W R4 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3',
#         #               'W R5 Fert N load Agri.W R1 Cereals Nitrogen losses Policies12and3']
        
#         # [important_names.append(name) for name in geo_vars if name not in important_names]
        
#         if reduced_dataset:
#             k_best = None
            
#             # random feature selection
#             n_keep = 500
#             keep_vars = random.sample(all_feature_names, n_keep-len(important_names)-1)
            
#             # manual feature selection
            
            
#             # key-word feature selection
#             # keep_vars = [name for name in all_feature_names if any(
#             #     [var in name.lower() for var in ["road","transport","emissions", "oil", "months"]])]
            
#             for name in control_names + important_names:
#                 if name not in keep_vars:
#                     keep_vars.append(name)
            
#             if 'Months' not in keep_vars:
#                 keep_vars.append('Months')
            
#             all_feature_names = keep_vars

#     case "P19": # 50% increase in perennial grassland use, loss in cereal land
        
#         # nc = 5
#         control_names = ['LU R1 Livestock.LU R1 Perennial Grasslands Policies',
#                           'LU R2 Livestock.LU R2 Perennial grassland Policies',
#                           'LU R3 Livestock.LU R3 Perennial grassland Policies',
#                           'LU R4 Livestock.LU R4 Perennial grassland Policies',
#                           'LU R5 Livestock.LU R5 Perennial grassland Policies']
        
#         # nc = 15
#         control_names = ['LU R1 Livestock.LU R1 Perennial Grasslands Policies',
#                           'LU R2 Livestock.LU R2 Perennial grassland Policies',
#                           'LU R3 Livestock.LU R3 Perennial grassland Policies',
#                           'LU R4 Livestock.LU R4 Perennial grassland Policies',
#                           'LU R5 Livestock.LU R5 Perennial grassland Policies',
#                           'LU R1 Agriculture.LU R1 Cereals Policies',
#                           'LU R2 Agriculture.LU R2 Cereals Policies',
#                           'LU R3 Agriculture.LU R3 Cereals Policies',
#                           'LU R4 Agriculture.LU R4 Cereals Policies',
#                           'LU R5 Agriculture.LU R5 Cereals Policies',
#                           'LU R1 Livestock.LU R1 livistock tot',
#                           'LU R2 Livestock.LU R2 livestock tot',
#                           'LU R3 Livestock.LU R3 Livestock tot',
#                           'LU R4 Livestock.LU R4 Livestock tot',
#                           'LU R5 Livestock.LU R5 Livestock tot'
#                           ]
        
#         important_names = ['W R1 Fert N load Agri.W R1 Cereals Nitrogen loss',
#                            'Income Latvia TOT.Income LU Cereals TOT Latvia (EUR)',
#                            'Climate.LV grassland emissions'
#                            ]
        
#         if reduced_dataset:
#             k_best = None
            
#             keep_vars = ['LU R1 Livestock.LU R1 Perennial grasslands', 'LU R2 Livestock.LU R2 Perennial grasslands', 'LU R3 Livestock.LU R3 Perennial grasslands', 'LU R4 Livestock.LU R4 Perennial grasslands', 'LU R5 Livestock.LU R5 Perennial grasslands',
#                          'W R1 Pieriga.W N losses R1', 'W R2 Vidzeme.W N losses R2', 'W R3 Kurzeme.W N losses R3', 'W R4 Zemgale.W N losses R4', 'W R5 Latgale.W N losses R5',
#                          'W R1 Fert N load Agri.W R1 Perenial grassland Nitrogen loss',  'W R2 Fert N load Agri.W R2 Perenial grassland Nitrogen loss', 'W R3 Fert N load Agri.W R3 Perennial grassland Nitrogen loss', 'W R4 Fert N load Agri.W R4 Agriculture Nitrogen loss', 'W R5 Fert N load Agri.W R5 Perennial grassland Nitrogen loss',
#                          'W R1 Fert N load Agri.W R1 Agriculture Nitrogen loss TOT', 'W R2 Fert N load Agri.W R2 Agriculture Nitrogen loss TOT', 'W R3 Fert N load Agri.W R3 Agriculture Nitrogen loss TOT', 'W R4 Fert N load Agri.W R4 Agriculture Nitrogen loss TOT', 'W R5 Fert N load Agri.W R5 Agriculture Nitrogen loss TOT',
#                          'LU R1 Agriculture.LU R1 Cereals 1', 'LU R2 Agriculture.LU R2 Cereals', 'LU R3 Agriculture.LU R3 Cereals', 'LU R4 Agriculture.LU R4 Cereals', 'LU R5 Agriculture.LU R5 Cereals',
#                          'W R1 Fert N load Agri.W R1 N Losses AGRI tonha', 'W R2 Fert N load Agri.W R2 N Losses AGRI tonha', 'W R3 Fert N load Agri.W R3 N Losses AGRI tonha', 'W R4 Fert N load Agri.W R4 N Losses AGRI tonha', 'W R5 Fert N load Agri.W R5 N Losses AGRI tonha',
#                          'W R1 Fert N load Agri.W R1 Cereals Nitrogen loss', 'W R2 Fert N load Agri.W R2 Cereals Nitrogen loss', 'W R3 Fert N load Agri.W R3 Cereals Nitrogen loss', 'W R4 Fert N load Agri.W R4 Cereals Nitrogen loss', 'W R5 Fert N load Agri.W R5 Cereals Nitrogen loss'
#                          ]
            
#             # keep_vars = [name for name in all_feature_names if any(
#             #     [var in name.lower() for var in ["nitrogen","perennial","perenial", "losses", "cereals", "load"]])]
            
#             for name in control_names + important_names:
#                 if name not in keep_vars:
#                     keep_vars.append(name)
            
#             all_feature_names = keep_vars
        
#     case "test":
        
#         control_names = ["u"]
        
#         important_names = ["x1", "x2"]
    
#     case _:
#         raise ValueError('Incorrect policy specified')