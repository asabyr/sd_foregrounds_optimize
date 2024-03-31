import numpy as np

### fiducial parameter values from Abitbol+2017 ###
### in order of default signals in fisher.py ###
params_all={'mu_amp': 2e-08, 'y_tot': 1.77e-06, 'kT_yweight': 1.245, 'DeltaT_amp': 0.00012,
     'Ad': 1.36e6, 'Bd': 1.53, 'Td': 21,'Acib': 346000.0, 'Bcib': 0.86,
     'Tcib': 18.8, 'EM': 300.0, 'As': 288.0, 'alps': -0.82, 'w2s': 0.2, 'Asd': 1.0, 'Aco': 1.0}


def convert_one_dict(param_values_one, param_keys_one, factor_one=True):
    """function to make a dictionary from an array of values,
    that can be an input to fisher calculations.
    
    args:
    param_values [array of floats] -- array of parameter values
    param_keys [array of strings] -- array of parameter names 
    (must match sd_foregrounds_optimize conventions)
    factor [bool] -- True/False, 
    whether the parameter values are specified as some factor of fiducial values.
    
    return:
    final_dict: dictionary that
    includes all sky parameters used in fisher (i.e makes a dictionary in the order that
    fisher.py needs and includes all parameters that are kept at fiducial values)

    """
    #amplitude parameters
    amps=np.array(['Ad','As','Acib','EM','Asd','Aco'])

    #convert to actual values if only factors specified
    param_dict={}
    for i in range(len(param_keys_one)):
        if param_keys_one[i] in amps or factor_one==True:
            param_dict[param_keys_one[i]]=param_values_one[i]*params_all[param_keys_one[i]]
        else:
            param_dict[param_keys_one[i]]=param_values_one[i]

    #add missing parameters
    for param, param_value in params_all.items():
        if param not in param_dict:
            param_dict[param]=param_value

    #put in right order
    final_dict={}
    for param, param_value in params_all.items():
        final_dict[param]=param_dict[param]

    return final_dict