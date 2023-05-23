import yaml
import os
global params 

def load_preset_parameters(paramsFile):
    """
    Args:
        paramsFile (string): Name of the yaml with pre set parameters for a specific run. Each variable contains a "desc" and "value".

    Returns:
       params_dict (dict): return yaml as dict.
    """
    print(20*"-" + paramsFile + 20*'-')
    # Read yaml and save in dict
    with open(os.path.expanduser('~')+'/MIMM/src/config/'+paramsFile) as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)
    return params_dict

def save_parameters_as_global_dict(config):
    global params
    params = config
    return params