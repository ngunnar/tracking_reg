import importlib
#from lib.pytracking.pytracking.parameter.eco.default import parameters

def eco_parameters():
    param_module = importlib.import_module('pytracking.parameter.eco.default')
    params = param_module.parameters()
    params.box_size = 10
    return params
