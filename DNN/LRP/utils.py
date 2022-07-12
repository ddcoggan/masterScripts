import copy
import pickle
import torch
import types
from . import layers
from . import rules
Rules = rules.Rules

def flatten_model(module): # flatten module to base operation like Conv2, ReLU, Linear, ...
    modules_list = []
    for m_1 in module.children():

        if len(list(m_1.children())) == 0:
            modules_list.append(m_1)
        else:
            modules_list = modules_list + flatten_model(m_1)
    return modules_list

def copy_module(module): # sometimes copy.deepcopy() does not work
    module = copy.deepcopy(pickle.loads(pickle.dumps(module)))
    module._forward_hooks.popitem()  # remove hooks from module copy
    module._backward_hooks.popitem()  # remove hooks from module copy
    return module

def redefine_nn(model, rule): # go over model layers and overload chosen instance methods (e.g. forward())
    lrp_rule = Rules(rule)
    list_of_layers = dir(layers) # list of redefined layers in layers module

    try:
        model_name = model.__class__.__name__
    except: # multi-gpus
        model_name = model.module.__class__.__name__

    flattened_model = flatten_model(model) # just for debugging
    for i in range(len(flattened_model)):
        print('[%d]: %s' % (i, flattened_model[i]))

    # for name, layer in enumerate(flattened_model):
    #     if isinstance(layer, torch.nn.MaxPool2d):
    #         flattened_model[name] = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    for layer_num, module in enumerate(flatten_model(model)):
        if  module.__class__.__name__ in list_of_layers:
            local_class = module.__class__ # current layer class. e.g., <class 'torch.nn.modules.conv.Conv2d'>
            layer_module_class = layers.__getattr__(local_class.__name__) # get the same redefined layer class. e.g., <class 'LRP.layers.Conv2d'>
            list_of_methods = [attr for attr in dir(layer_module_class) if attr[:2] != '__'] # methods which  was redefined. e.g., ['conv2d_forward']

            for l in list_of_methods: # overload object method from https://stackoverflow.com/questions/394770/override-a-method-at-instance-level
                setattr(module, l, types.MethodType(getattr(layer_module_class, l), module)) # set redefined methods to object

            if model_name == 'AlexNet':
                if layer_num == 0:
                    setattr(module, 'lrp_rule', Rules('z_rule_beta'))  # first layer always z_rule_beta
                elif layer_num < 6:
                    setattr(module, 'lrp_rule', Rules('z_rule_gamma'))
                elif layer_num >= 6 and layer_num < 13:
                    setattr(module, 'lrp_rule', Rules('z_rule_epsilon'))
                else:
                    setattr(module, 'lrp_rule', lrp_rule)
            elif model_name == 'VGG':
                if layer_num == 0:
                    setattr(module, 'lrp_rule', Rules('z_rule_beta'))  # first layer always z_rule_beta
                # elif layer_num < 17:
                elif layer_num < 19:
                    setattr(module, 'lrp_rule', Rules('z_rule_gamma'))
                # elif layer_num >= 17 and layer_num < 31:
                elif layer_num >= 19 and layer_num < 37: 
                    setattr(module, 'lrp_rule', Rules('z_rule_epsilon'))
                else:
                    setattr(module, 'lrp_rule', lrp_rule)

    return model
