import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from . import utils

class LRP():

    def __init__(self, model, rule='z_rule'):
        self.model = copy.deepcopy(model)
        self.model = self.model.eval()
        self.model = utils.redefine_nn(self.model, rule=rule)
        self.output = None

    def forward(self, input_):
        self.local_input = input_.clone().detach()
        self.local_input.requires_grad_(True)
        output = self.model(self.local_input)
        return output

    def relprop(self, input_, R=None):
        output = self.forward(input_) # after this step object will have local_input attribute, which will request gradient
        if R is None: # if input R (relevance) is None select max logit
            R = torch.zeros_like(output)
            R[torch.arange(len(output)), output.argmax(axis=1)] = 1.
        output.backward(R, retain_graph=True) # calculate gradients of the output with respect to inputs/parameters. The input of "backward" is to calculate a weighted sum of each element
        C = self.local_input.grad.clone().detach()
        assert C is not None, 'obtained relevance is None'
        self.local_input.grad = None
        R = C # *input_.clone().detach()
        R = R.data.cpu().numpy() # tensor to numpy
        return R

    def relprop_ff_driven(self, input_, R=None):
        output = self.forward(input_) # after this step object will have local_input attribute, which will request gradient
        if R is None: # if input R (relevance) is None select max logit
            R = output
        output.backward(R, retain_graph=True) # calculate gradients of the output with respect to inputs/parameters. The input of "backward" is to calculate a weighted sum of each element
        C = self.local_input.grad.clone().detach()
        assert C is not None, 'obtained relevance is None'
        self.local_input.grad = None
        R = C # *input_.clone().detach()
        R = R.data.cpu().numpy() # tensor to numpy
        return R

    def forward_layer_specific(self, input_):
        self.local_input = input_.clone().detach()
        self.local_input.requires_grad_(True)

        self.layerwise_grad_inputs = []
        def hook_fn(module, grad_input, grad_output):
            self.layerwise_grad_inputs.insert(0, grad_input[0].detach())
        def get_all_layers(model):
            for name, layer in model._modules.items():
                if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
                    get_all_layers(layer)
                else:
                    layer.register_backward_hook(hook_fn)

        get_all_layers(self.model)  # just register hooks
        output = self.model(self.local_input)

        return output

    def relprop_layer_specific(self, input_, R=None, layer_index=0):
        output = self.forward_layer_specific(input_)
        if R is None:  # if input R (relevance) is None select max logit
            R = torch.zeros_like(output)
            R[torch.arange(len(output)), output.argmax(axis=1)] = 1.
        output.backward(R, retain_graph=True) # calculate gradients of the output with respect to inputs/parameters. The input of "backward" is to calculate a weighted sum of each element
        R = self.layerwise_grad_inputs[layer_index]
        R = R.data.cpu().detach().numpy() # tensor to numpy
        return R

    def forward_ff_driven_layer_specific(self, input_, layer_index):
        self.local_input = input_.clone().detach()
        self.local_input.requires_grad_(True)

        check_index = []
        def hook_fn(module, input, output):
            index = len(check_index)
            if index == layer_index:
                input[0].backward(input[0].detach(), retain_graph=False)
            check_index.append(0)
        def get_all_layers(model):
            for name, layer in model._modules.items():
                if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
                    get_all_layers(layer)
                else:
                    layer.register_forward_hook(hook_fn)
        get_all_layers(self.model)  # just register hooks
        output = self.model(self.local_input)

        return output

    def relprop_ff_driven_layer_specific(self, input_, layer_index=-1):
        _ = self.forward_ff_driven(input_, layer_index)
        R = self.local_input.grad.clone().detach()
        self.local_input.grad = None
        R = R.data.cpu().numpy() # tensor to numpy
        return R

    __call__ = relprop


