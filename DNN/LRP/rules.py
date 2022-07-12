import torch
import numpy as np
import copy

class Rules(object):

    all_rules = ['z_rule', 'z_rule_epsilon', 'z_rule_gamma', 'z_rule_beta']

    def __init__(self, rule):
        assert isinstance(rule, str), 'rule parameter should be of type str'
        assert rule in self.all_rules, 'Rule "{}" not implemented. Implemented rules {}'.format(rule, self.all_rules)
        self.rule = rule

    def __call__(self, *args):
        if self.rule  == 'z_rule':
            return self.z_rule(*args, keep_bias=True)
        elif self.rule == 'z_rule_epsilon':
            return self.z_rule_epsilon(*args, keep_bias=True)
        elif self.rule == 'z_rule_gamma':
            return self.z_rule_gamma(*args, keep_bias=True)
        elif self.rule == 'z_rule_beta':
            return self.z_rule_beta(*args, keep_bias=True)
        else:
            ValueError('Rule "{}" not implemented'.format(rule))

    @staticmethod
    def z_rule(input, func, func_args, R, keep_bias=False):
        input.requires_grad_(True)
        if input.grad is not None: input.grad.zero_() # otherwise accumulation of gradient happening
        func_args = copy.deepcopy(func_args)
        if func_args.get('bias', None) is not None:
            if not keep_bias:
                func_args['bias'] = None

        with torch.enable_grad():
            Z = func(input, **func_args)
            Z = Z + 1e-9
            S = (R / Z).data
            (Z * S).sum().backward() # same as Z.backward(S)
            assert input.grad is not None
            C = input.grad
            Ri = (input * C).data
            # print(Ri.sum()) # just check

        return Ri

    @staticmethod
    def z_rule_epsilon(input, func, func_args, R, keep_bias=False):
        input.requires_grad_(True)
        if input.grad is not None: input.grad.zero_() # otherwise accumulation of gradient happening
        func_args = copy.deepcopy(func_args)
        if func_args.get('bias', None) is not None:
            if not keep_bias:
                func_args['bias'] = None

        epsilon = 0.25

        with torch.enable_grad():
            Z = func(input, **func_args)
            Z = Z + 1e-9 + epsilon * ((Z**2).mean(axis=tuple(range(1, Z.ndim)), keepdim=True)**.5).data
            # Z[Z == 0.] = 1e-9 # prevent from dividing by 0
            S = (R / Z).data
            (Z * S).sum().backward() # same as Z.backward(S)
            assert input.grad is not None
            C = input.grad
            Ri = (input * C).data

        return Ri

    @staticmethod
    def z_rule_gamma(input, func, func_args, R, keep_bias=False):
        input.requires_grad_(True)
        if input.grad is not None: input.grad.zero_() # otherwise accumulation of gradient happening
        func_args = copy.deepcopy(func_args)
        if func_args.get('bias', None) is not None:
            if not keep_bias:
                func_args['bias'] = None

        gamma = 0.25
        rho = lambda p: p + gamma * p.clamp(min=0)

        with torch.enable_grad():
            Z = func(input, **newlayer(func_args, rho))
            Z = Z + 1e-9
            S = (R / Z).data
            (Z * S).sum().backward() # same as Z.backward(S)
            assert input.grad is not None
            C = input.grad
            Ri = (input * C).data

        return Ri

    @staticmethod
    def z_rule_beta(input, func, func_args, R, keep_bias=False):
        input.requires_grad_(True)
        if input.grad is not None: input.grad.zero_() # otherwise accumulation of gradient happening
        func_args = copy.deepcopy(func_args)
        if func_args.get('bias', None) is not None:
            if not keep_bias:
                func_args['bias'] = None

        # imagenet_mean = [0.485, 0.456, 0.406]; imagenet_std = [0.229, 0.224, 0.225] # rgb
        imagenet_mean = [0.449]; imagenet_std = [0.226] # gray
        # imagenet_mean = [0.449 * 255]; imagenet_std = [0.226 * 255] # gray

        if  input.is_cuda:
            mean = torch.cuda.FloatTensor(imagenet_mean).reshape(1, -1, 1, 1)
            std = torch.cuda.FloatTensor(imagenet_std).reshape(1, -1, 1, 1)
        else:
            mean = torch.FloatTensor(imagenet_mean).reshape(1, -1, 1, 1)
            std = torch.FloatTensor(imagenet_std).reshape(1, -1, 1, 1)

        with torch.enable_grad():

            lb = (input.data*0 + (0-mean)/std).requires_grad_(True)
            hb = (input.data*0 + (1-mean)/std).requires_grad_(True)

            Z = func(input, **func_args)
            Z = Z + 1e-9
            Z -= func(lb, **newlayer(func_args, lambda p: p.clamp(min=0)))
            Z -= func(hb, **newlayer(func_args, lambda p: p.clamp(max=0)))
            # Z[Z == 0.] = 1e-9 # prevent from dividing by 0, it occurs a cuda OOM error for some reason..
            S = (R / Z).data    
            (Z * S).sum().backward() # same as Z.backward(S)
            assert input.grad is not None
            C, Cp, Cm = input.grad, lb.grad, hb.grad
            Ri = (input * C + lb * Cp + hb * Cm).data

        return Ri

# --------------------------------------------------------------
# Clone a layer and pass its parameters through the function g
# --------------------------------------------------------------
def newlayer(layer, g):

    if 'weight' in layer:
        layer = copy.deepcopy(layer)

        try: layer['weight'] = torch.nn.Parameter(g(layer['weight']))
        except AttributeError: pass

        try: layer['bias']   = torch.nn.Parameter(g(layer['bias']))
        except AttributeError: pass

    return layer
