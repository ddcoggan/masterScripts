import torch
from  torch.nn import modules
import torch.nn.functional as F
import numpy as np

def __getattr__(name): # to get layer modules classes as its attribute
    if name in globals(): # The globals() function returns a dictionary containing the variables defined in the global namespace
        return globals()[name]
    else:
        return AttributeError

#Definition of custom autograd function
class LRP_Standard(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, func, args, lrp_rule): # ctx is a context object that can be used to stash information for backward computation
        ctx.input = input.clone().detach()
        ctx.func = func
        ctx.args = args
        ctx.lrp_rule = lrp_rule
        return func(input, **args)

    @staticmethod
    def backward(ctx, R): # substitute backward pass with z-rule propagation. backward pass must return the same number of ouputs as the number of inputs in forward pass
        R = ctx.lrp_rule(ctx.input, ctx.func, ctx.args, R)
        return R, None, None, None

class LRP_ReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, func, args):
        ctx.args = args
        return func(input, **args)

    @staticmethod
    def backward(ctx, R):
        return R, None, None

class LRP_BatchNorm2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, func, args):
        ctx.input = input
        ctx.args = args
        ctx.output = func(input, **args)
        return ctx.output

    @staticmethod
    def backward(ctx, R):
        """
        Batch normalization can be considered as 3 distinct layers of subtraction, multiplication and then addition.
        The multiplicative scaling layer has no effect on LRP and functions as a linear activation layer

               x * (y - beta)     R
        Rin = ---------------- * ----
                  x - mu          y
        """
        beta = ctx.args['bias']
        mu = ctx.args['running_mean']
        Rout = ctx.input * (ctx.output - beta[None, ..., None, None]) * R / ((ctx.input - mu[None, ..., None, None]) * ctx.output + np.finfo(np.float32).eps)
        return Rout, None, None

class Conv2d(object):
    def forward(self, input):
        if self.padding_mode == 'circular':
            NotImplementedError

        return LRP_Standard.apply(input, F.conv2d, {'weight': self.weight, 'bias': self.bias, 'stride': self.stride, 'padding': self.padding, 'dilation': self.dilation, 'groups': self.groups}, self.lrp_rule)

class BatchNorm2d(object):
    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return LRP_BatchNorm2d.apply(input, F.batch_norm, {'running_mean': self.running_mean, 'running_var': self.running_var, 'weight': self.weight, 'bias': self.bias,
                                                           'training': self.training or not self.track_running_stats, 'momentum': exponential_average_factor, 'eps': self.eps})

class ReLU(object):
    def forward(self, input):
        return LRP_ReLU.apply(input, F.relu, {'inplace': self.inplace})

class Linear(object):
    def forward(self, input):
        return LRP_Standard.apply(input, F.linear, {'weight': self.weight, 'bias': self.bias}, self.lrp_rule)

class AvgPool2d(object):
    def forward(self, input):
        return LRP_Standard.apply(input, F.avg_pool2d, {'kernel_size': self.kernel_size, 'stride': self.stride, 'padding': self.padding,
                                                        'ceil_mode': self.ceil_mode, 'count_include_pad': self.count_include_pad, 'divisor_override': self.divisor_override})

class MaxPool2d(object):
    def forward(self, input):
        return LRP_Standard.apply(input, F.max_pool2d, {'kernel_size': self.kernel_size, 'stride': self.stride, 'padding': self.padding,
                                                        'ceil_mode': self.ceil_mode, 'dilation': self.dilation, 'return_indices': self.return_indices}, self.lrp_rule)
