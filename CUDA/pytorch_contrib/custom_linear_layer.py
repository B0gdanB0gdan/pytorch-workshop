import torch
import torch.nn as nn
import custom_linear
import math


class CustomLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return custom_linear.forward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()       
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = custom_linear.backward(grad_output, input, weight, bias)
        return grad_input, grad_weight, grad_bias
    

class CustomLinearLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(CustomLinearLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        self.bias = nn.Parameter(torch.empty(output_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
       
    def forward(self, input):
        return CustomLinearFunction.apply(input, self.weight, self.bias)