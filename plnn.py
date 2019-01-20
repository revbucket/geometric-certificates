import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import *

class PLNN(nn.Module):
    """ Simple piecewise neural net.
        Fully connected layers and ReLus only
    """
    def __init__(self, layer_sizes=None):
        super(PLNN, self).__init__()

        if layer_sizes is None:
            layer_sizes = [32, 64, 128, 64, 32, 10]
        self.layer_sizes = layer_sizes
        self.fcs = []
        for i in range(1, len(layer_sizes)):
            self.fcs.append(nn.Linear(layer_sizes[i-1],
                                      layer_sizes[i]))

    def relu_config(self, x, return_pre_relus=True):
        pre_relus = self.forward_by_layer(x)

        configs = [(pre_relu.squeeze() > 0).type(torch.float32)
                   for pre_relu in pre_relus]
        if return_pre_relus:
            return pre_relus, configs
        else:
            return configs


    def compute_polytope_config(self, configs, comparison_form_flag=False):

        lambdas = [torch.diag(config) for config in configs]
        js = [torch.diag(-2 * config + 1) for config in configs]

        # Compute Z_k = W_k * x + b_k for each layer
        wks = [self.fcs[0].weight]
        bks = [self.fcs[0].bias]
        for (i, fc) in enumerate(self.fcs[1:-1]):
            current_wk = wks[-1]
            current_bk = bks[-1]
            current_lambda = lambdas[i]
            precompute = fc.weight.matmul(current_lambda)
            wks.append(precompute.matmul(current_wk))
            bks.append(precompute.matmul(current_bk) + fc.bias)

        a_stack = []
        b_stack = []
        for j, wk, bk in zip(js, wks, bks):
            a_stack.append(j.matmul(wk))
            b_stack.append(-j.matmul(bk))

        polytope_A = torch.cat(a_stack, dim=0).detach().numpy()
        polytope_b = torch.cat(b_stack, dim=0).detach().numpy()

        if(comparison_form_flag):
            polytope_A, polytope_b = comparison_form(polytope_A, polytope_b)


        return {'poly_a': polytope_A,
                'poly_b': polytope_b,
                'configs': configs,
                }

    def compute_polytope(self, x):
        pre_relus, configs = self.relu_config(x, return_pre_relus=True)
        poly_out = self.compute_polytope_config(configs)
        poly_out['pre_relus'] = pre_relus
        return poly_out

    def forward_by_layer(self, x):
        pre_relus = []

        x = x.view(-1, self.layer_sizes[0])
        for fc in self.fcs[:-1]:
            x = fc(x)
            pre_relus.append(x.clone())
            x = F.relu(x)
        return pre_relus


    def forward(self, x):
        x = x.view(-1, self.layer_sizes[0])
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        return self.fcs[-1](x) # No ReLu on the last one

