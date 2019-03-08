import torch
import torch.nn as nn
import torch.nn.functional as F
from _polytope_ import Polytope, Face
import utilities as utils
from collections import OrderedDict
import numpy as np




class PLNN(nn.Module):
    #TODO: determine if building net addition was necessary
    # add some explanations for some methods
    """ Simple piecewise neural net.
        Fully connected layers and ReLus only
    """
    def __init__(self, layer_sizes=None, dtype=torch.FloatTensor):
        super(PLNN, self).__init__()

        if layer_sizes is None:
            layer_sizes = [32, 64, 128, 64, 32, 10]
        self.layer_sizes = layer_sizes
        self.dtype = dtype
        self.fcs = []
        self.net = self.build_network(layer_sizes)

    def build_network(self, layer_sizes):
        layers = OrderedDict()

        num = 1
        for size_pair in zip(layer_sizes, layer_sizes[1:]):
            size, next_size = size_pair
            layer = nn.Linear(size, next_size, bias=True).type(self.dtype)
            layers[str(num)] = layer
            self.fcs.append(layer)
            num = num + 1
            layers[str(num)] = nn.ReLU()
            num = num + 1

        del layers[str(num-1)]      # No ReLU for the last layer

        net = nn.Sequential(layers).type(self.dtype)
        print(self.layer_sizes)

        return net

    def get_parameters(self):
        params = []
        for fc in self.fcs:
            fc_params = [elem for elem in fc.parameters()]
            for param in fc_params:
                params.append(param)

        return params

    def config_str_to_config_list(self, config_str):
        """ Given str of configs, converts to list of torch tensors of right
            layer sizes
        """

        assert isinstance(config_str, str)
        assert len(config_str, sum(self.layer_sizes[1:-1]))
        splits = []
        running_idx = 0
        for el in self.layer_sizes[1:-1]:
            layer_config = config_str[running_idx:running_idx + el]
            layer_config = torch.Tensor([float(el) for el in layer_config])
            # Do some cuda nonsense here?
            splits.append(layer_config)
            running_idx += el
        return splits

    def relu_config(self, x, return_pre_relus=True):
        pre_relus = self.forward_by_layer(x)

        configs = [(pre_relu.squeeze() > 0).type(torch.float32)
                   for pre_relu in pre_relus]
        if return_pre_relus:
            return pre_relus, configs
        else:
            return configs

    def make_adversarial_constraints(self, configs, true_label):
        """ Given a config computes the linear map in terms of this config
            for all neurons INCLUDING the output neurons (logits) and generates
            the polytope constraints for the neuron config and
            constraints for each of the decision boundaries

            configs - as usual
            true_label -
        """
        # Find all adversarial constraints
        polytope_config = self.compute_polytope_config(configs)

        poly_a = utils.as_numpy(polytope_config['poly_a'])
        poly_b = utils.as_numpy(polytope_config['poly_b'])
        total_a = utils.as_numpy(polytope_config['total_a'])
        total_b = utils.as_numpy(polytope_config['total_b'])


        num_constraints = poly_a.shape[0]
        num_logits = total_a.shape[0]


        true_a = total_a[true_label]
        constraints_a = total_a - true_a

        true_b = total_b[true_label]
        constraints_b = -1*total_b + true_b

        # Append a row constraint for each of the logits except the true one
        facets = []
        for i in range(num_logits):
            if i == true_label:
                continue
            constraint_a_to_add = constraints_a[i]
            constraint_b_to_add = constraints_b[i]

            new_facet = Face(np.vstack((poly_a, constraint_a_to_add)),
                             np.hstack((poly_b, constraint_b_to_add)),
                             [num_constraints], config=None)
            new_facet.check_feasible()
            if new_facet.is_feasible:
                facets.append(new_facet)

        return facets



    def compute_polytope_config(self, configs, comparison_form_flag=True):

        lambdas = [torch.diag(config) for config in configs]
        js = [torch.diag(-2 * config + 1) for config in configs]

        # Compute Z_k = W_k * x + b_k for each layer
        wks = [self.fcs[0].weight]
        bks = [self.fcs[0].bias]
        for (i, fc) in enumerate(self.fcs[1:]):
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

        polytope_A = utils.as_numpy(torch.cat(a_stack, dim=0))
        polytope_b = utils.as_numpy(torch.cat(b_stack, dim=0))

        if(comparison_form_flag):
            polytope_A, polytope_b = utils.comparison_form(polytope_A, polytope_b)


        return {'poly_a': polytope_A,
                'poly_b': polytope_b,
                'configs': configs,
                'total_a': wks[-1],
                'total_b': bks[-1]
                }

    def compute_polytope(self, x, comparison_form_flag=True):
        pre_relus, configs = self.relu_config(x, return_pre_relus=True)
        poly_out = self.compute_polytope_config(configs, comparison_form_flag)
        poly_out['pre_relus'] = pre_relus
        return poly_out


    def compute_matrix(self, configs):
        M = torch.eye(self.layer_sizes[0])

        for config, fc, layer_size in zip(configs, self.fcs, self.layer_sizes):
            nullifier = torch.Tensor([config.numpy() for _ in range(0, layer_size)])
            M_layer_prime = fc.weight * torch.transpose(nullifier, 0, 1)
            M = torch.matmul(M_layer_prime, M)

        M = torch.matmul(self.fcs[-1].weight, M)
        return M

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

