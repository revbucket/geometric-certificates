import torch
import torch.nn as nn
import torch.nn.functional as F
from polytope import Polytope, Face
import utilities as utils


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
        num_logits = poly_b.shape[0]

        true_a = total_a[true_label]
        constraints_a = total_a - true_a

        true_b = total_b[true_label]
        constraints_b = total_b - true_b

        # Append a row constraint for each of the logits except the true one
        facets = []
        for i in range(num_logits):
            if i == true_label:
                continue
            constraint_a_to_add = constraints_a[i]
            constraint_b_to_add = constraints_b[i]
            new_facet = Face(np.stack(poly_a, constraint_a_to_add, dim=0),
                             np.stack(poly_b, constraint_b_to_add, dim=0),
                             [num_constraints], config=None)
            new_facet.check_feasible()
            if new_facet.is_feasible:
                facets.append(new_facet)

        return facets



    def compute_polytope_config(self, configs):

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

        polytope_A = torch.cat(a_stack, dim=0)
        polytope_b = torch.cat(b_stack, dim=0)
        return {'poly_a': polytope_A,
                'poly_b': polytope_b,
                'configs': configs,
                'total_a': wks[-1],
                'total_b': bks[-1]
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

