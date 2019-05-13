import torch
import torch.nn as nn
import torch.nn.functional as F
from _polytope_ import Polytope, Face
import utilities as utils
from collections import OrderedDict
import numpy as np

import time
import copy
import convex_adversarial.convex_adversarial as ca



class PLNN(nn.Module):
    #TODO: determine if building net addition was necessary
    # add some explanations for some methods
    """ Simple piecewise neural net.
        Fully connected layers and ReLus only
    """
    def __init__(self, layer_sizes=None, bias=True, dtype=torch.FloatTensor):
        super(PLNN, self).__init__()

        if layer_sizes is None:
            layer_sizes = [32, 64, 128, 64, 32, 10]
        self.layer_sizes = layer_sizes
        self.dtype = dtype
        self.fcs = []
        self.bias = bias
        self.net = self.build_network(layer_sizes)

    def build_network(self, layer_sizes):
        layers = OrderedDict()

        num = 1
        for size_pair in zip(layer_sizes, layer_sizes[1:]):
            size, next_size = size_pair
            layer = nn.Linear(size, next_size, bias=self.bias).type(self.dtype)
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

    def make_adversarial_constraints(self, configs, true_label,
                                     domain):
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
        flat_config = utils.flatten_config(configs)
        for i in range(num_logits):
            if i == true_label:
                continue
            constraint_a_to_add = constraints_a[i]
            constraint_b_to_add = constraints_b[i]

            new_facet = Face(np.vstack((poly_a, constraint_a_to_add)),
                             np.hstack((poly_b, constraint_b_to_add)),
                             [num_constraints], config=flat_config,
                             domain=domain, facet_type='decision')

            if new_facet.fast_domain_check():
                facets.append(new_facet)

        return facets



    def compute_polytope_config(self, configs, comparison_form_flag=False,
                                uncertain_constraints=None):

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

    def compute_polytope(self, x, comparison_form_flag=False):
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


    def compute_interval_bounds(self, domain_obj, on_off_format=False):
        """ For each neuron computes a bound for the range of values each
            pre-ReLU can take.
        ARGS:
            domain_obj : Domain - object used to hold bounding boxes
            on_off_format: boolean - if True, we return the more fine-grained
                                     list which displays if neurons are on or
                                     off, instead of stable
        RETURNS:
            returned_bounds : list of tensors giving pre-Relu bounds
            uncertain_set: list of tensors with 1 if uncertain about this
                           neuron in the list
            list of length (# fully connected layers - 1), where each element
            is a tensor of shape (num_neurons, 2) for the bounds for the preReLU
        """

        box = domain_obj.box_to_tensor()
        # setup + asserts
        assert all(box[:, 0] <= box[:, 1])

        midpoint_matrix = torch.Tensor([[1.0], [1.0]]) / 2.0
        ranges_matrix = torch.Tensor([[-1.0], [1.0]]) / 2.0
        returned_bounds = []
        dead_set = [] # list of tensors, 1 if always on or off
        working_bounds = box
        for fc in self.fcs[:-1]:
            weight, bias = fc.weight, fc.bias
            midpoint = torch.matmul(working_bounds, midpoint_matrix)
            ranges = torch.matmul(working_bounds, ranges_matrix)
            new_midpoint = torch.matmul(weight, midpoint)
            if bias is not None:
                new_midpoint = bias.view(-1, 1)
            new_ranges = torch.matmul(torch.abs(weight), ranges)

            pre_relus = torch.cat([new_midpoint - new_ranges,
                                   new_midpoint + new_ranges], 1)
            dead_set.append((pre_relus[:,0] * pre_relus[:,1]) >= 0)
            returned_bounds.append(pre_relus)
            working_bounds = F.relu(pre_relus)

        if on_off_format is False:
            return returned_bounds, dead_set
        else:
            on_off_list = []
            for el in returned_bounds:
                on_off_el = torch.LongTensor(el.shape[0])
                on_off_el[:] = 0
                on_off_el[el[:, 1] < 0] = -1
                on_off_el[el[:, 0] >= 0] = 1
                on_off_list.append(on_off_el)
            return on_off_list




    def compute_dual_lp_bounds(self, domain_obj):
        """ Use KW to actually find the bounds. Uses L_inf bounds to help
            get better bounds
        """
        low_bounds = torch.Tensor(domain_obj.box_low)
        high_bounds = torch.Tensor(domain_obj.box_high)
        midpoint = ((low_bounds + high_bounds) / 2.0).view(1, -1)
        box_bounds = (low_bounds, high_bounds)

        dual_net = ca.DualNetwork(self.net, midpoint, domain_obj.linf_radius,
                                  box_bounds=box_bounds).dual_net

        bounds, dead_set = [], []
        for el in dual_net:
            if isinstance(el, ca.DualReLU):
                bounds.append(torch.cat((el.zl.view(-1, 1), el.zu.view(-1, 1)),
                                        dim=1))
                dead_set.append(~el.I.squeeze())

        return bounds, dead_set

    def compute_dual_ia_bounds(self, domain_obj):
        """ Use both interval analysis and dual bounds to get best bounds """

        ia = self.compute_interval_bounds(domain_obj)[0]
        dd = self.compute_dual_lp_bounds(domain_obj)[0]

        bounds = []
        dead_set = []
        for i, d in zip(ia, dd):
            stacked = torch.stack((i, d))
            new_lows = torch.max(stacked[:, :, 0], dim=0)[0]
            new_highs = torch.min(stacked[:, :, 1], dim=0)[0]
            new_bounds = torch.stack((new_lows, new_highs), dim=1)
            bounds.append(new_bounds)
            dead_set.append((new_bounds[:, 0] * new_bounds[:, 1]) >= 0)
        return bounds, dead_set

    def fast_lip(self, c_vector, l_q, on_off_neurons):
        """
        Pytorch implementation of fast_lip. Might be buggy? Who knows?
        see : https://arxiv.org/pdf/1804.09699.pdf for details

        INPUTS:
            c_vector: tensor that multiplies the output vector:
                      we compute gradient of c^Tf(x)
            l_q : int - q_norm of lipschitzness that we compute
                        (is dual norm: e.g. if bounds come from an l_inf box,
                         this should be 1)
            on_off_neurons : list of LongTensors (entries in -1, 0 or 1)
                             corresponding to the set of
                             (off, uncertain, on, respectively) neurons
                             inside the domain
        RETURNS:
            upper bound on lipschitz constant
        """

        ######################################################################
        #   First generate inputs needed by fast_lip algorithm               #
        ######################################################################

        # --- split off active and uncertain neurons
        # -1 means off (don't care)
        #  0 means UNCERTAIN
        #  1 means ACTIVE

        active_neuron_list, uncertain_neuron_list = [], []
        for neuron_by_layer in on_off_neurons:
            active_neuron_list.append((neuron_by_layer == 1))
            uncertain_neuron_list.append((neuron_by_layer == 0))

        # --- get list of weights, initialize placeholders
        weights = [layer.weight for layer in self.fcs[:-1]]
        weights.append(c_vector.matmul(self.fcs[-1].weight).view(1, -1))

        constant_term = weights[0]
        lowers = [torch.zeros_like(constant_term)]
        uppers = [torch.zeros_like(constant_term)]


        ######################################################################
        #   Loop through layers using the _bound_layer_grad subroutine       #
        ######################################################################

        for i in range(len(weights) - 1):
            subroutine_out = self._bound_layers_grad(constant_term, lowers[-1],
                                                     uppers[-1],
                                                     weights[i + 1],
                                                     active_neuron_list[i],
                                                     uncertain_neuron_list[i])
            constant_term, lower, upper = subroutine_out
            lowers.append(lower)
            uppers.append(upper)

        ######################################################################
        #   Finalize and return the output                                   #
        ######################################################################

        low_bound = (constant_term + lowers[-1]).abs()
        upp_bound = (constant_term + uppers[-1]).abs()

        layerwise_max = torch.where(low_bound > upp_bound, low_bound, upp_bound)
        return torch.norm(layerwise_max, p=l_q).item()


    def _bound_layers_grad(self, constant_term, lower, upper, weight,
                           active_neurons, uncertain_neurons):
        """ Subroutine for fast_lip.
            Assume weight has shape [m, n]

        ARGS: (let's make sure the types and shapes all mesh)
            constant_term: floatTensor shape (n, n_0)
            lower: floatTensor shape (n, n_0)
            upper: floatTensor shape (n, n_0)
            weight: floatTensor shape (m, n)
            active_neurons: torch.Tensor shape (n,)
            uncertain_neurons: torch.Tensor shape (n,)
        RETURNS:
            new constant term, lower, and upper, each with shape (m, n_0)
        """

        # ASSERTS ON SHAPES FOR DEBUGGING
        n_0 = self.layer_sizes[0]
        n = weight.shape[1]
        assert constant_term.shape == (n, n_0)
        assert lower.shape == (n, n_0)
        assert upper.shape == (n, n_0)
        assert active_neurons.shape == (n,)
        assert uncertain_neurons.shape == (n,)



        # Make diagonals and split weights by +/-
        active_diag = torch.diag(active_neurons).float()
        uncertain_diag = torch.diag(uncertain_neurons).float()
        pos_weight, neg_weight = utils.split_tensor_pos(weight)


        # Compute the new constant_term
        new_constant_term = weight.matmul(active_diag).matmul(constant_term)

        # Make new upper bounds/lower bounds
        cons_low = constant_term + lower
        _, neg_cons_low = utils.split_tensor_pos(cons_low)
        cons_upp = constant_term + upper
        pos_cons_upp, _ = utils.split_tensor_pos(cons_upp)


        new_upper = (pos_weight.matmul(active_diag).matmul(upper) +
                     neg_weight.matmul(active_diag).matmul(lower) +
                     neg_weight.matmul(uncertain_diag).matmul(neg_cons_low) +
                     pos_weight.matmul(uncertain_diag).matmul(pos_cons_upp))

        new_lower = (pos_weight.matmul(active_diag).matmul(lower) +
                     neg_weight.matmul(active_diag).matmul(upper) +
                     pos_weight.matmul(uncertain_diag).matmul(neg_cons_low) +
                     neg_weight.matmul(uncertain_diag).matmul(pos_cons_upp))
        return new_constant_term, new_upper, new_lower


class PLNN_seq(PLNN):
    """ Simple piecewise neural net.
        Fully connected layers and ReLus only

        built from nn.Sequential
    """

    def __init__(self, sequential, layer_sizes, dtype=torch.FloatTensor):
        super(PLNN_seq, self).__init__(layer_sizes, dtype)
        self.fcs = [layer for layer in sequential if type(layer) == nn.Linear]

        self.net = sequential
