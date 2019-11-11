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

import full_lp as flp

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
        assert len(config_str) ==  sum(self.layer_sizes[1:-1])
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

    def make_adversarial_constraints(self, polytope, true_label,
                                     domain):
        """ Given a config computes the linear map in terms of this config
            for all neurons INCLUDING the output neurons (logits) and generates
            the polytope constraints for the neuron config and
            constraints for each of the decision boundaries

            configs - as usual
            true_label -

        """

        # Make all the adversarial_constraints:

        #if(x) = Ax + b (in R^#logits)
        # adversarial constraints are:
        #   f_true(x) - f_j(x) = 0 (for all j != true)
        #  ~ which is ~
        #   <a_true, x> + b_true - <a_j, x> - b_j = 0
        #  ~ which is ~
        # <a_true - a_j, x> = b_j - b_true


        total_a = polytope.linear_map['A']
        total_b = polytope.linear_map['b']

        num_logits = total_a.shape[0]

        facets = []
        true_a = total_a[true_label]
        true_b = total_b[true_label]

        for i in range(num_logits):
            if i == true_label:
                continue
            dec_bound = {'A': true_a - total_a[i],
                         'b': total_b[i] - true_b}
            new_facet = polytope.facet_constructor(None, facet_type='decision',
                                                   extra_tightness=dec_bound)
            if new_facet.fast_domain_check():
                facets.append(new_facet)
        return facets



    def compute_polytope_config(self, configs, comparison_form_flag=False,
                                uncertain_constraints=None, as_tensor=False):

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
        if as_tensor:
            return {'a_stack': a_stack,
                    'b_stack': b_stack,
                    'total_a': wks[-1],
                    'total_b': bks[-1]}

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





    def compute_polytope(self, x, comparison_form_flag=False, as_tensor=False):
        pre_relus, configs = self.relu_config(x, return_pre_relus=True)
        poly_out = self.compute_polytope_config(configs, comparison_form_flag,
                                                as_tensor=as_tensor)
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


    def compute_interval_bounds(self, domain_obj, compute_logit_bounds=False,
                                as_tensor=False):
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

        # Redoing this one more time


        # Redo this but doing it right :
        midpoint_matrix = torch.Tensor([[1.0], [1.0]]) / 2.0
        ranges_matrix = torch.Tensor([[-1.0], [1.0]]) / 2.0
        returned_bounds = []
        dead_set = [] # list of tensors, 1 if always on or off
        working_bounds = box
        current_low, current_high = box[:, 0], box[:, 1]

        if compute_logit_bounds:
            layers_to_check = self.fcs
        else:
            layers_to_check = self.fcs[:-1]

        for fc in layers_to_check:
            weight, bias = fc.weight, fc.bias

            weight_pos, weight_neg = utils.split_tensor_pos(weight)
            new_high = (torch.matmul(weight_pos, current_high) +
                        torch.matmul(weight_neg, current_low))
            new_low = (torch.matmul(weight_pos, current_low) +
                       torch.matmul(weight_neg, current_high))
            if bias is not None:
                new_high += bias
                new_low += bias
            returned_bounds.append(torch.stack([new_low, new_high], dim=1))

            current_low = F.relu(new_low)
            current_high = F.relu(new_high)

        if as_tensor:
            return returned_bounds
        else:
            return [utils.as_numpy(_) for _ in returned_bounds]


    def compute_improved_ia_bounds(self, domain_obj):
        """ Implements the improved interval bounds as presented here:
            https://arxiv.org/pdf/1809.03008.pdf (appendix C)
            [also done with gradients pushed through so we can build RS loss ]

            # CODE HEAVILY BORROWED FROM https://github.com/MadryLab/relu_stable/blob/master/models/MNIST_improved_ia.py
            # (but we're transposed from that code)
        """

        box = domain_obj.box_to_tensor()
        init_lows = box[:, 0]
        init_highs = box[:, 1]
        assert all(init_lows <= init_highs) # assert lows less than highs
        layers_to_check = self.fcs[:-1] # set the

        intermed_lows, intermed_highs = [], []

        # define the recursive call
        def recurs(layer_num, lows, highs, weights, biases):
            assert len(lows) == len(highs) == len(weights) == len(biases) == layer_num
            # current layer
            low = lows[0]
            high = highs[0]
            weight = weights[0]
            bias = biases[0]

            # Base case
            if layer_num == 1:
                weight_pos, weight_neg = utils.split_tensor_pos(weight)
                next_low = (torch.matmul(weight_pos, init_lows) +
                            torch.matmul(weight_neg, init_highs) + bias)
                next_high = (toch.matmul(weight_pos, init_highs) +
                             torch.matmul(weight_neg, init_lows) + bias)
                return next_low, next_high

            # Recursive case
            prev_weight = weights[1]
            prev_bias = biases[1]


            # Compute W_A, W_N (need to zero out COLUMNS here)
            w_a = torch.matmul(weight, (low > 0).diag_embed())
            w_n = weight - w_a
            w_n_pos, w_n_neg = utils.split_tensor_pos(w_n)

            w_prod = torch.matmul(w_a, prev_weight)
            b_prod = torch.matmul(w_a, prev_bias)

            # Compute prev layer bounds
            prev_low = (torch.matmul(w_n_pos, low) +
                        torch.matmul(w_n_neg, high) + bias)
            prev_high = (torch.matmul(w_n_pos, high) +
                         torch.matmul(w_n_neg, low) + bias)

            # Recurse
            deeper_lows, deeper_highs = recurs(layer_num - 1, lows[1:], highs[1:],
                                               [w_prod] + weights[2:],
                                               [b_prod] + biases[2:])
            return (prev_low + deeper_lows, prev_high + deeper_highs)


        # compute the lower and upper bounds for all neurons
        running_lows = [init_lows]
        running_highs = [init_highs]
        running_weights = [self.fcs[0].weight]
        running_biases = [self.fcs[0].bias]

        for layer_num, layer in enumerate(self.fcs[:-1]):
            new_lows, new_highs = recurs(layer_num + 1, running_lows, running_highs,
                                         running_weights, running_biases)
            running_lows = [new_lows] + running_lows
            running_highs = [new_highs] + running_highs
            running_weights = self.fcs[layer_num + 1].weight
            running_biases = self.fcs[layer_num + 1].bias
        return running_lows[::-1], running_highs[::-1]



    def compute_full_lp_bounds(self, domain_obj):
        """ Compute the full linear program values.
            Code here is in a different file
        """
        return flp.compute_full_lp_bounds(self, domain_obj)



    def compute_dual_lp_bounds(self, domain_obj):
        """ Use KW to actually find the bounds. Uses L_inf bounds to help
            get better bounds
        """
        low_bounds = torch.Tensor(domain_obj.box_low)
        high_bounds = torch.Tensor(domain_obj.box_high)
        midpoint = ((low_bounds + high_bounds) / 2.0).view(1, -1)
        box_bounds = (low_bounds, high_bounds)

        dual_net = ca.DualNetwork(self.net, midpoint, domain_obj.linf_radius,box_bounds=box_bounds).dual_net
        bounds, dead_set = [], []
        for el in dual_net:
            if isinstance(el, ca.DualReLU):
                bounds.append(torch.cat((el.zl.view(-1, 1), el.zu.view(-1, 1)),
                                        dim=1))
                dead_set.append(~el.I.squeeze())
        return bounds


    def compute_dual_ia_bounds(self, domain_obj):
        """ Use both interval analysis and dual bounds to get best bounds """

        ia = self.compute_interval_bounds(domain_obj)
        dd = self.compute_dual_lp_bounds(domain_obj)
        bounds = []
        for i, d in zip(ia, dd):
            stacked = torch.stack((i, d))
            new_lows = torch.max(stacked[:, :, 0], dim=0)[0]
            new_highs = torch.min(stacked[:, :, 1], dim=0)[0]
            new_bounds = torch.stack((new_lows, new_highs), dim=1)
            bounds.append(new_bounds)
        return bounds


    def fast_lip_all_vals(self, x, l_q, on_off_neurons):
        """ Does the fast_value for all possible c's """
        num_logits = self.fcs[-1].out_features
        if not isinstance(x, torch.Tensor):
            true_label = self(torch.Tensor(x)).max(1)[1].item()
        else:
            true_label = self(x).max(1)[1].item()

        c_vecs, lip_values = [], []
        for i in range(num_logits):
            if true_label == i:
                continue
            c_vec = torch.zeros(num_logits)
            c_vec[true_label] = 1.0
            c_vec[i] = -1.0
            lip_value = self.fast_lip(c_vec, l_q, on_off_neurons)
            c_vecs.append(c_vec)
            lip_values.append(lip_value)


        return c_vecs, lip_values


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
            constant_term, upper, lower = subroutine_out
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


class LinearRegionCollection(object):
    """ Takes a ReturnObj and builds a lot of linear regions and stores them
    """

    def __init__(self, plnn_obj, return_obj, objective_vec=None, 
                 do_setup=False):
        self.plnn_obj = plnn_obj
        self.return_obj = return_obj
        self.collection = {}
        for config in return_obj.seen_polytopes:
            self.collection[config] = LinearRegion(plnn_obj, config, 
                                                   return_obj=return_obj,
                                                   objective_vec=objective_vec, 
                                                   do_setup=do_setup)

    def get_maximum_lipschitz_constant(self):
        return max(_.get_lipschitz_constant() 
                   for _ in self.collection.values())

    def gradient_angle_list(self):
        """ Gets the gradient angles between neighboring linear regions """
        angle_list = {} 
        for (u, v) in self.return_obj.polytope_graph.keys():
            u_grad = self.collection[u].get_gradient() 
            v_grad = self.collection[v].get_gradient() 
            angle_list[(u, v)] = utils.angle(u_grad, v_grad)
        return angle_list

    def gradient_magnitude_diff_list(self, grad_fxn=None):
        """ Gets the magnitude of gradient difference 
            between neighboring linear regions
        """

        if grad_fxn is None:
            grad_fxn = lambda u, v: torch.norm(u - v).item() 
        output = {} 
        for (u, v) in self.return_obj.polytope_graph.keys():
            u_grad = self.collection[u].get_gradient()
            v_grad = self.collection[v].get_gradient()
            output[(u, v)] = grad_fxn(u_grad, v_grad)
        return output


    def get_greedy_lipschitz_components(self):
        """ Returns dict of str -> [str1, ..., ] mapping locally maximal
            linear regions to the set of regions that will greedily 
            approach this local max
        """
        # Let's just be really naive about this 

        def get_ascent_neighbor(node): 
            """ Gets the neighbor that has highest lipschitz constant 
                Returns None if nothing has higher than this one 
            """

            current = node.get_lipschitz_constant() 
            neighbors = [(_, _.get_lipschitz_constant()) 
                          for _ in node.get_neighbors()]
            max_neighbor = max(neighbors, key=lambda p: p[1])
            if max_neighbor[1] > current:
                return max_neighbor[0] 
            return None 

        def greedy_search_single_node(start_config):
            """ Start with a single sign_config and do greedy search
                to find max_lipschitz constant. Return the sign_config
                of the greedy search output 
            """
            current_node = self.collection[start_config]
            while True: 
                next_node = get_ascent_neighbor(current_node)
                if next_node is None:
                    break 
                else:
                    current_node = next_node
            return current_node.sign_config

        greedy_output = {} 
        for config in self.collection.keys(): 
            greedy_parent = greedy_search_single_node(config)
            if greedy_parent not in greedy_output:
                greedy_output[greedy_parent] = []
            greedy_output[greedy_parent].append(config)

        return greedy_output



class LinearRegion(object):
    """ Holds info and shortcuts to work with linear regions """
    @classmethod
    def process_return_obj(cls, plnn_obj, return_obj, objective_vec=None,
                           do_setup=False):
        """ Given a GeoCertReturn object, will build a linear region for
            all of the 'seen polytopes' and return the outputs in a
            dict keyed on teh sign_configs
        """

        output = {}
        for config in return_obj.seen_polytopes:
            output[config] = cls(plnn_obj, config,
                                 return_obj=return_obj,
                                 objective_vec=objective_vec,
                                 do_setup=do_setup)
        return output


    def __init__(self, plnn_obj, sign_config, return_obj=None,
                 objective_vec=None, do_setup=False):
        """ Initializes a Linear Region object
        ARGS:
            plnn_obj - the network this region is linear for
            sign_config - the neuron configuration of the region
            return_obj : GeoCertReturn object - if not None is an
                         output of GeoCert which contains info about
                         the linear regions.
        """
        super(LinearRegion, self).__init__()
        self.plnn_obj = plnn_obj
        self.sign_config = sign_config
        self.hex_config = hex(int(self.sign_config, 2))
        self.return_obj = return_obj
        self.objective_vec = objective_vec

        # setting up attributes to be stored later
        self._polytope_config = None
        self.polytope = None
        self.linear_map = None
        self.jacobian = None
        self.largest_sv = None

        if do_setup:
            self.setup()

    def __repr__(self):
        return "LinearRegion: %s" % self.hex_config

    def get_neighbors(self):
        """ If the return obj is not None, will error. Otherwise will
            return a list of neighboring LinearRegion objects
        """
        assert self.return_obj is not None
        neigbor_list = []
        for edge in self.return_obj.polytope_graph:
            if self.sign_config == edge[0]:
                neigbor_idx = 1
            elif self.sign_config == edge[1]:
                neigbor_idx = 0
            else:
                continue
            neigbor_list.append(edge[neigbor_idx])

        return [LinearRegion(self.plnn_obj, neigbor_config,
                             return_obj=self.return_obj,
                             objective_vec=self.objective_vec)
                for neigbor_config in neigbor_list]


    def _get_polytope_config(self):
        if self._polytope_config is not None:
            return self._polytope_config

        plnn_obj = self.plnn_obj
        config = plnn_obj.config_str_to_config_list(self.sign_config)
        self._polytope_config = plnn_obj.compute_polytope_config(config)
        return self._polytope_config


    def setup(self):
        self.get_polytope()
        self.get_linear_map()
        self.get_jacobian()
        self.get_largest_singular_value()


    def get_polytope(self):
        """ For this linear region will return the polytope for which
            the neural net satisfies the given neuron configuration
        """
        if self.polytope is not None:
            return self.polytope

        _polytope_config = self._get_polytope_config()
        self.polytope = {'A': _polytope_config['poly_a'],
                         'b': _polytope_config['poly_b']}
        return self.polytope


    def get_linear_map(self):
        """ For this linear region will return a torch.nn.Linear
            object corresponding to the linear map at this neuron
            configuration
        """
        if self.linear_map is not None:
            return self.linear_map
        _polytope_config = self._get_polytope_config()
        A = nn.Parameter(_polytope_config['total_a'])
        b = nn.Parameter(_polytope_config['total_b'])
        linear_map = nn.Linear(*A.shape)
        linear_map.weight = A
        linear_map.bias = b

        self.linear_map = linear_map
        return self.linear_map

    def get_jacobian(self):
        """ For this linear region will get the jacobian at this
            linear piece
        """
        if self.jacobian is not None:
            return self.jacobian

        linear_map = self.get_linear_map()
        self.jacobian = linear_map.weight
        return self.jacobian

    def get_largest_singular_value(self):
        """ Will return the largest singular value of the jacobian
            of this linear region
        """
        if self.largest_sv is not None:
            return self.largest_sv

        jacobian = self.get_jacobian()
        self.largest_sv = jacobian.svd().S[0].item()
        return self.largest_sv

    def get_gradient(self):
        assert self.objective_vec is not None
        return self.objective_vec.matmul(self.get_jacobian())

    def get_lipschitz_constant(self):
        if self.objective_vec is not None:
            return self.objective_vec.matmul(self.get_jacobian()).norm().item()
        else:
            return self.get_largest_singular_value()

