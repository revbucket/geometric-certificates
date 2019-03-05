
# =====================
# Imports
# =====================

from geocert import compute_boundary_batch, batch_GeoCert, incremental_GeoCert
from plnn import PLNN
from _polytope_ import Polytope, from_polytope_dict
import utilities as utils
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

##########################################################################
#                                                                        #
#                          Exp # 1                                       #
#                                                                        #
##########################################################################

# experiment to find a potentially better way to determine redundant constraints
# of the network
#
# Idea: look at matrices induced by modifiying ReLu configs

# ==================================
# Initialize Network
# ==================================

print('===============Initializing Network============')
layer_sizes = [2, 8, 6, 2]
network = PLNN(layer_sizes)

# ==================================
# Find Projection
# ==================================

# lp_norm = 'l_2'
#
# print('===============Finding Projections============')
# print('lp_norm: ', lp_norm)
# x_0 = torch.Tensor([[0.0], [0.0]])
# print('from point: ')
# print(x_0)
#
# ax = plt.axes()
# cwd = os.getcwd()

# Starting Point
x_0 = torch.Tensor([[0.0], [0.0]])
print('from point: ')
print(x_0)


# Get Relu Configuration
pre_relus, configs = network.relu_config(x_0, return_pre_relus=True)


# Get Polytope
polytope_dict = network.compute_polytope(x_0, comparison_form_flag=True)
polytope = from_polytope_dict(polytope_dict)
test = polytope.generate_facets()
polytope_list = [polytope, ]

# Get Faces
ax = plt.axes()

boundary,_  = compute_boundary_batch(polytope_list, 'fast_ReLu')
utils.plot_polytopes_2d(polytope_list, colors=None, alpha=0.7,
                        xylim=5, ax=ax, linestyle='dashed', linewidth=0)

# # View Tight constraints
tight_constraints = [facet.tight_list[0] for facet in boundary]
print(tight_constraints)


# Change ReLu activations and look at matrices properties
print('original matrix:', -1)
network.compute_matrix(configs)

for index in range(0, np.shape(polytope.ub_A)[0]):
    print('---------------')
    k = 0
    new_configs = []
    for config in configs:
        new_config = []
        for l, elem in enumerate(config):
            if k + l == index:
                new_config.append(1.0-elem)
            else:
                new_config.append(elem)
        new_configs.append(torch.Tensor(new_config))
        k += l+1

    print('matrix:', index)
    M = network.compute_matrix(new_configs)
    print(M)


# # Plot Polytope
#
# plt.autoscale()
# plt.show()
