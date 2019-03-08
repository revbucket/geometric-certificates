
# # =====================
# # Imports
# # =====================
#
# from geocert import compute_boundary_batch, batch_GeoCert, incremental_GeoCert
# from plnn import PLNN
# from _polytope_ import Polytope, from_polytope_dict
# import utilities as utils
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
#
# ##########################################################################
# #                                                                        #
# #                          Exp # 1                                       #
# #                                                                        #
# ##########################################################################
#
# # experiment to find a potentially better way to determine redundant constraints
# # of the network
# #
# # Idea: look at matrices induced by modifiying ReLu configs
#
# # ==================================
# # Initialize Network
# # ==================================
#
# print('===============Initializing Network============')
# layer_sizes = [2, 8, 6, 2]
# network = PLNN(layer_sizes)
#
# # ==================================
# # Find Projection
# # ==================================
#
# # lp_norm = 'l_2'
# #
# # print('===============Finding Projections============')
# # print('lp_norm: ', lp_norm)
# # x_0 = torch.Tensor([[0.0], [0.0]])
# # print('from point: ')
# # print(x_0)
# #
# # ax = plt.axes()
# # cwd = os.getcwd()
#
# # Starting Point
# x_0 = torch.Tensor([[0.0], [0.0]])
# print('from point: ')
# print(x_0)
#
#
# # Get Relu Configuration
# pre_relus, configs = network.relu_config(x_0, return_pre_relus=True)
#
#
# # Get Polytope
# polytope_dict = network.compute_polytope(x_0, comparison_form_flag=True)
# polytope = from_polytope_dict(polytope_dict)
# test = polytope.generate_facets()
# polytope_list = [polytope, ]
#
# # Get Faces
# ax = plt.axes()
#
# boundary,_  = compute_boundary_batch(polytope_list, 'fast_ReLu')
# utils.plot_polytopes_2d(polytope_list, colors=None, alpha=0.7,
#                         xylim=5, ax=ax, linestyle='dashed', linewidth=0)
#
# # # View Tight constraints
# tight_constraints = [facet.tight_list[0] for facet in boundary]
# print(tight_constraints)
#
#
# # Change ReLu activations and look at matrices properties
# print('original matrix:', -1)
# network.compute_matrix(configs)
#
# for index in range(0, np.shape(polytope.ub_A)[0]):
#     print('---------------')
#     k = 0
#     new_configs = []
#     for config in configs:
#         new_config = []
#         for l, elem in enumerate(config):
#             if k + l == index:
#                 new_config.append(1.0-elem)
#             else:
#                 new_config.append(elem)
#         new_configs.append(torch.Tensor(new_config))
#         k += l+1
#
#     print('matrix:', index)
#     M = network.compute_matrix(new_configs)
#     print(M)
#
#
# # # Plot Polytope
# #
# # plt.autoscale()
# # plt.show()



##########################################################################
#                                                                        #
#                          Exp # 2                                       #
#                                                                        #
##########################################################################

# Experiment for testing the use of MVIE for eliminating redundant constraints

from plnn import PLNN
import torch
import numpy as np
import matplotlib.pyplot as plt
from _polytope_ import from_polytope_dict


# ======================================================= #
#                     2d Example                          #
# ======================================================= #


# ==================================
# Generate Training Points
# ==================================

print('===============Generating Training Points============')
# random points at least 2r apart
input_dim = 2
m = 12
# np.random.seed(3)
x = [np.random.uniform(size=(input_dim))]
r = 0.16
while (len(x) < m):
    p = np.random.uniform(size=(input_dim))
    if min(np.abs(p - a).sum() for a in x) > 2 * r:
        x.append(p)
# r = 0.145
epsilon = r / 2

X = torch.Tensor(np.array(x))
torch.manual_seed(1)
y = (torch.rand(m) + 0.5).long()

# ==================================
# Initialize Network
# ==================================

print('===============Initializing Network============')
layer_sizes = [input_dim, 100, 50, 2]
network = PLNN(layer_sizes)
net = network.net

# ==================================
# Get Polytope
# ==================================

x_0 = torch.Tensor(x[0].reshape([input_dim, 1]))
poly_dict = network.compute_polytope(x_0)
polytope = from_polytope_dict(poly_dict)

import utilities as utils
Ci, di = utils.MVIE_ellipse(polytope.ub_A, polytope.ub_b)
polytope.redund_removal_approx_ellipse()
polytope.essential_constraints_ellipse()

# Polygon
fig = plt.figure()
ax = fig.add_subplot(111)

# The inner ellipse
theta = np.linspace(0, 2 * np.pi, 100)
x = Ci[0][0] * np.cos(theta) + Ci[0][1] * np.sin(theta) + di[0]
y = Ci[1][0] * np.cos(theta) + Ci[1][1] * np.sin(theta) + di[1]
ax.plot(x, y)
import utilities as utils
utils.plot_polytopes_2d([polytope,],ax=ax)

plt.autoscale()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# The outer ellipse
theta = np.linspace(0, 2 * np.pi, 100)
Ci = np.multiply(2, Ci)
x = Ci[0][0] * np.cos(theta) + Ci[0][1] * np.sin(theta) + di[0]
y = Ci[1][0] * np.cos(theta) + Ci[1][1] * np.sin(theta) + di[1]
ax.plot(x, y)

# The Hyperplanes
styles = ['-' if not bool else '--' for bool in polytope.redundant]
styles = []
for bool in polytope.redundant:
    if bool == True:
        styles.append('-')
    elif bool == False:
        styles.append('--')
    else:
        styles.append(':')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
utils.plot_hyperplanes(polytope.ub_A, polytope.ub_b, styles, ax)
xlim, ylim = utils.expand_xylim(5, xlim, ylim)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.show()

# # ======================================================= #
# #                     2d+ Example                         #
# # ======================================================= #
#
#
# fractions = []
# xs = [elem for elem in np.linspace(2,20,8)]
# dtype=torch.FloatTensor
#
# for input_dim in xs:
#     input_dim = int(input_dim)
#
#     # ==================================
#     # Initialize Network
#     # ==================================
#
#     print('===============Initializing Network============')
#     layer_sizes = [input_dim, 100, 50, 2]
#     network = PLNN(layer_sizes, dtype)
#     net = network.net
#
#     # ==================================
#     # Get Polytope
#     # ==================================
#
#     x_0 = torch.Tensor(np.random.uniform(size=(input_dim)).reshape([input_dim, 1]))
#     poly_dict = network.compute_polytope(x_0)
#     polytope = from_polytope_dict(poly_dict)
#     polytope.redund_removal_approx_ellipse()
#
#     bools = polytope.redundant
#     num_kept = np.sum([1 if not bool else 0 for bool in bools])
#     total = len(bools)
#
#     print(bools)
#     fractions.append(num_kept/total)
#
#
# plt.plot(xs, fractions)
# plt.show()