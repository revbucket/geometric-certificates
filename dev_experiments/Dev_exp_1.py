# =====================
# Imports
# =====================

from plnn import PLNN
import torch
import numpy as np
from _polytope_save import Polytope
import matplotlib.pyplot as plt

# ##########################################################################
# #                                                                        #
# #                          Dev_Exp_1_a                                   #
# #                                                                        #
# ##########################################################################
#
# # experiment to see how the percentage of essential constraints varies
# # as a function of dimension
#
#
# # ==================================
# # Network Structure Loop
# # ==================================
#
# dims = [int(d) for d in np.linspace(2, 50, 10)]
# ess_percentages_per_dim = []
# num_samples = 12
#
# for d in dims:
#
#     # =========================
#     # Initialize Network
#     # =========================
#     print('===============Initializing Network============')
#     layer_sizes = [d, 50, 50, 2]
#     network = PLNN(layer_sizes)
#
#     # Loop to run test for 'num_samples'
#     ess_percentages = []
#     num_redund = []
#     nums_essential = []
#     for _ in range(0, num_samples):
#         # =========================
#         # Find number of Red. Constraints
#         # =========================
#         x_0 = torch.Tensor(np.random.randn(1, d))
#         poly_dict = network.compute_polytope(x_0)
#         polytope = Polytope.from_polytope_dict(poly_dict)
#         essential_constrs, reject_reasons = polytope.generate_facets_configs(None, network, use_clarkson=False, use_ellipse=False)
#
#         num_constrs = np.shape(polytope.ub_A)[0]
#         num_essential = len(essential_constrs)
#         ess_percentage = (num_essential / num_constrs)*100
#
#         nums_essential.append(num_essential)
#         num_redund.append(num_constrs-num_essential)
#         ess_percentages.append(ess_percentage)
#     # Save Data per dimension
#     ess_percentages_per_dim.append(np.mean(ess_percentages))
#
# # =========================
# # Plot Percentages vs Dimensions
# # =========================
#
# plt.plot(dims, ess_percentages_per_dim)
# plt.title('Average Percentage Essential vs. Dimension')
# plt.xlabel('dimension')
# plt.ylabel('% Essential')
# plt.show()


# ##########################################################################
# #                                                                        #
# #                          Dev_Exp_1_b                                    #
# #                                                                        #
# ##########################################################################
#
# # experiment to see how the percentage of essential constraints varies
# # with dimension fixed and architecture varied
#
#
# # ==================================
# # Network Structure Loop
# # ==================================
#
# layer_widths = [int(d) for d in np.linspace(25, 300, 4)]
# num_samples = 1
#
# # Two examples (low dim, high dim)
# for input_dim in [2, 6, 8, 10, 12, 14, 18, 20]:
#     ess_percentages_per_width = []
#
#     # Loop over different layer widths
#     for width in layer_widths:
#
#         # =========================
#         # Initialize Network
#         # =========================
#         print('===============Initializing Network============')
#         layer_sizes = [input_dim, width, width, 2]
#         network = PLNN(layer_sizes)
#
#         # Loop to run test for 'num_samples'
#         ess_percentages = []
#         num_redund = []
#         nums_essential = []
#         for _ in range(0, num_samples):
#             # =========================
#             # Find number of Red. Constraints
#             # =========================
#             x_0 = torch.Tensor(np.random.randn(1, input_dim))
#             poly_dict = network.compute_polytope(x_0)
#             polytope = Polytope.from_polytope_dict(poly_dict)
#             essential_constrs, reject_reasons = polytope.generate_facets_configs(None, network, use_clarkson=False, use_ellipse=False)
#
#             num_constrs = np.shape(polytope.ub_A)[0]
#             num_essential = len(essential_constrs)
#             ess_percentage = (num_essential / num_constrs)*100
#
#             nums_essential.append(num_essential)
#             num_redund.append(num_constrs-num_essential)
#             ess_percentages.append(ess_percentage)
#         # Save Data per dimension
#         ess_percentages_per_width.append(np.mean(ess_percentages))
#
#     # =========================
#     # Plot Percentages vs Layer width
#     # =========================
#
#     plt.plot(layer_widths, ess_percentages_per_width, label='In Dim: '+str(input_dim))
#     plt.title('Avg. Percent. Essential vs. Layer width ')
#     plt.xlabel('Width (2 layer RELU)')
#     plt.ylabel('% Essential')
#     plt.legend()
#
# # Show Plot
# plt.show()

# ##########################################################################
# #                                                                        #
# #                          Dev_Exp_1_c                                    #
# #                                                                        #
# ##########################################################################
#
# # experiment to see how the percentage of essential constraints varies
# # with a few examples of network width and input dimension varied greatly
#
#
# # ==================================
# # Network Structure Loop
# # ==================================
#
# input_dims = [int(d) for d in np.linspace(25, 100, 6)]
# num_samples = 1
#
# # Widths examples
# for width in [50, 100, 300]:
#     ess_percentages_per_dim = []
#
#     # Loop over different input dims
#     for input_dim in input_dims:
#
#         # =========================
#         # Initialize Network
#         # =========================
#         print('===============Initializing Network============')
#         layer_sizes = [input_dim, width, width, 2]
#         network = PLNN(layer_sizes)
#
#         # Loop to run test for 'num_samples'
#         ess_percentages = []
#         num_redund = []
#         nums_essential = []
#         for _ in range(0, num_samples):
#             # =========================
#             # Find number of Red. Constraints
#             # =========================
#             x_0 = torch.Tensor(np.random.randn(1, input_dim))
#             poly_dict = network.compute_polytope(x_0)
#             polytope = Polytope.from_polytope_dict(poly_dict)
#             essential_constrs, reject_reasons = polytope.generate_facets_configs(None, network, use_clarkson=False, use_ellipse=False)
#
#             num_constrs = np.shape(polytope.ub_A)[0]
#             num_essential = len(essential_constrs)
#             ess_percentage = (num_essential / num_constrs)*100
#
#             nums_essential.append(num_essential)
#             num_redund.append(num_constrs-num_essential)
#             ess_percentages.append(ess_percentage)
#         # Save Data per dimension
#             ess_percentages_per_dim.append(np.mean(ess_percentages))
#
#     # =========================
#     # Plot Percentages vs Layer width
#     # =========================
#
#     plt.plot(input_dims, ess_percentages_per_dim, label='Width: '+str(width))
#     plt.title('Avg. Percent. Essential vs. Layer width ')
#     plt.xlabel('Width (2 layer RELU)')
#     plt.ylabel('% Essential')
#     plt.legend()
#
# # Show Plot
# plt.show()

# ##########################################################################
# #                                                                        #
# #                          Dev_Exp_1_d                                    #
# #                                                                        #
# ##########################################################################
#
# # experiment to see how the percentage of essential constraints varies
# # with a few examples of different network depth (num neurons fixed) and
# # input dimension varied
#
#
# # ==================================
# # Network Structure Loop
# # ==================================
#
# input_dims = [int(d) for d in np.linspace(2, 20, 6)]
# num_samples = 10
# num_neurons_fixed = 200
#
# # Widths examples
# for depth in [2, 4, 5, 8, 10]:
#     ess_percentages_per_dim = []
#
#     # Loop over different input dims
#     for input_dim in input_dims:
#
#         # =========================
#         # Initialize Network
#         # =========================
#         print('===============Initializing Network============')
#         width = int(num_neurons_fixed/depth)
#         layer_sizes  = [input_dim,] + [width for _ in range(0, depth)] + [2]
#         print(layer_sizes)
#         network = PLNN(layer_sizes)
#
#         # Loop to run test for 'num_samples'
#         ess_percentages = []
#         num_redund = []
#         nums_essential = []
#         for _ in range(0, num_samples):
#             # =========================
#             # Find number of Red. Constraints
#             # =========================
#             x_0 = torch.Tensor(np.random.randn(1, input_dim))
#             poly_dict = network.compute_polytope(x_0)
#             polytope = Polytope.from_polytope_dict(poly_dict)
#             essential_constrs, reject_reasons = polytope.generate_facets_configs(None, network, use_clarkson=False, use_ellipse=False)
#
#             num_constrs = np.shape(polytope.ub_A)[0]
#             num_essential = len(essential_constrs)
#             ess_percentage = (num_essential / num_constrs)*100
#
#             nums_essential.append(num_essential)
#             num_redund.append(num_constrs-num_essential)
#             ess_percentages.append(ess_percentage)
#         # Save Data per dimension
#         ess_percentages_per_dim.append(np.mean(ess_percentages))
#
#     # =========================
#     # Plot Percentages vs Layer width
#     # =========================
#
#     plt.plot(input_dims, ess_percentages_per_dim, label='Depth: '+str(depth))
#     plt.title('Avg. Percent. Essential vs. Input Dim ')
#     plt.xlabel('Input Dim')
#     plt.ylabel('% Essential')
#     plt.legend()
#
# # Show Plot
# plt.show()

##########################################################################
#                                                                        #
#                          Dev_Exp_1_e                                    #
#                                                                        #
##########################################################################

# experiment to see how the percentage of essential constraints varies
# as initial points distance from the origin is increased


# ==================================
# Network Structure Loop
# ==================================

input_dim = 20
num_samples = 20
num_neurons_fixed = 200

# Loop over different distances
variances = np.logspace(-2, 8, 10)
ess_percentages_per_var = []
distances = []

for var in variances:
    # =========================
    # Initialize Network
    # =========================
    print('===============Initializing Network============')
    layer_sizes = [input_dim, 50, 50, 2]
    network = PLNN(layer_sizes)

    # Loop to run test for 'num_samples'
    ess_percentages = []
    num_redund = []
    nums_essential = []
    for _ in range(0, num_samples):
        # =========================
        # Find number of Red. Constraints
        # =========================
        x_0_np = np.multiply( np.sqrt(var), np.random.randn(1, input_dim))
        x_0 = torch.Tensor(x_0_np)
        distances.append(np.linalg.norm(x_0_np))
        poly_dict = network.compute_polytope(x_0)
        polytope = Polytope.from_polytope_dict(poly_dict)
        essential_constrs, reject_reasons = polytope.generate_facets_configs(None, network, use_clarkson=False, use_ellipse=False)

        num_constrs = np.shape(polytope.ub_A)[0]
        num_essential = len(essential_constrs)
        ess_percentage = (num_essential / num_constrs)*100

        nums_essential.append(num_essential)
        num_redund.append(num_constrs-num_essential)
        ess_percentages.append(ess_percentage)
    # Save Data per dimension
    ess_percentages_per_var.append(np.mean(ess_percentages))

# =========================
# Plot Percentages vs Layer width
# =========================

plt.plot(variances, ess_percentages_per_var)
plt.title('Avg. Percent. Essential vs. ||x_0||')
plt.xlabel('L2 Norm')
plt.ylabel('% Essential')
plt.semilogx()
plt.legend()

# Show Plot
plt.show()