# =====================
# Imports
# =====================

from geocert import compute_boundary_batch, compute_l_inf_ball_batch
from plnn import PLNN
from _polytope_ import Polytope, from_polytope_dict
import utilities as utils
import matplotlib.pyplot as plt
import numpy as np



# ##########################################################################
# #                                                                        #
# #                             Test 1                                     #
# #                                                                        #
# ##########################################################################
#
# # test to see if code works for simple union of glued polytopes
#
# # =====================
# # Create Polytopes
# # =====================
#
# # Two glued 2d-triangle polytopes
#
# ub_A = np.asarray([[-1, 0, 1], [1, -1, 0]]).transpose()
# ub_b = np.asarray([1, 0, 0])
# P1 = Polytope(ub_A, ub_b)
#
# ub_A = np.asarray([[1, 0, -1], [1, -1, 0]]).transpose()
# ub_b = np.asarray([1, 0, 0])
# P2 = Polytope(ub_A, ub_b)
#
# poly_list = [P1, P2]
#
# # ===========
# # Find Projection
# # ===========
#
# unshared_facets = compute_boundary_batch(poly_list)
#
# x_0 = np.asarray([[0], [0.4]])
# t = compute_l_inf_ball_batch(poly_list, x_0)
#
# print('max linf size:', t)




# ##########################################################################
# #                                                                        #
# #                             Test 2                                     #
# #                                                                        #
# ##########################################################################

# test to see if code works for moderate number of glued polytopes

# ==================================
# Gather Polytopes
# ==================================

print('===============Collecting Polytopes============')
layer_sizes = [2, 8, 4, 1]
network = PLNN(layer_sizes)
num_pts = 100
xylim = 0.5

unique_relu_configs_list, unique_bin_acts, _, _ = utils.get_unique_relu_configs(network, xylim, num_pts)
print(len(unique_bin_acts))
color_dict = utils.get_color_dictionary(unique_bin_acts)
polytope_list = []
colors = []

for relu_configs, unique_act in zip(unique_relu_configs_list, unique_bin_acts):
    polytope_dict = network.compute_polytope_config(relu_configs, True)
    polytope = from_polytope_dict(polytope_dict)
    polytope_list.append(polytope)
    colors.append(color_dict[unique_act])


# =================================
# Find Projection
# =================================

print('===============Finding Projection============')
x_0 = np.asarray([[0.0], [0.0]])
print('from: ', x_0)

t, boundary, unshared_facets = compute_l_inf_ball_batch(polytope_list, x_0, 'slow')
print('the final projection value:', t)


# ------------------------------
# Plot Polytopes, boundary, and linf norm
# ------------------------------

ax = plt.gca()
alpha = 0.75
xylim = 2.0

utils.plot_polytopes_2d(polytope_list, colors, alpha, xylim, ax)
utils.plot_facets_2d(boundary, xylim=xylim, ax=ax)
utils.plot_facets_2d(unshared_facets, xylim=xylim, ax=ax, color='red')
# plot_network_polytopes_sloppy(network, 1.0, 150)
utils.plot_linf_norm(x_0, t, ax=ax)

plt.show()


