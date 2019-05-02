# =====================
# Imports
# =====================

from plnn import PLNN
import torch
import numpy as np
from _polytope_save import Polytope
import utilities as utils
import matplotlib.pyplot as plt
from torch.autograd import Variable

##########################################################################
#                                                                        #
#                          Dev_Exp                                       #
#                                                                        #
##########################################################################

# experiments to test redundancy tests derived from the dual of L2 constrained
# redundancy Linear Program
# PRIMAL:
#   max <a_i, x> - b_i
#  ||x ||_2 <= t
#
# check OPT >< 0

# ==================================
# Create Polytope with L2 bound
# ==================================

# ---------Initialize Network-------------------
print('===============Initializing Network============')
d = 2
layer_sizes = [d, 10, 10, 2]
network = PLNN(layer_sizes)

# ----------Poly + Red. Constraints---------------
x_0_np = np.random.randn(1, d)
x_0_np = np.zeros((1,d))
x_0 = torch.Tensor(x_0_np)
poly_dict = network.compute_polytope(x_0)
polytope = Polytope.from_polytope_dict(poly_dict)
essential_constrs, reject_reasons = polytope.generate_facets_configs(None, network, use_clarkson=False, use_ellipse=False)

essential_indices = []
for constr in essential_constrs:
    essential_indices.append(constr.tight_list[0])

# ----------L2 bound: t ---------------
ptope = utils.Polytope_2(polytope.ub_A, polytope.ub_b)
vertices = utils.ptope.extreme(ptope)
distances = []
for vertex in vertices:
    distances.append(np.linalg.norm(x_0_np-vertex))
t = np.max(distances)

# # ==================================
# # Dual Method 1 (sphere slicing)
# # ==================================
# A = polytope.ub_A
# b = polytope.ub_b
#
# red_indices = []
# for i, (a_i, b_i) in enumerate(zip(A, b)):
#     lhs = t*np.linalg.norm(a_i) - b_i
#     if lhs < 0:
#         red_indices.append(i)



# ==================================
# Dual Method 2 (secondary dual formulation)
# ==================================
# min <lamb.T, b> + t*max |(a_i.T-lamb.T*A)[j]| - b_i
# lamb                 j

A = torch.Tensor(polytope.ub_A)
n = np.shape(A)[0]
b = torch.Tensor(polytope.ub_b).view((n, 1))
lamb = Variable(torch.Tensor(np.zeros((1, n))))
red_indices = []


for i, (a_i, b_i) in enumerate(zip(A, b)):
    # minimize using gradients
    # lhs = dual_func_2(lamb, A, b, a_i, b_i, t)
    # dual_func_2 = torch.mm(lamb, b) + t*torch.max(torch.abs(a_i - torch.mm(lamb, A)))\
    #               - b_i + torch.mm(x_0, a_i.view(d,1))

    inf_norm_sqrt = t*torch.max(torch.abs(a_i)) - b_i + torch.mm(x_0, a_i.view(d,1))

    # l1_norm = t*torch.sum(torch.abs(a_i)) - b_i

    # check redundancy
    if utils.as_numpy(inf_norm_sqrt) < 0:
        red_indices.append(i)

# # ==================================
# # Dual Method 3 (primary dual formulation)
# # ==================================
# # min <lamb.T, b> + t*||lamb.T*A||_2
# # lamb
#
# A = polytope.ub_A
# b = polytope.ub_b
# red_indices = []
#
# for i, (a_i, b_i) in enumerate(zip(A, b)):
#     # one step gradient
#     lamb = None
#     # find redundancy
#     lhs = np.inner(lamb, b) + t * np.linalg.norm(np.matmul(lamb, A)) - b_i + t * np.linalg.norm(a_i)
#     if lhs < 0:
#         red_indices.append(i)




# ----------Plot Everything ---------------
x_0_np = x_0_np[0]
poly_list = [polytope]
ax = plt.axes()
xylim = [-5, 5]
utils.plot_polytopes_2d(poly_list, ax=ax, xylim=xylim)
utils.plot_2d_lines(polytope.ub_A, polytope.ub_b, xylim, color='black')
utils.plot_2d_lines(polytope.ub_A[red_indices], polytope.ub_b[red_indices], xylim, color='red')
utils.plot_l2_norm(x_0_np, t, ax=ax)
utils.plot_linf_norm(x_0_np, t, ax=ax)
ax.scatter(x_0_np[0], x_0_np[1], s=10)
plt.show()