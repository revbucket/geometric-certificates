import torch
import numpy as np
import matplotlib.pyplot as plt
from plnn import PLNN
from geocert import batch_GeoCert
import utilities as utils

from geocert import compute_boundary_batch, batch_GeoCert, incremental_GeoCert
from plnn import PLNN
from _polytope_ import Polytope
import utilities as utils
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# ==================================== #
# Experiment
# ==================================== #

# Testing to see if random sampling from the polytope plus projection + ray shooting can find
# many non-redundant constraints

fractions = []
fractions2 = []
xs = [int(elem) for elem in np.linspace(2, 60,  10)]
dtype=torch.FloatTensor

for input_dim in xs:
    input_dim = int(input_dim)

    # ==================================
    # Initialize Network
    # ==================================

    print('===============Initializing Network============')
    layer_sizes = [input_dim, 10, 10, 2]
    network = PLNN(layer_sizes, dtype)
    net = network.net

    # ==================================
    # Get Polytope
    # ==================================

    x_0 = torch.Tensor(np.random.uniform(size=(input_dim)).reshape([input_dim, 1]))
    poly_dict = network.compute_polytope(x_0)
    polytope = Polytope.from_polytope_dict(poly_dict)

    # ==================================
    # Get Boundary
    # ==================================

    boundary = polytope.generate_facets_naive(check_feasible=True)

    essential_indices = []
    for face in boundary:
        essential_indices.append(face.tight_list[0])

    print('essential_indices')
    print(essential_indices)
    print('true number:')
    true_num = np.shape(boundary)[0]
    print(true_num)

    # ==================================
    # Find Non-redundant Constraints
    # ==================================

    center = utils.as_numpy(x_0).T
    n = np.shape(polytope.ub_A)[0]
    d = (polytope.ub_b-np.matmul(polytope.ub_A, center.T).reshape(-1,))/np.diagonal(np.matmul(polytope.ub_A, polytope.ub_A.T))
    projections = np.tile(center, (n, 1)) + np.matmul(np.diag(d), polytope.ub_A)

    approx_ess_indices = []
    for i, projection in enumerate(projections):
        if polytope.is_point_feasible(projection):
            approx_ess_indices.append(i)

    print('approx essential_indices')
    print(approx_ess_indices)
    print('approx number:')
    approx_num = np.shape(approx_ess_indices)[0]
    print(approx_num)
    fractions.append(approx_num/true_num)

    # ==================================
    # Find Non-redundant Constraints
    # ==================================
    #(improved by finding t separated point using LP)

    # Find interior point using LP
    center = polytope._interior_point()
    n = np.shape(polytope.ub_A)[0]
    d = (polytope.ub_b-np.matmul(polytope.ub_A, center.T).reshape(-1,))/np.diagonal(np.matmul(polytope.ub_A, polytope.ub_A.T))
    projections = np.tile(center, (n, 1)) + np.matmul(np.diag(d), polytope.ub_A)

    approx_ess_indices = []
    for i, projection in enumerate(projections):
        if polytope.is_point_feasible(projection):
            approx_ess_indices.append(i)

    print('approx essential_indices')
    print(approx_ess_indices)
    print('approx number:')
    approx_num = np.shape(approx_ess_indices)[0]
    print(approx_num)
    fractions2.append(approx_num/true_num)

    # ==================================
    # Find Non-redundant Constraints
    # ==================================
    # Bouncing beam method


    # # ----------Plot Constraints and Stuff---------------
    # x_0_np = utils.as_numpy(x_0)
    # poly_list = [polytope]
    # ax = plt.axes()
    # xylim = [-1, 1]
    # utils.plot_polytopes_2d(poly_list, ax=ax, xylim=xylim)
    # # utils.plot_hyperplanes(polytope.ub_A[essential_indices], polytope.ub_b[essential_indices], color='red', ax=ax)
    # utils.plot_hyperplanes(polytope.ub_A[approx_ess_indices], polytope.ub_b[approx_ess_indices], color='black', ax=ax)
    # ax.scatter(x_0_np[0], x_0_np[1], s=10)
    #
    # for i, projection in enumerate(projections):
    #     if i in essential_indices:
    #         ax.plot([center[0], projection[0]], [center[1], projection[1]])
    # ax.scatter(center[0], center[1], s=30)
    #
    # print('===================')
    #
    # plt.show()


plt.plot(xs, fractions, label='x_0')
plt.plot(xs, fractions2, label='max_center')
plt.legend()
plt.show()