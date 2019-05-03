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



# ==================================== #
# Experiment
# ==================================== #

# Testing to see if random sampling from the polytope plus projection + ray shooting can find
# many non-redundant constraints

fractions = []
powers = []
xs = [elem for elem in np.linspace(2, 2,  1)]
dtype=torch.FloatTensor

for input_dim in xs:
    input_dim = int(input_dim)

    # ==================================
    # Initialize Network
    # ==================================

    print('===============Initializing Network============')
    layer_sizes = [input_dim, 50, 30, 30, 2]
    network = PLNN(layer_sizes, dtype)
    net = network.net

    # ==================================
    # Get Polytope
    # ==================================

    x_0 = torch.Tensor(np.random.uniform(size=(input_dim)).reshape([input_dim, 1]))
    poly_dict = network.compute_polytope(x_0)
    polytope_true = Polytope.from_polytope_dict(poly_dict)

    # ==================================
    # Get Boundary
    # ==================================

    boundary = polytope_true.generate_facets_naive(check_feasible=True)

    essential_indices = []
    for face in boundary:
        essential_indices.append(face.tight_list[0])

    print('essential_indices')
    print(essential_indices)
    print('true number:')
    true_num = np.shape(boundary)[0]
    print(true_num)

    print('===================')


    # ==================================
    # Find Non-redundant Constraints
    # ==================================




plt.show()




