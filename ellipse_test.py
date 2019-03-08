import torch
import numpy as np
import matplotlib.pyplot as plt
from plnn import PLNN
from _polytope_ import from_polytope_dict
from geocert import batch_GeoCert
import utilities as utils

from geocert import compute_boundary_batch, batch_GeoCert, incremental_GeoCert
from plnn import PLNN
from _polytope_ import Polytope, from_polytope_dict
import utilities as utils
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Experiment to see if Min. Vol. Inscr. Ellipse touches many of the non-redundant constraints

# ======================================================= #
#                     2d+ Example                         #
# ======================================================= #


fractions = []
xs = [elem for elem in np.linspace(2, 30,  7)]
dtype=torch.FloatTensor

for input_dim in xs:
    input_dim = int(input_dim)

    # ==================================
    # Initialize Network
    # ==================================

    print('===============Initializing Network============')
    layer_sizes = [input_dim, 50, 10, 2]
    network = PLNN(layer_sizes, dtype)
    net = network.net

    # ==================================
    # Get Polytope
    # ==================================

    x_0 = torch.Tensor(np.random.uniform(size=(input_dim)).reshape([input_dim, 1]))
    poly_dict = network.compute_polytope(x_0)
    polytope = from_polytope_dict(poly_dict)

    # ==================================
    # Get Boundary
    # ==================================

    _, boundary, shared_facets = batch_GeoCert([polytope,], x_0)

    essential_indices = []
    for face in boundary:
        essential_indices.append(face.tight_list[0])
    print('essential_indices')
    print(essential_indices)

    # ==================================
    # Find Non-Redundant Constraints (random sampling + projection_
    # ==================================

    print('true number:')
    print(np.shape(boundary))

    print('===================')



plt.plot(xs, fractions)
plt.show()