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
xs = [elem for elem in np.linspace(70, 70,  1)]
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
    polytope_true = from_polytope_dict(poly_dict)

    # ==================================
    # Get Boundary
    # ==================================

    _, boundary, shared_facets = batch_GeoCert([polytope_true,], x_0)


    essential_indices = []
    for face in boundary:
        essential_indices.append(face.tight_list[0])
    print('essential_indices')
    print(essential_indices)

    print('true number:')
    true_num = np.shape(boundary)[0]
    print(true_num)

    previous = 0
    power = 2
    max_iter = 2
    my_iter = 0
    new = 1
    max = 1
    max_power = 2
    while my_iter < max_iter:
        # ==================================
        # Find Non-Redundant Constraints (random sampling + projection)
        # ==================================
        polytope = from_polytope_dict(poly_dict)
        num_pts = int(input_dim**power)
        num_pts = 10
        burn = input_dim*polytope.ub_A.shape[0]
        burn = 100

        polytope.essential_constraints_rand_walk(burn=burn, count=num_pts)

        nums = []
        for index in essential_indices:
            if not polytope.redundant[index] and polytope.redundant[index] is not None:
                nums.append(1)
        num_found = np.sum(nums)
        print('num found:')
        print(num_found)
        fraction = num_found/true_num
        print(fraction)

        if num_found > max:
            max = num_found
            max_power = power
        previous = new
        new = num_found
        my_iter += 1
        power = power + 0.2

    fractions.append(fraction)
    powers.append(power)

    print('===================')


plt.plot(xs, fractions)
plt.plot(xs, powers)
plt.show()




