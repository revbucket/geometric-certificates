import torch
import numpy as np
import matplotlib.pyplot as plt
from plnn import PLNN
from _polytope_ import from_polytope_dict
from geocert import batch_GeoCert

# Experiment to see if Min. Vol. Inscr. Ellipse touches many of the non-redundant constraints

# ======================================================= #
#                     2d+ Example                         #
# ======================================================= #


fractions = []
xs = [elem for elem in np.linspace(2,2,8)]
dtype=torch.FloatTensor

for input_dim in xs:
    input_dim = int(input_dim)

    # ==================================
    # Initialize Network
    # ==================================

    print('===============Initializing Network============')
    layer_sizes = [input_dim, 100, 50, 2]
    network = PLNN(layer_sizes, dtype)
    net = network.net

    # ==================================
    # Get Polytope
    # ==================================

    x_0 = torch.Tensor(np.random.uniform(size=(input_dim)).reshape([input_dim, 1]))
    poly_dict = network.compute_polytope(x_0)
    polytope = from_polytope_dict(poly_dict)
    polytope.essential_constraints_ellipse()

    # ==================================
    # Constraint Determination
    # ==================================

    bools = polytope.redundant
    num_non_redund = np.sum([1 if bool == False else 0 for bool in bools])

    print('num_non_redund found:')
    print(num_non_redund)

    _, boundary, shared_facets = batch_GeoCert([polytope,], x_0)
    print('true number:')
    print(np.shape(boundary))


plt.plot(xs, fractions)
plt.show()