
# ==================================
# Experiment 8
# ==================================

# Compare performance of GeoCert vs. ReluPlex on certifying robustness of ACAS XU network

# =====================
# Imports
# =====================

import torch
import os
import torch.nn as nn
from plnn import PLNN_seq, PLNN
from geocert_oop import IncrementalGeoCert
import matplotlib.pyplot as plt
import numpy as np
from _polytope_ import from_polytope_dict, Face
import utilities as utils
import time


def constr_heuristic_ellipse(self, n, redundant_list=None):
    ''' Finds non-redundant constraints by finding MVIE and then checking which constriants are tight
        at the boundary of the ellipse + projecting onto constraint from point if not tight

        Removes redundant constraint by finding an approximation to the
        minimum volume circumscribing ellipsoid. Done by solving maximum
        volume inscribed ellipsoid and multiplying by dimenion n. Ellipse
        E(P, c) is defined by Pos. Def. matrix P and center c.

        returns:   'redundant_list' [True if red. | False if non-red | None if unknown]
    '''

    if redundant_list is None:
        redundant_list = [None for _ in range(0, np.shape(self.ub_A)[0])]

    # Find min. vol. inscribed ellipse
    time_1 = time.time()
    P, c = utils.MVIE_ellipse(self.ub_A, self.ub_b)
    P = np.asarray(P)
    c = np.asarray(c)
    time_2 = time.time()
    print('ellipse:', time_2-time_1)

    # Approximate max. vol. circum. ellipse (provable bounds polytope)
    P_outer = np.multiply(n, P)

    # Find non-Redundant constraints
    # solving: max <a_i, y> <= b_i for all y in E(P, c))
    # y^* = P.T*a_i/||P.T*a_i||_2

    # Remove Redundant constraints
    # constraint a_i redundant if below holds:
    # max <a_i, y> <= b_i for all y in E(P, c))
    #
    # equivalent to: ||P.T*a_i||_2 + a_i.T*c <= b_i
    # (max has a closed form)

    potent_faces = [Face(self.ub_A, self.ub_b, tight_list=[i]) for i in range(0, np.shape(self.ub_A)[0])]
    non_redund = []
    redund = []

    for face in potent_faces:

        # Finds Non-redundant constraints
        lhs = np.linalg.norm(np.matmul(P.T, face.ub_A[face.tight_list].T)) + np.dot(face.ub_A[face.tight_list], c)
        rhs = face.ub_b[face.tight_list]

        if utils.fuzzy_equal(lhs, rhs, tolerance=1e-7):
            non_redund.append(face.tight_list[0])
            redundant_list[face.tight_list[0]] = False

        # Finds Redundant constraints
        lhs = np.linalg.norm(np.matmul(P_outer.T, face.ub_A[face.tight_list].T)) + np.dot(face.ub_A[face.tight_list], c)
        rhs = face.ub_b[face.tight_list]
        if lhs <= rhs:
            redund.append(face.tight_list[0])
            redundant_list[face.tight_list[0]] = True

    time_3 = time.time()
    print('ellipse_support:', time_3-time_2)
    print('ellipse_total:', time_3-time_1)

    print('redund_ellipse:', redund)
    print('non_redund_ellipse:', non_redund)

    return redundant_list



# =====================
# Load Network
# =====================

print('===============Initializing Network============')
cwd = os.getcwd()
folderpath = cwd + "/data/"
filepath = folderpath + "acas_xu_net"
sequential = torch.load(filepath)

layer_shape = lambda layer: layer.weight.detach().numpy().shape
layer_sizes = [layer_shape(layer)[1] for layer in sequential if type(layer) == nn.Linear] + [layer_shape(sequential[-1])[0]]
dtype = torch.FloatTensor
network = PLNN_seq(sequential, layer_sizes, dtype)
net = network.net


# print('===============Initializing Network============')
# layer_sizes = [5, 50, 8, 2]
# network = PLNN(layer_sizes)
# net = network.net


# ==================================
# Get Polytope
# ==================================
input_dim = layer_sizes[0]
x_0 = torch.Tensor(np.random.uniform(size=(input_dim)).reshape([input_dim, 1]))
poly_dict = network.compute_polytope(x_0)
polytope = from_polytope_dict(poly_dict)


# ==================================
# Constraint Determination Exact
# ==================================

time_4 = time.time()
facets, reject_reasons = polytope.generate_facets_configs([], network, check_feasible=True)
time_5 = time.time()
print('clarkson:', time_5-time_4)
print('non_redund_true_clarkson:', [facet.tight_list[0] for facet in facets])
print('num non_redund_true_clarkson:', len(facets))

# time_4 = time.time()
# facets, reject_reasons = polytope.generate_facets_configs([], network, use_clarkson=False, check_feasible=True)
# time_5 = time.time()
# print('non-clarkson:', time_5-time_4)
#
# print('non_redund_true:', [facet.tight_list[0] for facet in facets])
# print('num non_redund_true:', len(facets))

# ==================================
# Constraint Determination Heuristic
# ==================================

list = constr_heuristic_ellipse(polytope, input_dim)

num_redund = np.sum([1 for elem in list if elem and elem is not None])
num_non_redund = np.sum([1 for elem in list if not elem and elem is not None])
num_unknown = np.sum([1 for elem in list if not elem and elem is None])
total = len(list)

print('num_redund:', num_redund)
print('num_non_redund:', num_non_redund)
print('num_unknown:', num_unknown)
print('total:', total)
print('total_prime:', num_non_redund+num_redund+num_unknown)
print('percentage: ', (num_non_redund+num_redund)/total*100, '%')


# # ==================================
# # Find Projections
# # ==================================
#
# lp_norm = 'l_2'
# ts = []
# input_dim = layer_sizes[0]
# pts = [np.random.uniform(0, 1, [1, input_dim]),]
#
#
# for pt in pts:
#     print('===============Finding Projection============')
#     print('lp_norm: ', lp_norm)
#     x_0 = torch.Tensor(pt.reshape([1, input_dim])).type(dtype)
#     print('from point: ')
#     print(x_0)
#
#     ax = plt.axes()
#     cwd = os.getcwd()
#     plot_dir = cwd + '/plots/incremental_geocert/'
#
#     geocert = IncrementalGeoCert(network)
#     t, cw_bound = geocert.min_dist(x_0, lp_norm, False)
#     print('the final projection value:', t)
#     print('carlin-wagner bound', cw_bound)
#     ts.append(t)
