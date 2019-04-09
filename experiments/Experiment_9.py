# ==================================
# Experiment 9
# ==================================

# Compare performance of GeoCert vs. ReluPlex on similar network

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
from _polytope_save import from_polytope_dict, Face
import utilities as utils
import time
import torch.optim as optim
from torch.autograd import Variable

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

    print('ellipse_total:', time.time()-time_1)

    print('redund_ellipse:', redund)
    print('non_redund_ellipse:', non_redund)

    return redundant_list


# =====================
# Load Network
# =====================

print('===============Initializing Network============')
cwd = os.getcwd()
folderpath = cwd + "/data/"
filepath = folderpath + "acas_xu_net_copy_paste"
sequential = torch.load(filepath)

layer_shape = lambda layer: layer.weight.detach().numpy().shape
layer_sizes = [layer_shape(layer)[1] for layer in sequential if type(layer) == nn.Linear] + [layer_shape(sequential[-1])[0]]
dtype = torch.FloatTensor
network = PLNN_seq(sequential, layer_sizes, dtype)
net = network.net


# print('===============Initializing Network============')
cwd = os.getcwd()
network = PLNN(layer_sizes)
net = network.net
dtype = torch.FloatTensor


# # ==================================
# # Generate Training Points
# # ==================================
#
# print('===============Generating Training Points============')
# # random points at least 2r apart
# input_dim = layer_sizes[0]
# m = 2000
# np.random.seed(3)
# x = [np.random.uniform(size=(input_dim))]
# r = 0.01
# while(len(x) < m):
#     print('got one')
#     p = np.random.uniform(size=(input_dim))
#     # if min(np.max(np.sum(np.abs(x), axis=1)) for a in x) > r:
#     #     x.append(p)
#     if min(np.linalg.norm(p-a, ord=2) for a in x) > r:
#         x.append(p)
#
# print('done')
#
# filepath = folderpath + "x_l2.npy"
# np.save(filepath, x)
# quit()

# ==================================
# Load Training Points
# ==================================

m = 2000
filepath = folderpath + "x_l2.npy"
x = np.load(filepath)

mins = []
for i, pt in enumerate(x):
    mins.append(min(np.linalg.norm(pt-a, ord=2) for a in [elem for j, elem in enumerate(x) if j != i]))
print(np.average(mins))

X = torch.Tensor(np.array(x))
torch.manual_seed(1)
y = (torch.rand(m)+0.5).long()

print('===============Generating Test Points============')

x_test = []
for pt in x:
    x_test.append(pt+np.ones([1, 5])*0.0001/np.linalg.norm(np.ones([1,5])))
pts = torch.Tensor(x_test)

# ==================================
# Train Network
# ==================================

print('===============Training Network============')
opt = optim.Adam(net.parameters(), lr=1e-3)
for i in range(3000):
    out = net(Variable(X))
    l = nn.CrossEntropyLoss()(out, Variable(y))
    err = (out.max(1)[1].data != y).float().mean()
    if i % 100 == 0:
        print(l.data[0], err)
    opt.zero_grad()
    (l).backward()
    opt.step()

# ==================================
# Find Projections
# ==================================

lp_norm = 'l_2'
ts = []
input_dim = layer_sizes[0]


plot_dir = cwd+'/plots/incremental_geocert/'
geocert = IncrementalGeoCert(network, display=False, config_fxn='v2', save_dir=plot_dir)

start_time = time.time()
times = []

for pt in pts:
    print('===============Finding Projection============')
    print('lp_norm: ', lp_norm)
    x_0 = torch.Tensor(pt.reshape([1, input_dim])).type(dtype)
    print('from point: ')
    print(x_0)

    t, cw_bound, adver_examp = geocert.min_dist(x_0, lp_norm, compute_upper_bound=False)
    print()
    print('===============Projection Found============')
    print('the final projection value:', t)
    print('carlin-wagner bound', cw_bound)
    ts.append(t)


    intermed_time = time.time()
    diff = intermed_time-start_time
    print('ITER TIME:', diff)
    times.append(diff)
    start_time = intermed_time

    # ==================================
    # Check Projection is Adv. Example
    # ==================================

    print()
    print('==========Is Adversarial?===========')

    orig_output = net.forward(x_0)
    print('orig_pt')
    print(x_0)
    print('orig_output')
    print(orig_output)

    adver_examp = torch.Tensor(adver_examp.reshape([1, input_dim])).type(dtype)
    print('adv_example:')
    print(adver_examp)

    new_output = net.forward(adver_examp)
    print('new output:')
    print(new_output)

    print()
    print('======================================================')

end_time = time.time()

print('==========================================================')
print('~~~~PROJECTIONS COMPLETE~~~~')
print('TOTAL TIME (s):', end_time-start_time)
print('TIMES:')
print(times)