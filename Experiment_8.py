

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
from geocert import incremental_GeoCert
import matplotlib.pyplot as plt
import numpy as np


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
print(net)


print('===============Initializing Network============')
layer_sizes = [2, 50, 8, 2]
network = PLNN(layer_sizes)
net = network.net
print(net)


# ==================================
# Find Projections
# ==================================

lp_norm = 'l_2'
ts = []
input_dim = layer_sizes[0]
pts = [np.random.uniform(0, 1, [1, input_dim]),]


for pt in pts:
    print('===============Finding Projection============')
    print('lp_norm: ', lp_norm)
    x_0 = torch.Tensor(pt.reshape([1, input_dim])).type(dtype)
    print('from point: ')
    print(x_0)

    ax = plt.axes()
    cwd = os.getcwd()
    plot_dir = cwd + '/plots/incremental_geocert/'

    t = incremental_GeoCert(lp_norm, network, x_0, ax, plot_dir, plot_iter=None)

    print('the final projection value:', t)
    ts.append(t)