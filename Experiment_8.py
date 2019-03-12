

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
from plnn import PLNN_seq
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
layer_shapes = [layer_shape(layer)[0] for layer in sequential if type(layer) == nn.Linear] + [layer_shape(sequential[-1])[1]]
dtype = torch.FloatTensor
plnn = PLNN_seq(sequential, layer_shapes, dtype)
net = plnn.net


# ==================================
# Find Projections
# ==================================

lp_norm = 'l_inf'
ts = []
input_dim = layer_shapes[0]
pts = np.random.uniform(0, 1, [1, input_dim])

for pt in pts:
    print('===============Finding Projection============')
    print('lp_norm: ', lp_norm)
    x_0 = torch.Tensor(pt.reshape([input_dim, 1]))
    print(x_0)
    print('from point: ')
    print(x_0)

    ax = plt.axes()
    cwd = os.getcwd()
    print(cwd)
    plot_dir = cwd + '/plots/incremental_geocert/'

    t = incremental_GeoCert(lp_norm, plnn, x_0, ax, plot_dir, plot_iter=1)

    print('the final projection value:', t)
    ts.append(t)