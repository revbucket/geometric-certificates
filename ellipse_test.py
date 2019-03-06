from plnn import PLNN
import torch
import numpy as np
import matplotlib.pyplot as plt
from _polytope_ import from_polytope_dict

# ==================================
# Generate Training Points
# ==================================

print('===============Generating Training Points============')
# random points at least 2r apart
input_dim = 2
m = 12
# np.random.seed(3)
x = [np.random.uniform(size=(input_dim))]
r = 0.16
while (len(x) < m):
    p = np.random.uniform(size=(input_dim))
    if min(np.abs(p - a).sum() for a in x) > 2 * r:
        x.append(p)
# r = 0.145
epsilon = r / 2

X = torch.Tensor(np.array(x))
torch.manual_seed(1)
y = (torch.rand(m) + 0.5).long()

# ==================================
# Initialize Network
# ==================================

print('===============Initializing Network============')
layer_sizes = [input_dim, 50, 8, 2]
network = PLNN(layer_sizes)
net = network.net

# ==================================
# Get Polytope
# ==================================

x_0 = torch.Tensor(x[0].reshape([input_dim, 1]))
poly_dict = network.compute_polytope(x_0)
polytope = from_polytope_dict(poly_dict)

import utilities as utils
Ci, di = utils.MVIE_ellipse(polytope.ub_A, polytope.ub_b)
polytope.redund_removal_ellipse()

# Polygon
fig = plt.figure()
ax = fig.add_subplot(111)

# The inner ellipse
theta = np.linspace(0, 2 * np.pi, 100)
x = Ci[0][0] * np.cos(theta) + Ci[0][1] * np.sin(theta) + di[0]
y = Ci[1][0] * np.cos(theta) + Ci[1][1] * np.sin(theta) + di[1]
ax.plot(x, y)
import utilities as utils
utils.plot_polytopes_2d([polytope,],ax=ax)

plt.autoscale()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# # The outer ellipse
# theta = np.linspace(0, 2 * np.pi, 100)
# Ci = np.multiply(2, Ci)
# x = Ci[0][0] * np.cos(theta) + Ci[0][1] * np.sin(theta) + di[0]
# y = Ci[1][0] * np.cos(theta) + Ci[1][1] * np.sin(theta) + di[1]
# ax.plot(x, y)

# The Hyperplanes
styles = ['--' if bool else '-' for bool in polytope.redundant]
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
utils.plot_hyperplanes(polytope.ub_A, polytope.ub_b, styles, ax)
xlim, ylim = utils.expand_xylim(5, xlim, ylim)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.show()