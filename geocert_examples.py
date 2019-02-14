# =====================
# Imports
# =====================

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


# ##########################################################################
# #                                                                        #
# #                          Example 1                                     #
# #                                                                        #
# ##########################################################################

# run batch 'GeoCert' method on a collection of simple perfectly glued polytopes

# =====================
# Create Polytopes
# =====================

# Two glued 2d-triangle polytopes

ub_A = np.asarray([[-1, 0, 1], [1, -1, 0]]).transpose()
ub_b = np.asarray([1, 0, 0])
P1 = Polytope(ub_A, ub_b)

ub_A = np.asarray([[1, 0, -1], [1, -1, 0]]).transpose()
ub_b = np.asarray([1, 0, 0])
P2 = Polytope(ub_A, ub_b)

poly_list = [P1, P2]

# ===========
# Find Projection
# ===========

unshared_facets = compute_boundary_batch(poly_list)

x_0 = np.asarray([[0], [0.4]])
norm = 'l_inf'
t, _, _ = batch_GeoCert(poly_list, x_0, norm)

print('max linf size:', t)



##########################################################################
#                                                                        #
#                             Example 2                                  #
#                                                                        #
##########################################################################

# run batch 'GeoCert' on a moderate number of perfectly glued polytopes.
# polytopes generated from the collection of perfectly glued polytopes
# defined by the ReLu configurations of a network with Gaussian random weights.
# unique ReLu configs gathered by uniform sampling of (2*xylim)x(2*xylim) box
# centered at the origin

# ==================================
# Gather Polytopes
# ==================================

print('===============Collecting Polytopes============')
layer_sizes = [2, 10, 8, 1]
network = PLNN(layer_sizes)
num_pts = 200
xylim = 0.8

unique_relu_configs_list, unique_bin_acts, _, _ = utils.get_unique_relu_configs(network, xylim, num_pts)
print('number of polytopes:', len(unique_bin_acts))

polytope_list = []
colors = []
color_dict = utils.get_color_dictionary(unique_bin_acts)

for relu_configs, unique_act in zip(unique_relu_configs_list, unique_bin_acts):
    polytope_dict = network.compute_polytope_config(relu_configs, True)
    polytope = from_polytope_dict(polytope_dict)
    polytope_list.append(polytope)
    colors.append(color_dict[unique_act])

# colors = utils.get_spaced_colors(200)[0:len(polytope_list)]

# =================================
# Find Projection
# =================================

print('===============Finding Projection============')
x_0 = np.asarray([[0.0], [0.0]])
print('from point: ')
print(x_0)

norm = 'l_2'
t, boundary, unshared_facets = batch_GeoCert(polytope_list, x_0, norm, 'fast_ReLu')
print('the final projection value:', t)


# ------------------------------
# Plot Polytopes, boundary, and linf norm
# ------------------------------

fig = plt.figure()
ax = plt.axes()
alpha = 0.6
xylim = 0.8

utils.plot_polytopes_2d(polytope_list, colors, alpha, xylim, ax)
utils.plot_facets_2d(boundary, xylim=xylim, ax=ax, linestyle='solid')
# utils.plot_facets_2d(unshared_facets, xylim=xylim, ax=ax, linestyle='dashed', color='black')
utils.plot_l2_norm(x_0, t, ax=ax, linewidth=2.0, edgecolor='red')

plt.autoscale()
cwd = os.getcwd()
plot_dir = cwd + '/plots/batch_geocert/'
filename = plot_dir + 'batch_geocert' + '.svg'
fig.savefig(filename, bbox_inches='tight')
plt.show()



##########################################################################
#                                                                        #
#                          Example 3                                     #
#                                                                        #
##########################################################################

# run incremental 'GeoCert' from the point [0,0]. Network is simple ReLu net with
# Gaussian random weights. Finds maximal l_p ball within which the class label
# remains the same.


# ==================================
# Initialize Network
# ==================================

print('===============Initializing Network============')
layer_sizes = [2, 8, 6, 2]
network = PLNN(layer_sizes)

# ==================================
# Find Projection
# ==================================

lp_norm = 'l_2'

print('===============Finding Projection============')
print('lp_norm: ', lp_norm)
x_0 = torch.Tensor([[0.0], [0.0]])
print('from point: ')
print(x_0)

ax = plt.axes()
cwd = os.getcwd()
print(cwd)
plot_dir = cwd + '/plots/incremental_geocert/'

t = incremental_GeoCert(lp_norm, network, x_0, ax, plot_dir)

print('the final projection value:', t)


# ##########################################################################
# #                                                                        #
# #                          Example 4                                     #
# #                                                                        #
# ##########################################################################

# apply incremental geocert to a normal and l1-regularized classifier. Finds maximal l_p balls
# for random points in R^2.

# ==================================
# Generate Training Points
# ==================================

print('===============Generating Training Points============')
# random points at least 2r apart
m = 12
np.random.seed(3)
x = [np.random.uniform(size=(2))]
r = 0.16
while(len(x) < m):
    p = np.random.uniform(size=(2))
    if min(np.abs(p-a).sum() for a in x) > 2*r:
        x.append(p)
# r = 0.145
epsilon = r/2

X = torch.Tensor(np.array(x))
torch.manual_seed(1)
y = (torch.rand(m)+0.5).long()

# ==================================
# Initialize Network
# ==================================

print('===============Initializing Network============')
# layer_sizes = [2, 8, 2]
layer_sizes = [2, 50, 8, 2]
network = PLNN(layer_sizes)
net = network.net


# ==================================
# Train Network
# ==================================

# print('===============Training Network============')
# opt = optim.Adam(net.parameters(), lr=1e-3)
# for i in range(1000):
#     out = net(Variable(X))
#     l = nn.CrossEntropyLoss()(out, Variable(y))
#     err = (out.max(1)[1].data != y).float().mean()
#     if i % 100 == 0:
#         print(l.data[0], err)
#     opt.zero_grad()
#     (l).backward()
#     opt.step()

def l1_loss(net):

    return sum([_.norm(p=1) for _ in net.parameters() if _.dim() > 1])


print('===============Training Network with Regularization============')
opt = optim.Adam(net.parameters(), lr=1e-3)
for i in range(1000):
    out = net(Variable(X))
    l = nn.CrossEntropyLoss()(out, Variable(y)).view([1])

    l1_scale = torch.Tensor([1e-4])
    l += l1_scale * l1_loss(net).view([1])

    err = (out.max(1)[1].data != y).float().mean()
    opt.zero_grad()
    (l).backward()
    opt.step()

print('error: ', err)



# ==================================
# Visualize:  classifier boundary
# ==================================

XX, YY = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
y0 = network(X0)
ZZ = (y0[:,0] - y0[:,1]).resize(100, 100).data.numpy()

_, ax = plt.subplots(figsize=(8,8))
ax.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-1000,1000,3))
ax.scatter(X.numpy()[:,0], X.numpy()[:,1], c=y.numpy(), cmap="coolwarm", s=70)
ax.axis("equal")
ax.axis([0, 1, 0, 1])

plt.show()

# ==================================
# Visualize: baseline classifier ReLu regions
# ==================================

print('===============Collecting Polytopes============')
num_pts = 200
xylim = 1.0

unique_relu_configs_list, unique_bin_acts, _, _ = utils.get_unique_relu_configs(network, xylim, num_pts)
print('number of polytopes:', len(unique_bin_acts))
color_dict = utils.get_color_dictionary(unique_bin_acts)
polytope_list = []


for relu_configs, unique_act in zip(unique_relu_configs_list, unique_bin_acts):
    polytope_dict = network.compute_polytope_config(relu_configs, True)
    polytope = from_polytope_dict(polytope_dict)
    polytope_list.append(polytope)
    # colors.append(color_dict[unique_act])
colors = utils.get_spaced_colors(200)[0:len(polytope_list)]
x_0 = torch.Tensor([[0.3], [0.5]])

print('===============Finding Classification Boundary Facets============')

true_label = int(network(x_0).max(1)[1].item())  # what the classifier outputs

adversarial_facets = []
for polytope in polytope_list:
    polytope_adv_constraints = network.make_adversarial_constraints(polytope.config,
                                                                    true_label)

    for facet in polytope_adv_constraints:
        adversarial_facets.append(facet)


# ------------------------------
# Plot Polytopes, boundary, and lp norm
# ------------------------------

ax = plt.gca()
alpha = 0.6
xylim = 1.0

utils.plot_polytopes_2d(polytope_list, colors, alpha, xylim, ax)
utils.plot_facets_2d(adversarial_facets, xylim=xylim, ax=ax, color='black', linestyle='dashed')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.show()


# ==================================
# Find Projections
# ==================================

lp_norm = 'l_2'
ts = []
pts = x

for pt in pts:
    print('===============Finding Projection============')
    print('lp_norm: ', lp_norm)
    x_0 = torch.Tensor(pt.reshape([2, 1]))
    print(x_0)
    print('from point: ')
    print(x_0)

    ax = plt.axes()
    cwd = os.getcwd()
    print(cwd)
    plot_dir = cwd + '/plots/incremental_geocert/'

    t = incremental_GeoCert(lp_norm, network, x_0, ax, plot_dir)

    print('the final projection value:', t)
    ts.append(t)

# ==================================
# Visualize: incremental geocert projections
# ==================================

XX, YY = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
y0 = network(X0)
ZZ = (y0[:,0] - y0[:,1]).resize(100,100).data.numpy()

_, ax = plt.subplots(figsize=(8,8))
ax.contourf(XX,YY,-ZZ, cmap="coolwarm", levels=np.linspace(-1000,1000,3))
ax.scatter(X.numpy()[:,0], X.numpy()[:,1], c=y.numpy(), cmap="coolwarm", s=70)
ax.axis("equal")
ax.axis([0, 1, 0, 1])

for pt, t, y in zip(pts, ts, y.numpy()):
    if lp_norm == 'l_2':
        utils.plot_l2_norm(pt, t, ax=ax)
    elif lp_norm == 'l_inf':
        utils.plot_linf_norm(pt, t, ax=ax)
    else:
        raise NotImplementedError

print('average_linf:', np.average(ts))

plt.show()