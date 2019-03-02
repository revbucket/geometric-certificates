# ==================================
# Experiment 7
# ==================================

# estimating the number of polytopes between a random mnist digit and the
# classification boundary
# ======NOTE: this experiment is not completed==================

# =====================
# Imports
# =====================

from geocert import compute_boundary_batch, batch_GeoCert, incremental_GeoCert
from plnn import PLNN
from _polytope_ import Polytope, from_polytope_dict
import utilities as utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import foolbox



# ==================================
# MNIST
# ==================================

#TODO: implement for MNIST

# ==================================
# Random Scattered Points
# ==================================

# random points at least 2r apart
xylim = 1.0
m = 60
# np.random.seed(3)
x = [xylim*np.random.uniform(size=(2))]
r = 0.16/3.5
while(len(x) < m):
    p = xylim*np.random.uniform(size=(2))
    if min(np.abs(p-a).sum() for a in x) > 2*r:
        x.append(p)
# r = 0.145
epsilon = r/2

X = torch.Tensor(np.array(x))
torch.manual_seed(1)
y = (torch.rand(m)+0.5).long()

plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", s=70)


# ==================================
# Initialize Network
# ==================================

import torchvision.models as models
resnet18 = models.resnet18(pretrained=True).eval()  # for CPU, remove cuda()

print('===============Initializing Network============')
layer_sizes = [2, 8, 4, 2]
network = PLNN(layer_sizes)
net = network.net

filename = str(layer_sizes) + '_boundary.svg'
plt.savefig('plots/figures/Exp_7/' + filename)

# ==================================
# Train Network
# ==================================

print('===============Training Network============')
torch.manual_seed(1)
opt = optim.Adam(net.parameters(), lr=1e-3)
for i in range(3000):
    out = net(Variable(X))
    l = nn.CrossEntropyLoss()(out, Variable(y))
    err = (out.max(1)[1].data != y).float().mean()
    opt.zero_grad()
    (l).backward()
    opt.step()

print('Final Error:')
print(err)

# ==================================
# Do PGD
# ==================================

# get source image and label
index = 0
image = X[index, :].numpy()
label = y[index].numpy()

# apply attack on source image
bounds = (-100, 100)    #TODO: change for MNIST (min and max of input value)
num_classes = 2 #TODO: change for MNIST
fmodel = foolbox.models.PyTorchModel(net.eval(), bounds=bounds, num_classes=num_classes)
# attack = foolbox.attacks.FGSM(fmodel)
# attack = foolbox.attacks.L2BasicIterativeAttack(fmodel)
attack = foolbox.attacks.LinfinityBasicIterativeAttack(fmodel)
epsilon = 0.3   #TODO: change this
print('Original Point:')
print(image)
print('Original Class:')
print(label)
print('Original Pred')
print(out.detach().numpy()[0])
adversarial = attack(image, label, epsilon=epsilon)
print('Classification Change Example:')
print(adversarial)
print('Original Labels:')
print(y)
print('Original Preds:')
print(out.detach().numpy())

# ==================================
# Get classifier boundary
# ==================================
xylim = [-.5, 1.5]
XX, YY = np.meshgrid(np.linspace(xylim[0], xylim[1], 100), np.linspace(xylim[0], xylim[1], 100))
X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
y0 = network(X0)
ZZ = (y0[:, 0] - y0[:, 1]).resize(100, 100).data.numpy()

fig, ax = plt.subplots(figsize=(8, 8))
levels_param = np.linspace(-1000, 1000, 3)
contour_obj = ax.contourf(XX, YY, -ZZ, cmap="coolwarm", levels=levels_param)
ax.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y.numpy(), cmap="coolwarm", s=30)
paths = contour_obj.collections[0].get_paths()[0]
decision_boundary = paths.vertices
filename = str(layer_sizes) + '_class_regions.svg'
plt.savefig('plots/figures/Exp_7/' + filename)

fig, ax = plt.subplots(figsize=(8, 8))
filename = str(layer_sizes) + '_boundary.svg'
min_level = min(y0[:, 0] - y0[:, 1]).detach();
max_level = max(y0[:, 0] - y0[:, 1]).detach()
levels_param = np.linspace(min_level, max_level, 15)
contour_obj = ax.contourf(XX, YY, -ZZ, cmap="coolwarm")
cbar = fig.colorbar(contour_obj)
plt.savefig('plots/figures/Exp_7/' + filename)


# ==================================
# Get Polytopes and Adv. Boundaries
# ==================================

print('===============Collecting Polytopes============')
num_pts = 200

unique_relu_configs_list, unique_bin_acts, _, _ = utils.get_unique_relu_configs(network, xylim, num_pts)
print('number of polytopes:', len(unique_bin_acts))
color_dict = utils.get_color_dictionary(unique_bin_acts)
polytope_list = []

for relu_configs, unique_act in zip(unique_relu_configs_list, unique_bin_acts):
    polytope_dict = network.compute_polytope_config(relu_configs, True)
    polytope = from_polytope_dict(polytope_dict)
    polytope_list.append(polytope)
    # colors.append(color_dict[unique_act])
num_colors = max(200, len(polytope_list))
colors = utils.get_spaced_colors(num_colors)[0:len(polytope_list)]
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
# Plot Polytopes, PGD projection, and Adversarial Facets
# ------------------------------

plt.clf()
ax = plt.axes()
alpha = 0.6
utils.plot_polytopes_2d(polytope_list, colors, alpha, xylim, ax)

utils.plot_l2_norm(image,t=epsilon)
plt.scatter(image[0], image[1], marker='*', markersize=12)

utils.plot_facets_2d(adversarial_facets, xylim=xylim, ax=ax, color='red', linestyle='solid', linewidth=0.5)
plt.xlim(xylim[0], xylim[1])
plt.ylim(xylim[0], xylim[1])
filename = str(layer_sizes) + '_polytopes.svg'
plt.savefig('plots/figures/Exp_7/' + filename)


