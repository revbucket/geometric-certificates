
# =====================
# Imports
# =====================
import sys
sys.path.append('mister_ed')
#sys.path.append('../mister_ed') # library for adversarial examples
from collections import defaultdict
import geocert_oop as geo
from domains import Domain
from plnn import PLNN
import _polytope_ as _poly_
from _polytope_ import Polytope, Face
import utilities as utils
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


from cvxopt import solvers, matrix
import adversarial_perturbations as ap
import prebuilt_loss_functions as plf
import loss_functions as lf
import adversarial_attacks as aa
import utils.pytorch_utils as me_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import mnist.mnist_loader as  ml
MNIST_DIM = 784


##################################################################################
#                                                                                #
#                                TRAIN OR LOAD NEURAL NET                        #
#                                                                                #
##################################################################################

# Define functions to train and evaluate a network

def l1_loss(net):
    return sum([_.norm(p=1) for _ in net.parameters() if _.dim() > 1])

def l2_loss(net):
    return sum([_.norm(p=2) for _ in net.parameters() if _.dim() > 1])


def train(net, trainset, num_epochs):
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
    for epoch in range(num_epochs):
        err_acc = 0
        err_count = 0
        for data, labels in trainset:
            output = net(Variable(data.view(-1, 784)))
            l = nn.CrossEntropyLoss()(output, Variable(labels)).view([1])
            # l1_scale = torch.Tensor([2e-3])
            # l += l1_scale * l1_loss(net).view([1])

            err_acc += (output.max(1)[1].data != labels).float().mean()
            err_count += 1
            opt.zero_grad()
            (l).backward()
            opt.step()
        print("(%02d) error:" % epoch, err_acc / err_count)


def test_acc(net, valset):
    err_acc = 0
    err_count = 0
    for data, labels in valset:
        n = data.shape[0]
        output = net(Variable(data.view(-1, 784)))
        err_acc += (output.max(1)[1].data != labels).float().mean() * n
        err_count += n

    print("Accuracy of: %.03f" % (1 - (err_acc / err_count).item()))


NETWORK_NAME = '17_mnist_small.pkl'
ONE_SEVEN_ONLY = True

if ONE_SEVEN_ONLY:
    trainset = ml.load_single_digits('train', [1, 7], batch_size=16,
                                      shuffle=False)
    valset = ml.load_single_digits('val', [1, 7], batch_size=16,
                                      shuffle=False)
else:
    trainset = ml.load_mnist_data('train', batch_size=128, shuffle=False)
    valset = ml.load_mnist_data('val', batch_size=128, shuffle=False)


try:
    network = pickle.load(open(NETWORK_NAME, 'rb'))
    net = network.net
    print("Loaded pretrained network")
except:
    print("Training a new network")

    network = PLNN([MNIST_DIM, 10, 50, 10, 2])
    net = network.net
    train(net, trainset, 10)
    pickle.dump(network, open(NETWORK_NAME, 'wb'))

test_acc(net, valset)

##############################################################################
#                                                                            #
#   NOW DO LOOP -- with and without domain bounds                            #
#                                                                            #
##############################################################################
l2_file = open('l2_counts_DOMAIN.txt', 'w+')
linf_file= open('linf_counts_DOMAIN.txt', 'w+')



for i, mb in enumerate(valset):
    for j, ex in enumerate(mb[0]):
        print("--------------------STARTING %02d, %02d" % (i, j))
        for lp_file, lp_str, bound in [(l2_file, 'l_2', 1.5),
                            (linf_file, 'l_inf', 0.2)]:

            try:
                cert_obj = geo.IncrementalGeoCert(network, config_fxn='parallel',
                                  config_fxn_kwargs={'num_jobs': 1},
                                  hyperbox_bounds=[0, 1.0],
                                  verbose=True)
                output = cert_obj.count_regions(ex.view(-1, 1), bound, lp_norm=lp_str,
                                                compute_upper_bound=False)
                lp_file.write(str(output))
                lp_file.write('\n')
                lp_file.flush()
            except Exception as err:
                print("<<<<<<ERROR")
                print(err)
                print(">>>>>>ERROR")








