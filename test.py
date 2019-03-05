import numpy as np
import torch
from plnn import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from cvxopt import matrix

# apply incremental geocert to a normal and l1-regularized classifier. Finds maximal l_p balls
# for random points in R^2.

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
while(len(x) < m):
    p = np.random.uniform(size=(input_dim))
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
layer_sizes = [input_dim, 50, 8, 2]
network = PLNN(layer_sizes)
net = network.net

# ==================================
# Get Polytope
# ==================================

x_0 = torch.Tensor(x[0].reshape([input_dim, 1]))
poly_dict = network.compute_polytope(x_0)


# =======================================================

# Figures 8.3 and 8.4, pages 412 and 416.
# Ellipsoidal approximations.

from math import log, pi
from cvxopt import blas, lapack, solvers, matrix, sqrt, mul, cos, sin
solvers.options['show_progress'] = False
try: import pylab
except ImportError: pylab_installed = False
else: pylab_installed = True

# Extreme points (with first one appended at the end)
X = matrix([ 0.55,  0.25, -0.20, -0.25,  0.00,  0.40,  0.55,
             0.00,  0.35,  0.20, -0.10, -0.30, -0.20,  0.00 ], (7,2))
m = X.size[0] - 1

# Inequality description G*x <= h with h = 1
G, h = matrix(0.0, (m,2)), matrix(0.0, (m,1))
G = (X[:m,:] - X[1:,:]) * matrix([0., -1., 1., 0.], (2,2))
h = (G * X.T)[::m+1]
G = mul(h[:,[0,0]]**-1, G)
h = matrix(1.0, (m,1))

print(G)
print(h)

print(np.shape(G))
print(np.shape(h))

G = matrix(poly_dict['poly_a'])
h = matrix(poly_dict['poly_b'])

print(G)
print(h)


print(np.shape(G))
print(np.shape(h))

# Maximum volume enclosed ellipsoid
#
# minimize    -log det B
# subject to  ||B * gk||_2 + gk'*c <= hk,  k=1,...,m
#
# with variables  B and c.
#
# minimize    -log det L
# subject to  ||L' * gk||_2^2 / (hk - gk'*c) <= hk - gk'*c,  k=1,...,m
#
# L lower triangular with positive diagonal and B*B = L*L'.
#
# minimize    -log x[0] - log x[2]
# subject to   g( Dk*x + dk ) <= 0,  k=1,...,m
#
# g(u,t) = u'*u/t - t
# Dk = [ G[k,0]   G[k,1]  0       0        0
#        0        0       G[k,1]  0        0
#        0        0       0      -G[k,0]  -G[k,1] ]
# dk = [0; 0; h[k]]
#
# 5 variables x = (L[0,0], L[1,0], L[1,1], c[0], c[1])

D = [ matrix(0.0, (3,5)) for k in range(m) ]
for k in range(m):
    D[k][ [0, 3, 7, 11, 14] ] = matrix( [G[k,0], G[k,1], G[k,1],
        -G[k,0], -G[k,1]] )
d = [matrix([0.0, 0.0, hk]) for hk in h]

def F(x=None, z=None):
    if x is None:
        return m, matrix([ 1.0, 0.0, 1.0, 0.0, 0.0 ])
    if min(x[0], x[2], min(h-G*x[3:])) <= 0.0:
        return None

    y = [ Dk*x + dk for Dk, dk in zip(D, d) ]

    f = matrix(0.0, (m+1,1))
    f[0] = -log(x[0]) - log(x[2])
    for k in range(m):
        f[k+1] = y[k][:2].T * y[k][:2] / y[k][2] - y[k][2]

    Df = matrix(0.0, (m+1,5))
    Df[0,0], Df[0,2] = -1.0/x[0], -1.0/x[2]

    # gradient of g is ( 2.0*(u/t);  -(u/t)'*(u/t) -1)
    for k in range(m):
        a = y[k][:2] / y[k][2]
        gradg = matrix(0.0, (3,1))
        gradg[:2], gradg[2] = 2.0 * a, -a.T*a - 1
        Df[k+1,:] =  gradg.T * D[k]
    if z is None: return f, Df

    H = matrix(0.0, (5,5))
    H[0,0] = z[0] / x[0]**2
    H[2,2] = z[0] / x[2]**2

    # Hessian of g is (2.0/t) * [ I, -u/t;  -(u/t)',  (u/t)*(u/t)' ]
    for k in range(m):
        a = y[k][:2] / y[k][2]
        hessg = matrix(0.0, (3,3))
        hessg[0,0], hessg[1,1] = 1.0, 1.0
        hessg[:2,2], hessg[2,:2] = -a,  -a.T
        hessg[2, 2] = a.T*a
        H += (z[k] * 2.0 / y[k][2]) *  D[k].T * hessg * D[k]

    return f, Df, H

sol = solvers.cp(F)
L = matrix([sol['x'][0], sol['x'][1], 0.0, sol['x'][2]], (2,2))
c = matrix([sol['x'][3], sol['x'][4]])

if pylab_installed:
    pylab.figure(2, facecolor='w')

    # polyhedron
    for k in range(m):
        edge = X[[k,k+1],:] + 0.1 * matrix([1., 0., 0., -1.], (2,2)) * \
            (X[2*[k],:] - X[2*[k+1],:])
        pylab.plot(edge[:,0], edge[:,1], 'k')


    # 1000 points on the unit circle
    nopts = 1000
    angles = matrix( [ a*2.0*pi/nopts for a in range(nopts) ], (1,nopts) )
    circle = matrix(0.0, (2,nopts))
    circle[0,:], circle[1,:] = cos(angles), sin(angles)

    # ellipse = L * circle + c
    ellipse = L * circle + c[:, nopts*[0]]
    ellipse2 = 2.0 * L * circle + c[:, nopts*[0]]

    pylab.plot(ellipse2[0,:].T, ellipse2[1,:].T, 'k-')
    pylab.fill(ellipse[0,:].T, ellipse[1,:].T, facecolor = '#F0F0F0')
    pylab.title('Maximum volume inscribed ellipsoid (fig 8.4)')
    pylab.axis('equal')
    pylab.axis('off')

    pylab.show()



# # ------------------------------------------------------------------------------
# import numpy as np
# from _polytope_ import Polytope, Face
# import utilities as utils
# import matplotlib.pyplot as plt
#
# # ---------------------------
# # Test Redundant Removal PGD
# # ---------------------------
#
# # ub_A = np.asarray([[1, 1], [1,  1], [1,  1], [1,  1]])
# # ub_b = np.asarray([0.5, 1, 2, 3])
#
# m = 10; n = 2
# ub_A = np.random.normal(0.0, 1.0, (m,n))
# ub_b = np.random.normal(0.0, 1.0, m)
#
# P = Polytope(ub_A, ub_b)
# t = np.linalg.norm(np.asarray([0.5,0.5]))
# x_0 = np.asarray([0, 0])
# P.redund_removal_pgd_l2(t, x_0)
# indices = P.redundant
#
# xylim = [-5, 5]
# styles = ['--' if index else '-' for index in indices]
# utils.plot_l2_norm(x_0, t)
# plt.xlim(xylim); plt.ylim(xylim)
# utils.plot_hyperplanes(ub_A, ub_b, styles)
#
#
# plt.show()
