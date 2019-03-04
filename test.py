
import numpy as np
from _polytope_ import Polytope, Face
import utilities as utils
import matplotlib.pyplot as plt


# ub_A = np.asarray([[1, 1], [1,  1], [1,  1], [1,  1]])
# ub_b = np.asarray([0.5, 1, 2, 3])

m = 10; n = 2
ub_A = np.random.normal(0.0, 1.0, (m,n))
ub_b = np.random.normal(0.0, 1.0, m)

P = Polytope(ub_A, ub_b)
t = np.linalg.norm(np.asarray([0.5,0.5]))
x_0 = np.asarray([0, 0])
P.redund_removal_pgd(t, x_0)
indices = P.redundant

xylim = [-5, 5]
styles = ['--' if index else '-' for index in indices]
utils.plot_l2_norm(x_0, t)
plt.xlim(xylim); plt.ylim(xylim)
utils.plot_hyperplanes(ub_A, ub_b, styles)


plt.show()
