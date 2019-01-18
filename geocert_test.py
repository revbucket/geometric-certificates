# =====================
# Imports
# =====================

from geocert import compute_boundary_batch,compute_l_inf_ball_batch
from plnn import PLNN
from polytope import Polytope
import numpy as np

# ==========================================================
# Test 1:
# ==========================================================

# test to see if code works for simple union of glued polytopes

# =====================
# Create Polytopes
# =====================

ub_A = np.asarray([[-1, 0, 1], [1, -1, 0]]).transpose()
ub_b = np.asarray([1, 0, 0])
P1 = Polytope(ub_A, ub_b)

ub_A = np.asarray([[1, 0, -1], [1, -1, 0]]).transpose()
ub_b = np.asarray([1, 0, 0])
P2 = Polytope(ub_A, ub_b)

# P3 =
# P4 =

poly_list = [P1, P2]

# ===========
# Debugging
# ===========

unshared_facets = compute_boundary_batch(poly_list)

x_0 = np.asarray([[0], [0.75]])
t = compute_l_inf_ball_batch(poly_list, x_0)

print(t)
print(7.0/8.0)
