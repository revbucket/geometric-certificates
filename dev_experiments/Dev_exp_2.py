# =====================
# Imports
# =====================

from plnn import PLNN
import torch
import os
import numpy as np
from geocert_oop import IncrementalGeoCert
from utilities import Polytope_2
import polytope as poly_lib
from _polytope_ import Polytope
import matplotlib.pyplot as plt

# ##########################################################################
# #                                                                        #
# #                          Dev_Exp_2                                     #
# #                                                                        #
# ##########################################################################

# Experiment to investigate the distribution of volume of polytopes for
# example where sampled polytopes have all constraints as essential

# ===============================================
# Find polytopes using Geocert
# ===============================================

# ------ Initialize Net ---------
print('===============Initializing Network============')
input_dim = 10
layer_sizes = [input_dim, 50, 50, 2]
network = PLNN(layer_sizes)
x_0_np = np.random.randn(1, input_dim)
x_0 = torch.Tensor(x_0_np)

# ------ Run Geocert ---------
cwd = os.getcwd()
plot_dir = cwd+'/plots/incremental_geocert/'
config_fxn_kwargs = {'use_clarkson': False,
                     'use_ellipse': False}
geocert = IncrementalGeoCert(network, display=False, config_fxn='serial',
                             save_dir=plot_dir, config_fxn_kwargs=config_fxn_kwargs)
lp_norm = 'l_2'
t, cw_bound,cw_examp, adver_examp, elem = geocert.min_dist(x_0, lp_norm, compute_upper_bound=False
                                            ,max_iter=2)
polytopes = geocert.seen_to_polytope_map.values()


# ===============================================
#  Estimate volume of polytopes found
# ===============================================

# for poly in polytopes:
#     a_polytope = Polytope_2(poly.ub_A, poly.ub_b)
#     extreme_pts = poly_lib.extreme(a_polytope)
#     print('num_ex_pts:', len(extreme_pts))

volumes = []
for poly in polytopes:
    ptope = Polytope_2(poly.ub_A, poly.ub_b)
    volume = poly_lib.volume(ptope)
    volumes.append(volume)
    print('VOLUME:', volume)


# # =========================
# # Find number of Red. Constraints
# # =========================

# distances.append(np.linalg.norm(x_0_np))
# poly_dict = network.compute_polytope(x_0)
# polytope = Polytope.from_polytope_dict(poly_dict)
# essential_constrs, reject_reasons = polytope.generate_facets_configs(None, network, use_clarkson=False, use_ellipse=False)
#
# num_constrs = np.shape(polytope.ub_A)[0]
# num_essential = len(essential_constrs)
# ess_percentage = (num_essential / num_constrs)*100

# =========================
# Plots
# =========================
