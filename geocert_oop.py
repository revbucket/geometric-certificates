"""  OOP refactor of geocert so I get a better feel for how the ICML
    implementation went -mj (3/1/19)
"""
import sys
sys.path.append('mister_ed')
import adversarial_perturbations as ap
import prebuilt_loss_functions as plf

import prebuilt_loss_functions as plf
import loss_functions as lf
import adversarial_attacks as aa
import utils.pytorch_utils as me_utils

from _polytope_ import Polytope, Face
import utilities as utils
import torch
import numpy as np
import heapq
import time
import matplotlib.pyplot as plt

import torch.nn.functional as F

##############################################################################
#                                                                            #
#                               BATCHED GEOCERT                              #
#                                                                            #
##############################################################################


class BatchGeoCert(object):

    def __init__(self, polytope_list, comp_method='slow',
                 verbose=True):
        self.polytope_list = polytope_list
        for poly in self.polytope_list:
            poly.to_comparison_form()



        assert comp_method in ['slow', 'unstable', 'fast_ReLu']
        self.comp_method = comp_method

        self.verbose = verbose

    def _safety_setup(self, x ):
        """
        Just make sure this is well-posed.
        For now, return True iff x is in at least one of the polytopes

        Technically we also require the union of polytopes to be perfectly
        glued, but this is harder to check, so we leave it for now
        """
        return any(p.is_point_feasible(x) for p in self.polytope_list)


    def compute_boundaries(self):
        """ Generates a list of the shared and unshared (n-1 dimensional) faces
            of the boundary of the polytope list
        ARGS:
            None
        RETURNS:
            (unshared_facets, shared_facets) where each is a list of Face
            objects
        """
        # First gather all the feasible facets from the polytope list
        total_facets = [f for poly in self.polytope_list for
                        f in poly.generate_facets_naive(check_feasible=True)]

        if self.verbose:
            print('num total facets:', len(total_facets))

        unshared_facets, shared_facets = [], []

        # Next handle techniques for which comparison method we use
        same_facet_dict = {'slow': Face.check_same_facet_pg_slow,
                             'unstable': Face.check_same_facet_pg,
                             'fast_ReLu': Face.check_same_facet_config}
        same_facet = same_facet_dict[self.comp_method]


        # Loop through all facets, figure out which are shared/unshared
        for og_facet in total_facets:
            bool_unshared = [same_facet(og_facet, ex_facet)
                             for ex_facet in unshared_facets]
            bool_shared = [same_facet(og_facet, ex_facet)
                           for ex_facet in shared_facets]

            # Don't add or remove anything already accounted for in shared list
            if any(bool_shared):
                continue

            # Remove the shared facet from the 'unshared list', add to 'shared'
            elif any(bool_unshared):
                index = bool_unshared.index(True)
                shared_facets.append(unshared_facets.pop(index))

            # Otherwise, just add this facet to the 'unshared list'
            else:
                unshared_facets.append(og_facet)

        return unshared_facets, shared_facets


    def min_dist(self, x, norm='l_261'):
        """ Returns the minimum distance from self.x to the boundary of the
            polytopes.
        ARGS:
            None
        RETURNS:
            if self.x is not contained in the list of polytopes, returns
            (-1, None, None)
            else, returns (min_dist, boundary, shared_facets) where:
            - min_dist is the minimum l_p dist to the boundary,
            - boundary is a list of facets that define the boundary
            - shared_facets is a list of facets that are shared amongst the
                            polytopes
        """
        assert norm in ['l_inf', 'l_2']

        if not self._safety_setup(x):
            return -1, None, None

        if self.verbose:
            print('----------Computing Boundary----------')
        boundary, shared_facets = self.compute_boundaries()

        dist_fxn = {'l_inf': Face.linf_dist,
                    'l_2': Face.l2_dist}[norm]

        x = x.reshape(-1, 1)
        min_dist = min(dist_fxn(facet, x)[0] for facet in boundary)
        return min_dist, boundary, shared_facets



###############################################################################
#                                                                             #
#                               INCREMENTAL GEOCERT                           #
#                                                                             #
###############################################################################


class HeapElement(object):
    """ Wrapper of the element to be pushed around the priority queue
        in the incremental algorithm
    """
    def __init__(self, lp_dist, facet,
                 facet_type='facet',
                 exact_or_estimate='exact'):
        self.lp_dist = lp_dist
        self.facet = facet

        assert facet_type in ['facet', 'decision', 'domain']
        self.facet_type = facet_type
        self.exact_or_estimate = exact_or_estimate
        self.projection = None

    def __lt__(self, other):
        return self.lp_dist < other.lp_dist

    def decision_bound(self):
        return self.facet_type == 'decision'

    def domain_bound(self):
        return self.facet_type == 'domain'


class IncrementalGeoCert(object):
    def __init__(self, net, verbose=True, display=False, save_dir=None,
                 ax=None, config_fxn='serial',
                 config_fxn_kwargs=None,
                 domain_bounds=None):

        # Direct input state
        self.lp_norm = None # filled in later
        self.true_label = None # filled in later
        self.lp_dist = None # filled in later
        self.net = net
        self.verbose = verbose
        self.display = display
        self.save_dir = save_dir
        self.ax = ax
        facet_config_map = {'serial': Polytope.generate_facets_configs,
                            'parallel': Polytope.generate_facets_configs_parallel2}
        self.facet_config_fxn = facet_config_map[config_fxn]
        if config_fxn_kwargs is None:
            if config_fxn == 'parallel':
                config_fxn_kwargs = None
            else:
                config_fxn_kwargs = {'use_clarkson': True,
                                     'use_ellipse': False}
        self.config_fxn_kwargs = config_fxn_kwargs or {}

        # Things to keep track of
        self.seen_to_polytope_map = {} # binary config str -> Polytope object
        self.seen_to_facet_map = {} # binary config str -> Facet list
        self.pq = [] # Priority queue that contains HeapElements
        self.upper_bound = [None] # wrapped in a dynamic object
        self.dead_constraints = [None] # wrapped in a dynamic object

        # Handle domain bounds
        # Is either None, a single pair (lo, hi), or a list of pairs of length
        # input-dimension(net). Provides box constraints for the domain of the
        # search. These constraints get thrown in to the polytope and facet
        assert domain_bounds is None or\
               (isinstance(domain_bounds, list) and
                all(isinstance(_, tuple) for _ in domain_bounds))
        self.domain_bounds = domain_bounds


    def _verbose_print(self, *args):
        if self.verbose:
            print(*args)


    def _compute_upper_bounds(self, x, true_label, lp_dist,
                              extra_attack_kwargs=None):
        """ Runs an adversarial attack to compute an upper bound on the
            distance to the decision boundary.

            In the l_inf case, we compute the constraints that are always
            on or off in the specified upper bound

        """
        self._verbose_print("Starting upper bound computation")

        start = time.time()
        upper_bound, adv_ex = self._pgd_upper_bound(x, true_label, self.lp_norm,
                                              extra_kwargs=extra_attack_kwargs)

        if upper_bound is None:
            self._verbose_print("Upper bound failed in %.02f seconds" %
                                (time.time() - start))
        else:
            self._verbose_print("Upper bound of %s in %.02f seconds" %
                                (upper_bound, time.time() - start))
            self._update_dead_constraints(x, upper_bound)

        return upper_bound, adv_ex

    def _pgd_upper_bound(self, x, true_label, lp_norm, num_repeats=64,
                         extra_kwargs=None):

        ######################################################################
        #   Setup attack object                                              #
        ######################################################################
        norm = {'l_inf': 'inf', 'l_2': 2}[lp_norm]
        linf_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf',
                                                        'lp_bound': 1.0})
        normalizer = me_utils.IdentityNormalize()

        loss_fxn = plf.VanillaXentropy(self.net, normalizer)

        pgd_attack = aa.PGD(self.net, normalizer, linf_threat, loss_fxn,
                            manual_gpu=False)
        attack_kwargs = {'num_iterations': 1000,
                         'random_init': 0.25,
                         'signed': False,
                         'verbose': False}

        if isinstance(extra_kwargs, dict):
            attack_kwargs.update(extra_kwargs)

        ######################################################################
        #   Setup 'minibatch' of randomly perturbed examples to try          #
        ######################################################################

        new_x = x.view(1, -1).repeat(num_repeats, 1)
        labels = [true_label for _ in range(num_repeats)]
        labels = torch.Tensor(labels).long()

        ######################################################################
        #   Run the attack and collect the best (if any) successful example  #
        ######################################################################

        pert_out = pgd_attack.attack(new_x, labels, **attack_kwargs)
        pert_out = pert_out.binsearch_closer(self.net, normalizer, labels)
        success_out = pert_out.collect_successful(self.net, normalizer,
                                          success_def='alter_top_logit')
        success_idxs = success_out['success_idxs']
        if success_idxs.numel() == 0:
            return None, None

        diffs = pert_out.delta.data.index_select(0, success_idxs)
        max_idx = me_utils.batchwise_norm(diffs, norm, dim=0).min(0)[1].item()
        best_adv = success_out['adversarials'][max_idx].squeeze()
        upper_bound = (best_adv - x.view(-1)).abs().max().item()
        self.upper_bound[0] = upper_bound # assign dynamically
        return self.upper_bound[0], best_adv


    def _update_dead_constraints(self, x, upper_bound_val):
        if self.lp_norm is not 'l_inf':
            return

        upper_bound = upper_bound_val # this is a float
        x = torch.Tensor(x)
        low_box = (x.view(-1) - upper_bound).unsqueeze(-1)
        high_box = (x.view(-1) + upper_bound).unsqueeze(-1)

        box_bounds = torch.cat((low_box, high_box), -1)

        # modify with domain bounds too
        box_bounds = self._intersect_box_domain(box_bounds)
        box_bounds = self.net.compute_interval_bounds(box_bounds)[1]
        dead_constraints = utils.cat_config(box_bounds)

        # Wrap this in a list to update dynamically as we go
        self.dead_constraints[0] = dead_constraints.astype(np.bool)


    def _intersect_box_domain(self, box):
        """ Given an (N x 2) tensor with upper and lower box bounds, we
            intersect with the domain box bounds and return the tighter box
        ARGS:
            box : Tensor (N x 2): lower and upper box bounds
        RETURNS:
            box of same shape as input, but possibly smaller
        """
        if self.domain_bounds is None:
            return box

        if len(self.domain_bounds) == 1:
            lo, hi = self.domain_bounds[0]
        else:
            domain_bounds = torch.Tensor(numpy.vstack(self.domain_bounds))
            lo, hi = domain_bounds[:, 0], domain_bounds[:, 1]

        box[:, 0] = F.relu(box[:, 0] - lo) + lo  # max(lo, box)
        box[:, 1] = hi - F.relu(hi - box[:, 1])  # min(hi, box)
        return box


    def _update_step(self, poly, popped_facet):
        """ Given the next polytope from the popped heap, does the following:
            1) Gets and stashes all the facets of the provided polytope
            2) For each facet, computes lp distance to x and adds to heap
            3) Adds this polytope's adversarial constraints to the heap too
        ARGS:
            poly : _polytope_.Polytope object - which linear region we're
                       exploring
            popped_facet : _polytope_.Face object - which face we came from
                           (because we don't want to add it into the pq again)
        RETURNS:
            None, but modifies the heap
        """

        ######################################################################
        #   Generate new facets and check feasibility/interior point-ness    #
        ######################################################################
        upper_bound_dict = None
        if self.upper_bound is not None:
            upper_bound_dict = {'upper_bound': self.upper_bound,
                                'x': self.x.cpu().numpy(),
                                'lp_norm': self.lp_norm,
                                }

        new_facets, rejects = self.facet_config_fxn(poly,
                                              self.seen_to_polytope_map,
                                              self.net, upper_bound_dict,
                                              **self.config_fxn_kwargs)
        self._verbose_print("Num facets: ", len(new_facets))
        self._verbose_print("REJECT DICT: ", rejects)

        ######################################################################
        #   Add new feasible facets to priority queue                        #
        ######################################################################
        handled_popped_facet = (popped_facet == None)
        for facet in new_facets:
            if facet.facet_type == 'domain':
                continue
            if (not handled_popped_facet) and\
                popped_facet.facet.check_same_facet_config(facet):
                handled_popped_facet = True
                continue
            facet_distance, projection = self.lp_dist(facet, self.x)
            heap_el = HeapElement(facet_distance, facet,
                                  facet_type=facet.facet_type,
                                  exact_or_estimate='exact')
            heap_el.projection = projection

            if (self.upper_bound[0] is None or
                facet_distance < self.upper_bound[0]):
                heapq.heappush(self.pq, heap_el)

        poly_config = utils.flatten_config(poly.config)

        self.seen_to_polytope_map[poly_config] = poly
        self.seen_to_facet_map[poly_config] = new_facets

        ######################################################################
        #   Handle adversarial constraints                                   #
        ######################################################################

        adv_constraints = self.net.make_adversarial_constraints(poly.config,
                              self.true_label, domain_bounds=self.domain_bounds)

        # Step 3) Adds the adversarial constraints
        for facet in adv_constraints:
            facet_distance, projection = self.lp_dist(facet, self.x)
            heap_el = HeapElement(facet_distance, facet, facet_type='decision',
                                  exact_or_estimate='exact')
            heap_el.projection = projection
            if (self.upper_bound[0] is None or
                facet_distance < self.upper_bound[0]):
                heapq.heappush(self.pq, heap_el)
            else:
                if len(self.pq) == 0:
                    raise Exception("WHAT IS GOING ON???")

            # HEURISTIC: IMPROVE UPPER BOUND IF POSSIBLE
            if (self.upper_bound[0] is None or
                facet_distance < self.upper_bound[0]):
                self.upper_bound[0] = facet_distance

                # also update dead constraints for all facets
                self._update_dead_constraints(self.x.cpu().numpy(),
                                              self.upper_bound[0])


    def min_dist(self, x, lp_norm='l_2', compute_upper_bound=False):
        """ Returns the minimum distance between x and the decision boundary.
            Plots things too, I guess...
        """

        ######################################################################
        #   Step 0: Clear and setup state                                    #
        ######################################################################

        # Computed state
        assert lp_norm in ['l_2', 'l_inf']
        self.lp_norm = lp_norm
        self.x = x
        self.true_label = int(self.net(x).max(1)[1].item()) # classifier(x)
        self.lp_dist = {'l_2': Face.l2_dist,
                        'l_inf': Face.linf_dist}[self.lp_norm]


        # Things to keep track of
        self.seen_to_polytope_map = {} # binary config str -> Polytope object
        self.seen_to_facet_map = {} # binary config str -> Facet list
        self.pq = [] # Priority queue that contains HeapElements
        self.upper_bound = [None]
        self.dead_constraints = [None] # wrapped in a dynamic object

        if compute_upper_bound is not False:
            adv_bound, adv_ex = self._compute_upper_bounds(x, self.true_label,
                                                           lp_norm,
                                       extra_attack_kwargs=compute_upper_bound)

        ######################################################################
        #   Step 1: handle the initial polytope                              #
        ######################################################################
        self._verbose_print('---Initial Polytope---')
        p_0_dict = self.net.compute_polytope(self.x, comparison_form_flag=False)

        p_0 = Polytope.from_polytope_dict(p_0_dict,
                                          domain_bounds=self.domain_bounds,
                                         dead_constraints=self.dead_constraints)
        self._update_step(p_0, None)

        ######################################################################
        #   Step 2: Repeat until we hit a decision boundary                  #
        ######################################################################

        index = 0
        if len(self.pq) == 0:
            print("Messed up in the first block??")
        prev_min_dist = min(self.pq, key=lambda el: el.lp_dist).lp_dist
        while True:
            min_el_dist = min(self.pq, key=lambda el: el.lp_dist).lp_dist
            pop_el = heapq.heappop(self.pq)
            # If popped el is part of decision boundary, we're done!
            if pop_el.decision_bound():
                break
            # Otherwise, open up a new polytope and explore
            else:
                prev_min_dist = pop_el.lp_dist
                self._verbose_print('---Opening New Polytope---')
                self._verbose_print('Bounds ', pop_el.lp_dist, "  |  ",
                                     self.upper_bound)
                popped_facet = pop_el.facet
                configs = popped_facet.get_new_configs(self.net)
                configs_flat = utils.flatten_config(configs)


                # If polytope has already been seen, don't add it again
                if configs_flat not in self.seen_to_polytope_map:
                    new_poly_dict = self.net.compute_polytope_config(configs,
                                                                     False)
                    new_poly = Polytope.from_polytope_dict(new_poly_dict,
                                              domain_bounds=self.domain_bounds,
                                         dead_constraints=self.dead_constraints)
                    self._update_step(new_poly, pop_el)
                else:
                    self._verbose_print("We've already seen that polytope")

            if index % 1 == 0 and self.display:
                self.plot_2d(pop_el.lp_dist, iter=index)
            index += 1

        self._verbose_print('----------Minimal Projection Generated----------')
        self._verbose_print("DIST: ", pop_el.lp_dist)
        if self.display:
            self.plot_2d(pop_el.lp_dist, iter=index)
        best_example = pop_el.projection
        return pop_el.lp_dist, adv_bound, adv_ex, best_example, pop_el




    def plot_2d(self, t, n_colors=200, iter=0):
        ''' Plots the current search boundary based on the heap, the seen polytopes,
            the current minimal lp ball, and any classification boundary facets
        '''
        plt.figure(figsize=[10, 10])
        ax = self.ax if self.ax is not None else plt.axes()

        polytope_list = [self.seen_to_polytope_map[elem]
                         for elem in self.seen_to_polytope_map]
        facet_list = [heap_elem.facet for heap_elem in self.pq
                      if not heap_elem.decision_bound()]
        boundary_facet_list = [heap_elem.facet for heap_elem in self.pq
                               if heap_elem.decision_bound()]
        colors = utils.get_spaced_colors(n_colors)[0:len(polytope_list)]

        xylim = 50.0
        utils.plot_polytopes_2d(polytope_list, colors=colors, alpha=0.7,
                              xylim=5, ax=ax, linestyle='dashed', linewidth=0)

        utils.plot_facets_2d(facet_list, alpha=0.7,
                             xylim=xylim, ax=ax, linestyle='dashed',
                            linewidth=3, color='black')

        utils.plot_facets_2d(boundary_facet_list, alpha=0.7,
                             xylim=xylim, ax=ax, linestyle='dashed',
                             linewidth=3, color='red')

        if self.lp_norm == 'l_inf':
            utils.plot_linf_norm(self.x, t, linewidth=1, edgecolor='red', ax=ax)
        elif self.lp_norm == 'l_2':
            utils.plot_l2_norm(self.x, t, linewidth=1, edgecolor='red', ax=ax)
        else:
            raise NotImplementedError

        plt.autoscale()
        new_xlims = plt.xlim()
        new_ylims = plt.ylim()


        if (min(new_xlims) > -xylim) and \
           (max(new_xlims) < xylim) and \
           (min(new_ylims) > -xylim) and \
           (max(new_ylims) < xylim):
            pass
        else:
            plt.xlim(-xylim, xylim)
            plt.ylim(-xylim, xylim)
        if self.save_dir is not None:
            filename = self.save_dir + str(iter) + '.png'
            plt.savefig(filename)
        plt.close()



