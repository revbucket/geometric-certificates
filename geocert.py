"""  OOP refactor of geocert so I get a better feel for how the ICML
    implementation went -mj (3/1/19)
"""
import sys
import itertools
#sys.path.append('mister_ed')
import mister_ed.adversarial_perturbations as ap
#import prebuilt_loss_functions as plf

#import prebuilt_loss_functions as plf
#import loss_functions as lf
#import adversarial_attacks as aa
#import utils.pytorch_utils as me_utils

from _polytope_ import Polytope, Face
import utilities as utils
from .domains import Domain
from .plnn import PLNN
import inspect
print("PLNN", inspect.getfile(PLNN))
import torch
import numpy as np
import heapq
import time
import matplotlib.pyplot as plt

import torch.nn.functional as F
import joblib
import multiprocessing as mp

from dataclasses import dataclass, field
from typing import Any
from multiprocessing.managers import SyncManager
from threading import Thread
from queue import PriorityQueue, Empty



""" Different from the standard Geocert in that we use multiprocessing


    Multiprocessing flow works like this:
    - First compute the domain and upper bounds and all that nonsense
    - Next handle the first linear region locally to push some stuff onto the
      pq

    - Initialize a bunch of processes that have two phases:
        PROCESS SETUP:
            - load a copy of the net
            - keep track of the most recent domain
            - keep track of the true label

        PROCESS LOOP:
            - Reread and copy the domain onto memory
            - Reread and copy the dead neurons onto memory

            - Given an element off the queue (config + tight constraint),
              list all the facets that would need to be added to the PQ
            - quickly reject what we can (using domain knowledge)
            - quickly reject what we can (using the shared seen-dict)
            - compute feasible/domain bounds on everything else
            - make the new feasible domains available to the main pq

        TERMINATION:
            - if popped adversarial constraint,
    SHARED MEMORY:
        - domain
        - seen_to_polytope_map
        - dead_neurons
        - valid domain
        - priority queue

    LOCAL PROCESS MEMORY :
        - net


"""

############################################################################
#                                                                          #
#                           HELPER CLASSES                                 #
#                                                                          #
############################################################################

def verbose_print(*args, verbose=True):
    if verbose:
        print(*args)


class PQElement:
    priority: float # IS THE LP DIST OR 'POTENTIAL' VALUE
    config: Any=field(compare=False) # Configs for neuron region
    tight_constraint: Any=field(compare=False) # which constraint is tight
    facet_type: Any=field(compare=False) # is decision or nah?
    projection: Any=field(compare=False)

    def __lt__(self, other):
        return self.priority < other.priority


class GeoCertReturn:
    """ Object that encapsulates the output from GeoCert """
    def __init__(self, original=None, original_shape=None, 
                 best_dist=None, best_ex=None, adv_bound=None,
                 adv_ex=None, seen_polytopes=None, missed_polytopes=None,
                 polytope_graph=None, lower_bound_times=None,
                 upper_bound_times=None, status=None, problem_type=None,
                 radius=None, num_regions=None):

        self.original = original
        self.original_shape = original_shape

        # If computed the minimal distance adversarial example...
        self.best_dist = best_dist # this is the distance
        self.best_ex = best_ex  # and this is the example itself

        # If Upper bound Adv.Attack was performed...
        self.adv_bound = adv_bound # this is the adv.ex distance
        self.adv_ex = adv_ex  # this is the adversarial example itself

        # dict of binary strings corresponding to feasible polytopes seen by geocert
        self.seen_polytopes = seen_polytopes

        # dict of binary strings corresponding to infeasible polytopes checked
        self.missed_polytopes = missed_polytopes

        # dict of pairs of binary strings representing the edges of the graph
        self.polytope_graph = polytope_graph

        # list of pairs of (time, lower/upper_bound)
        self.lower_bound_times = lower_bound_times
        self.upper_bound_times = upper_bound_times

        self.status = status # return status ['TIMEOUT', 'FAILURE', 'SUCCESS']
        self.problem_type = problem_type # in ['min_dist', 'decision_problem', 'count_regions']

        self.radius = radius
        self.num_regions = num_regions

    def __repr__(self):
        """ Method to print out results"""
        output_str = 'GeoCert Return Object\n'
        output_str += '\tProblem Type: ' + self.problem_type + '\n'
        output_str += '\tStatus: %s\n' % self.status

        if self.status == 'TIMEOUT':
            return output_str

        if self.problem_type == 'min_dist':
            output_str += '\tRobustness: %.04f' % self.best_dist
        elif self.problem_type in ['decision_problem', 'count_regions']:
            output_str += '\tRadius %.02f\n' % self.radius
            if self.problem_type == 'count_regions':
                output_str += '\tNum Linear Regions: %s' % self.num_regions
        return output_str

    def display_images(self, include_diffs=True, include_pgd=False,
                       figsize=(12, 12)):
        """ Shorthand method to display images found by GeoCert.
            Useful when doing things with GeoCert in jupyter notebooks
        ARGS:
            include_diffs : boolean - if True, we'll display the differences
                            between the original and GeoCert image
                            (diffs scaled up by 5x!)
            include_pgd : boolean - if True, we'll also display the image
                          found by PGD (useful upper bound)
        RETURNS:
            None, but inline displays the images in the order
            [original | diff | geoCert | PGD]
        """
        if self.best_ex is None:
            # No Geocert image => do nothing
            return

        # Build the display row of numpy elements
        original_np = utils.as_numpy(self.original.reshape(self.original_shape))
        best_ex_np = utils.as_numpy(self.best_ex.reshape(self.original_shape))
        display_row = [original_np, best_ex_np]
        label_row = ['original', 'geoCert']
        if include_diffs:
            diff_np = np.clip(0.5 + (best_ex_np - original_np) * 5, 0.0, 1.0)
            display_row.insert(1, diff_np)
            label_row.insert(1, 'difference x5 (+0.5)')

        if include_pgd and self.adv_ex is not None:
            adv_ex_np = utils.as_numpy(self.adv_ex.reshape(self.original_shape))
            display_row.append(adv_ex_np)
            label_row.append('PGD')
        # Make sure everything has three dimensions (CxHxW)
        # --- determine if grayscale or not
        grayscale = (original_np.squeeze().ndim == 2)
        if grayscale:
            num_channels = 1
            imshow_kwargs = {'cmap': 'gray'}
        else:
            num_channels = 3
            imshow_kwargs = {}

        # --- determine height/width
        h, w = original_np.squeeze().shape[-2:]

        for i in range(len(display_row)):
            display_row[i] = display_row[i].reshape((num_channels, h, w))

        # Concatenate everything into a single row, and display
        # --- concatenate row together
        cat_row = np.concatenate(display_row, -1)
        if grayscale: 
            cat_row = cat_row.squeeze()
        plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
        plt.axis('off')
        plt.imshow(cat_row, **imshow_kwargs)

        # -- add labels underneath the images
        for label_idx, label in enumerate(label_row):
            x_offset = (0.33 + label_idx) * w 
            plt.text(x_offset, h + 1, label)
        plt.show()






##############################################################################
#                                                                            #
#                           MAIN GEOCERT CLASS                               #
#                                                                            #
##############################################################################

class GeoCert(object):

    bound_fxn_selector = {'ia': PLNN.compute_interval_bounds,
                          'dual_lp': PLNN.compute_dual_lp_bounds,
                          'full_lp': PLNN.compute_full_lp_bounds}

    def __init__(self, net, hyperbox_bounds=None,
                 verbose=True, neuron_bounds='ia',
                 # And for 2d inputs, some kwargs for displaying things
                 display=False, save_dir=None, ax=None):

        """ To set up a geocert instance we need to know:
        ARGS:
            net : PLNN instance - the network we're verifying
            hyperbox_bounds: if not None, is a tuple of pair of numbers
                             (lo, hi) that define a valid hyperbox domain
            neuron_bounds: string - which technique we use to compute
                                    preactivation bounds. ia is interval
                                    analysis, full_lp is the full linear
                                    program, and dual_lp is the Kolter-Wong
                                    dual approach
            verbose: bool - if True, we print things
            THE REST ARE FOR DISPLAYING IN 2D CASES
        """
        ##############################################################
        #   First save the kwargs                                    #
        ##############################################################
        self.net = net
        self.hyperbox_bounds = hyperbox_bounds
        self.verbose = verbose
        assert neuron_bounds in ['ia', 'dual_lp', 'full_lp']
        self.neuron_bounds = neuron_bounds
        self.bound_fxn = self.bound_fxn_selector[neuron_bounds]

        # DISPLAY PARAMETERS
        self.display = display
        self.save_dir = save_dir
        self.ax = ax


        # And intialize the per-run state
        self._reset_state()


    def _reset_state(self):
        """ Clears out the state of things that get set in a min_dist run """
        # Things that are saved as instances for a run
        self.lp_norm = None # filled in later
        self.true_label = None # filled in later
        self.lp_dist = None # filled in later
        self.seen_to_polytope_map = {} # binary config str -> Polytope object
        self.pq = [] # Priority queue that contains HeapElements
        self.dead_constraints = None
        self.on_off_neurons = None
        self.domain = None # keeps track of domain and upper bounds
        self.config_history = None # keeps track of all seen polytope configs

        self.x = None
        self.x_np = None


    def _setup_state(self, x, lp_norm, potential):
        """ Sets up the state to be used on a per-run basis
        Shared between min_dist_multiproc and decision_problem_multiproc

        Sets instance variables and does asserts
        """
        assert lp_norm in ['l_2', 'l_inf']
        self.lp_norm = lp_norm
        self.x = x
        self.x_np = utils.as_numpy(x)
        self.true_label = int(self.net(x).max(1)[1].item())
        dist_selector = {'l_2'  : Face.l2_dist_gurobi,
                         'l_inf': Face.linf_dist_gurobi}
        self.lp_dist = dist_selector[self.lp_norm]
        self.domain = Domain(x.numel(), x)
        if self.hyperbox_bounds is not None:
            self.domain.set_original_hyperbox_bound(*self.hyperbox_bounds)
            self._update_dead_constraints()
        assert potential in ['lp', 'lipschitz']
        if self.net.layer_sizes[-1] > 2 and potential == 'lipschitz':
            raise NotImplementedError("Lipschitz potential buggy w/ >2 classes!")


    def _verbose_print(self, *args):
        """ Print method that leverages self.verbose -- makes code cleaner """
        if self.verbose:
            print(*args)


    def _compute_upper_bounds(self, x, true_label,
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
        ub_time = time.time() - start
        if upper_bound is None:
            self._verbose_print("Upper bound failed in %.02f seconds" % ub_time)

        else:
            self._verbose_print("Upper bound of %s in %.02f seconds" %
                                (upper_bound, ub_time))
            self._update_dead_constraints()

        return upper_bound, adv_ex, ub_time


    def _pgd_upper_bound(self, x, true_label, lp_norm, num_repeats=64,
                         extra_kwargs=None):
        """ Runs PGD attack off of many random initializations to help generate
            an upper bound.

            Sets self.upper_bound as the lp distance to the best (of the ones we
            found) adversarial example

            Also returns both the upper bound and the supplied adversarial
            example
        """

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
                         'random_init': 0.4,
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

        # Use the GPU to build adversarial attacks if we can
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            new_x = new_x.cuda()
            labels = labels.cuda()
            self.net.cuda()


        ######################################################################
        #   Run the attack and collect the best (if any) successful example  #
        ######################################################################

        pert_out = pgd_attack.attack(new_x, labels, **attack_kwargs)
        pert_out = pert_out.binsearch_closer(self.net, normalizer, labels)
        success_out = pert_out.collect_successful(self.net, normalizer,
                                          success_def='alter_top_logit')
        success_idxs = success_out['success_idxs']
        if USE_GPU:
            best_adv = best_adv.cpu()
            labels = labels.cpu()
            self.net.cpu()

        if success_idxs.numel() == 0:
            return None, None

        diffs = pert_out.delta.data.index_select(0, success_idxs)
        max_idx = me_utils.batchwise_norm(diffs, norm, dim=0).min(0)[1].item()
        best_adv = success_out['adversarials'][max_idx].squeeze()

        # Set both l_inf and l_2 upper bounds
        l_inf_upper_bound = (best_adv - x.view(-1)).abs().max().item()
        self.domain.set_l_inf_upper_bound(l_inf_upper_bound)
        l_2_upper_bound = torch.norm(best_adv - x.view(-1), p=2).item()
        self.domain.set_l_2_upper_bound(l_2_upper_bound)

        upper_bound = {'l_inf': l_inf_upper_bound,
                       'l_2': l_2_upper_bound}[self.lp_norm]

        return upper_bound, best_adv


    def _update_dead_constraints(self):
        # Compute new bounds
        new_bounds = self.bound_fxn(self.net, self.domain)

        # Change to dead constraint form
        self.dead_constraints = utils.ranges_to_dead_neurons(new_bounds)
        self.on_off_neurons = utils.ranges_to_on_off_neurons(new_bounds)


    def run(self, x, lp_norm='l_2', compute_upper_bound=False,
            potential='lp', problem_type='min_dist', decision_radius=None,
            collect_graph=False, max_runtime=None):
        """
        Main method for running GeoCert. This method handles each of the three
        problem types, as specified by the problem_type argument:
            - min_dist : computes the minimum distance point (under the specified
                         lp_norm), x', for which net(x) != net(x')
            - decision_problem : answers yes/no whether or not an adversarial
                                 example exists within a radius of decision_radius
                                 from the specified point x. Will return early if
                                 finds an adversarial example within the radius
                                 (which may not be the one with minimal distance!)
            - count_regions : like decision_problem, explores the region specified
                              by decision_radius, but will not stop early and instead
                              explore the entire region
        ARGS:
            x : numpy array or tensor - vector that we wish to certify
                                        robustness for
            lp_norm: string - needs to be 'l_2' or 'l_inf'
            compute_upper_bound : None, True, or dict - if None, no upper bound
                                  to pointwise robustness is computed. If not
                                  None, should be either True (to use default
                                  attack params) or a dict specifying extra
                                  kwargs to use in the PGD attack (see examples)
            potential : string - needs to be 'lp' or 'lipschitz', affects which
                                 potential function to be used in ordering facets
            problem_type : string - must be in ['min_dist', 'decision_problem',
                                                'count_regions']
            collect_graph: bool - if True, we collect the graph of linear regions
                                  and return it
            max_runtime : None or numeric - if not None, is a limit on the runtime
        RETURNS:
            GeoCertReturn object which has attributes regarding the output data


        """
        ######################################################################
        #   Step 0: Clear and setup state                                    #
        ######################################################################

        # 0.A) Establish clean state for a new run
        original_shape = x.shape 
        x = x.view(-1)

        self._reset_state() # clear out the state first
        self._setup_state(x, lp_norm, potential)
        start_time = time.time()


        # 0.B) Setup objects to gather bound updates with timing info
        # Upper bound times that the domain queue updater knows about
        upper_bound_times = [] # (time, bound)
        # Lower bound times that the workers know about
        lower_bound_times = [] # (time, bound)


        # 0.C) Compute upper bounds to further restrict search
        adv_bound, adv_ex, ub_time = None, None, None
        if problem_type == 'min_dist':
            # If finding min dist adv.ex, possibly run a PGD attack first
            if compute_upper_bound is not False:
                ub_out = self._compute_upper_bounds(x, self.true_label,
                                       extra_attack_kwargs=compute_upper_bound)
                adv_bound, adv_ex, ub_time = ub_out
                if adv_bound is not None:
                    upper_bound_times.append((time.time() - start_time, adv_bound))
                    self.domain.set_upper_bound(adv_bound, lp_norm)
        if problem_type in ['decision_problem', 'count_regions']:
            # If searching the entire provided domain, set up asymmetric domain
            assert decision_radius is not None
            self.domain.set_upper_bound(decision_radius, lp_norm)

        # 0.D) Set up priority queues
        sync_pq = []
        pq_decision_bounds = []

        # 0.E) Set up the objects to collect seen/missed polytopes and connections
        seen_polytopes = {}
        missed_polytopes = {}
        if collect_graph:
            polytope_graph = {}
        else:
            polytope_graph = None

        # 0.F) Set up heuristic dicts to hold info on domain, fixed neurons,
        #      and lipschitz constant
        heuristic_dict = {}
        heuristic_dict['domain'] = self.domain
        heuristic_dict['dead_constraints'] = self.dead_constraints

        if potential == 'lipschitz':
            # Just assume binary classifiers for now
            # on_off_neurons = self.net.compute_interval_bounds(self.domain, True)
            dual_lp = utils.dual_norm(lp_norm)
            c_vector, lip_value = self.net.fast_lip_all_vals(x, dual_lp,
                                                             self.on_off_neurons)
            self._verbose_print("LIPSCHITZ CONSTANTS", lip_value)
            self._verbose_print(c_vector[0].dot(self.net(x).squeeze()) / lip_value[0])
        else:
            lip_value = None
            c_vector = None
        heuristic_dict['fast_lip'] = lip_value
        heuristic_dict['c_vector'] = c_vector

        # 0.G) Set up return object to be further populated later
        # (mutable objects for all dynamic kwargs to GeoCertReturn make this ok)
        return_obj = GeoCertReturn(original=x,
                                   original_shape=original_shape,
                                   best_dist=None,
                                   best_ex=None,
                                   adv_bound=adv_bound,
                                   adv_ex=adv_ex,
                                   seen_polytopes=seen_polytopes,
                                   missed_polytopes=missed_polytopes,
                                   polytope_graph=polytope_graph,
                                   lower_bound_times=lower_bound_times,
                                   upper_bound_times=upper_bound_times,
                                   status=None,
                                   problem_type=problem_type,
                                   radius=decision_radius)
        ######################################################################
        #   Step 1: handle the initial polytope                              #
        ######################################################################

        # NOTE: The loop doesn't quite work here, so have to do the first part
        #       (aka emulate update_step_build_poly) manually.
        #       1) Build the original polytope
        #       2) Add polytope to seen polytopes
        #
        self._verbose_print('---Initial Polytope---')
        p_0_dict = self.net.compute_polytope(self.x)
        p_0 = Polytope.from_polytope_dict(p_0_dict, self.x_np,
                                          domain=self.domain,
                                          dead_constraints=self.dead_constraints,
                                          gurobi=True,
                                          lipschitz_ub=lip_value,
                                          c_vector=c_vector)
        seen_polytopes[utils.flatten_config(p_0.config)] = True

        update_step_handle_polytope(self.net, self.x_np, self.true_label,
                                    sync_pq, seen_polytopes, self.domain,
                                    self.dead_constraints, p_0, self.lp_norm,
                                    pq_decision_bounds, potential, missed_polytopes,
                                    problem_type, polytope_graph,
                                    heuristic_dict, upper_bound_times,
                                    start_time, max_runtime,
                                    verbose=self.verbose)

        if problem_type == 'decision_problem':
            # If a decision problem and found a decision bound in the first polytope
            # (which must also be in the 'restricted domain'), then we can return
            try:
                best_decision_bound = heapq.heappop(pq_decision_bounds)
                # Will error here^ unless found a decision bound
                return_obj.status = 'SUCCESS'
                return return_obj # note, not guaranteed to be optimal!
            except IndexError:
                pass


        ######################################################################
        #   Step 2: Loop until termination                                   #
        ######################################################################
        proc_args = (self.net, self.x_np, self.true_label, sync_pq,
                     seen_polytopes, heuristic_dict, self.lp_norm,
                     pq_decision_bounds,
                     potential, missed_polytopes, problem_type,
                     polytope_graph, lower_bound_times, start_time,
                     upper_bound_times, max_runtime)

        update_step_worker(*proc_args, **{'proc_id': 0,
                                          'verbose': self.verbose})


        ######################################################################
        #   Step 3: Collect the best thing in the decision queue and return  #
        ######################################################################

        overran_time = ((max_runtime is not None) and\
                        (time.time() - start_time > max_runtime))
        if overran_time:
            return_obj.status = 'TIMEOUT'
            return return_obj

        if problem_type == 'min_dist':
            best_decision_bound = heapq.heappop(pq_decision_bounds)
        elif problem_type in ['decision_problem', 'count_regions']:
            try:
                best_decision_bound = heapq.heappop(pq_decision_bounds)
            except IndexError:
                if problem_type == 'decision_problem':
                    self._verbose_print("DECISION PROBLEM FAILED")
                    return_obj.status = 'FAILURE'
                else:
                    self._verbose_print("COUNTED %s LINEAR REGIONS" % len(seen_polytopes))
                    return_obj.status = 'SUCCESS'
                    return_obj.num_regions = len(seen_polytopes)
                return return_obj

        return_obj.best_dist = best_decision_bound.priority
        return_obj.best_ex = best_decision_bound.projection
        return_obj.status = 'SUCCESS'
        return return_obj




##############################################################################
#                                                                            #
#                           FUNCTIONAL VERSION OF UPDATES                    #
#                            (useful for multiprocessing)                    #
##############################################################################



def update_step_worker(piecewise_net, x, true_label, pqueue, seen_polytopes,
                       heuristic_dict, lp_norm,
                       pq_decision_bounds, potential,
                       missed_polytopes, problem_type,
                       polytope_graph, lower_bound_times, start_time,
                       upper_bound_times, max_runtime,
                       proc_id=None, verbose=True):
    """ Setup for the worker objects
    ARGS:
        network - actual network object to be copied over into memory
        everything else is a manager
    """
    assert problem_type in ['min_dist', 'decision_problem', 'count_regions']
    # with everything set up, LFGD
    while True:
        output = update_step_loop(piecewise_net, x, true_label, pqueue,
                                  seen_polytopes, heuristic_dict,
                                  lp_norm,
                                  pq_decision_bounds,
                                  potential, missed_polytopes,
                                  problem_type, polytope_graph,
                                  lower_bound_times, start_time, proc_id,
                                  upper_bound_times, max_runtime,
                                  verbose=verbose)
        if output is not True: # Termination condition
            return output

        if (max_runtime is not None) and (time.time() - start_time) > max_runtime:
            return output



def update_step_loop(piecewise_net, x, true_label, pqueue, seen_polytopes,
                     heuristic_dict, lp_norm,
                     pq_decision_bounds, potential, missed_polytopes,
                     problem_type, polytope_graph,
                     lower_bound_times, start_time, proc_id, upper_bound_times,
                     max_runtime, verbose=True):
    """ Inner loop for how to update the priority queue. This handles one
        particular thing being popped off the PQ
    """
    # Build the polytope to pop from the queue

    poly_out = update_step_build_poly(piecewise_net, x, pqueue, seen_polytopes,
                                      heuristic_dict, lp_norm,
                                      pq_decision_bounds, potential,
                                      problem_type,
                                      lower_bound_times, start_time, proc_id,
                                      verbose=verbose)

    if isinstance(poly_out, bool): # bubble up booleans
        return poly_out

    new_poly, domain, dead_constraints = poly_out

    # Build facets, reject what we can, and do optimization on the rest
    return update_step_handle_polytope(piecewise_net, x, true_label, pqueue,
                                       seen_polytopes, domain, dead_constraints,
                                       new_poly, lp_norm,
                                       pq_decision_bounds, potential,
                                       missed_polytopes,
                                       problem_type, polytope_graph,
                                       heuristic_dict, upper_bound_times,
                                       start_time, max_runtime,
                                       verbose=verbose)



def update_step_build_poly(piecewise_net, x, pqueue, seen_polytopes,
                           heuristic_dict, lp_norm,
                           pq_decision_bounds, potential,
                           problem_type, lower_bound_times, start_time,
                           proc_id, verbose=True):
    """ Component method of the loop.
        1) Pops the top PQ element off and rejects it as seen before if so
        2) Collect the domain/heuristics
        3) builds the new polytope and returns the polytope
    """
    ##########################################################################
    #   Step 1: pop something off the queue                                  #
    ##########################################################################
    try:
        item = heapq.heappop(pqueue)
    except IndexError:
        return False

    #priority, config, tight_constraint, proj, facet_type = item
    if item.priority < 0: #item.priority < 0: # Termination condition -- bubble up the termination
        return False
    if item.facet_type == 'decision': # Termination condition -- bubble up
        heapq.heappush(pq_decision_bounds, item)
        #pq_decision_bounds.put(item)

        return False

    # Update the lower bound queue
    lower_bound_times.append(((time.time() - start_time), item.priority))


    new_configs = utils.get_new_configs(item.config, item.tight_constraint)
    if utils.flatten_config(new_configs) in seen_polytopes:
        return True # No need to go further, but don't terminate!
    else:
        seen_polytopes[utils.flatten_config(new_configs)] = True



    ##########################################################################
    #   Step 2: Gather the domain and dead neurons                           #
    ##########################################################################

    domain = heuristic_dict['domain']
    current_upper_bound = domain.current_upper_bound(lp_norm) or 1e10

    verbose_print("(p%s) Popped: %.06f  | %.06f" %
                  (proc_id, item.priority, current_upper_bound),
                  verbose=verbose)
    assert isinstance(domain, Domain)
    dead_constraints = heuristic_dict['dead_constraints']
    lipschitz_ub = heuristic_dict['fast_lip']
    c_vector = heuristic_dict['c_vector']

    ##########################################################################
    #   Step 3: Build polytope and return                                    #
    ##########################################################################

    new_poly_dict = piecewise_net.compute_polytope_config(new_configs, False)
    new_poly = Polytope.from_polytope_dict(new_poly_dict, x,
                                           domain=domain,
                                           dead_constraints=dead_constraints,
                                           lipschitz_ub=lipschitz_ub,
                                           c_vector=c_vector)

    return new_poly, domain, dead_constraints




def update_step_handle_polytope(piecewise_net, x, true_label, pqueue,
                                seen_polytopes, domain, dead_constraints,
                                new_poly, lp_norm,
                                pq_decision_bounds, potential,
                                missed_polytopes,
                                problem_type, polytope_graph, heuristic_dict,
                                upper_bound_times, start_time, max_runtime,
                                verbose=True):
    """ Component method of the loop
        1) Makes facets, rejecting quickly where we can
        2) Run convex optimization on everything we can't reject
        3) Push the updates to the process-safe objects
    """

    ##########################################################################
    #   Step 1: Make new facets while doing fast rejects                     #
    ##########################################################################
    new_facets, rejects = new_poly.generate_facets_configs(seen_polytopes,
                                                           missed_polytopes)

    if problem_type != 'count_regions':
        adv_constraints = piecewise_net.make_adversarial_constraints(new_poly,
                                                            true_label, domain)
    else:
        adv_constraints = []


    ##########################################################################
    #   Step 2: Compute the min-dists/feasibility checks using LP/QP         #
    ##########################################################################

    # -- compute the distances
    chained_facets = itertools.chain(new_facets, adv_constraints)
    parallel_args = [(_, x) for _ in chained_facets]
    dist_selector = {'l_2': Face.l2_dist_gurobi,
                     'l_inf': Face.linf_dist_gurobi}
    lp_dist = dist_selector[lp_norm]
    dist_fxn = lambda el: (el[0], lp_dist(*el))

    outputs = [dist_fxn(_) for _ in parallel_args]
    updated_domain = False

    # -- collect the necessary facets to add to the queue
    current_upper_bound = domain.current_upper_bound(lp_norm)
    pq_elements_to_push = []
    fail_count = 0
    for facet, (dist, proj) in outputs:
        try:
            new_facet_conf = utils.flatten_config(facet.get_new_configs())
        except:
            new_facet_conf = None
        if dist is None:
            rejects['optimization infeasible'] += 1
            if facet.facet_type == 'decision':
                continue
            # Handle infeasible case

            missed_polytopes[new_facet_conf] = True
            fail_count += 1
            continue
        if polytope_graph is not None:
            edge = (utils.flatten_config(new_poly.config),
                    new_facet_conf)
            polytope_graph[edge] = dist

        if current_upper_bound is not None and dist > current_upper_bound:
            #Handle the too-far-away facets
            continue

        rejects['optimization successful'] += 1
        new_pq_element = PQElement()
        for k, v in {'priority': dist,
                     'config': new_poly.config,
                     'tight_constraint': facet.tight_list[0],
                     'projection': proj,
                     'facet_type': facet.facet_type}.items():
            setattr(new_pq_element, k, v)

        pq_elements_to_push.append(new_pq_element)

        if facet.facet_type == 'decision':
            if problem_type == 'decision_problem':
                # If in decision_problem style, just return
                heapq.heappush(pq_decision_bounds, new_pq_element)
                return True


            updated_domain = True
            # If also a decision bound, update the upper_bound
            domain.set_upper_bound(dist, lp_norm)
            # update l_inf bound in l_2 case as well
            if lp_norm == 'l_2':
                new_linf = abs(proj - x).max()
                domain.set_upper_bound(new_linf, 'l_inf')
            current_upper_bound = domain.current_upper_bound(lp_norm)

    ##########################################################################
    #   Step 3: Process all the updates and return                           #
    ##########################################################################

    # -- push objects to priority queue
    for pq_element in pq_elements_to_push:
        heapq.heappush(pqueue, pq_element)


    # -- call the update domain to try and compute tighter stable neurons
    if updated_domain:
        update_domain(domain, piecewise_net, x, heuristic_dict, potential,
                      lp_norm, None, upper_bound_times, start_time,
                      max_runtime, verbose=verbose)
    return True





def update_domain(new_domain, piecewise_net, x, heuristic_dict, potential,
                  lp_norm,  bound_fxn, upper_bound_times, start_time,
                  max_runtime, verbose=True):
    linf_radius = new_domain.linf_radius or 1e10
    l2_radius = new_domain.l2_radius or 1e10
    verbose_print('-' * 20, "DOMAIN UPDATE | L_inf %.06f | L_2 %.06f" %
                  (linf_radius, l2_radius), verbose=verbose)

    # Record the update in the upper_bound_times log
    attr = {'l_inf': 'linf_radius',
            'l_2':   'l2_radius'}[lp_norm]

    upper_bound_times.append((time.time() - start_time,
                              getattr(new_domain, attr)))


    # Update the current domain and change the heuristic dict
    heuristic_dict['domain'] = new_domain


    # And use the domain to compute the new dead constraints
    new_bounds = piecewise_net.compute_interval_bounds(new_domain)
    # new_bounds = bound_fxn(piecewise_net, domain)
    dead_constraints = utils.ranges_to_dead_neurons(new_bounds)
    on_off_neurons = utils.ranges_to_on_off_neurons(new_bounds)
    heuristic_dict['dead_constraints'] = dead_constraints

    # Use the domain to update the lipschitz bound on everything
    # (this can only shrink as we shrink the domain)
    if potential == 'lipschitz':
        # Just assume binary classifiers for now
        dual_lp = utils.dual_norm(lp_norm)
        c_vector, lip_value = piecewise_net.fast_lip_all_vals(x, dual_lp,
                                                              on_off_neurons)
        heuristic_dict['fast_lip'] = lip_value
