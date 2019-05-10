"""  OOP refactor of geocert so I get a better feel for how the ICML
    implementation went -mj (3/1/19)
"""
import sys
import itertools
sys.path.append('mister_ed')
import adversarial_perturbations as ap
import prebuilt_loss_functions as plf

import prebuilt_loss_functions as plf
import loss_functions as lf
import adversarial_attacks as aa
import utils.pytorch_utils as me_utils

from _polytope_ import Polytope, Face
import utilities as utils
from domains import Domain
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

    def convert_to_PQElement(self, *args):

        pq_el = PQElement()
        convert_dict = {'priority': self.lp_dist,
                        'config': self.facet.config,
                        'tight_constraint': self.facet.tight_list[0],
                        'facet_type': self.facet.facet_type,
                        'projection': self.projection}
        for k, v in convert_dict.items():
            setattr(pq_el, k, v)
        return pq_el


class PQElement:
    priority: float # IS THE LP DIST
    config: Any=field(compare=False) # Configs for neuron region
    tight_constraint: Any=field(compare=False) # which constraint is tight
    facet_type: Any=field(compare=False) # is decision or nah?
    projection: Any=field(compare=False)

    def __lt__(self, other):
        return self.priority < other.priority




class PQManager(SyncManager):
    pass

PQManager.register("PriorityQueue", PriorityQueue)



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
class IncrementalGeoCertMultiProc(object):
    def __init__(self, net, config_fxn='serial',
                 config_fxn_kwargs=None, hyperbox_bounds=None,
                 verbose=True, num_processes=4,
                 # And for 2d inputs, some kwargs for displaying things
                 display=False, save_dir=None, ax=None):

        """ To set up a geocert instance we need to know:
        ARGS:
            net : PLNN instance - the network we're verifying
            config_fxn: how we're generating facets (parallel or serial)
            hyperbox_bounds: if not None, is a tuple of pair of numbers
                             (lo, hi) that define a valid hyperbox domain
            verbose: bool - if True, we print things
            THE REST ARE FOR DISPLAYING IN 2D CASES
        """
        ##############################################################
        #   First save the kwargs                                    #
        ##############################################################
        self.net = net
        config_map = {'serial': Polytope.generate_facets_configs,
                      'parallel': Polytope.generate_facets_configs_parallel}
        self.facet_config_fxn = config_map[config_fxn]
        if config_fxn_kwargs is None:
            if config_fxn == 'parallel':
                config_fxn_kwargs = None
            else:
                config_fxn_kwargs = {'use_clarkson': True,
                                     'use_ellipse': False}
        self.config_fxn_kwargs = config_fxn_kwargs or {}
        self.hyperbox_bounds = hyperbox_bounds
        self.verbose = verbose

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
        self.dead_constraints = [None] # wrapped in a dynamic object
        self.domain = None # keeps track of domain and upper bounds
        self.config_history = None # keeps track of all seen polytope configs

    def _verbose_print(self, *args):
        """ Print method that leverages self.verbose -- makes code cleaner """
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
            self._update_dead_constraints()

        return upper_bound, adv_ex


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
        bounds = self.net.compute_dual_ia_bounds(self.domain)
        tensor_dead_constraints = bounds[1]
        self.dead_constraints = utils.cat_config(tensor_dead_constraints)



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

        #######################################################################
        #   Generate new facets in this polytope and do fast rejection checks #
        #######################################################################
        # Compute polytope boundaries, rejecting where we can
        new_facets, rejects = self.facet_config_fxn(poly,
                                              self.seen_to_polytope_map)
        initial_len = len(new_facets)
        #self._verbose_print("Num facets: ", len(new_facets))
        self._verbose_print("REJECT DICT: ", rejects)

        # Compute adversarial facets, rejecting where we can
        adv_constraints = self.net.make_adversarial_constraints(poly.config,
                              self.true_label, self.domain)

        ######################################################################
        #   Compute min-dists/feasibilities using LP/QP methods              #
        ######################################################################
        chained_facets = itertools.chain(new_facets, adv_constraints)
        parallel_args = [(_, self.x) for _ in chained_facets]
        dist_fxn = lambda el: (el[0], self.lp_dist(*el))


        if (len(parallel_args) > 1 and
            self.config_fxn_kwargs.get('num_jobs', 1) > 1):
            # Do optimizations in parallel
            map_fxn = joblib.delayed(dist_fxn)
            n_jobs = self.config_fxn_kwargs['num_jobs']
            outputs = joblib.Parallel(n_jobs=n_jobs)(map_fxn(_)
                                                     for _ in parallel_args)
        else:
            # Do optimizations in serial
            outputs = [dist_fxn(_) for _ in parallel_args]


        ######################################################################
        #   With distances in hand, make HeapElements                        #
        ######################################################################
        current_upper_bound = self.domain.current_upper_bound(self.lp_norm)
        num_pushed = 0
        for facet, (dist, proj) in outputs:
            if dist is None:
                continue
            else:
                heap_el = HeapElement(dist, facet,
                                      facet_type=facet.facet_type,
                                      exact_or_estimate='exact')
                heap_el.projection = proj

            if (current_upper_bound is None or
                dist < current_upper_bound):
                # If feasible and worth considering push onto heap
                heapq.heappush(self.pq, heap_el)
                num_pushed += 1

                if facet.facet_type == 'decision':
                    # If also a decision bound, update the upper bound/domain
                    self.domain.set_upper_bound(dist, self.lp_norm)
                    # Update l_inf bound in l_2 case anyway
                    if self.lp_norm == 'l_2':
                        xnp = np.array(self.x.squeeze())
                        new_linf = abs(proj - xnp).max()
                        self.domain.set_upper_bound(new_linf, 'l_inf')

                    current_upper_bound =\
                                   self.domain.current_upper_bound(self.lp_norm)



                    self._update_dead_constraints()
        print("Pushed %s/%s facets" % (num_pushed, initial_len))


        # Mark that we've handled this polytope
        poly_config = utils.flatten_config(poly.config)
        self.seen_to_polytope_map[poly_config] = poly



    def min_dist(self, x, lp_norm='l_2', compute_upper_bound=False):
        """ Returns the minimum distance between x and the decision boundary.
            Plots things too, I guess...

        ARGS:
            x - tensor or numpy array of the point we're computing robustness
                for
            lp_norm - under which norm we're computing pointwise robustness
            compute_upper_bound - if False, we don't compute an upper bound
                                  if True, we compute an upper bound using
                                  default kwargs. Otherwise, is a dict with
                                  the kwargs (used for the attack kwargs)
        """

        ######################################################################
        #   Step 0: Clear and setup state                                    #
        ######################################################################
        self._reset_state() # clear out the state first

        # Computed state
        assert lp_norm in ['l_2', 'l_inf']
        self.lp_norm = lp_norm
        self.x = x
        self.true_label = int(self.net(x).max(1)[1].item()) # classifier(x)
        self.lp_dist = {'l_2': Face.l2_dist,
                        'l_inf': Face.linf_dist}[self.lp_norm]
        self.domain = Domain(x.numel(), x)
        if self.hyperbox_bounds is not None:
            self.domain.set_original_hyperbox_bound(*self.hyperbox_bounds)
            self._update_dead_constraints()

        upper_bound_attr = {'l_2': 'l2_radius',
                            'l_inf': 'linf_radius'}[lp_norm]



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
                                         domain=self.domain,
                                         dead_constraints=self.dead_constraints)
        self._update_step(p_0, None)

        ######################################################################
        #   Step 2: Repeat until we hit a decision boundary                  #
        ######################################################################

        index = 0
        prev_min_dist = min(self.pq, key=lambda el: el.lp_dist).lp_dist
        while True:
            pop_el = heapq.heappop(self.pq)
            # If popped el is part of decision boundary, we're done!
            if pop_el.decision_bound():
                break
            # Otherwise, open up a new polytope and explore
            else:
                prev_min_dist = pop_el.lp_dist

                popped_facet = pop_el.facet
                configs = popped_facet.get_new_configs()
                configs_flat = utils.flatten_config(configs)


                # If polytope has already been seen, don't add it again
                if configs_flat not in self.seen_to_polytope_map:
                    self._verbose_print('---Opening New Polytope---')
                    self._verbose_print('Bounds ', pop_el.lp_dist, "  |  ",
                                     getattr(self.domain, upper_bound_attr))
                    new_poly_dict = self.net.compute_polytope_config(configs,
                                                                     False)
                    new_poly = Polytope.from_polytope_dict(new_poly_dict,
                                                           domain=self.domain,
                                         dead_constraints=self.dead_constraints)
                    self._update_step(new_poly, pop_el)
                else:
                    pass
                    #self._verbose_print("We've already seen that polytope")

            if index % 1 == 0 and self.display:
                self.plot_2d(pop_el.lp_dist, iter=index)
            index += 1

        self._verbose_print('----------Minimal Projection Generated----------')
        self._verbose_print("DIST: ", pop_el.lp_dist)
        if self.display:
            self.plot_2d(pop_el.lp_dist, iter=index)
        best_example = pop_el.projection
        return pop_el.lp_dist, adv_bound, adv_ex, best_example, pop_el


    def min_dist_multiproc(self, x, lp_norm='l_2', compute_upper_bound=False,
                           num_proc=2):
        ######################################################################
        #   Step 0: Clear and setup state                                    #
        ######################################################################
        self._reset_state() # clear out the state first

        # Computed state
        assert lp_norm in ['l_2', 'l_inf']
        self.lp_norm = lp_norm
        self.x = x

        print()
        self.true_label = int(self.net(torch.Tensor(x)).max(1)[1].item())
        self.lp_dist = {'l_2': Face.l2_dist,
                        'l_inf': Face.linf_dist}[self.lp_norm]
        self.domain = Domain(x.numel(), x)
        if self.hyperbox_bounds is not None:
            self.domain.set_original_hyperbox_bound(*self.hyperbox_bounds)
            self._update_dead_constraints()

        upper_bound_attr = {'l_2': 'l2_radius',
                            'l_inf': 'linf_radius'}[lp_norm]



        if compute_upper_bound is not False:
            adv_bound, adv_ex = self._compute_upper_bounds(x, self.true_label,
                                                           lp_norm,
                                       extra_attack_kwargs=compute_upper_bound)

        ######################################################################
        #   Step 1: Set up things needed for multiprocessing                 #
        ######################################################################

        # Set up the priorityqueue
        pq_manager = PQManager()
        pq_manager.start()
        sync_pq = pq_manager.PriorityQueue()

        pq_decision_bounds = pq_manager.PriorityQueue()

        # Set up the seen_polytopes
        manager = mp.Manager()
        seen_polytopes = manager.dict()

        # Set up heuristic dicts
        heuristic_dict = manager.dict()
        heuristic_dict['domain'] = self.domain
        heuristic_dict['dead_constraints'] = self.dead_constraints


        # Set up domain updater queue
        domain_update_queue = mp.Queue()


        ######################################################################
        #   Step 2: handle the initial polytope                              #
        ######################################################################
        self._verbose_print('---Initial Polytope---')
        p_0_dict = self.net.compute_polytope(self.x, comparison_form_flag=False)

        p_0 = Polytope.from_polytope_dict(p_0_dict,
                                         domain=self.domain,
                                         dead_constraints=self.dead_constraints)
        self._update_step(p_0, None)

        # and update the shared stuff
        # (i)  take stuff from the pq and put onto the process safe pq
        for el in self.pq:
            pq_el = el.convert_to_PQElement()
            sync_pq.put(pq_el)

        # (ii) update the heuristic dict
        heuristic_dict['domain'] = self.domain
        heuristic_dict['dead_constraints'] = self.dead_constraints

        # (iii) update the seen polytopes
        for k in self.seen_to_polytope_map.keys():
            seen_polytopes[k] = True

        ######################################################################
        #   Step 3: Setup threads and start looping                          #
        ######################################################################


        '''
        update_processes = [mp.Process(target=update_step_worker,
                                       args=(self.net, self.x, self.true_label,
                                             pq, seen_polytopes, heuristic_dict,
                                             self.lp_norm, domain_update_queue))
                            for _ in range(num_processes)]

        domain_queue_process = mp.Process(target=domain_queue_updater,
                                          args=(self.net, heuristic_dict,
                                                domain_update_queue))
        for proc in itertools.chain(update_processes, [domain_queue_process]):
            proc.start()

        for proc in itertools.chain(update_processes, [domain_queue_process]):
            proc.join()
        '''


        update_procs = [mp.Process(target=update_step_worker,
                                 args=(self.net, self.x, self.true_label,
                                 sync_pq, seen_polytopes, heuristic_dict,
                                 self.lp_norm, domain_update_queue,
                                 pq_decision_bounds)) for i in range(num_proc)]


        dq_proc = mp.Process(target=domain_queue_updater,
                          args=(self.net, heuristic_dict, domain_update_queue))
        [_.start() for _ in update_procs]
        dq_proc.start()
        # update_step_worker(self.net, self.x, self.true_label, sync_pq,
        #                    seen_polytopes, heuristic_dict, self.lp_norm,
        #                    domain_update_queue, pq_decision_bounds)


        [_.join() for _ in update_procs]
        print("DONE WITH LOOP. WAITING FOR DOMAIN QUEUE UPDATER TO FINISH")
        dq_proc.join()
        print("DONE WITH DOMAIN QUEUEU UPDATER")

        ######################################################################
        #   Step 4: Collect the best thing in the decision queue and return  #
        ######################################################################

        best_decision_bound = pq_decision_bounds.get()
        best_dist = best_decision_bound.priority
        best_example = best_decision_bound.projection
        return best_dist, adv_bound, adv_ex, best_example


def update_step_worker(piecewise_net, x, true_label, pqueue, seen_polytopes,
                       heuristic_dict, lp_norm, domain_update_queue,
                       pq_decision_bounds):
    """ Setup for the worker objects
    ARGS:
        network - actual network object to be copied over into memory
        everything else is a manager
    """

    # with everything set up, LFGD
    while True:
        output = update_step_loop(piecewise_net, x, true_label, pqueue,
                                  seen_polytopes, heuristic_dict,
                                  lp_norm, domain_update_queue,
                                  pq_decision_bounds)
        if output is not True: # Termination condition
            return output



def update_step_loop(piecewise_net, x, true_label, pqueue, seen_polytopes,
                     heuristic_dict, lp_norm, domain_update_queue,
                     pq_decision_bounds):
    """ Inner loop for how to update the priority queue. This handles one
        particular thing being popped off the PQ
    """

    ##########################################################################
    #   Step 1: pop something off the queue                                  #
    ##########################################################################
    item = pqueue.get()
    print("ITEM", item.priority)
    if item.priority < 0: # Termination condition -- bubble up the termination
        return False
    if item.facet_type == 'decision': # Termination condition -- bubble up
        for i in range(100): # Just flood the pq with -1's later
            stop_pq_el = PQElement()
            stop_pq_el.priority = -1
            pqueue.put(stop_pq_el)
            domain_update_queue.put(None)
            pq_decision_bounds.put(item)
        return False


    config = item.config
    tight_constraint = item.tight_constraint

    new_configs = utils.get_new_configs(config, tight_constraint)
    print("HASH CONFIG", hash(utils.flatten_config(new_configs)))
    if utils.flatten_config(new_configs) in seen_polytopes:
        return True # No need to go further, but don't terminate!
    else:
        seen_polytopes[utils.flatten_config(new_configs)] = True

    ##########################################################################
    #   Step 2: Gather the domain and dead neurons                           #
    ##########################################################################

    domain = heuristic_dict['domain']
    assert isinstance(domain, Domain)
    dead_constraints = heuristic_dict['dead_constraints']

    ##########################################################################
    #   Step 3: Generate facets and do fast rejects                          #
    ##########################################################################


    new_poly_dict = piecewise_net.compute_polytope_config(new_configs, False)
    new_poly = Polytope.from_polytope_dict(new_poly_dict,
                                           domain=domain,
                                           dead_constraints=dead_constraints)
    new_facets, rejects = new_poly.generate_facets_configs_parallel(seen_polytopes)
    initial_len = len(new_facets)

    adv_constraints = piecewise_net.make_adversarial_constraints(new_configs,
                                                            true_label, domain)

    ##########################################################################
    #   Step 4: Compute the min-dists/feasibility checks using LP/QP         #
    ##########################################################################

    # -- compute the distances
    chained_facets = itertools.chain(new_facets, adv_constraints)
    parallel_args = [(_, x) for _ in chained_facets]
    lp_dist = {'l_2': Face.l2_dist,
               'l_inf': Face.linf_dist}[lp_norm]
    dist_fxn = lambda el: (el[0], lp_dist(*el))

    outputs = [dist_fxn(_) for _ in parallel_args]
    updated_domain = False

    # -- collect the necessary facets to add to the queue
    current_upper_bound = domain.current_upper_bound(lp_norm)
    pq_elements_to_push = []
    for facet, (dist, proj) in outputs:
        if dist is None or dist > current_upper_bound:  #
            continue
        new_pq_element = PQElement()
        for k, v in {'priority': dist,
                     'config': new_configs,
                     'tight_constraint': facet.tight_list[0],
                     'projection': proj,
                     'facet_type': facet.facet_type}.items():
            setattr(new_pq_element, k, v)
        pq_elements_to_push.append(new_pq_element)

        if facet.facet_type == 'decision':
            updated_domain = True
            # If also a decision bound, update the upper_bound
            domain.set_upper_bound(dist, lp_norm)
            # update l_inf bound in l_2 case as well
            if lp_norm == 'l_2':
                xnp = np.array(x.squeeze())
                new_linf = abs(proj - xnp).max()
                domain.set_upper_bound(new_linf, 'l_inf')
            current_upper_bound = domain.current_upper_bound(lp_norm)


    ##########################################################################
    #   Step 5: Process all the updates and return                           #
    ##########################################################################

    # -- push objects to priority queue
    for pq_element in pq_elements_to_push:
        pqueue.put(pq_element)

    # -- push domain to be handled by a separate thread
    if updated_domain:
        domain_update_queue.put(domain)



    return True



def domain_queue_updater(piecewise_net, heuristic_dict, domain_update_queue):
    """ Separate thread to handle updating the domain/dead constraints """

    domain = heuristic_dict['domain']
    while True:
        new_domain = domain_update_queue.get()
        if new_domain == None:
            break

        # Update the current domain and change the heuristic dict

        domain.set_hyperbox_bound(new_domain.box_low,
                                  new_domain.box_high)

        domain.set_l_inf_upper_bound(new_domain.linf_radius)
        domain.set_l_2_upper_bound(new_domain.l2_radius)
        heuristic_dict['domain'] = domain

        # And use the domain to compute the new dead constraints
        dead_constraints = piecewise_net.compute_dual_ia_bounds(domain)[1]
        heuristic_dict['dead_constraints'] = utils.cat_config(dead_constraints)


