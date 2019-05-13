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
from queue import PriorityQueue



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
    priority: float # IS THE LP DIST OR 'POTENTIAL' VALUE
    config: Any=field(compare=False) # Configs for neuron region
    tight_constraint: Any=field(compare=False) # which constraint is tight
    facet_type: Any=field(compare=False) # is decision or nah?
    projection: Any=field(compare=False)

    def __lt__(self, other):
        return self.priority < other.priority


class PQManager(SyncManager):
    pass

PQManager.register("PriorityQueue", PriorityQueue)



class IncrementalGeoCertMultiProc(object):
    def __init__(self, net, hyperbox_bounds=None,
                 verbose=True,
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

        self.x = None
        self.x_np = None


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
        new_facets, rejects = poly.generate_facets_configs_parallel(
                                                      self.seen_to_polytope_map)

        initial_len = len(new_facets)
        #self._verbose_print("Num facets: ", len(new_facets))
        self._verbose_print("REJECT DICT: ", rejects)

        # Compute adversarial facets, rejecting where we can
        adv_constraints = self.net.make_adversarial_constraints(poly,
                              self.true_label, self.domain)

        ######################################################################
        #   Compute min-dists/feasibilities using LP/QP methods              #
        ######################################################################
        chained_facets = itertools.chain(new_facets, adv_constraints)
        parallel_args = [(_, self.x) for _ in chained_facets]
        dist_fxn = lambda el: (el[0], self.lp_dist(*el))

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



    def min_dist_multiproc(self, x, lp_norm='l_2', compute_upper_bound=False,
                           num_proc=2, optimizer='gurobi'):
        ######################################################################
        #   Step 0: Clear and setup state                                    #
        ######################################################################
        self._reset_state() # clear out the state first

        # Computed state
        assert lp_norm in ['l_2', 'l_inf']
        self.lp_norm = lp_norm
        self.x = x
        self.x_np = x.view(-1).detach().cpu().numpy()

        self.true_label = int(self.net(torch.Tensor(x)).max(1)[1].item())
        dist_selector = {('l_2', 'mosek')   : Face.l2_dist,
                         ('l_2', 'gurobi')  : Face.l2_dist_gurobi,
                         ('l_inf', 'mosek') : Face.linf_dist,
                         ('l_inf', 'gurobi'): Face.linf_dist_gurobi}
        self.lp_dist = dist_selector[(self.lp_norm, optimizer)]
        self.domain = Domain(x.numel(), x)
        if self.hyperbox_bounds is not None:
            self.domain.set_original_hyperbox_bound(*self.hyperbox_bounds)
            self._update_dead_constraints()

        upper_bound_attr = {'l_2': 'l2_radius',
                            'l_inf': 'linf_radius'}[lp_norm]

        assert optimizer in ['mosek', 'gurobi']
        if optimizer == 'mosek' and num_proc > 1:
            raise Exception("Multiprocessing doesn't work with mosek!")


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

        p_0 = Polytope.from_polytope_dict(p_0_dict, self.x_np,
                                         domain=self.domain,
                                         dead_constraints=self.dead_constraints,
                                         gurobi=True)
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


        # Do this on one processor if num_jobs is one

        if num_proc == 1:
            dq_thread = Thread(target=domain_queue_updater,
                               args=(self.net, heuristic_dict,
                                     domain_update_queue))
            proc_args = (self.net, self.x_np, self.true_label, sync_pq,
                         seen_polytopes, heuristic_dict, self.lp_norm,
                         domain_update_queue, pq_decision_bounds, optimizer)
            # TEST SINGLE PROC

            # proc = mp.Process(target=update_step_worker, args=proc_args)

            dq_thread.start()
            update_step_worker(*proc_args)
            dq_thread.join()
        else:
            procs = [mp.Process(target=update_step_worker,
                                args=(self.net, self.x_np, self.true_label,
                                sync_pq, seen_polytopes, heuristic_dict,
                                self.lp_norm, domain_update_queue,
                                pq_decision_bounds, optimizer),
                                kwargs={'proc_id': i})
                     for i in range(num_proc)]


            dq_proc = mp.Process(target=domain_queue_updater,
                                 args=(self.net, heuristic_dict,
                                       domain_update_queue))
            dq_proc.start()
            [_.start() for _ in procs]
            [_.join() for _ in procs]
            dq_proc.join()


        ######################################################################
        #   Step 4: Collect the best thing in the decision queue and return  #
        ######################################################################

        best_decision_bound = pq_decision_bounds.get()
        best_dist = best_decision_bound.priority
        best_example = best_decision_bound.projection
        return best_dist, adv_bound, adv_ex, best_example




##############################################################################
#                                                                            #
#                           WORKER FUNCTIONS                                 #
#                                                                            #
##############################################################################



def update_step_worker(piecewise_net, x, true_label, pqueue, seen_polytopes,
                       heuristic_dict, lp_norm, domain_update_queue,
                       pq_decision_bounds, optimizer, proc_id=None):
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
                                  pq_decision_bounds, optimizer,
                                  proc_id=proc_id)
        if output is not True: # Termination condition
            return output



def update_step_loop(piecewise_net, x, true_label, pqueue, seen_polytopes,
                     heuristic_dict, lp_norm, domain_update_queue,
                     pq_decision_bounds, optimizer, proc_id=None):
    """ Inner loop for how to update the priority queue. This handles one
        particular thing being popped off the PQ
    """

    ##########################################################################
    #   Step 1: pop something off the queue                                  #
    ##########################################################################
    item = pqueue.get()

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
    if utils.flatten_config(new_configs) in seen_polytopes:
        return True # No need to go further, but don't terminate!
    else:
        seen_polytopes[utils.flatten_config(new_configs)] = True

    if proc_id is None:
        print("ITEM", item.priority)
    else:
        print("(p%s) ITEM" % proc_id, item.priority)
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
    new_poly = Polytope.from_polytope_dict(new_poly_dict, x,
                                           domain=domain,
                                           dead_constraints=dead_constraints)
    new_facets, rejects = new_poly.generate_facets_configs_parallel(seen_polytopes)
    initial_len = len(new_facets)

    adv_constraints = piecewise_net.make_adversarial_constraints(new_poly,
                                                            true_label, domain)

    ##########################################################################
    #   Step 4: Compute the min-dists/feasibility checks using LP/QP         #
    ##########################################################################

    # -- compute the distances
    chained_facets = itertools.chain(new_facets, adv_constraints)
    parallel_args = [(_, x) for _ in chained_facets]
    dist_selector = {('l_2', 'mosek')   : Face.l2_dist,
                     ('l_2', 'gurobi')  : Face.l2_dist_gurobi,
                     ('l_inf', 'mosek') : Face.linf_dist,
                     ('l_inf', 'gurobi'): Face.linf_dist_gurobi}
    lp_dist = dist_selector[(lp_norm, optimizer)]
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


