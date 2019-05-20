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
from plnn import PLNN
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



##############################################################################
#                                                                            #
#                           MAIN GEOCERT CLASS                               #
#                                                                            #
##############################################################################



class IncrementalGeoCertMultiProc(object):

    bound_fxn_selector = {'ia': PLNN.compute_interval_bounds,
                          'dual_lp': PLNN.compute_dual_lp_bounds,
                          'full_lp': PLNN.compute_full_lp_bounds}

    def __init__(self, net, hyperbox_bounds=None,
                 verbose=True, neuron_bounds='full_lp',
                 # And for 2d inputs, some kwargs for displaying things
                 display=False, save_dir=None, ax=None):

        """ To set up a geocert instance we need to know:
        ARGS:
            net : PLNN instance - the network we're verifying
            config_fxn: how we're generating facets (parallel or serial)
            hyperbox_bounds: if not None, is a tuple of pair of numbers
                             (lo, hi) that define a valid hyperbox domain
            neuron_bounds: string

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

        assert neuron_bounds in ['ia', 'dual_lp', 'full_lp']
        self.neuron_bounds = neuron_bounds
        self.bound_fxn = self.bound_fxn_selector[neuron_bounds]
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


    def _setup_state(self, x, lp_norm, num_proc, optimizer, potential):
        """ Sets up the state to be used on a per-run basis
        Shared between min_dist_multiproc and decision_problem_multiproc

        Sets instance variables and does asserts
        """
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
        assert optimizer in ['mosek', 'gurobi']
        if optimizer == 'mosek' and num_proc > 1:
            raise Exception("Multiprocessing doesn't work with mosek!")

        assert potential in ['lp', 'lipschitz']



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

    def min_dist_multiproc(self, x, lp_norm='l_2', compute_upper_bound=False,
                           num_proc=2, optimizer='gurobi', potential='lp',
                           problem_type='min_dist', decision_radius=None,
                           collect_graph=False, max_runtime=None):
        ######################################################################
        #   Step 0: Clear and setup state                                    #
        ######################################################################
        self._reset_state() # clear out the state first
        self._setup_state(x, lp_norm, num_proc, optimizer, potential)

        start_time = time.time()
        # Upper bound times that the domain queue updater knows about
        upper_bound_times = mp.Queue() # (time, bound)

        # Lower bound times that the workers know about
        lower_bound_times = mp.Queue() # (time, bound)


        # Compute upper bound
        adv_bound, adv_ex, ub_time = None, None, None
        if problem_type == 'min_dist':
            if compute_upper_bound is not False:
                ub_out = self._compute_upper_bounds(x, self.true_label, lp_norm,
                                       extra_attack_kwargs=compute_upper_bound)
                adv_bound, adv_ex, ub_time = ub_out
                if adv_bound is not None:
                    upper_bound_times.put((time.time() - start_time, adv_bound))
        if problem_type in ['decision_problem', 'count_regions']:
            assert decision_radius is not None
            self.domain.set_upper_bound(decision_radius, lp_norm)


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
        status_dict = manager.dict()
        status_dict['kill_max_runtime_thread'] = False
        seen_polytopes = manager.dict()
        missed_polytopes = manager.dict()
        if collect_graph:
            polytope_graph = manager.dict()
        else:
            polytope_graph = None

        # Set up heuristic dicts
        heuristic_dict = manager.dict()
        heuristic_dict['domain'] = self.domain
        heuristic_dict['dead_constraints'] = self.dead_constraints

        if potential == 'lipschitz':
            # Just assume binary classifiers for now
            # on_off_neurons = self.net.compute_interval_bounds(self.domain, True)
            dual_lp = utils.dual_norm(lp_norm)
            c_vector, lip_value = self.net.fast_lip_all_vals(x, dual_lp,
                                                             self.on_off_neurons)
        else:
            lip_value = None
            c_vector = None

        heuristic_dict['fast_lip'] = lip_value
        heuristic_dict['c_vector'] = c_vector
        status_dict['num_proc'] = num_proc
        for i in range(num_proc):
            set_waiting_status(status_dict, i, False)


        # Set up domain updater queue
        domain_update_queue = mp.Queue()



        # Start the domain update thread
        dq_thread = Thread(target=domain_queue_updater,
                           args=(self.net, self.x, heuristic_dict,
                                 domain_update_queue, potential, lp_norm,
                                 self.bound_fxn, upper_bound_times, start_time,
                                 max_runtime))
        dq_thread.start()

        # Start max runtime thread
        if max_runtime is not None:
            runtime_thread = Thread(target=max_runtime_thread,
                                    args=(start_time, max_runtime, sync_pq,
                                          domain_update_queue,
                                          pq_decision_bounds, status_dict))
        else:
            dummy_thread = lambda: None
            runtime_thread = Thread(target=dummy_thread)
        runtime_thread.start()
        ######################################################################
        #   Step 2: handle the initial polytope                              #
        ######################################################################

        # NOTE: The loop doesn't quite work here, so have to do the first part
        #       (aka emulate update_step_build_poly) manually.
        #       1) Build the original polytope
        #       2) Add polytope to seen polytopes
        #
        print(len(self.dead_constraints), sum(self.dead_constraints))
        self._verbose_print('---Initial Polytope---')
        p_0_dict = self.net.compute_polytope(self.x, comparison_form_flag=False)

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
                                    domain_update_queue, pq_decision_bounds,
                                    optimizer, potential, missed_polytopes,
                                    status_dict, problem_type, polytope_graph)

        if problem_type == 'decision_problem':
            try:
                best_decision_bound = pq_decision_bounds.get_nowait()
                best_dist = best_decision_bound.priority
                best_example = best_decision_bound.projection

                dq_thread.join()
                status_dict['kill_max_runtime_thread'] = True
                runtime_thread.join()
                return (best_dist, adv_bound, adv_ex, best_example, ub_time,
                        seen_polytopes, missed_polytopes, polytope_graph,
                        lower_bound_times, upper_bound_times)
            except Empty:
                pass


        ######################################################################
        #   Step 3: Setup threads and start looping                          #
        ######################################################################
        # Do this on one processor if num_jobs is one

        proc_args = (self.net, self.x_np, self.true_label, sync_pq,
                     seen_polytopes, heuristic_dict, self.lp_norm,
                     domain_update_queue, pq_decision_bounds, optimizer,
                     potential, missed_polytopes, problem_type, status_dict,
                     polytope_graph, lower_bound_times, start_time)
        if num_proc == 1:
            update_step_worker(*proc_args, **{'proc_id': 0})
        else:
            procs = [mp.Process(target=update_step_worker,
                                args=proc_args, kwargs={'proc_id': i})
                     for i in range(num_proc)]
            [_.start() for _ in procs]
            [_.join() for _ in procs]

        # Stop the domain update thread
        dq_thread.join()
        status_dict['kill_max_runtime_thread'] = True
        runtime_thread.join()

        ######################################################################
        #   Step 4: Collect the best thing in the decision queue and return  #
        ######################################################################

        overran_time = ((max_runtime is not None) and\
                        (time.time() - start_time > max_runtime))
        if problem_type == 'min_dist' and not overran_time:
            best_decision_bound = pq_decision_bounds.get()
        else:
            try:
                best_decision_bound = pq_decision_bounds.get_nowait()
            except:
                if overran_time:
                    print("TIMEOUT")
                elif problem_type == 'decision_problem':
                    print("DECISION PROBLEM FAILED")
                else:
                    print("COUNTED %s LINEAR REGIONS" % len(seen_polytopes))

                return (None, adv_bound, adv_ex, None, ub_time, seen_polytopes,
                        missed_polytopes, polytope_graph,
                        lower_bound_times, upper_bound_times)

        best_dist = best_decision_bound.priority
        best_example = best_decision_bound.projection
        return  (best_dist, adv_bound, adv_ex, best_example, ub_time,
                 seen_polytopes, missed_polytopes, polytope_graph,
                 lower_bound_times, upper_bound_times)




##############################################################################
#                                                                            #
#                           FUNCTIONAL VERSION OF UPDATES                    #
#                            (useful for multiprocessing)                    #
##############################################################################



def update_step_worker(piecewise_net, x, true_label, pqueue, seen_polytopes,
                       heuristic_dict, lp_norm, domain_update_queue,
                       pq_decision_bounds, optimizer, potential,
                       missed_polytopes, problem_type, status_dict,
                       polytope_graph, lower_bound_times, start_time,
                       proc_id=None):
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
                                  lp_norm, domain_update_queue,
                                  pq_decision_bounds, optimizer,
                                  potential, missed_polytopes, status_dict,
                                  problem_type, polytope_graph,
                                  lower_bound_times, start_time, proc_id)
        if output is not True: # Termination condition
            return output



def update_step_loop(piecewise_net, x, true_label, pqueue, seen_polytopes,
                     heuristic_dict, lp_norm, domain_update_queue,
                     pq_decision_bounds, optimizer, potential, missed_polytopes,
                     status_dict, problem_type, polytope_graph,
                     lower_bound_times, start_time, proc_id):
    """ Inner loop for how to update the priority queue. This handles one
        particular thing being popped off the PQ
    """
    # Build the polytope to pop from the queue

    poly_out = update_step_build_poly(piecewise_net, x, pqueue, seen_polytopes,
                                      heuristic_dict, lp_norm,
                                      domain_update_queue,
                                      pq_decision_bounds, potential,
                                      status_dict, problem_type,
                                      lower_bound_times, start_time, proc_id)

    if isinstance(poly_out, bool): # bubble up booleans
        return poly_out

    new_poly, domain, dead_constraints = poly_out

    # Build facets, reject what we can, and do optimization on the rest
    return update_step_handle_polytope(piecewise_net, x, true_label, pqueue,
                                       seen_polytopes, domain, dead_constraints,
                                       new_poly, lp_norm, domain_update_queue,
                                       pq_decision_bounds, optimizer, potential,
                                       missed_polytopes, status_dict,
                                       problem_type, polytope_graph)



def update_step_build_poly(piecewise_net, x, pqueue, seen_polytopes,
                           heuristic_dict, lp_norm, domain_update_queue,
                           pq_decision_bounds, potential, status_dict,
                           problem_type, lower_bound_times, start_time,
                           proc_id):
    """ Component method of the loop.
        1) Pops the top PQ element off and rejects it as seen before if so
        2) Collect the domain/heuristics
        3) builds the new polytope and returns the polytope
    """
    ##########################################################################
    #   Step 1: pop something off the queue                                  #
    ##########################################################################
    try:
        if problem_type in ['decision_problem', 'count_regions']:
            item = pqueue.get_nowait()
        else:
            item = pqueue.get()
    except Empty:
        set_waiting_status(status_dict, proc_id, True)
        status_items = status_dict.items()
        if all_waiting(status_items):
            kill_processes(None, pqueue, domain_update_queue,
                           pq_decision_bounds, status_dict)
            return False
        item = pqueue.get()
        set_waiting_status(status_dict, proc_id, False)


    if item.priority < 0: # Termination condition -- bubble up the termination
        return False
    if item.facet_type == 'decision': # Termination condition -- bubble up
        kill_processes(item, pqueue, domain_update_queue, pq_decision_bounds,
                       status_dict)
        return False

    # Update the lower bound queue
    lower_bound_times.put(((time.time() - start_time), item.priority))

    config = item.config
    tight_constraint = item.tight_constraint

    new_configs = utils.get_new_configs(config, tight_constraint)
    if utils.flatten_config(new_configs) in seen_polytopes:
        return True # No need to go further, but don't terminate!
    else:
        seen_polytopes[utils.flatten_config(new_configs)] = True



    ##########################################################################
    #   Step 2: Gather the domain and dead neurons                           #
    ##########################################################################

    domain = heuristic_dict['domain']
    current_upper_bound = domain.current_upper_bound(lp_norm) or 1e10

    print("(p%s) Popped: %.06f  | %.06f" %
          (proc_id, item.priority, current_upper_bound))
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
                                new_poly, lp_norm, domain_update_queue,
                                pq_decision_bounds, optimizer, potential,
                                missed_polytopes, status_dict,
                                problem_type, polytope_graph):
    """ Component method of the loop
        1) Makes facets, rejecting quickly where we can
        2) Run convex optimization on everything we can't reject
        3) Push the updates to the process-safe objects
    """

    ##########################################################################
    #   Step 1: Make new facets while doing fast rejects                     #
    ##########################################################################

    new_facets, rejects = new_poly.generate_facets_configs_parallel(seen_polytopes,
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
    fail_count = 0
    try_count = len(outputs)
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
            rejects['above upper bound']
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
                # If in decision_problem style, kill all processes
                return kill_processes(new_pq_element, pqueue,
                                      domain_update_queue, pq_decision_bounds,
                                      status_dict)

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
        pqueue.put(pq_element)

    # -- push domain to be handled by a separate thread
    if updated_domain:
        domain_update_queue.put(domain)

    # print("REJECTS", dict(rejects))
    return True



def domain_queue_updater(piecewise_net, x, heuristic_dict, domain_update_queue,
                         potential, lp_norm, bound_fxn, upper_bound_times,
                         start_time, max_runtime):
    """ Separate thread to handle updating the domain/dead constraints """

    domain = heuristic_dict['domain']
    attr = {'l_inf': 'linf_radius',
            'l_2':   'l2_radius'}[lp_norm]


    while True:
        new_domain = domain_update_queue.get()
        if new_domain == None:
            break

        linf_radius = new_domain.linf_radius or 1e10
        l2_radius = new_domain.l2_radius or 1e10
        print('-' * 20, "DOMAIN UPDATE | L_inf %.06f | L_2 %.06f" %
              (linf_radius, l2_radius))

        # Record the update in the upper_bound_times log
        if getattr(new_domain, attr) < getattr(domain, attr):
            upper_bound_times.put((time.time() - start_time,
                                   getattr(new_domain, attr)))


        # Update the current domain and change the heuristic dict

        domain.set_hyperbox_bound(new_domain.box_low,
                                  new_domain.box_high)

        domain.set_l_inf_upper_bound(new_domain.linf_radius)
        domain.set_l_2_upper_bound(new_domain.l2_radius)
        heuristic_dict['domain'] = domain




        # And use the domain to compute the new dead constraints
        new_bounds = bound_fxn(piecewise_net, domain)
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



def kill_processes(item, pqueue, domain_update_queue, pq_decision_bounds,
                   status_dict):
    """ Puts a poison pill in all the pqs that should bubble kill all
        processes on the next loop iteration
    """
    for i in range(100): # Just flood the pq with -1's later
        stop_pq_el = PQElement()
        stop_pq_el.priority = -1
        pqueue.put(stop_pq_el)
        kill_domain_update_queue(domain_update_queue)
        if item is not None:
            pq_decision_bounds.put(item)
    return True


def max_runtime_thread(start_time, max_runtime, pqueue, domain_update_queue,
                       pq_decision_bounds, status_dict):
    """ Thread to kill all processes if we run over the max allotted time """
    ready_to_kill = lambda : time.time() - start_time > max_runtime
    while True:
        if ready_to_kill() or status_dict['kill_max_runtime_thread']:
            kill_processes(None, pqueue, domain_update_queue,
                           pq_decision_bounds, status_dict)
            return
        else:
            time.sleep(0.5)


def kill_domain_update_queue(domain_update_queue):
    """ puts a poison pill onto the domain_update_queue"""
    domain_update_queue.put(None)


def set_waiting_status(status_dict, proc_id, waiting_status):
    status_dict['waiting:%s' % proc_id] = waiting_status

def all_waiting(status_items):
    return all([_[1] for _ in status_items if _[0].startswith('waiting:')])
