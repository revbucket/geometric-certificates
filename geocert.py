""" File that contains the algorithm for geometric certificates in piecewise
    linear neural nets or general unions of perfectly glued polytopes

"""

from _polytope_ import Polytope, Face, from_polytope_dict
import utilities as utils
import torch
import numpy as np
import heapq
import matplotlib.pyplot as plt
import copy

##############################################################################
#                                                                            #
#                               BATCHED GEOCERT                              #
#                                                                            #
##############################################################################

# Batched algorithm is when the union of polytopes is specified beforehand

def compute_boundary_batch(polytope_list, comparison_method = 'slow'):
    """ Takes in a list of polytopes and outputs the facets that define the
        boundary
    """

    total_facets = [facet for poly in polytope_list for facet in poly.generate_facets(check_feasible=True)]

    print('num total facets:', len(total_facets))

    unshared_facets = []
    shared_facets = []


    for og_facet in total_facets:

        if comparison_method == 'slow':
            bool_unshared = [og_facet.check_same_facet_pg_slow(ex_facet)
                             for ex_facet in unshared_facets]

            bool_shared = [og_facet.check_same_facet_pg_slow(ex_facet)
                           for ex_facet in shared_facets]

        elif comparison_method == 'unstable':
            bool_unshared = [og_facet.check_same_facet_pg(ex_facet)
                   for ex_facet in unshared_facets]
            bool_shared = [og_facet.check_same_facet_pg(ex_facet)
                   for ex_facet in shared_facets]

        elif comparison_method == 'fast_ReLu':
            # Uses information of ReLu activations to check if two facets
            # are the same
            bool_unshared = [og_facet.check_same_facet_config(ex_facet)
                   for ex_facet in unshared_facets]
            bool_shared = [og_facet.check_same_facet_config(ex_facet)
                   for ex_facet in shared_facets]
        else:
            raise NotImplementedError

        if any(bool_shared):
            continue
        elif any(bool_unshared):
            index = bool_unshared.index(True)
            shared_facet = unshared_facets[index]
            unshared_facets.remove(shared_facet)
            shared_facets.append(shared_facet)
        else:
            unshared_facets.append(og_facet)

    return unshared_facets, shared_facets



def compute_l_inf_ball_batch(polytope_list, x, comp_method = 'slow'):
    """ Computes the linf distance from x to the boundary of the union of polytopes

        Comparison method options: {slow | unstable | fast_ReLu}
    """

    # First check if x is in one of the polytopes
    if not any(poly.is_point_feasible(x) for poly in polytope_list):
        return -1

    print('----------Computing Boundary----------')
    boundary, shared_facets = compute_boundary_batch(polytope_list, comp_method)

    dist_to_boundary = [facet.linf_dist(x) for facet in boundary]

    return min(dist_to_boundary), boundary, shared_facets

def compute_l2_ball_batch(polytope_list, x, comp_method = 'slow'):
    """ Computes the l2 distance from x to the boundary of the union of polytopes

        Comparison method options: {slow | unstable | fast_ReLu}
    """

    # First check if x is in one of the polytopes
    if not any(poly.is_point_feasible(x) for poly in polytope_list):
        return -1

    print('----------Computing Boundary----------')
    boundary, shared_facets = compute_boundary_batch(polytope_list, comp_method)

    dist_to_boundary = [facet.l2_dist(x) for facet in boundary]

    return min(dist_to_boundary), boundary, shared_facets



##########################################################################
#                                                                        #
#                           INCREMENTAL GEOCERT                          #
#                                                                        #
##########################################################################
#TODO: why are polytopes being seeen more than once?
class HeapElement(object):
    """ Wrapper of the element to be pushed around the priority queue
        in the incremental algorithm
    """
    def __init__(self, lp_dist, facet,
                 decision_bound=False,
                 exact_or_estimate='exact'):
        self.lp_dist = lp_dist
        self.facet = facet
        self.decision_bound = decision_bound
        self.exact_or_estimate = exact_or_estimate

    def __lt__(self, other):
        return self.lp_dist < other.lp_dist


def incremental_geocert(lp_norm, net, x, ax, plot_dir, n_colors=200):
    """ Computes l_inf distance to decision boundary in incremental steps of
        expanding the search space

        lp_norm: options include    =>  {'l_2' | 'l_inf'}
    """
    true_label = int(net(x).max(1)[1].item()) # what the classifier outputs
    seen_to_polytope_map = {} # binary config str -> Polytope object
    seen_to_facet_map = {} # binary config str -> Facet list
    pq = [] # Priority queue that contains HeapElements


    ###########################################################################
    #   Initialization phase: compute polytope containing x                   #
    ###########################################################################
    print('---Initial Polytope---')
    p_0_dict = net.compute_polytope(x, True)
    p_0 = from_polytope_dict(p_0_dict)
    geocert_update_step(lp_norm, net, x, p_0, None, pq, true_label,
                        seen_to_polytope_map, seen_to_facet_map)



    ##########################################################################
    #   Incremental phase -- repeat until we hit a decision boundary         #
    ##########################################################################
    index = 0

    while True:
        # Pop a facet from the heap
        pop_el = heapq.heappop(pq)

        # If only an estimate, make it exact and push it back onto the heap
        if pop_el.exact_or_estimate == 'estimate':
            exact_lp_dist = pop_el.facet.lp_dist(x)
            new_heap_el = HeapElement(exact_lp_dist, pop_el.facet,
                                      decision_bound=pop_el.decision_bound,
                                      exact_or_estimate='exact')
            heapq.heappush(pq, new_heap_el)

        # If popped element is part of the decision boundary then DONE
        if pop_el.decision_bound:
            print('----------Minimal Projection Generated----------')
            geocert_plot_step(lp_norm, seen_to_polytope_map, pq, pop_el.lp_dist,
                              x, plot_dir, n_colors, iter=index)
            return pop_el.lp_dist

        # Otherwise, find ReLu configuration on other side of the facet
        # and expand the search space
        else:
            print('---Opening New Polytope---')

            popped_facet = pop_el.facet
            pre_relus, post_relus = net.relu_config(torch.Tensor(popped_facet.interior))
            tight_boolean = [utils.fuzzy_equal(elem, 0.0, tolerance=1e-6)
                             for activations in pre_relus for elem in activations]
            orig_configs = popped_facet.config
            new_configs = get_new_configs(tight_boolean, orig_configs)
            new_configs_flat = utils.flatten_config(new_configs)

            # If polytope has already been seen, don't add it again
            if new_configs_flat not in seen_to_polytope_map:
                new_polytope_dict = net.compute_polytope_config(new_configs, True)
                new_polytope = from_polytope_dict(new_polytope_dict)
                geocert_update_step(lp_norm, net, x, new_polytope, popped_facet, pq, true_label,
                                    seen_to_polytope_map, seen_to_facet_map)

            else:
                print('weve already seen that polytope')

        if(index % 1 == 0 ):
            geocert_plot_step(lp_norm, seen_to_polytope_map, pq, pop_el.lp_dist,
                              x, plot_dir, n_colors, iter=index)
        index = index + 1


def geocert_update_step(lp_norm, net, x, polytope, popped_facet, pr_queue, true_label,
                        seen_to_polytope_map, seen_to_facet_map):
    ''' Given next polytope from popped heap element: finds new polytope facets,
        pushes facets to the heap, and updates seen maps
    '''

    polytope_facets = polytope.generate_facets_configs(seen_to_polytope_map, check_feasible=True)
    print('num facets: ', len(polytope_facets))

    polytope_config = utils.flatten_config(polytope.config)
    polytope_adv_constraints = net.make_adversarial_constraints(polytope.config,
                                                           true_label)
    seen_to_polytope_map[polytope_config] = polytope
    seen_to_facet_map[polytope_config] = polytope_facets



    for facet in polytope_facets:
        if popped_facet is not None:
            if not (popped_facet.check_same_facet_config(facet)):
                # Only add to heap if new face isn't the popped facet
                lp_dist = get_lp_dist(lp_norm, facet, x)
                heap_el = HeapElement(lp_dist, facet, decision_bound=False,
                                      exact_or_estimate='exact')
                heapq.heappush(pr_queue, heap_el)
        else:
            # For first time use, popped facet doesn't exist
            lp_dist = get_lp_dist(lp_norm, facet, x)
            heap_el = HeapElement(lp_dist, facet, decision_bound=False,
                                  exact_or_estimate='exact')
            heapq.heappush(pr_queue, heap_el)

    for facet in polytope_adv_constraints:
        lp_dist = get_lp_dist(lp_norm, facet, x)
        heap_el = HeapElement(lp_dist, facet, decision_bound=True,
                              exact_or_estimate='exact')
        heapq.heappush(pr_queue, heap_el)

def get_lp_dist(lp_norm, facet, x):
    if lp_norm == 'l_2':
        return facet.l2_dist(x)
    elif lp_norm == 'l_inf':
        return facet.linf_dist(x)
    else:
        raise NotImplementedError


def get_new_configs(tight_boolean_configs, orig_configs):
    ''' Function takes original ReLu configs and flips the activation of
        the ReLu at index specified in 'tight_boolean_configs'.
    '''

    new_configs = copy.deepcopy(orig_configs)

    for i, tight_bools in enumerate(tight_boolean_configs):
        for j, tight_bool in enumerate(tight_bools):
            if tight_bool == 1:
                if orig_configs[i][j] == 1.0:
                    new_configs[i][j] = 0.0
                elif orig_configs[i][j] == 0.0:
                    new_configs[i][j] = 1.0
                else:
                    raise AttributeError

    return new_configs


def geocert_plot_step(lp_norm, seen_to_polytope_map, facet_heap_elems,
                      t, x, plot_dir, n_colors, ax =None, iter=0):
    ''' Plots the current search boundary based on the heap, the seen polytopes,
        the current minimal lp ball, and any classification boundary facets
    '''
    plt.figure(figsize=[10, 10])
    if ax is None:
        ax = plt.axes()
    polytope_list = [seen_to_polytope_map[elem] for elem in seen_to_polytope_map]
    facet_list = [heap_elem.facet for heap_elem in facet_heap_elems if not heap_elem.decision_bound]
    boundary_facet_list = [heap_elem.facet for heap_elem in facet_heap_elems if heap_elem.decision_bound]
    colors = utils.get_spaced_colors(n_colors)[0:len(polytope_list)]

    xylim = 50.0

    utils.plot_polytopes_2d(polytope_list, colors=colors, alpha=0.7,
                          xylim=5, ax=ax, linestyle='dashed', linewidth=0)

    utils.plot_facets_2d(facet_list, alpha=0.7,
                   xylim=xylim, ax=ax, linestyle='dashed', linewidth=3, color='black')

    utils.plot_facets_2d(boundary_facet_list, alpha=0.7,
                   xylim=xylim, ax=ax, linestyle='dashed', linewidth=3, color='red')

    if lp_norm == 'l_inf':
        utils.plot_linf_norm(x, t, linewidth=1, edgecolor='red', ax=ax)
    elif lp_norm == 'l_2':
        utils.plot_l2_norm(x, t, linewidth=1, edgecolor='red', ax=ax)
    else:
        raise NotImplementedError

    plt.autoscale()
    new_xlims = plt.xlim()
    new_ylims = plt.ylim()

    if min(new_xlims) > -xylim and max(new_xlims) < xylim and min(new_ylims) > -xylim and max(new_ylims) < xylim:
        pass
    else:
        plt.xlim(-xylim, xylim)
        plt.ylim(-xylim, xylim)
    filename = plot_dir + str(iter) + '.svg'
    plt.savefig(filename)
    plt.close()




