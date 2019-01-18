""" File that contains the algorithm for geometric certificates in piecewise
    linear neural nets or general unions of perfectly glued polytopes

"""

from polytope import Polytope, Face, from_polytope_dict
import utilities as utils
import torch
import numpy as np
import heapq

##############################################################################
#                                                                            #
#                               BATCHED GEOCERT                              #
#                                                                            #
##############################################################################

# Batched algorithm is when the union of polytopes is specified beforehand

def compute_boundary_batch(polytope_list):
    """ Takes in a list of polytopes and outputs the facets that define the
        boundary
    """

    total_facets = [facet for facet in
                    poly.generate_facets(check_feasible=True)
                    for poly in polytope_list]


    unshared_facets = []
    for og_facet in total_facets:
        if any(og_facet.check_same_facet_pg(ex_facet)
               for ex_facet in unshared_facets):
            continue
        else:
            unshared_facets.append(og_facet)

    return unshared_facets


def compute_l_inf_ball_batch(polytope_list, x):
    """ Computes the distance from x to the boundary of the union of polytopes
    """

    # First check if x is in one of the polytopes
    if not any(poly.is_point_feasible(x) for poly in polytope_list):
        return -1


    boundary = compute_boundary_batch(polytope_list)

    dist_to_boundary = [facet.linf_dist(x) for facet in boundary]

    return min(dist_to_boundary)



##########################################################################
#                                                                        #
#                           INCREMENTAL GEOCERT                          #
#                                                                        #
##########################################################################

class HeapElement(object):
    """ Wrapper of the element to be pushed around the priority queue
        in the incremental algorithm
    """
    def __init__(self, linf_dist, facet,
                 decision_bound=False,
                 exact_or_estimate='exact'):
        self.linf_dist = linf_dist
        self.facet = facet
        self.decision_bound = decision_bound
        self.exact_or_estimate = exact_or_estimate

    def __cmp__(self, other):
        return cmp(self.l_inf_dist, other.l_inf_dist)


def incremental_geocert(net, x):
    """ Computes l_inf distance to decision boundary in
    """

    seen_to_polytope_map = {} # binary config str -> Polytope object
    seen_to_facet_map = {} # binary config str -> Facet list

    pq = [] # Priority queue that contains HeapElements


    ###########################################################################
    #   Initialization phase: compute polytope containing x                   #
    ###########################################################################

    p_0_dict = plnn.compute_polytope(x)
    p_0 = from_polytope_dict(p_0_dict)
    p_0_facets = p_0.generate_facets(check_feasible=True)
    p_0_config = utils.flatten_config(p_0_dict['config'])

    seen_to_polytope_map[p_0_config] = p_0
    seen_to_facet_map[p_0_config] = p_0_facets
    for facet in p_0_facets:
        linf_dist = facet.linf_dist(x)
        heap_el = HeapElement(linf_dist, facet, decision_bound=False,
                              exact_or_estimate='exact')
        heapq.heappush(pq, heap_el)

    # ADD ADVERSARIAL CONFIGS
    ##########################################################################
    #   Incremental phase -- repeat until we hit a decision boundary         #
    ##########################################################################

    pop_el = heapq.heappop(pq)

    # If only an estimate, make it exact and push it back onto the heap
    if pop_el.exact_or_estimate == 'estimate':
        exact_linf = pop_el.facet.linf_dist(x)
        new_heap_el = HeapElement(exact_linf, pop_el.facet,
                                  decision_bound=pop_el.decision_bound,
                                  exact_or_estimate='exact')
        heapq.heappush

        # BROKEN BUT PUSHING ANYWAY



