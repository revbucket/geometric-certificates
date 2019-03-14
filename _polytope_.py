import utilities as utils
import torch
import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from cvxopt import matrix, solvers
import copy
import pickle
from _facet_ import Face
import joblib 

##########################################################################
#                                                                        #
#                   POLYTOPE AND FACE CLASSES                            #
#                                                                        #
##########################################################################

def fast_fxn(j):
    x = len(j) 
    return x

def handle_facet(facet_index, ub_b, ub_A, redundant, seen_dict, upper_bound, 
                 check_feasible, net, config):        
    facets = []
    reject_reasons = {}
    if facet_index in redundant:
        return facets, reject_reasons 
    facet = Face(ub_A, ub_b, [facet_index], config)

    if upper_bound is not None:
        reject_status, reason = facet.reject_via_upper_bound(upper_bound)
        if reject_status:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            return facets, reject_reasons

    if check_feasible:
        facet.check_feasible()
    if not facet.is_feasible:                
        reject_reasons['infeasible'] =\
                                 reject_reasons.get('infeasible', 0) + 1
    else:
        assert facet_index not in redundant

    facet.check_facet()
    if facet.is_facet:
        # Check to see if facet is shared with a seen polytope                
        new_configs = facet.get_new_configs(net)
        new_configs_flat = utils.flatten_config(new_configs)
        shared_facet_bools = [new_configs_flat == other_config_flat
                              for other_config_flat in seen_dict]
        if not any(shared_facet_bools):
            facets.append(facet)                

    return facets, reject_reasons


def handle_facet_2(ub_A, ub_b, tight_idx, config, upper_bound_dict, 
                   seen_dict, net):
    facet = Face(ub_A, ub_b, [tight_idx], config=config)

    if upper_bound_dict is not None:
        reject_status, reason = facet.reject_via_upper_bound(upper_bound_dict)
        if reject_status:
            return (False, reason)

    if True: #check_feasible:
        facet.check_feasible()
    if not facet.is_feasible:
        return (False, 'infeasible')

    facet.check_facet() 
    if facet.is_facet:
        new_configs = facet.get_new_configs(net)
        new_configs_flat = utils.flatten_config(new_configs)
        shared_facet_bools = [new_configs_flat == other_config_flat
                               for other_config_flat in seen_dict]
        if not any(shared_facet_bools):
            return (True, facet)                
        else:
            return (False, 'shared')
    else:
        return (False, 'not-facet') 

def handle_facet_3(ub_A, ub_b, tight_idx, config, upper_bound_dict):
    facet = Face(ub_A, ub_b, [tight_idx], config=config)
    #print("PID: ", os.getpid(), " TIGHT LIST:", tight_idx)
    if upper_bound_dict is not None:
        reject_status, reason = facet.reject_via_upper_bound(upper_bound_dict)
        if reject_status:
            return (False, reason)

    if True: #check_feasible:
        facet.check_feasible()
    if not facet.is_feasible:
        return (False, 'infeasible')

    facet.check_facet() 
    return (True, facet)


def handle_facet_3_star(args):
    try:
        return handle_facet_3(*args)
    except:
        print("ERRORR!?!?!?!?!?!?!?!?!??!")

def finish_handle_facet_3(facet, config, seen_dict, net):


    if facet.is_facet:
        new_configs = facet.get_new_configs(net)
        new_configs_flat = utils.flatten_config(new_configs)
        shared_facet_bools = [new_configs_flat == other_config_flat
                               for other_config_flat in seen_dict]
        if not any(shared_facet_bools):
            return (True, facet)                
        else:
            return (False, 'shared')
    else:
        return (False, 'not-facet') 

    

def from_polytope_dict(polytope_dict):
    return Polytope(polytope_dict['poly_a'],
                    polytope_dict['poly_b'],
                    config=polytope_dict['configs'])

class Polytope(object):
    def __init__(self, ub_A, ub_b, config=None, interior_point=None):
        """ Polytopes are of the form Ax <= b
            with no strict equality constraints"""

        if isinstance(ub_A, torch.Tensor):
            ub_A = ub_A.cpu().detach().numpy()
        if isinstance(ub_b, torch.Tensor):
            ub_b = ub_b.cpu().detach().numpy()
        self.ub_A = utils.as_numpy(ub_A)
        self.ub_b = utils.as_numpy(ub_b).squeeze()
        self.config = config
        self.interior_point = interior_point

    def generate_facets(self, check_feasible=False):
        """ Generates all (n-1) dimensional facets of polytope
        """
        num_constraints = self.ub_A.shape[0]
        facets = []
        for i in range(num_constraints):
            facet = Face(self.ub_A, self.ub_b, [i], config=self.config)
            if check_feasible:
                facet.check_feasible()
            facet.check_facet()

            if facet.is_facet:
                facets.append(facet)
        return facets



    def generate_facets_configs_parallel_2(self, seen_polytopes_dict, net, 
                                         check_feasible=False, 
                                         upper_bound_dict=None, 
                                         use_clarkson=False):

        ######################################################################
        #   Set up global variables and subprocess to be pooled out          #
        ######################################################################

        ##################################################################
        #   Set up pool and offload all the work, and then merge results #
        ##################################################################

        global uba
        uba = self.ub_A
        
        global ubb 
        ubb = self.ub_b 

        global conf
        conf = self.config 

        global ubdict 
        ubdict = upper_bound_dict
        
        
        maplist = [] 

        for idx in range(self.ub_A.shape[0]):
            # new_uba = np.array(self.ub_A, copy=True)
            # new_ubb = np.array(self.ub_b, copy=True)
            # new_config = copy.deepcopy(self.config)
            # new_upper_bound_dict = copy.deepcopy(upper_bound_dict)
            maplist.append((uba, ubb, idx, conf, ubdict))
            
        
            #results = [pool.apply_async(handle_facet_3, el) for el in maplist]
        
        outputs = joblib.Parallel(n_jobs=20)(joblib.delayed(handle_facet_3_star)(arg) for arg in maplist)
        
        if True:
            #proclist = [pool.apply_async(handle_facet_3_star, (args,)) for args in maplist]
            #outputs = [res.get() for res in proclist]
            #outputs = pool.map(handle_facet_3_star, maplist)
            new_outputs = [] 
            for status, output in outputs:
                if status:
                    new_outputs.append(finish_handle_facet_3(output, self.config, 
                                                             seen_polytopes_dict, 
                                                             net))
                else:
                    new_outputs.append((status, output))
            facets = []
            reject_dict = {} 
            for status, output in new_outputs:
                if status:
                    facets.append(output)
                else:
                    reject_dict[output] = reject_dict.get(output, 0) + 1
            return facets, reject_dict


    def generate_facets_configs_parallel(self, seen_polytopes_dict, net, 
                                         check_feasible=False, 
                                         upper_bound_dict=None, 
                                         use_clarkson=True):

        ######################################################################
        #   Set up global variables and subprocess to be pooled out          #
        ######################################################################
        
        global pool_ub_A 
        pool_ub_A = self.ub_A 

        global pool_ub_b 
        pool_ub_b = self.ub_b 

        global pool_redundant_set 
        pool_redundant_set = set() 
        if check_feasible and use_clarkson:
            pool_redundant_set = self.clarkson_redundancy_set(
                                                            self.interior_point)
            print("Clarkson found %s redundant constraints" % \
                  len(pool_redundant_set))

        global pool_upper_bound_dict 
        pool_upper_bound_dict = upper_bound_dict

        global pool_seen_polytope_dict
        pool_seen_polytope_dict = seen_polytopes_dict

        global pool_check_feasible
        pool_check_feasible = check_feasible

        global pool_net 
        pool_net = net 

        global pool_config 
        pool_config = self.config


        ##################################################################
        #   Set up pool and offload all the work, and then merge results #
        ##################################################################
        
        pool = mp.Pool(processes=4)
        maplist = [] 
        for idx in range(self.ub_A.shape[0]):
            maplist.append((idx, pool_ub_b, pool_ub_A, pool_redundant_set, 
                            pool_seen_polytope_dict, pool_upper_bound_dict, 
                            check_feasible, pool_net, pool_config))

        facets_out = []
        rejects_out = [] 
        results = [pool.apply_async(handle_facet, maplist_el) for 
                   maplist_el in maplist]            

        facets_out, rejects_out = zip(*[result.get() for result in results])
            

        #facets_out, rejects_out = zip(*res)
        #facets_out, rejects_out = zip(*pool.map(handle_facet, maplist))

        faces_total = [facet for facet_list in facets_out 
                             for facet in facet_list]
        rejects_total = {}
        for reject in rejects_out:
            for k in reject:
                rejects_total[k] = rejects_total.get(k, 0) + reject[k]

        return faces_total, rejects_total


    def generate_facets_configs(self, seen_polytopes_dict, net, check_feasible=False,
                                upper_bound_dict=None, use_clarkson=True):
        """ Generates all (n-1) dimensional facets of polytope which aren't
            shared with other polytopes in list. (for ReLu nets)
        """

        num_constraints = self.ub_A.shape[0]
        facets = []
        reject_reasons = {}

        redundant_set = set()
        if check_feasible and use_clarkson:
            redundant_set = self.clarkson_redundancy_set(self.interior_point)
            print("Clarkson found %s redundant constraints" % len(redundant_set))

        for i in range(num_constraints):
            if i in redundant_set:
                continue

            facet = Face(self.ub_A, self.ub_b, [i], config=self.config)

            if upper_bound_dict is not None:
                reject_status, reason = facet.reject_via_upper_bound(upper_bound_dict)
                if reject_status:
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                    continue

            if check_feasible:
                facet.check_feasible()
            if not facet.is_feasible:                
                reject_reasons['infeasible'] = reject_reasons.get('infeasible', 0) + 1
            else:
                assert i not in redundant_set

            facet.check_facet()
            if facet.is_facet:
                # Check to see if facet is shared with a seen polytope                
                new_configs = facet.get_new_configs(net)
                new_configs_flat = utils.flatten_config(new_configs)
                shared_facet_bools = [new_configs_flat == other_config_flat
                                      for other_config_flat in seen_polytopes_dict]
                if not any(shared_facet_bools):
                    facets.append(facet)                

        return facets, reject_reasons



    def generate_facets_configs_2(self, seen_polytopes_dict, net, 
                                  check_feasible=False, upper_bound_dict=None,
                                  use_clarkson=True):
        """ Generates all (n-1) dimensional facets of polytope which aren't
            shared with other polytopes in list. (for ReLu nets)

            Strategies to calculate redundant constraints:
            (1) Remove unnecessary constraints with upper bound dict             
            (2) Remove redundant constraints with Clarksons 
            (3) Check upperbound + 
            (4) Check feasibility with uniform polytope sampling
            ORDER: (1) -> (3) -> (2) [and do (4) before (2)]
        """

        ######################################################################
        #   Step 0: Setup things we'll need                                  #
        ######################################################################
        num_constraints = self.ub_A.shape[0]
        relevant_facets = [] 

        # True for unnecessary constraints, false ow. Shared amongst facets
        removal_list = np.full(num_constraints, False) 
        reject_reasons = dict() # For logging        

        # Make all the facets first, and then only select the ones that matter
        base_facets = [Face(self.ub_A, self.ub_b, [i], 
                            config=self.config, removal_list=removal_list) 
                       for i in range(num_constraints)]
        ######################################################################
        #   Step 1: Remove unnecessary constraints with upper bound dict     #
        ######################################################################
        if upper_bound_dict is not None:
            for i, facet in enumerate(base_facets):
                status, reason = facet.reject_via_upper_bound(upper_bound_dict)
                if status:
                    removal_list[i] = True # automatically broadcasted to facets
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1


        ######################################################################
        #   Step 2: Secondary upper bound checks                             #
        ######################################################################        
        # TODO: maybe this is fast?

        ######################################################################
        #   Step Final-1: Remove redundant constraints with Clarksons        #
        ######################################################################

        redundant_list = self.clarkson_with_removal(removal_list)   

        clarkson_count = sum(redundant_list & (~removal_list))
        #print("Clarkson found %02d redundant constraints" % clarkson_count)
        reject_reasons['redundant'] = clarkson_count


        removal_list |= redundant_list # in-place 'OR'


        ######################################################################
        #   Step Final: Remove facets that have been seen before             #
        ######################################################################
        surviving_facets = []
        for i, facet in enumerate(base_facets):
            if removal_list[i]:
                continue 
            else:
                facet.check_facet() 
                if facet.is_facet:
                    new_configs = facet.get_new_configs(net)
                    new_configs_flat = utils.flatten_config(new_configs)

                    seen_this_facet = False
                    for other_config_flat in seen_polytopes_dict:
                        if new_configs == other_config_flat:
                            seen_this_facet = True
                            break 
                    if not seen_this_facet: 
                        surviving_facets.append(facet)

        return surviving_facets, reject_reasons


    def is_point_feasible(self, x):
        """ Returns True if point X satisifies all constraints, false otherwise
        """
        lhs = np.matmul(self.ub_A, x)
        bools = [lhs.reshape((lhs.size,)) <= self.ub_b][0]
        return all(bools)


    def linf_dist(self, x):
        """ Takes a feasible point x and returns the minimum l_inf distance to
            a boundary point of the polytope.

            If x is not feasible, return -1
        """

        # l_inf dist to each polytope is (b_i - a_i^T x) / ||a_i||_1
        # l_inf dist to all bounds is (b - Ax) * diag(1/||a_i||_1)

        slack = self.ub_b - self.ub_A.matmul(x)
        norms = np.diag(1.0 / np.linalg.norm(self.ub_A, ord=1, axis=1))
        dists = slack.matmul(norms)
        argmin_0 = np.argmin(dists)[0]
        return dists[argmin_0], argmin_0


    def to_comparison_form(self, copy=False):
        """ Converts this A,b into comparison form. If copy is true, returns a
            new object of this type """
        comp_A, comp_b = utils.comparison_form(self.ub_A, self.ub_b)
        self.ub_A = comp_A
        self.ub_b = comp_b
        return self


    def clarkson_with_removal(self, removal_list):
        """ Returns the set of indices of constraints which are redundant, done
            using Clarkson's algorithm: potentially much faster.

            By being careful, we can do better than regular Clarksons. Since we 
            have access to a 'removal list' where things that are removed come 
            from not having a projection within the upper-bound-ball.
            The idea here is that we don't have to use these removed constraints 
            in our LP, but we can use them in our rayshoot. 
        ARGS:
            removal_list : boolean np array of length m or None.
        RETURNS:
            boolean numpy array with True in the indices which correspond to 
            redundant constraints
        """

        ##################################################################
        #   Setups of things                                             #
        ##################################################################
        
        num_constraints = self.ub_A.shape[0]        
        active_indices = np.full(num_constraints, False)
        interior_point = self._interior_point()

        ##################################################################
        #   Loop through constraints                                     #
        ##################################################################

        for i in range(num_constraints):
            if removal_list[i]:
                # No need to check removal status of rejected constraints
                continue
            else:
                true_dix = lambda arr: [i for i, el in enumerate(arr) if el]
                redundant, opt_pt = self._clarkson_lp_removal(i, active_indices, 
                                                              removal_list)
                if not redundant:
                    # If not redundant, then a rayshoot returns an essential idx
                    active_idx = self._clarkson_rayshoot(interior_point, opt_pt)
                    if removal_list[active_idx]:
                       pass 
                       # If hits a removed index, no need to add to active idxs
                    else:
                        active_indices[active_idx] = True 
                        #redundant_constraints[i] = True
        return ~active_indices


    def _clarkson_lp_removal(self, i, active_indices, removal_list):
        """ For index i, does the linear_program:
        max <A[i], x> 
        st. A'X <= b'  [for A':= A[active_indices], b':=b[active_indices]]
            <A[i], x> <= b[i] + 1
        And then returns (is_redundant, optimal_point)
        Where is_redundant is True iff the optimal value is < b[i]
        """

        c = self.ub_A[i] 
        if not any(active_indices): # base case, nothing is redundant
            return (False, self.interior_point + c) # Just need the direction

        real_active_indices = active_indices & (~removal_list)

        selected_ub_A = self.ub_A[real_active_indices]
        selected_ub_b = self.ub_b[real_active_indices]

        constraint_to_check = self.ub_A[i]
        val_to_check = self.ub_b[i] + 1.0

        linprog_a = np.vstack((selected_ub_A, constraint_to_check))
        linprog_b = np.hstack((selected_ub_b, val_to_check))

        bounds = [(None, None) for _ in c]
        linprog_result = opt.linprog(-c,  # max A[i]x
                                     A_ub=linprog_a, 
                                     b_ub=linprog_b,
                                     bounds=bounds,
                                     method='interior-point')
        if False:
            print("-" * 40)
            print("ITER %02d" % i )
            print(-c)
            print(linprog_a)
            print(linprog_b)
            print("OBJECTIVE OUT", -linprog_result.fun)
            print("-" * 40)
        return (-linprog_result.fun <= self.ub_b[i], linprog_result.x)




    def clarkson_redundancy_set(self, interior_point=None):

        """ Returns the set of indices of constraints which are redundant, done
            using Clarkson's algorithm: potentially much faster.

            By being careful, we can do better than regular Clarksons. 
        ARGS:
            interior_point : np.Array - if not None, is a point for which
                             ub_A * interior_point < ub_b [strictly]. If None,
                             we'll find such an interior point (but this costs
                             1 LP)
            removal_list : boolean np array of length m or None.
        RETURNS:
            set of integers which correspond to constraint indices which are
            redundant
        """

        ######################################################################
        #   Setup stuff                                                      #
        ######################################################################
        
        num_constraints = self.ub_A.shape[0]        
        redundant_constraints = set()
        active_indices = np.full(num_constraints, False)
        # Need to find an interior point from which to shoot rays
        if interior_point is None:
            interior_point = self._interior_point()


        #######################################################################
        #   Loop through all constraints and find the ones that are redundant #
        #######################################################################
        
        for index_to_test in range(num_constraints):
            redundant, opt_pt = self._clarkson_lp(index_to_test, active_indices)
            if not redundant:
                rayshoot_idx = self._clarkson_rayshoot(interior_point, opt_pt)
                active_indices[rayshoot_idx] = True
            else:
                redundant_constraints.add(index_to_test)
        return redundant_constraints


    def _clarkson_rayshoot(self, interior_point, optimal_point):
        """ Does the ray shoot subroutine to return an 'essential index' of a
            constraint.
        ARGS:
            interior_point : np.array - vector that's strictly feasible
            optimal_point : np.array - sol'n to LP test, is output of
                            self._clarkson_lp
        RETURNS:
            index of the nonredundant constraint found here

        NOTES:

        Math: searching for smallest scalar c >0 such that
        A(z + c * d) <= b, but is tight for (at least) one index
        where z is the initial point, and d:= (optimal_point - initial_point)

        So min_i (b-Az) / (Ad) over the indices for which Ad > 0 should cut it
        """
        direction = (optimal_point - interior_point).squeeze()
        ub_a_direction = np.matmul(self.ub_A, direction).squeeze()
        numerator = (self.ub_b - np.matmul(self.ub_A, interior_point)).squeeze()
        min_index = None
        min_eps = float('inf')
        tolerance = 1e-10
        for idx in range(len(numerator)):
            # only care about things that are in the right direction
            # and numerically stable
            if ub_a_direction[idx] <= tolerance:
                continue
            else:
                val = numerator[idx] / ub_a_direction[idx]
                if val < min_eps:
                    min_eps = val
                    min_index = idx
        return min_index



    def _clarkson_lp(self, index_to_test, active_indices):
        """ Single redundancy check for a facet.
        ARGS:
            index_to_test :
            active_indices : boolean array of length (# constraints) which is
                             True only on the active indices
        RETURNS:
            (<bool>, optimal_point)
            If the constraint is redundant, then <bool> is True.
            Otherwise, retruns (False, the optimal point) that solves the LP
        """
        c = self.ub_A[index_to_test]
        if not any(active_indices): # base case
            return (False, self.interior_point + c)        
        bounds = [(0.0, 1.0) for _ in c]
        active_indices[index_to_test] = True # just temporarily

        selected_ub_A = self.ub_A[active_indices]
        
        # Need to find index of index_to_test in active_indices
        index_map = sum(active_indices[:index_to_test])
        
        selected_ub_b = self.ub_b[active_indices] # maybe need [0] first

        upper_bound = self.ub_b[index_to_test] # maybe need [0] first
        selected_ub_b[index_map] += 1.0
        linprog_result = opt.linprog(-c,
                                     A_ub=selected_ub_A,
                                     b_ub=selected_ub_b,
                                     bounds=bounds,
                                     method='interior-point')

        active_indices[index_to_test] = False # Reset back to normal

        if (linprog_result.status == 3) or (-linprog_result.fun > upper_bound):
            # Not a redundant constraint 
            return (False, linprog_result.x) # Maybe something weird here?

        else:
            return (True, linprog_result.x) 


    def _interior_point(self):
        """ Finds an interior point of this polytope if its full dimension
            and sets the self.interior_point attribute
        """

        # If we've already computed this, don't do it again
        if self.interior_point is not None:
            return self.interior_point

        # Otherwise, do the linprog:
        # Do max t
        # st. Ax + t <= b
        #          -t <= 0
        # and if has positive objective, then x is an interior point

        m, n = self.ub_A.shape

        c = np.zeros(n + 1)
        c[n] = -1.0

        A_ub = self.ub_A
        A_ub = np.hstack((A_ub, np.ones((m, 1))))
        bottom_row = np.zeros((1, n + 1))
        bottom_row[0][-1] = -1

        A_ub = np.vstack((A_ub, bottom_row))

        b_ub = self.ub_b

        try:
            b_ub = np.hstack((b_ub, np.array([[0]])))            
        except:
            b_ub = np.hstack((b_ub, np.array([0])))       

        bounds = [(None, None) for _ in c]
        linprog_result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                                     method='interior-point')

        if linprog_result.status in [0, 4]:
            if linprog_result.fun < 0:
                self.interior_point = linprog_result.x[:-1]
                assert np.max(np.matmul(self.ub_A, self.interior_point) - self.ub_b) < 0
            else:
                self.interior_point = 'not-full-dimension'
                # This is weird, and shouldn't ever happen in the polytope case
        elif linprog_result.status == 3:
            print("UNBOUNDED INTERIOR POINT:", m, n, c, linprog_result.x)
            self.interior_point = linprog_result.x[:-1]
        elif linprog_result.status == 2:
            self.interior_point = 'infeasible'

        if self.interior_point is None: 
            print("NO INTERIOR POINT?", linprog_result.status, 
                  linprog_result.fun, linprog_result.x)


        return self.interior_point


