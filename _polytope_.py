import utilities as utils
import torch
import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import copy
import pickle
import joblib

import time

#########################################################################
#                                                                       #
#               algorithm helpers                                       #
#                                                                       #
#########################################################################




##########################################################################
#                                                                        #
#                   POLYTOPE AND FACE CLASSES                            #
#                                                                        #
##########################################################################



class Polytope(object):
    def __init__(self, ub_A, ub_b, config=None, interior_point=None,
                 domain_bounds=None, _domain_structure=None):
        """ Polytopes are of the form Ax <= b
            with no strict equality constraints"""

        if isinstance(ub_A, torch.Tensor):
            ub_A = ub_A.cpu().detach().numpy().astype(np.double)
        if isinstance(ub_b, torch.Tensor):
            ub_b = ub_b.cpu().detach().numpy().astype(np.double)
        self.ub_A = utils.as_numpy(ub_A).astype(np.double)
        self.ub_b = utils.as_numpy(ub_b).squeeze().astype(np.double)
        self.config = config
        self.interior_point = interior_point

        # Domain constraints:
        # Either have no domain constraints
        # -- xor --
        # domain constraints cooked into ub_A already or not
        assert (domain_bounds == None) or (_domain_structure == None)
        num_constraints = self.ub_A.shape[1]

        # If need to cook in domain constraints into ub_A, then do so
        if domain_bounds is not None:
            domain_a, domain_b = self.convert_domain_bounds(domain_bounds,
                                                            self.ub_A.shape[1])
            self.ub_A = np.vstack((self.ub_A, domain_a))
            self.ub_b = np.hstack((self.ub_b, domain_b))

        # If domain constraints not provided, build them and set them
        if _domain_structure is None:
            _domain_structure = {'num_constraints': num_constraints,
                                 'domain_bounds': domain_bounds}
        self._domain_structure = _domain_structure



    @classmethod
    def from_polytope_dict(cls, polytope_dict, domain_bounds=None):
        """ Alternate constructor of Polytope object """
        return cls(polytope_dict['poly_a'],
                   polytope_dict['poly_b'],
                   config=polytope_dict['configs'],
                   domain_bounds=domain_bounds)


    @classmethod
    def convert_domain_bounds(cls, domain_bounds, n):
        """ Given domain bounds, which can take 3 forms, converts them into
            a constraint matrix and vector pair (A,b) such that the domain is
            {x | Ax <= b}
        ARGS:
            domain_bounds: None, [(lo, hi)], [(lo_1, hi_1), ..., (lo_n, hi_n)]
                - If None, no domain bounds
                - If a singleton list, has that bound for all dimensionss
                - If length n list, has custom bounds for every set
            n: int - dimension of polytope
        RETURNS:
            (None, None) or (A (array[2n][n]), b (array[2n]))
        """
        if domain_bounds is None:
            return (None, None)
        else:
            A_upper = np.eye(n)
            A_lower = -1 * np.eye(n)
            assert isinstance(domain_bounds, list)
            if len(domain_bounds) == 1:
                domain_low, domain_high = domain_bounds[0]
                b_lower = -1 * domain_low * np.zeros(n)
                b_upper = domain_high * np.zeros(n)
            else:
                b_lower = -1 * np.array([_[0] for _ in domain_bounds])
                b_upper = np.array([_[1] for _ in domain_bounds])

            domain_a = np.vstack((A_upper, A_lower))
            domain_b = np.hstack((b_upper, b_lower))

            return (domain_a, domain_b)




    ##########################################################################
    #       GENERATES FACETS FOR USE IN GEOCERT ALGORITHM                    #
    ##########################################################################
    """ The following methods all are variations on a theme where the function
        aims to output a list of Faces of this polytope (represented as Face
        objects) which can be checked for the following:
        (i)    - feasibility: Facets need to have a feasible region
        (ii)   - (n-1) dimension : Facets need to be FACES and not lower dim
        (iii)  - boundedness : If an upper bound is supplied, the facet needs to
                               have minimal projection closer than this upper
                               bound
        (iV)   - novelty : If a record of previously seen faces was supplied,
                           the output faces can't be in that record
    """

    def handle_single_facet(self, tight_idx, upper_bound_dict, facet=None):
        """ Function that takes a polytope description and a tight index and
            checks if this is rejectable or feasible and a facet.

            If the output[0] of this is True, then facet is feasible and dim (n-1)
        """
        if facet is None:
            facet = Face(self.ub_A, self.ub_b, [tight_idx], config=self.config,
                         _domain_structure=self._domain_structure)

        #print("PID: ", os.getpid(), " TIGHT LIST:", tight_idx)
        if upper_bound_dict is not None:
            reject_status, reason = facet.reject_via_upper_bound(upper_bound_dict)
            if reject_status:
                return (False, reason)

        facet.check_feasible()
        if not facet.is_feasible:
            return (False, 'infeasible')

        facet.check_facet()
        if not facet.is_facet:
            return (False, 'not-facet')

        return (True, facet)


    def scrub_seen_facets(self, facet_list, seen_dict, net):

        assert all(facet.is_facet and facet.is_feasible for facet in facet_list)

        output_facets = []
        num_seen = 0
        # remove all domain bounded facets first
        facet_list = [_ for _ in facet_list if _.facet_type != 'domain']
        for facet in facet_list:
            new_configs = utils.flatten_config(facet.get_new_configs(net))
            if new_configs in seen_dict:
                num_seen += 1
            else:
                output_facets.append(facet)

        return output_facets, num_seen



    def generate_facets_naive(self, check_feasible=False):
        """ Generates all (n-1) dimensional facets of polytope

        IMPLEMENTATION NOTES: Most naive implementation, very slow, not useful
                              in practice. Useful in Batch implementation though
        """
        num_constraints = self.ub_A.shape[0]
        facets = []
        for i in range(num_constraints):
            facet = Face(self.ub_A, self.ub_b, [i], config=self.config,
                         _domain_structure=self._domain_structure)
            if check_feasible:
                facet.check_feasible()
            facet.check_facet()

            if facet.is_facet:
                facets.append(facet)
        return facets


    def generate_facets_configs_parallel(self, seen_dict, net,
                                         upper_bound_dict=None,
                                         num_jobs=8):

        """ Does Facet checking in parallel using joblib to farm out multiple
            jobs to various processes (possibly on differing processors)
        IMPLEMENTATION NOTES:
            Uses joblib to farm out checking of feasibility, facet-ness,
            boundedness and then does the novelty check in-serial
        """
        ##################################################################
        #   Set up pool and offload all the work, and then merge results #
        ##################################################################
        maplist = [(i, upper_bound_dict) for i in range(self.ub_A.shape[0])]


        # handle_single_facet checks boundedness, feasibility, dimensionality
        map_fxn = joblib.delayed(utils.star_arg(self.handle_single_facet))
        outputs = joblib.Parallel(n_jobs=num_jobs)(map_fxn(_) for _ in maplist)

        reject_dict = {}
        surviving_facets = []
        for is_okay, output in outputs:
            if not is_okay: # if already rejected, record why
                reject_dict[output] = reject_dict.get(output, 0) + 1
            else:
                surviving_facets.append(output)

        facets, num_seen = self.scrub_seen_facets(surviving_facets, seen_dict,
                                                  net)
        if num_seen > 0:
            reject_dict['seen before'] = num_seen
        return facets, reject_dict

    def generate_facets_configs(self, seen_dict, net, upper_bound_dict=None,
                                use_clarkson=True, use_ellipse=False):
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

        # True for unnecessary constraints, false ow. Shared amongst facets
        removal_list = np.full(num_constraints, False)
        reject_reasons = dict() # For logging

        # Make all the facets first, and then only select the ones that matter
        base_facets = [Face(self.ub_A, self.ub_b, [i],
                            config=self.config, removal_list=removal_list,
                            _domain_structure=self._domain_structure)
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
        #   Step 2: Use min-vol enclosing ellipse to further reject          #
        ######################################################################
        if use_ellipse:
            current_facets = [facet for i, facet in enumerate(base_facets)
                              if not removal_list[i]]
            results = self.reject_via_ellipse(current_facets)
            for result in results:
                status, reason, i = result
                if status:
                    removal_list[i] = True # automatically broadcasted to facets
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1


        ######################################################################
        #   Step Final-1: Remove redundant constraints                       #
        ######################################################################

        if use_clarkson:
            redundant_list = self.clarkson_with_removal(removal_list)

            clarkson_count = sum(redundant_list & (~removal_list))
            print("Clarkson found %s redundant constraints" % clarkson_count)
            reject_reasons['redundant'] = clarkson_count
            removal_list |= redundant_list # in-place 'OR'

            # Everything output as not-redundant by Clarkson is feasible
            # But still need to check dimensionality of what's left
            for i, facet in enumerate(base_facets):
                if removal_list[i]:
                    continue
                facet.is_feasible = True
                facet.check_facet()
                if not facet.is_facet:
                    removal_list[i] = True
                    reject_reasons['not-facet'] =\
                                          reject_reasons.get('not-facet', 0) + 1

        else:
            print("NOT USING CLARKSON")
            for i, facet in enumerate(base_facets):
                if removal_list[i]:
                    continue
                # Check feasibility
                facet.check_feasible()
                if not facet.is_feasible:
                    reject_reasons['infeasible'] =\
                                         reject_reasons.get('infeasible', 0) + 1
                    removal_list[i] = True
                    continue
                # Check dimensionality
                facet.check_facet()
                if not facet.is_facet:
                    reject_reasons['not-facet'] =\
                                         reject_reasons.get('not_factet', 0) + 1
                    removal_list[i] = True
                    continue


        ######################################################################
        #   Step Final: Remove facets that have been seen before             #
        ######################################################################
        surviving_facets = [_ for i, _ in enumerate(base_facets) if not
                            removal_list[i]]
        facets, num_seen = self.scrub_seen_facets(surviving_facets, seen_dict,
                                                  net)
        if num_seen > 0:
            reject_reasons['seen before'] = num_seen
        return facets, reject_reasons



    ##########################################################################
    #                                                                        #
    #                       POLYTOPE HELPER METHODS                          #
    #                                                                        #
    ##########################################################################


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
        print("SHOULDNT BE DOING THIS ANYMORE")
        comp_A, comp_b = utils.comparison_form(self.ub_A, self.ub_b)
        self.ub_A = comp_A
        self.ub_b = comp_b
        return self


    ##########################################################################
    #   CLARKSON TECHNIQUES                                                  #
    ##########################################################################

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
                redundant, opt_pt = self._clarkson_lp_removal(i, active_indices,
                                                              removal_list)
                if not redundant:
                    # If not redundant, then a rayshoot returns an essential idx
                    active_idx = self._clarkson_rayshoot(interior_point, opt_pt)
                    if removal_list[active_idx]:
                       # If hits a removed index, no need to add to active idxs
                       pass
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
            #TODO: this doesn't work when self.interior_point is NONE
            return (False, self.interior_point + c) # Just need the direction

        real_active_indices = active_indices & (~removal_list)

        selected_ub_A = self.ub_A[real_active_indices]
        selected_ub_b = self.ub_b[real_active_indices]

        constraint_to_check = self.ub_A[i]
        val_to_check = self.ub_b[i] + 1.0

        linprog_a = np.vstack((selected_ub_A, constraint_to_check))
        linprog_b = np.hstack((selected_ub_b, val_to_check))

        bounds = [(None, None) for _ in c]
        try:
            linprog_result = solvers.lp(matrix(-c), matrix(linprog_a),
                                        matrix(linprog_b), solver='glpk')
            return (linprog_result['primal objective'] < self.ub_b[i],
                    np.array(linprog_result['x']).squeeze())

        except:
            linprog_result = opt.linprog(-c,  # max A[i]x
                                         A_ub=linprog_a,
                                         b_ub=linprog_b,
                                         bounds=bounds,
                                         method='interior-point')
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


        try:
            linprog_result = solvers.lp(matrix(c), matrix(A_ub), matrix(b_ub),
                                        solver='glpk')
            if linprog_result['status'] == 'optimal':
                if linprog_result['primal objective'] < 0:
                    self.interior_point = np.array(linprog_result['x'])[:-1].squeeze()
                else:
                    self.interior_point = 'not-full-dimension'
            else:
                print("LINPROG STATUS", linprog_result['status'])
                self.interior_point = 'infeasible'
        except ValueError: # If cvxopt fails...
            linprog_result = opt.linprog_result(c, A_ub=A_ub, b_ub=b_ub,
                                                bounds=bounds,
                                                method='interior-point')
            if linprog_result.fun < 0:
                self.interior_point = linprog_result.x[:-1].squeeze()
            else:
                self.interior_point = 'not-full-dimension'

        return self.interior_point


    def reject_via_ellipse(self, facets):
        ''' Finds non-redundant constraints by finding MVIE and then checking which constriants are tight
            at the boundary of the ellipse + projecting onto constraint from point if not tight

            Removes redundant constraint by finding an approximation to the
            minimum volume circumscribing ellipsoid. Done by solving maximum
            volume inscribed ellipsoid and multiplying by dimenion n. Ellipse
            E(P, c) is defined by Pos. Def. matrix P and center c.

            returns:   'redundant_list' [True if red. | False if non-red | None if unknown]
        '''
        # Find min. vol. inscribed ellipse
        P, c = utils.MVIE_ellipse(self.ub_A, self.ub_b)
        P = np.asarray(P)
        c = np.asarray(c)

        # Approximate max. vol. circum. ellipse (provable bounds polytope)
        n = np.shape(self.ub_A)[1]
        P_outer = np.multiply(n, P)

        # Remove Redundant constraints
        # constraint a_i redundant if below holds:
        # max <a_i, y> <= b_i for all y in E(P, c))
        #
        # equivalent to: ||P.T*a_i||_2 + a_i.T*c <= b_i
        # (max has a closed form)

        results = []
        for facet in facets:
            # Find Redundant constraints
            lhs = np.linalg.norm(np.matmul(P_outer.T, self.ub_A[facet.tight_list].T)) + np.dot(
                self.ub_A[facet.tight_list], c)
            rhs = self.ub_b[facet.tight_list]
            if lhs <= rhs:
                results.append((True, 'ellipse_upper_bound', facet.tight_list))

        return results




##############################################################################
#                                                                            #
#                               FACE CLASS                                   #
#                                                                            #
##############################################################################



class Face(Polytope):
    def __init__(self, poly_a, poly_b, tight_list, config=None,
                 removal_list=None, domain_bounds=None, _domain_structure=None):
        super(Face, self).__init__(poly_a, poly_b, config=config,
                                   domain_bounds=domain_bounds,
                                   _domain_structure=_domain_structure)
        self.poly_a = poly_a
        self.poly_b = poly_b
        self.a_eq = self.poly_a[tight_list]
        self.b_eq = self.poly_b[tight_list]
        self.tight_list = tight_list
        self.is_feasible = None
        self.is_facet = None
        self.interior = None
        self.removal_list = removal_list

        # Set facet type
        assert self._domain_structure is not None
        num_constraints = self._domain_structure['num_constraints']
        if any(tight_el >= num_constraints for tight_el in tight_list):
            self.facet_type = 'domain'
        else:
            self.facet_type = 'facet'


    def check_feasible(self):
        """ Checks if this polytope is feasible and stores the result
        Simply checks the linear program:
            min 0
            st. Ax <= b
                A_eq x = b_eq

        """
        if self.is_feasible is not None:
            return self.is_feasible
        # Set up feasibility check Linear program
        c = np.zeros(self.poly_a.shape[1])

        A_ub = self.poly_a
        b_ub = self.poly_b

        cvxopt_out = solvers.lp(matrix(c), matrix(A_ub), matrix(b_ub),
                                A=matrix(self.a_eq), b=matrix(self.b_eq),
                                solver='glpk')

        self.is_feasible = (cvxopt_out['status'] == 'optimal')
        return self.is_feasible


    def check_facet(self):
        """ Checks if this polytope is a (n-1) face and stores the result"""

        # if already computed, return that
        if self.is_facet is not None:
            return self.is_facet

        # if not feasible, then return False
        if self.is_feasible is not None and not self.is_feasible:
            self.is_facet = False
            return self.is_facet

        m, n = self.poly_a.shape
        # Dimension check of (n-1) facet
        # Do min 0
        # st. Ax + t <= b
        #          t  > 0
        # and if is_feasible, then good
        c = np.zeros(n + 1)
        # c[-1] = -1

        # SPEED UP WITH REMOVAL LIST!
        if self.removal_list is not None:
            map_idx = sum(~self.removal_list[:self.tight_list[0]])
            saved_row = self.poly_a[self.tight_list[0]]

            new_poly_a = self.poly_a[~self.removal_list]
            new_poly_a = np.vstack((new_poly_a, np.zeros((1, new_poly_a.shape[1]))))
            new_poly_a = np.hstack((new_poly_a, np.ones((new_poly_a.shape[0], 1))))
            new_poly_a[-1][-1] = -1
            new_poly_a[map_idx, -1] = 0     # remove affect of t on tight constraints

            new_poly_b = self.poly_b[~self.removal_list]
            new_poly_b = np.hstack((new_poly_b, 0))
        else:
            new_poly_a = np.ones([m, n+1])
            new_poly_a[:, :-1] = self.poly_a
            new_poly_a[self.tight_list, -1] = 0
            new_poly_b = self.poly_b
        #------



        bounds = [(None, None) for _ in range(n)] + [(1e-11, None)]
        lower_bound_a = np.zeros(n + 1)
        lower_bound_a[-1] = -1
        lower_bound_b = 1e-7
        new_poly_a = np.vstack((new_poly_a, lower_bound_a))
        new_poly_b = np.hstack((new_poly_b, lower_bound_b))
        # Map indices real quick:
        m2, n2 = self.a_eq.shape
        a_eq_new = np.zeros([m2, n+1])
        a_eq_new[:, :-1] = self.a_eq


        # Setup and solve the linear program using scipy
        linprog_result = solvers.lp(matrix(c), matrix(new_poly_a),
                                    matrix(new_poly_b),
                                    A=matrix(a_eq_new), b=matrix(self.b_eq),
                                    solver='glpk')
        if linprog_result['status'] == 'optimal':
            self.is_facet = True
            self.interior = np.array(linprog_result['x'][0:-1])
        else:
            print("WHAT??? ", linprog_result['status'])
        return self.is_facet


    def _same_hyperplane(self, other):
        """ Given two facets, checks if they lie in different hyperplanes
            Returns True if they lie in the same hyperplane
        """
        # if either is not a facet, then return False
        if not (self.check_facet() and other.check_facet()):
            return False
        # if they lie in different hyperplanes, then return False
        self_tight = self.tight_list[0]
        self_a = self.poly_a[self_tight, :]
        self_b = self.poly_b[self_tight]

        other_tight = other.tight_list[0]
        other_a = other.poly_a[other_tight, :]
        other_b = other.poly_b[other_tight]

        return utils.is_same_hyperplane_nocomp(self_a, self_b, other_a, other_b)


    def _same_tight_constraint(self, other):
        """ Given two facets, checks if their tight constraints are the same
            Returns True if they are the same
        """
        # if either is not a facet, then return False
        if not (self.check_facet() and other.check_facet()):
            return False
        # if they lie in different hyperplanes, then return False
        self_tight = self.tight_list[0]
        self_a = self.poly_a[self_tight, :]
        self_b = self.poly_b[self_tight]

        other_tight = other.tight_list[0]
        other_a = other.poly_a[other_tight, :]
        other_b = other.poly_b[other_tight]

        return utils.is_same_tight_constraint(self_a, self_b, other_a, other_b)


    def check_same_facet_pg(self, other):
        """ Checks if this facet is the same as the other facet. Assumes
            that self and other are perfectly glued if they intersect at all.
            Uses LP to check if intersection of facets is (n-1) dimensional.
        """
        # if either is not a facet, then return False
        if not (self.check_facet() and other.check_facet()):
            return False

        if not self._same_hyperplane(other):
            return False

        # now just return True if their intersection is dimension (n-1)

        new_tight_list = np.add(other.tight_list, self.poly_b.shape)

        new_face = Face(np.vstack((self.poly_a, other.poly_a)),
                        np.hstack((self.poly_b, other.poly_b)),
                        tight_list=np.hstack((self.tight_list, new_tight_list)))
        return new_face.check_facet()


    def check_same_facet_pg_slow(self, other):
        """ Checks if this facet is the same as the other facet. Assumes
            that self and other are perfectly glued if they intersect at all

            Method uses PyPi library 'polytope' to compare vertices of the
            faces
        """

        # if either is not a facet, then return False
        if not (self.check_facet() and other.check_facet()):
            return False

        if not self._same_hyperplane(other):
            return False

        P1 = utils.Polytope_2(self.ub_A, self.ub_b)
        P2 = utils.Polytope_2(other.ub_A, other.ub_b)

        V1 = utils.ptope.extreme(P1)
        V2 = utils.ptope.extreme(P2)

        V1_ = []
        if(V1 is not None and V2 is not None):
            for vertex in V1:
                flag = utils.fuzzy_equal(np.matmul(self.a_eq, vertex), self.b_eq)
                if flag:
                    V1_.append(vertex)
            V2_ = []
            for vertex in V2:
                flag = utils.fuzzy_equal(np.matmul(other.a_eq, vertex), other.b_eq)
                if flag:
                    V2_.append(vertex)

            if len(V1_) == len(V2_):
                flags = []
                for vertex in V1_:
                    flags_2 = []
                    for vertex_2 in V2_:
                        if np.allclose(vertex, vertex_2, atol=1e-6):
                            flags_2.append(True)
                        else:
                            flags_2.append(False)
                    flags.append(any(flags_2))
                return all(flags)
            else:
                return False
        else:
            return  False

    def check_same_facet_config(self, other):
        #TODO: fix numerical issues (shared facets aren't being eliminated)

        """ Potentially faster technique to check facets are the same
            The belief here is that if both (self, other) are facets, with their
            neuron configs specified, then if they have the same hyperplane and
            have config hamming distance 1 (plus condition explained below),
            then they are the same facet

            must account for the case where each polytope can be simulatenously
            glued (not perfectly glued) to more than one hyperplane. two
            faces can come from polytopes with ReLu hamming distance one
            and yet their intersection is not a (n-1) dim. face
        """
        if not (self.check_facet() and other.check_facet()):
            return False

        if not self._same_hyperplane(other):
            return False

        if self.config is None or other.config is None:
            return self.check_same_facet_pg(other)

        ReLu_distance_check = (utils.config_hamming_distance(self.config, other.config) == 1)


        return ReLu_distance_check and not self._same_tight_constraint(other)

    def get_new_configs(self, net):
        ''' Function takes original ReLu configs and flips the activation of
            the ReLu at index specified in 'tight_boolean_configs'.
        '''



        # New and improved version:
        # Looks at the tight list, maps the tight index to the 2d
        # coordinate in the config and flips the index_map
        assert self.interior is not None

        orig_configs = self.config
        tight_idx = self.tight_list[0]
        flip_i, flip_j = utils.index_to_config_coord(orig_configs, tight_idx)
        new_configs = copy.deepcopy(orig_configs)
        new_configs[flip_i][flip_j] = int(1 - new_configs[flip_i][flip_j])

        return new_configs



        #TODO: this could be improved if tight index was consistent with
        # order of ReLu configs, but it isn't for some reason?
        # Solution: find tight ReLu activation by running interior pt through net
        orig_configs = self.config
        pre_relus, post_relus = net.relu_config(torch.Tensor(self.interior))
        tight_boolean_configs = [utils.fuzzy_equal(elem, 0.0, tolerance=1e-9)
                         for activations in pre_relus for elem in activations]

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

    def linf_dist(self, x):
        #TODO: this method doesn't  seem to always correctly find the projection onto a facet

        """ Returns the l_inf distance to point x using LP
            as well as the optimal value of the program"""

        # set up the linear program
        # min_{t,v} t
        # s.t.
        # 1)  A(x + v) <= b        (<==>)    Av <= b - Ax
        # 2)  A_eq(x + v) =  b_eq  (<==>)    A_eq v = b_eq - A_eq x
        # 3)  v <= t * 1           (<==>)    v_i - t <= 0
        # 4) -v <= t * 1           (<==>)   -v_i - t <= 0

        n = np.shape(self.poly_a)[1]
        x = utils.as_numpy(x).reshape(n, -1)

        # optimization variable is [t, v]
        m = self.poly_a.shape[0]
        c = np.zeros(n+1)
        c[0] = 1

        # Constraint 1
        constraint_1a = np.hstack((np.zeros((m, 1)), self.poly_a))
        constraint_1b = self.poly_b - np.matmul(self.poly_a, x)[:, 0]

        # Constraint 2
        constraint_2a = constraint_1a[self.tight_list, :]
        constraint_2b = self.poly_b[self.tight_list] - np.matmul(self.poly_a[self.tight_list, :], x)


        # Constraint 3
        constraint_3a = np.hstack((-1*np.ones((n, 1)), np.identity(n)))
        constraint_3b = np.zeros(n)

        # Constraint 4
        constraint_4a = np.hstack((-1*np.ones((n, 1)), -1*np.identity(n)))
        constraint_4b = np.zeros(n)

        ub_a = np.vstack((constraint_1a, constraint_3a, constraint_4a))
        ub_b = np.hstack((constraint_1b, constraint_3b, constraint_4b))


        bounds = [(0, None)] + [(None, None) for _ in range(len(x))]

        # Solve linprog

        cvxopt_out = solvers.lp(matrix(c), matrix(ub_a), matrix(ub_b),
                               A=matrix(constraint_2a), b=matrix(constraint_2b),
                               solver='glpk')
        if cvxopt_out['status'] == 'optimal':
            return cvxopt_out['primal objective'], \
                   (x + np.array(cvxopt_out['x'])[1:])
        else:
            raise Exception("LINF DIST FAILED?")


    def l2_dist(self, x):
        """ Returns the l_2 distance to point x using LP
            as well as the optimal value of the program"""

        # set up the quadratic program
        # min_{v} v^T*v
        # s.t.
        # 1)  A(x + v) <= b        (<==>)    Av <= b - Ax
        # 2)  A_eq(x + v) =  b_eq  (<==>)    A_eq v = b_eq - A_eq x

        n = np.shape(self.poly_a)[1]
        x = utils.as_numpy(x).reshape(n, -1)

        P = matrix(np.identity(n))
        G = matrix(self.poly_a)
        h = matrix(self.poly_b - np.matmul(self.poly_a, x)[:, 0])
        q = matrix(np.zeros([n, 1]))
        A = matrix(self.a_eq)
        b = matrix(self.b_eq - np.matmul(self.a_eq, x))

        quad_program_result = solvers.qp(P, q, G, h, A, b)

        if quad_program_result['status'] == 'optimal' or quad_program_result['status'] == 'unknown':
            v = np.array(quad_program_result['x'])
            return np.linalg.norm(v), x + v.reshape(n,-1)
        else:
            raise Exception("QPPROG FAILED: " + quad_program_result['status'])



    def l2_projection(self, x, lp_norm):
        """ Computes the l2 distance between point x and the hyperplane that
            this face lies on. Serves as a lower bound to the l2 distance
            {y | <a,y> =b}.
            This is equivalent to (b - <a,x>)/ ||a||_*
            for ||.||_* being the dual norm
        """
        dual_norm = {'l_2': None, 'l_inf': 1}[lp_norm]

        eq_a = self.a_eq
        eq_b = self.b_eq
        return (eq_b - np.matmul(eq_a, x)) / np.linalg.norm(eq_a, ord=dual_norm)

    def reject_via_upper_bound(self, upper_bound_dict):
        """ Takes in a dict w/ keys:
            {'upper_bound': float of upper bound of dist to decision bound,
             'x': point we're verifying robustness for,
             'norm': ['l_2'| 'l_inf'], which norm we care about
             (optional) 'hypercube': [lo, hi] If present, describes
                         the dimension of the [lo,hi]^n hypercube that is the
                         domain of our problem

            Returns TRUE if we can reject this facet, FALSE O.w.
        """
        lp_norm = upper_bound_dict['lp_norm']
        x = upper_bound_dict['x']
        dual_norm = {'l_2': None, 'l_inf': 1}[lp_norm]
        eq_a = self.a_eq
        eq_b = self.b_eq.item()
        proj = (eq_b - np.dot(x, eq_a.T)) / np.linalg.norm(eq_a, ord=dual_norm)

        if proj > upper_bound_dict['upper_bound']:
            return True, 'upper_bound'

        if 'hypercube' in upper_bound_dict:
            lo, hi = upper_bound_dict['hypercube']
            central_point = np.ones_like(x.cpu().numpy()) * ((lo + hi) / 2.0)
            central_point = central_point.squeeze()
            overshot = -1 if (np.matmul(eq_a, central_point).item() >= eq_b) else 1
            corner_signs = (np.sign(eq_a) * overshot).T.squeeze()
            corner = central_point + corner_signs * (lo + hi) / 2.0
            overshot_2 = -1 if (np.matmul(eq_a, corner).item() >= eq_b) else 1

            if overshot == overshot_2:
                return True, 'hypercube'


        return False, None


    def get_inequality_constraints(self):

        """ Converts equality constraints into inequality constraints
            and gathers them all together
        """

        A = np.vstack((self.poly_a, np.multiply(self.a_eq, -1.0)))
        b = np.hstack((self.poly_b, np.multiply(self.b_eq, -1.0)))

        return A, b

