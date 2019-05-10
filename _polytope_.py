import utilities as utils
import torch
import numpy as np
import scipy.optimize as opt
from collections import defaultdict
from mosek import iparam


from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
solvers.options['mosek'] = {iparam.log: 0,
                            iparam.max_num_warnings: 0}
import copy

import joblib

import time


################################################################################
#                                                                              #
#                           POLYTOPE CLASS                                     #
#                                                                              #
################################################################################


class Polytope(object):


    ######################################################################
    #                                                                    #
    #                       Polytope Initializations                     #
    #                                                                    #
    ######################################################################

    def __init__(self, ub_A, ub_b, config=None, interior_point=None,
                 domain=None, dead_constraints=None):
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
        self.domain = domain # is a domain object now
        self.dead_constraints = dead_constraints


    @classmethod
    def from_polytope_dict(cls, polytope_dict, domain=None,
                           dead_constraints=None):
        """ Alternate constructor of Polytope object """
        return cls(polytope_dict['poly_a'],
                   polytope_dict['poly_b'],
                   config=polytope_dict['configs'],
                   domain=domain,
                   dead_constraints=dead_constraints)



    ######################################################################
    #                                                                    #
    #                   FACET GENERATION TECHNIQUES                      #
    #                                                                    #
    ######################################################################
    def generate_facets_naive(self, check_feasible=False):
        """ Generates all (n-1) dimensional facets of polytope

        NOTES: Most naive implementation, and uses no heuristics.
               Doesn't care about domain, is very slow, and not useful for
               verifying nets. Useful in Batch implementation though
        """
        num_constraints = self.ub_A.shape[0]
        facets = []

        for i in range(num_constraints):
            facet = Face(self.ub_A, self.ub_b, [i])
            if check_feasible:
                check = facet.check_feasible()
                if check: facets.append(facet)
        return facets



    def generate_facets_configs_parallel(self, seen_dict):
        """ Does Facet checking in parallel using joblib to farm out multiple
            jobs to various processes (possibly on differing processors)
        NOTES:
            Main technique using all the heuristics that don't solve any LPs and
            then farms out the things that need LP's to be done in parallel.

            Uses joblib to farm out checking of feasibility, facet-ness,
            boundedness and then does the novelty check in-serial

        ORDER OF OPERATIONS:
            (1): Removes all possible facets that are tight on
                 'dead constraints'
            (2): Removes all possible facets that don't have a feasible
                 point in the box constraint
            (3): Removes all possible facets that have minimal projection
                 outside the upper bound
            (4): Farms out the remaining facets for feasibility/interior point
                 checks
        """

        ######################################################################
        #   First set things up                                              #
        ######################################################################
        num_facets = self.ub_A.shape[0]
        potential_facets = [_ for _ in range(num_facets)]
        reject_dict = defaultdict(int)

        domain_feasible = self.domain.feasible_facets(self.ub_A, self.ub_b)

        ######################################################################
        #   Step 1: Remove facets that are tight on dead constraints         #
        #   Step 2: Remove facets that are infeasible within domain          #
        ######################################################################
        new_potential_facets = []
        for idx in potential_facets:
            if self._is_dead(idx):
                reject_dict['dead_constraints'] += 1
            elif idx not in domain_feasible:
                reject_dict['domain_infeasible'] += 1
            else:
                new_potential_facets.append(idx)
        potential_facets = new_potential_facets

        ######################################################################
        #   Step 3: Remove facets that aren't feasible within upper bound    #
        ######################################################################
        upper_bound_proj = self.domain.minimal_facet_projections(self.ub_A,
                                                                 self.ub_b)
        new_potential_facets = []
        for idx in potential_facets:
            if idx not in upper_bound_proj:
                reject_dict['upper_bound'] += 1
            else:
                new_potential_facets.append(idx)
        potential_facets = new_potential_facets


        ######################################################################
        #   Step 4: Construct the facet objects                              #
        ######################################################################
        facets = [self.facet_constructor(idx) for idx in potential_facets]

        #####################################################################
        #   Step 5: Remove all facets that have been seen before            #
        #####################################################################
        facets_to_check, num_seen = self.scrub_seen_facets(facets, seen_dict)
        if num_seen > 0:
            reject_dict['seen before'] += num_seen

        return facets_to_check, dict(reject_dict)




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
        num_constraints = self._domain_structure['num_constraints']

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
        #           and dead constraints                                     #
        ######################################################################
        if upper_bound_dict is not None:
            for i, facet in enumerate(base_facets):
                if self._is_dead(i):
                    reject_reasons['dead_constraints'] =\
                         reject_reasons.get('dead_constraints', 0) + 1
                    # removal_list[i] = True # <--- Reason about this later
                    continue

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
    #                            HELPER METHODS                              #
    #                                                                        #
    ##########################################################################

    def _is_dead(self, i):
        """ Just a quick check for deadness of a constraint. We don't need
            to build faces for neurons that we know to be fixed to be on or off
        ARGS:
            i : int - constraint index
        RETURNS:
            False if no known dead constraints or we're not sure about this
            constraint. True o.w.
        """
        return (self.dead_constraints is not None and
                self.dead_constraints[i])

    def facet_constructor(self, tight_idx):
        return Face(self.ub_A, self.ub_b, [tight_idx], config=self.config,
                     domain=self.domain, facet_type='facet')

    def facet_feasibility_check(self, tight_idx, facet=None):
        """ Checks if a particular facet (with specified tight_idx) is feasible
        """
        if facet is None:
            facet = Face(self.ub_A, self.ub_b, [tight_idx], config=self.config,
                         domain=self.domain,
                         dead_constraints=self.dead_constraints)
        # facet.check_facet_feasible()
        return (True, facet) # TEST TEST TEST
        if not facet.is_feasible:
            return (False, 'infeasible')

        if not facet.is_facet:
            return (False, 'not-facet')

        return (True, facet)


    def scrub_seen_facets(self, facet_list, seen_dict):
        """ Removes facets that we've seen before """

        # assert all(facet.is_facet and facet.is_feasible for facet in facet_list)
        output_facets = []
        num_seen = 0
        # remove all domain bounded facets first
        for facet in facet_list:
            new_configs = utils.flatten_config(facet.get_new_configs())
            if new_configs in seen_dict:
                num_seen += 1
            else:
                output_facets.append(facet)

        return output_facets, num_seen


    ##########################################################################
    #                                                                        #
    #                           JUNK METHODS???                              #
    #                                                                        #
    ##########################################################################


    def is_point_feasible(self, x):
        """ Returns True if point X satisifies all constraints, false otherwise
        """
        lhs = np.matmul(self.ub_A, x).reshape((np.shape(self.ub_A)[0],))
        less_than = [lhs <= self.ub_b][0]
        _, fuzzy_equal = utils.fuzzy_vector_equal_plus(lhs, self.ub_b)
        return all(less_than | fuzzy_equal)

    def is_point_fesible_plus(self, x):
        """ Same as above, but also returns extra information
        """
        lhs = np.matmul(self.ub_A, x)
        bools = [lhs.reshape((lhs.size,)) <= self.ub_b][0]
        return all(bools), bools


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

    def redund_removal_pgd_l2(self, t, x_0):
        ''' Removes redundant constraint based on PGD based upper bound 't'
            on the largest l_2 ball from initial point x_0

            modifies:   'self.redundant'
        '''

        # Remove Redundant constraints
        # (constraint a_i redundant if ||projection(x_0, a_i)-x_0||_2 >= t
        potent_faces = [Face(self.ub_A, self.ub_b, tight_list=[i]) for i in range(0, np.shape(self.ub_A)[0])]
        is_redund = lambda face: (np.linalg.norm(face.get_hyperplane_proj_2(x_0)-x_0) >= t)

        for face in potent_faces:
            if is_redund(face):
                self.redundant[face.tight_list[0]] = True


    def redund_removal_approx_ellipse(self):
        ''' Removes redundant constraint by finding an approximation to the
            minimum volume circumscribing ellipsoid. Done by solving maximum
            volume inscribed ellipsoid and multiplying by dimenion n. Ellipse
            E(P, c) is defined by Pos. Def. matrix P and center c.

            modifies:   'self.redundant'
        '''

        # Find min. vol. inscribed ellipse
        P, c = utils.MVIE_ellipse(self.ub_A, self.ub_b)
        P = np.asarray(P)
        c = np.asarray(c)


        # Approximate max. vol. circum. ellipse (provable bounds polytope)
        P = np.multiply(self.n, P)


        # Remove Redundant constraints
        # constraint a_i redundant if below holds:
        # max <a_i, y> <= b_i for all y in E(P, c))
        #
        # equivalent to: ||P.T*a_i||_2 + a_i.T*c <= b_i
        # (max has a closed form)

        potent_faces = [Face(self.ub_A, self.ub_b, tight_list=[i]) for i in range(0, np.shape(self.ub_A)[0])]

        for face in potent_faces:
            lhs = np.linalg.norm(np.matmul(P.T, face.ub_A[face.tight_list].T))+np.dot(face.ub_A[face.tight_list], c)
            rhs = face.ub_b[face.tight_list]
            if lhs <= rhs:
                self.redundant[face.tight_list[0]] = True



    def essential_constraints_ellipse(self):
        ''' Finds non-redundant constraints by finding MVIE and then checking which constriants are tight
            at the boundary of the ellipse + projecting onto constraint from point if not tight

            modifies:   'self.redundant'
        '''

        # Find min. vol. inscribed ellipse
        P, c = utils.MVIE_ellipse(self.ub_A, self.ub_b)
        P = np.asarray(P)
        c = np.asarray(c)

        # Find non-Redundant constraints
        # solving: max <a_i, y> <= b_i for all y in E(P, c))
        # y^* = P.T*a_i/||P.T*a_i||_2
        potent_faces = self.get_potent_facets()

        for face in potent_faces:
            project = np.matmul(P.T, face.ub_A[face.tight_list].T)
            project = project/np.linalg.norm(np.matmul(P.T, face.ub_A[face.tight_list].T))

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
                 domain=None, dead_constraints=None, removal_list=None,
                 facet_type=None):
        super(Face, self).__init__(poly_a, poly_b, config=config, domain=domain,
                                   dead_constraints=dead_constraints)
        self.a_eq = self.ub_A[tight_list]
        self.b_eq = self.ub_b[tight_list]
        self.tight_list = tight_list
        self.is_feasible = None
        self.is_facet = None
        self.interior = None
        self.removal_list = removal_list

        assert facet_type in [None, 'decision', 'facet']
        self.facet_type = facet_type



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
        c = np.zeros(self.ub_A.shape[1])

        A_ub = self.ub_A
        b_ub = self.ub_b

        cvxopt_out = solvers.lp(matrix(c), matrix(A_ub), matrix(b_ub),
                                A=matrix(self.a_eq), b=matrix(self.b_eq),
                                solver='glpk')

        self.is_feasible = (cvxopt_out['status'] == 'optimal')
        return self.is_feasible


    def check_facet_feasible(self):
        """ Solves an LP to find an interior point of the polytope
            Returns (False, 'infeasible') if infeasible
            Returns (False, 'not-facet') if no interior point
            Returns (True, None) otherwise
        """
        ######################################################################
        #   If already computed, just return that result                     #
        ######################################################################

        if (self.is_feasible is not None and
            self.is_feasible and
            self.is_facet is not None and
            self.is_facet):
            return True

        ######################################################################
        #   If not already computed, find an interior point using an LP      #
        ######################################################################

        # maximize t
        # st Ax + t <= b
        #         t >= 0
        A = self.ub_A
        b = self.ub_b

        # If box constraints over domain, add these domain constraints
        if self.domain.box_low is not None:
            assert len(self.tight_list) == 1
            tight_idx = self.tight_list[0]
            d_A, d_b = self.domain.box_constraints()
            # Append domain constraints
            A = np.vstack((A, d_A))
            b = np.hstack((b, d_b))


        # Add variable t
        A = np.vstack((A, np.zeros(A.shape[1]))) # stack 0 row onto A
        b = np.hstack((b, np.zeros(1)))

        # Modify contraints such that Ax + t <= b
        # except for the tight constraint
        # (also add t >= 0)
        A = np.hstack((A, np.ones((A.shape[0], 1))))
        A[self.tight_list, -1] = 0 # tight constraint has no t
        A[-1][-1] = -1 # t >= 0

        a_eq = A[self.tight_list]

        # Build constraint set
        c = np.zeros(A.shape[1])
        c[-1] = -1

        linprog_result = solvers.lp(matrix(c), matrix(A), matrix(b),
                            A=matrix(a_eq), b=matrix(self.b_eq), solver='glpk')
        if linprog_result['status'] == 'optimal':
            self.is_feasible = True
            if linprog_result['primal objective'] < 0:
                self.is_facet = True
                self.interior = np.array(linprog_result['x'][0:-1])
        elif linprog_result['status'] == 'primal infeasible':
            self.is_feasible = self.is_facet = False

        return self.is_facet


    def get_new_configs(self):
        ''' Function takes original ReLu configs and flips the activation of
            the ReLu at index specified in 'tight_boolean_configs'.
        '''

        # New and improved version:
        # Looks at the tight list, maps the tight index to the 2d
        # coordinate in the config and flips the index_map
        #assert self.interior is not None

        orig_configs = self.config
        tight_idx = self.tight_list[0]
        flip_i, flip_j = utils.index_to_config_coord(orig_configs, tight_idx)
        new_configs = copy.deepcopy(orig_configs)
        new_configs[flip_i][flip_j] = int(1 - new_configs[flip_i][flip_j])

        return new_configs

    def fast_domain_check(self):
        """ Does the fast checks to see if we can reject this facet based on
            the domain.
        Returns:
            True if we cannot reject this without checking an LP/QP
            False if we can for sure reject this
        """
        domain = self.domain
        # Do checks to see if this hyperplane intersects domain
        domain_feasible = domain.feasible_facets(self.ub_A, self.ub_b,
                                               indices_to_check=self.tight_list)

        if len(domain_feasible) == 0:
            return False

        # Do checks to see if this hyperplane has projection inside ball
        projection = domain.minimal_facet_projections(self.ub_A, self.ub_b,
                                               indices_to_check=self.tight_list)
        return len(projection) > 0



    def linf_dist(self, x):
        """ Computes the l_infinity distance to point x using LP
            The linear program is as follows

            min_{t, v} t
            such that
            1) A(x + v) <= b        (<==>)  Av <= b - Ax
            2) -t <= v_i <= t       (<==>)  v_i - t <= 0  AND -v_i -t <= 0
            3) (x + v) in Domain
            5) t <= upper_bound
            4) A_eq(x + v) = b_eq   (<==>)


            so if A has shape (m,n) and domain constraints have shape (d, n)
            - (n + 1) variables
            - (m + 2n + d) inequality constraints
            - 1 equality constraint
        """

        ######################################################################
        #       Setup things needed for linprog                              #
        ######################################################################

        m, n = self.ub_A.shape
        zero_m_col = np.zeros((m, 1))
        zero_n_col = np.zeros((n, 1))
        x_row = utils.as_numpy(x).squeeze()
        x_col = x_row.reshape(n, 1)


        ######################################################################
        #       Build constraints row by row                                 #
        ######################################################################

        # VARIABLES ARE (v, t)
        a_constraints = []
        b_constraints = []

        # Constraint 1 has shape (m, n+1)
        constraint_1a = np.hstack((self.ub_A, zero_m_col))
        constraint_1b = (self.ub_b - self.ub_A.dot(x_col).squeeze()).reshape(-1)

        assert constraint_1a.shape == (m, n + 1)
        assert constraint_1b.shape == (m,)

        a_constraints.append(constraint_1a)
        b_constraints.append(constraint_1b)

        # Constraint 2 has shape (2n, n+1)
        constraint_2a_left = np.vstack((np.eye(n), -1 * np.eye(n)))
        constraint_2a = np.hstack((constraint_2a_left,
                                   -1 * np.ones((2 * n, 1))))
        constraint_2b = np.zeros(2 * n)

        assert constraint_2a.shape == (2 * n, n + 1)
        assert constraint_2b.shape == (2 * n,)
        a_constraints.append(constraint_2a)
        b_constraints.append(constraint_2b)


        # Constraint 3 is added by the domain
        # If a full box, should have shape (2n, n + 1)
        d_a, d_b = self.domain.original_box_constraints()
        x_dx_low = x_row[self.domain.unmodified_bounds_low]
        x_dx_high = x_row[self.domain.unmodified_bounds_high]
        if d_a is not None:
            d_a_rows = d_a.shape[0]
            constraint_d_a = np.hstack((d_a, np.zeros((d_a_rows, 1))))
            constraint_d_b = d_b + np.hstack((x_dx_low, -x_dx_high))

            assert constraint_d_a.shape == (d_a_rows, n + 1)
            assert constraint_d_b.shape == (d_a_rows,)

            a_constraints.append(constraint_d_a)
            b_constraints.append(constraint_d_b)


        # Constraint 4 is upper bound constraint
        if self.domain.linf_radius is not None:
            constraint_4a = np.zeros((1, n + 1))
            constraint_4a[0][-1] = 1
            constaint_4b = np.array(self.domain.linf_radius)
            a_constraints.append(constraint_4a)
            b_constraints.append(constaint_4b)

        # Constraint 5 is equality constraint, should have (1, n+1)
        a_eq = matrix(np.hstack((self.a_eq, np.zeros((1, 1)))))
        b_eq = matrix(self.b_eq - self.a_eq.dot(x_row))




        # Objective should have length (n + 1)
        c = matrix(np.zeros(n + 1))
        c[-1] = 1


        ub_a = matrix(np.vstack(a_constraints))
        ub_b = matrix(np.hstack(b_constraints))


        # DOMAIN FREE STUFF?
        # dfree_a = matrix(np.vstack([constraint_1a, constraint_2a]))
        # dfree_b = matrix(np.hstack([constraint_1b, constraint_2b]))
        # dfree_start = time.time()
        # dfree_out = solvers.lp(c, dfree_a, dfree_b, A=a_eq, b=b_eq, solver='glpk')
        # dfree_end = time.time()
        # print("DFREE SOLVED IN %.03f" % (dfree_end - dfree_start))

        start = time.time()
        cvxopt_out = solvers.lp(c, ub_a, ub_b, A=a_eq, b=b_eq, solver='mosek')
        end = time.time()
        #print("LP SOLVED IN %.03f" % (end -start))

        if cvxopt_out['status'] == 'optimal':
            return cvxopt_out['primal objective'], \
                   (x_row + np.array(cvxopt_out['x'])[:-1].squeeze())
        elif cvxopt_out['status'] in ['primal infeasible', 'unknown']:
            return None, None
        else:
            print("About to fail...")
            print("CVXOPT status", cvxopt_out['status'])
            raise Exception("LINF DIST FAILED?")



    def l2_dist(self, x):
        """ Returns the l_2 distance to point x using LP
            as well as the optimal value of the program"""
        m, n = self.ub_A.shape
        x_row = utils.as_numpy(x).squeeze()
        x_col = x_row.reshape(-1, 1)

        # set up the quadratic program
        # min_{v} v^Tv             (<==>)     v^T I v
        # s.t.
        # 1)  A(x + v) <= b        (<==>)    Av <= b - Ax
        # 2)  A_eq(x + v) =  b_eq  (<==>)    A_eq v = b_eq - A_eq x



        # Setup objective
        P = matrix(np.identity(n))
        q = matrix(np.zeros([n, 1]))

        # Inequality constraints
        # Need to add domain constraints too
        d_a, d_b = self.domain.original_box_constraints()
        x_dx_low = x_row[self.domain.unmodified_bounds_low]
        x_dx_high = x_row[self.domain.unmodified_bounds_high]
        if d_a is not None:
            d_a_rows = d_a.shape[0]
            constraint_d_a = d_a
            constraint_d_b = d_b + np.hstack((x_dx_low, -x_dx_high))

            assert constraint_d_a.shape == (d_a_rows, n)
            assert constraint_d_b.shape == (d_a_rows,)

        G = matrix(np.vstack([self.ub_A, constraint_d_a]))
        h = matrix(np.hstack([self.ub_b - self.ub_A.dot(x_row),
                              constraint_d_b]))

        # Equality constraints
        A = matrix(self.a_eq)
        b = matrix(self.b_eq - self.a_eq.dot(x_row))

        quad_start = time.time()
        quad_program_result = solvers.qp(P, q, G, h, A, b, solver='mosek')
        quad_end = time.time()
        # print("QP SOLVED IN %.03f seconds" % (quad_end - quad_start))

        if quad_program_result['status'] == 'optimal': # or quad_program_result['status'] == 'unknown':
            v = np.array(quad_program_result['x'])
            return np.linalg.norm(v), x_row + v.squeeze()
        else:
            return None, None
            raise Exception("QPPROG FAILED: " + quad_program_result['status'])


    ##########################################################################
    #                                                                        #
    #               METHODS FOR BATCH VERSION ONLY                           #
    #                                                                        #
    ##########################################################################


    def _same_hyperplane(self, other):
        """ Given two facets, checks if they lie in different hyperplanes
            Returns True if they lie in the same hyperplane
        """
        # if either is not a facet, then return False
        if not (self.check_facet_feasible() and other.check_facet_feasible()):
            return False
        # if they lie in different hyperplanes, then return False
        self_tight = self.tight_list[0]
        self_a = self.ub_A[self_tight, :]
        self_b = self.ub_b[self_tight]

        other_tight = other.tight_list[0]
        other_a = other.ub_A[other_tight, :]
        other_b = other.ub_b[other_tight]

        return utils.is_same_hyperplane_nocomp(self_a, self_b, other_a, other_b)


    def _same_tight_constraint(self, other):
        """ Given two facets, checks if their tight constraints are the same
            Returns True if they are the same
        """
        # if either is not a facet, then return False
        if not (self.check_facet_feasible() and other.check_facet_feasible()):
            return False

        # if they lie in different hyperplanes, then return False
        self_tight = self.tight_list[0]
        self_a = self.ub_A[self_tight, :]
        self_b = self.ub_b[self_tight]

        other_tight = other.tight_list[0]
        other_a = other.ub_A[other_tight, :]
        other_b = other.ub_b[other_tight]

        return utils.is_same_tight_constraint(self_a, self_b, other_a, other_b)


    def check_same_facet_pg(self, other):
        """ Checks if this facet is the same as the other facet. Assumes
            that self and other are perfectly glued if they intersect at all.
            Uses LP to check if intersection of facets is (n-1) dimensional.
        """
        # if either is not a facet, then return False
        if not (self.check_facet_feasible() and other.check_facet_feasible()):
            return False

        if not self._same_hyperplane(other):
            return False

        # now just return True if their intersection is dimension (n-1)

        new_tight_list = np.add(other.tight_list, self.ub_b.shape)

        new_face = Face(np.vstack((self.ub_A, other.ub_A)),
                        np.hstack((self.ub_b, other.ub_b)),
                        tight_list=np.hstack((self.tight_list, new_tight_list)))
        return new_face.check_facet_feasible()


    def check_same_facet_pg_slow(self, other):
        """ Checks if this facet is the same as the other facet. Assumes
            that self and other are perfectly glued if they intersect at all

            Method uses PyPi library 'polytope' to compare vertices of the
            faces
        """

        # if either is not a facet, then return False
        if not (self.check_facet_feasible() and other.check_facet_feasible()):
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
            return False

    def check_same_facet_config(self, other):
        #TODO: fix numerical issues (shared facets aren't being eliminated)

        """ Potentially faster technique to check facets are the same
            The belief here is that if both (self, other) are facets, with their
            neuron configs specified, then if they have the same hyperplane and
            have config hamming distance 1 (plus condition explained below),
            then they are the same facet

            Extra Condtion: must account for the case where two polytopes
            can be simulatenously glued (not perfectly glued) to the same
            hyperplane. two faces can come from these polytopes with ReLu
            hamming distance one and yet their intersection is not a (n-1)
            dim. face.
        """
        if not (self.is_facet is True) and (other.is_facet is True):
            return False

        if not self._same_hyperplane(other):
            return False

        if self.config is None or other.config is None:
            return self.check_same_facet_pg(other)

        ReLu_distance_check = (utils.config_hamming_distance(self.config, other.config) == 1)


        return ReLu_distance_check and not self._same_tight_constraint(other)


