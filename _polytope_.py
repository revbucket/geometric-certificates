import utilities as utils
import torch
import numpy as np
import scipy.optimize as opt
from collections import defaultdict

import gurobipy as gb
from mosek import iparam
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
solvers.options['mosek'] = {iparam.log: 0,
                            iparam.max_num_warnings: 0}
import copy
import time

class GurobiSquire(object):
    """ Wrapper to hold attributes for the gurobi model since gb.Model won't let
        me set attributes
    """
    def __init__(self):
        pass




################################################################################
#                           POLYTOPE CLASS                                     #
#                                                                              #
################################################################################


class Polytope(object):


    ######################################################################
    #                                                                    #
    #                       Polytope Initializations                     #
    #                                                                    #
    ######################################################################

    def __init__(self, ub_A, ub_b, x_np, config=None, interior_point=None,
                 domain=None, dead_constraints=None, gurobi=True,
                 linear_map=None, lipschitz_ub=None, c_vector=None):
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
        self.gurobi = gurobi
        self.gurobi_model = None
        self.gurobi_squire = None
        self.x_np = x_np
        self.linear_map = linear_map
        self.lipschitz_ub = lipschitz_ub
        self.lipschitz_constrs = []
        if c_vector is None:
            self.c_vector = c_vector
        else:
            self.c_vector = [utils.as_numpy(_) for _ in c_vector]


    @classmethod
    def from_polytope_dict(cls, polytope_dict, x_np, domain=None,
                           dead_constraints=None,
                           gurobi=True,
                           lipschitz_ub=None,
                           c_vector=None):
        """ Alternate constructor of Polytope object """

        linear_map = {'A': utils.as_numpy(polytope_dict['total_a']),
                      'b': utils.as_numpy(polytope_dict['total_b'])}


        return cls(polytope_dict['poly_a'],
                   polytope_dict['poly_b'],
                   x_np,
                   config=polytope_dict['configs'],
                   domain=domain,
                   dead_constraints=dead_constraints,
                   gurobi=gurobi,
                   linear_map=linear_map,
                   lipschitz_ub=lipschitz_ub,
                   c_vector=c_vector)



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



    def generate_facets_configs_parallel(self, seen_dict, missed_dict=None):
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


        #####################################################################
        #   Step 4: Remove all facets that have been seen before            #
        #####################################################################

        # Also remove the infeasible facets
        potential_facets, num_seen, num_missed = \
                            self.scrub_seen_idxs(potential_facets, seen_dict,
                                                 missed_dict)

        if num_seen > 0:
            reject_dict['seen before'] += num_seen

        if num_missed > 0:
            reject_dict['missed before'] += num_missed


        ####################################################################
        #   Step 5: Set up the Gurobi model for reusable optimization      #
        ####################################################################
        self._build_gurobi_model()#indices_to_include=potential_facets)

        ######################################################################
        #   Step 6: Construct the facet objects                              #
        ######################################################################

        facets = [self.facet_constructor(idx) for idx in potential_facets]
        return facets, reject_dict




    ##########################################################################
    #                                                                        #
    #                            HELPER METHODS                              #
    #                                                                        #
    ##########################################################################

    def _build_gurobi_model(self, indices_to_include=None):
        """ Builds a gurobi model with all the constraints added (except those
            specified in indices_to_ignore
        ARGS:
            indices_to_include: if not None, is np.array of indices to include
                                if None, includes all indices
        RETURNS:
            None, but modifies self.gurobi_model
        """
        model = gb.Model()
        dim = self.ub_A.shape[1]
        # add variables
        if self.domain.box_low is not None:
            try:
                lo_minus_x = self.domain.box_low - self.x_np
                lb_i = lambda i: lo_minus_x[i]
            except Exception as err:
                print("ERROR HERE")
                print(self.domain.box_low.__class__)
                print(self.x_np.__class__)
                raise err
        else:
            lb_i = lambda i: -gb.GRB.INFINITY

        if self.domain.box_high is not None:
            hi_minus_x = self.domain.box_high - self.x_np
            ub_i = lambda i: hi_minus_x[i]
        else:
            ub_i = lambda i: gb.GRB.INFINITY

        # --- variables representing 'v' in LP/QP projections
        v_vars = [model.addVar(lb=lb_i(i), ub=ub_i(i), name='v%s' % i)
                  for i in range(dim)]

        # --- variables representing t for linprogs etc
        aux_vars = []
        aux_vars.append(model.addVar(lb=0, ub=gb.GRB.INFINITY, name='t'))

        model.update()

        # add constraints
        # --- constraints for being in the polytope
        Ax = self.ub_A.dot(self.x_np)
        ub_b_minus_Ax = self.ub_b - Ax
        if indices_to_include is not None:
            a_rows = self.ub_A[indices_to_include, :]
            b_rows = ub_b_minus_Ax[indices_to_include]
        else:
            a_rows = self.ub_A
            b_rows = ub_b_minus_Ax
        m = a_rows.shape[0]

        _ = model.addConstrs(gb.LinExpr(a_rows[i], v_vars) <= b_rows[i]
                             for i in range(m))

        # --- constraints for being in the domain
        # if self.domain.box_low is not None:
        #     x_minus_lo = self.x_np - self.domain.box_low
        #     _ = model.addConstrs(v_vars[i] <= x_minus_lo[i] for i in range(dim))
        # if self.domain.box_high is not None:
        #     hi_minus_x = self.domain.box_high - self.x_np
        #     _ = model.addConstrs(v_vars[i] <= hi_minus_x[i] for i in range(dim))

        model.update()

        model.setParam('OutputFlag', False)
        self.gurobi_model = model
        self.gurobi_squire = GurobiSquire()


    def _is_feasible(self):
        """ Runs a gurobi check to see if this is a feasible model"""
        if self.gurobi_model is None:
            self._build_gurobi_model()

        self.gurobi_model.setObjective(0)
        self.gurobi_model.update()
        self.gurobi_model.optimize()

        return self.gurobi_model.Status



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


    def facet_constructor(self, tight_idx, facet_type='facet',
                          extra_tightness=None):

        return Face(self.ub_A, self.ub_b, [tight_idx], config=self.config,
                     domain=self.domain, facet_type=facet_type, x_np=self.x_np,
                     extra_tightness=extra_tightness,
                     gurobi_model=self.gurobi_model,
                     gurobi_squire=self.gurobi_squire,
                     linear_map=self.linear_map,
                     lipschitz_ub=self.lipschitz_ub,
                     c_vector=self.c_vector)



    def scrub_seen_idxs(self, idx_list, seen_dict, missed_dict=None):
        """ Removes facets we've seen before, where idx_list is which idx is
            tight. Also removes the cache-miss polytopes
        """
        if missed_dict is None:
            missed_dict = {}
        output_idxs, num_seen_before, num_missed_before = [], 0, 0
        for idx in idx_list:
            flip_i, flip_j = utils.index_to_config_coord(self.config, idx)
            new_configs = copy.deepcopy(self.config)
            new_configs[flip_i][flip_j] = int(1 - new_configs[flip_i][flip_j])
            new_flat = utils.flatten_config(new_configs)
            if new_flat in seen_dict:
                num_seen_before += 1
            elif new_flat in missed_dict:
                num_missed_before += 1
            else:
                output_idxs.append(idx)

        return output_idxs, num_seen_before, num_missed_before



##############################################################################
#                                                                            #
#                               FACE CLASS                                   #
#                                                                            #
##############################################################################



class Face(Polytope):
    def __init__(self, poly_a, poly_b, tight_list, x_np, config=None,
                 domain=None, dead_constraints=None, removal_list=None,
                 facet_type=None, gurobi_model=None, gurobi_squire=None,
                 extra_tightness=None,
                 linear_map=None, lipschitz_ub=None, c_vector=None):
        super(Face, self).__init__(poly_a, poly_b, x_np, config=config,
                                   domain=domain,
                                   dead_constraints=dead_constraints)

        if tight_list[0] is None:
            assert extra_tightness is not None
            self.a_eq = extra_tightness['A'].reshape((1, -1))
            self.b_eq = extra_tightness['b']
        else:
            self.a_eq = self.ub_A[tight_list]
            self.b_eq = self.ub_b[tight_list]
        self.tight_list = tight_list
        self.is_feasible = None
        self.is_facet = None
        self.interior = None
        self.removal_list = removal_list

        assert facet_type in [None, 'decision', 'facet']
        self.facet_type = facet_type

        self.gurobi_model = gurobi_model
        self.gurobi_squire = gurobi_squire
        self.extra_tightness = extra_tightness

        self.linear_map = linear_map
        self.lipschitz_ub = lipschitz_ub
        self.c_vector = c_vector


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

        if self.tight_list[0] is None: # adversarial constraint here
            dec_A = self.extra_tightness['A']
            dec_b = self.extra_tightness['b']
            A = np.vstack((self.ub_A, dec_A.reshape(1, -1)))
            b = np.hstack((self.ub_b, dec_b))
            checklist = [A.shape[0] - 1]
        else: # regular facet here
            A = self.ub_A
            b = self.ub_b
            checklist = self.tight_list

        domain_feasible = domain.feasible_facets(A, b, checklist)
        if len(domain_feasible) == 0:
            return False

        # Do checks to see if this hyperplane has projection inside ball
        projection = domain.minimal_facet_projections(A, b, checklist)

        return len(projection) > 0


    ##########################################################################
    #                                                                        #
    #                       GUROBI DISTANCE FUNCTIONS                        #
    #                                                                        #
    ##########################################################################

    def linf_dist_gurobi(self, x):
        """ Computes the l_infinity distance to point x using gurobi
        """
        v_vars = [v for v in self.gurobi_model.getVars()
                  if v.VarName.startswith('v')]
        # First do a hacky check to see if this model has already had LP setup
        if not hasattr(self.gurobi_squire, 'linf_dist_setup'):
            self.gurobi_squire.linf_dist_setup = True

            # If hasn't been set up yet, set up the general LP constraints
            # --- add -t<= v_i <= t
            t = self.gurobi_model.getVarByName('t')
            for v_var in v_vars:
                self.gurobi_model.addConstr(-t <= v_var)
                self.gurobi_model.addConstr(v_var <= t)

            # --- add t <= upper_bound
            self.gurobi_model.addConstr(t <= self.domain.linf_radius)

            # --- add objective
            # --- --- if self.lipschitz_ub is not None, we incorporate this
            #         objective as follows
            #
            if self.lipschitz_ub is None:
                self.gurobi_model.setObjective(t, gb.GRB.MINIMIZE)
            else:
                """
                If self.lipschitz_ub is not None, then we incorporate this
                objective as follows
                Recall our setting is
                min_{y in F} ||y -x|| + z
                    s.t. z >= |c_j^T(f(y) - f(DB))|  / L_j
                                                     for all j != true label
                    (and f(DB)=0 and c_j^Tf(y) >= 0 and linear)
                Then we need to compute g_j(y) := c_j^Tf(y) / L_j
                for each j (as a linear functional)
                But the minimization works like (letting y = x + v)
                min_{x+v in F} ||v||_infty + z
                s.t. z >= c_j^Tf(x+v) / L_j
                and c_j^Tf(x+v) = a_j^T(x +v) + b_j = a_j^Tv + (b_j + a_j^Tx)
                so  z >= a_j^Tv + (b_j + a_j^Tx)
                and if f(y) = Ay + b
                where a_j := c_j^TA/L_j and b_j = c_j^Tb/L_j
                """


                # First step is to compute the a_j/b_j for each c vector
                lin_A = self.linear_map['A']
                lin_b = self.linear_map['b']

                a_js, b_js = [], []
                for lip_val, c_vec in zip(self.lipschitz_ub, self.c_vector):
                    a_js.append(c_vec.dot(lin_A) / lip_val)
                    b_js.append(c_vec.dot(lin_b) / lip_val)

                # Then we can add the constraint of z to everything
                # (lip_var >= a_j^T v + (b_j + a_j^Tx))
                lip_var = self.gurobi_model.addVar(lb=0, name='lip_var')
                lipschitz_constrs = []
                for j in range(len(a_js)):
                    a_j, b_j = a_js[j], b_js[j]
                    linexpr_j = gb.LinExpr(a_j, v_vars)
                    const_j = b_j + a_j.dot(self.x_np)
                    lip_constr = (lip_var >= linexpr_j + const_j)
                    lipschitz_constrs.append(lip_constr)
                self.gurobi_squire.lipschitz_constrs = lipschitz_constrs


            self.gurobi_model.update()


        # Now we can remove any equality constraints already set
        try:
            self.gurobi_model.remove(self.gurobi_model.getConstrByName('facet'))
            self.gurobi_model.update()
        except gb.GurobiError:
            pass

        # Add the new equality constraint
        if self.facet_type == 'facet':
            tight_row = self.ub_A[self.tight_list[0], :]
            tight_b = self.ub_b[self.tight_list[0]] - tight_row.dot(self.x_np)
        elif self.facet_type == 'decision':
            tight_row = self.extra_tightness['A']
            tight_b = self.extra_tightness['b'] - tight_row.dot(self.x_np)

        self.gurobi_model.addConstr(gb.LinExpr(tight_row, v_vars) == tight_b,
                                    name='facet')

        # Now branch to handle the lipschitz cases...
        t_var = self.gurobi_model.getVarByName('t')
        if self.lipschitz_ub is None:
            self.gurobi_model.setObjective(t_var, gb.GRB.MINIMIZE)
            self.gurobi_model.update()
            self.gurobi_model.optimize()
            if self.gurobi_model.Status != 2:
                return None, None
            else:
                obj_value = self.gurobi_model.getObjective().getValue()
                opt_point =  self.x_np + np.array([v.X for v in v_vars])
                return obj_value, opt_point

        lip_var = self.gurobi_model.getVarByName('lip_var')
        self.gurobi_model.setObjective(t_var + lip_var, gb.GRB.MINIMIZE)
        objs_opts = []
        opt_times = []
        for lip_constr in self.gurobi_squire.lipschitz_constrs:
            try:
                self.gurobi_model.remove(self.gurobi_model.getConstrByName('lipschitz'))
            except gb.GurobiError:
                pass
            start_time = time.time()
            self.gurobi_model.addConstr(lip_constr)
            self.gurobi_model.update()
            self.gurobi_model.optimize()
            opt_times.append('%.04f' % (time.time() - start_time))

            if self.gurobi_model.Status == 3:
                return None, None
            objs_opts.append((self.gurobi_model.getObjective().getValue(),
                              np.array([v.X for v in v_vars])))
        # print("OPT TIMES: ", ' '.join(opt_times))

        min_pair = min(objs_opts, key=lambda pair: pair[0])
        return min_pair[0], min_pair[1] + self.x_np





    def l2_dist_gurobi(self, x):
        """ Returns the l_2 distance to point x, and projection using Gurobi"""

        v_vars = [v for v in self.gurobi_model.getVars()
                  if v.VarName.startswith('v')]
        # Swap out the inequality constraints if any exist
        try:
            self.gurobi_model.remove(self.gurobi_model.getConstrByName('facet'))
            # if passes, then already set up
        except gb.GurobiError:
            # if fails, then not set up yet
            # --- add objective
            obj_expr = gb.quicksum(v * v for v in v_vars)
            self.gurobi_model.setObjective(obj_expr, gb.GRB.MINIMIZE)
            self.gurobi_model.update()

        # --- add facet constraint
        if self.facet_type == 'facet':
            tight_row = self.ub_A[self.tight_list[0], :]
            tight_b = self.ub_b[self.tight_list[0]] - tight_row.dot(self.x_np)
        else:
            tight_row = self.extra_tightness['A']
            tight_b = self.extra_tightness['b'] - tight_row.dot(self.x_np)

        self.gurobi_model.addConstr(gb.LinExpr(tight_row, v_vars) == tight_b,
                                    name='facet')

        # Now solve and return value, output
        self.gurobi_model.update()
        self.gurobi_model.optimize()

        if self.gurobi_model.Status == 2: # OPTIMAL STATUS
            # --- get objective
            obj_value = self.gurobi_model.getObjective().getValue()

            # --- get variables and add to x
            opt_point =  self.x_np + np.array([v.X for v in v_vars])
            return obj_value, opt_point
        else:
            return None, None



    ##########################################################################
    #                                                                        #
    #                        MOSEK DISTANCE FUNCTIONS                        #
    #                                                                        #
    ##########################################################################

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
        a_eq = matrix(np.hstack((self.a_eq, np.zeros((1,1)))))
        b_eq = matrix((self.b_eq - self.a_eq.dot(x_row)).astype(np.double))




        # Objective should have length (n + 1)
        c = matrix(np.zeros(n + 1))
        c[-1] = 1


        ub_a = matrix(np.vstack(a_constraints))
        ub_b = matrix(np.hstack(b_constraints))


        start = time.time()
        cvxopt_out = solvers.lp(c, ub_a, ub_b, A=a_eq, b=b_eq, solver='mosek')
        end = time.time()
        # print("LP SOLVED IN %.03f" % (end -start))

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
            as well as the optimal value of the program
        set up the quadratic program
        min_{v} v^Tv             (<==>)     v^T I v
        s.t.
        1)  A(x + v) <= b        (<==>)    Av <= b - Ax
        2)  A_eq(x + v) =  b_eq  (<==>)    A_eq v = b_eq - A_eq x
        """
        m, n = self.ub_A.shape
        x_row = utils.as_numpy(x).squeeze()
        x_col = x_row.reshape(-1, 1)

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
        A = matrix(self.a_eq.astype(np.double))
        b = matrix((self.b_eq - self.a_eq.dot(x_row)).astype(np.double))

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



