import utilities as utils
import torch
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

##########################################################################
#                                                                        #
#                   POLYTOPE AND FACE CLASSES                            #
#                                                                        #
##########################################################################

def from_polytope_dict(polytope_dict):
    return Polytope(polytope_dict['poly_a'],
                    polytope_dict['poly_b'],
                    config=polytope_dict['configs'])

class Polytope(object):
    def __init__(self, ub_A, ub_b, config=None):
        """ Polytopes are of the form Ax <= b
            with no strict equality constraints"""

        if isinstance(ub_A, torch.Tensor):
            ub_A = ub_A.cpu().detach().numpy()
        if isinstance(ub_b, torch.Tensor):
            ub_b = ub_b.cpu().detach().numpy()
        self.ub_A = utils.as_numpy(ub_A)
        self.ub_b = utils.as_numpy(ub_b)
        self.config = config

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

    def generate_facets_configs(self, seen_polytopes_dict, check_feasible=False):
        """ Generates all (n-1) dimensional facets of polytope which aren't
            shared with other polytopes in list. (for ReLu nets)
        """

        num_constraints = self.ub_A.shape[0]
        facets = []
        for i in range(num_constraints):
            facet = Face(self.ub_A, self.ub_b, [i], config=self.config)
            if check_feasible:
                facet.check_feasible()
            facet.check_facet()

            #TODO: fix this code section below (no need to add already seen facets)

            # # Don't add facets which are connected to seen polytopes (don't want to add facet twice)
            # configs_flat = utils.flatten_config(facet.config)
            # close_seen_bools = [utils.string_hamming_distance(configs_flat, other_config_flat) == 1
            #               for other_config_flat in seen_polytopes_dict ]
            #
            # shared_facet_bools = [facet.tight_list[0] == utils.hamming_indices(configs_flat, other_config_flat)[0]
            #                        for seen_bool, other_config_flat in zip(close_seen_bools,seen_polytopes_dict)
            #                        if seen_bool]
            # print('------')
            # print([facet.tight_list[0]
            #                        for seen_bool, other_config_flat in zip(close_seen_bools,seen_polytopes_dict)
            #                        if seen_bool])
            # print([utils.hamming_indices(configs_flat, other_config_flat)[0]
            #                        for seen_bool, other_config_flat in zip(close_seen_bools,seen_polytopes_dict)
            #                        if seen_bool])
            # print('------')
            # if facet.is_facet and not any(shared_facet_bools):

            if facet.is_facet:
                facets.append(facet)

            # else:
            #     if(facet.is_facet):
            #         print('shared facet!')
            #     else:
            #         print('not a facet')

        return facets

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
        norms = np.diag(1.0 / np.linalg(self.ub_A, ord=1, axis=1))
        dists = slack.matmul(norms)
        argmin_0 = np.argmin(dists)[0]
        return dists[argmin_0], argmin_0






class Face(Polytope):
    def __init__(self, poly_a, poly_b, tight_list, config=None):
        super(Face, self).__init__(poly_a, poly_b, config=config)
        self.poly_a = poly_a
        self.poly_b = poly_b
        self.a_eq = self.poly_a[tight_list]   # EDIT: self.a_eq = self.poly_a[tight_list]
        self.b_eq = self.poly_b[tight_list]
        self.tight_list = tight_list
        self.is_feasible = None
        self.is_facet = None
        self.interior = None

    def check_feasible(self):
        """ Checks if this polytope is feasible and stores the result"""

        if self.is_feasible is not None:
            return self.is_feasible

        # Set up feasibility check Linear program
        c = np.zeros(self.poly_a.shape[1])
        tight_indices = np.array(sorted(self.tight_list))


        bounds = [(None, None) for _ in c]

        linprog_result = opt.linprog(c,
                                     A_ub=self.poly_a,
                                     b_ub=self.poly_b,
                                     A_eq=self.a_eq,
                                     b_eq=self.b_eq,
                                     bounds=bounds)
        is_feasible = (linprog_result.status == 0)
        self.is_feasible = is_feasible
        return self.is_feasible

    def check_facet(self):
        """ Checks if this polytope is a (n-1) face and stores the result"""
        if self.is_facet is not None:
            return self.is_facet

        if self.is_feasible is not None and not self.is_feasible:
            self.is_facet = False
            return self.is_facet

        m, n = self.poly_a.shape
        # Dimension check of (n-1) facet
        # Do min -t
        # st. Ax + t <= b
        #         -t <= 0
        # and if the optimal value is < 0 then (n-1) dimensional
        c = np.zeros(n + 1)
        c[-1] = -1

        new_poly_a = np.ones([m, n+1])
        new_poly_a[:, :-1] = self.poly_a
        new_poly_a[self.tight_list, -1] = 0     # remove affect of t on tight constraints
        bounds = [(None, None) for _ in range(n)]
        bounds.append((0, None))

        m2, n2 = self.a_eq.shape
        a_eq_new = np.zeros([m2, n+1])
        a_eq_new[:, :-1] = self.a_eq

        linprog_result = opt.linprog(c,
                                     A_ub=new_poly_a,
                                     b_ub=self.poly_b,
                                     A_eq=a_eq_new,
                                     b_eq=self.b_eq,
                                     bounds=bounds, method='interior-point')

        if (linprog_result.status == 0 and
            linprog_result.fun < 0):
            self.is_facet = True
            self.interior = linprog_result.x[0:-1]

        elif linprog_result == 3:
            print('error, unbounded, in linprog for check facet')

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

        return utils.is_same_hyperplane(self_a, self_b, other_a, other_b)

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



    def linf_dist(self, x):
        #TODO: this method doesn't  seem to always correctly find the projection onto a facet

        """ Returns the l_inf distance to point x using LP"""

        # set up the linear program
        # min_{t,v} t
        # s.t.
        # 1)  A(x + v) <= b        (<==>)    Av <= b - Ax
        # 2)  A_eq(x + v) =  b_eq  (<==>)    A_eq v = b_eq - A_eq x
        # 3)  v <= t * 1           (<==>)    v_i - t <= 0
        # 4) -v <= t * 1           (<==>)   -v_i - t <= 0

        # optimization variable is [t, v]
        n = x.shape[0]
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

        bounds = [(0, None)] + [(None, None) for _ in range(n)]


        # Solve linprog
        linprog_result = opt.linprog(c, A_ub=ub_a, b_ub=ub_b,
                                        A_eq=constraint_2a,
                                        b_eq=constraint_2b,
                                        bounds=bounds)

        if linprog_result.status == 0:
            return linprog_result.fun
        else:
            raise Exception("LINPROG FAILED: " + linprog_result.message)


    def l2_dist(self, x):
        """ Returns the l_2 distance to point x using LP"""

        # set up the quadratic program
        # min_{v} v^T*v
        # s.t.
        # 1)  A(x + v) <= b        (<==>)    Av <= b - Ax
        # 2)  A_eq(x + v) =  b_eq  (<==>)    A_eq v = b_eq - A_eq x

        n = np.shape(x)[0]
        P = matrix(np.identity(n))
        G = matrix(self.poly_a)
        h = matrix(self.poly_b - np.matmul(self.poly_a, x)[:, 0])
        q = matrix(np.zeros([n, 1]))
        A = matrix(self.a_eq)
        b = matrix(self.b_eq - np.matmul(self.a_eq, x))

        quad_program_result = solvers.qp(P, q, G, h, A, b)

        if quad_program_result['status'] == 'optimal' or quad_program_result['status'] == 'unknown':
            x = np.array(quad_program_result['x'])
            return np.linalg.norm(x)
        else:
            print(quad_program_result)
            raise Exception("QPPROG FAILED: " + quad_program_result['status'])



    def get_inequality_constraints(self):

        """ Converts equality constraints into inequality constraints
            and gathers them all together
        """

        A = np.vstack((self.poly_a, np.multiply(self.a_eq, -1.0)))
        b = np.hstack((self.poly_b, np.multiply(self.b_eq, -1.0)))

        return A, b

