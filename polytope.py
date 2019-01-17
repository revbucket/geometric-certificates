import utilities as utils
import torch

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
        self.ub_A = ub_A
        self.ub_b = ub_b

        self.eq_A = None
        self.eq_b = None
        self.config = None

    def generate_facets(self, check_feasible=False):
        num_constraints = self.ub_A.shape[0]
        facets = []
        for i in range(num_constraints):
            facet = Face(self.ub_A, self.ub_b, set([i]), config=config)
            if check_feasible:
                facet.check_feasible()
            facet.check_facet()
            if facet.is_facet:
                facets.append(facet)
        return facets

    def is_point_feasible(self, x):
        """ Returns True if point X satisifies all constraints, false otherwise
        """
        pass



    def linf_dist(self, x):
        """ Takes a feasible point x and returns the minimum l_inf distance to
            a boundary point of the polytope.

            If x is not feasible, return -1
        """
        pass



class Face(Polytope):
    def __init__(self, poly_a, poly_b, tight_set, config=None):
        super(Polytope, self).__init__(ub_A, ub_b, )
        self.poly_a = poly_a
        self.poly_b = poly_b
        self.a_eq = self.poly_a[tight_indices]
        self.b_eq = self.poly_b[tight_indices]
        self.tight_set = tight_set
        self.config = config
        self.is_feasible = None
        self.is_facet = None
        self.interior = None

    def check_feasible(self):
        """ Checks if this polytope is feasible and stores the result"""

        if self.is_feasible is not None:
            return self.is_feasible

        # Set up feasibility check Linear program
        c = np.zeros(self.poly_b.shape)
        tight_indices = np.array(sorted(self.tight_set))


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
        m, n = self.poly_a.shape
        # Dimension check of (n-1) facet
        # Do min -t
        # st. Ax + t <= b
        #         -t <= 0
        # and if the optimal value is > 0 then (n-1) dimensional
        c = np.zeros(n + 1)
        c[-1] = -1

        new_poly_a = np.ones(m, n+1)
        new_poly_a[:-1, :-1] = self.poly_a
        bounds = [(None, None) for _ in range(n)]
        bounds.append((0, None))

        linprog_result = opt.linprog(c,
                                     A_ub=new_poly_a,
                                     b=self.poly_b,
                                     A_eq=self.a_eq,
                                     B_eq=self.b_eq,
                                     bounds=bounds)

        if (linprog_result.status == 0 and
            linprog_result.fun > 0):
            self.is_facet = True
            self.interior = linprog_result.x
        return self.is_facet


    def _same_hyperplane(self, other):
        """ Given two facets, checks if they lie in different hyperplanes
            Returns True if they lie in the same hyperplane
        """
        # if either is not a facet, then return False
        if not (self.check_facet() and other.check_facet()):
            return False
        # if they lie in different hyperplanes, then return False
        self_tight = list(self.tight_set)[0]
        self_a = self.poly_a[self_tight, :]
        self_b = self.poly_b(self_tight)

        other_tight = list(other.tight_set)[0]
        other_a = other.poly_a[other_tight, :]
        other_b = other.poly_b[other_tight]

        return utils.is_same_hyperplane(self_a, self_b, other_a, other_b)



    def check_same_facet_pg(self, other):
        """ Checks if this facet is the same as the other facet. Assumes
            that self and other are perfectly glued if they intersect at all

        """
        # if either is not a facet, then return False
        if not (self.check_facet() and other.check_facet()):
            return False

        if not self._same_hyperplane(other):
            return False

        # now just return True if their intersection is dimension (n-1)
        new_face = Face(np.concatenate(self.poly_a, other.poly_a),
                        np.concatenate(self.poly_b, other.poly_b),
                        tight_set = self.tight_set)
        return new_face.check_facet()


    def check_same_facet_config(self, other):
        """ Potentially faster technique to check facets are the same
            The belief here is that if both (self, other) are facets, with their
            neuron configs specified, then if they have the same hyperplane and
            have config hamming distance 1, then they are the same facet
        """
        if not (self.check_facet() and other.check_facet()):
            return False

        if not self._same_hyperplane(other):
            return False

        if self.config is None or other.config is None:
            return self.check_same_facet_pg(other)

        return utils.config_hamming_distance(self.config, other.config) == 1




