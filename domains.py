"""
File to contain stuff about domain restrictions. We can typically speed things
up a good deal by using domain restrictions, but these are complicated. We
can also incorporate upper bounds into this object.

"""
import numpy as np
import numbers
import utilities as utils
import torch



class Domain(object):
    """ Can support combinations of box + l2 bounds """
    def __init__(self, dimension, x):
        """ For now just set the dimension of the ambient space
            and the central point (which can be none)"""
        self.dimension = dimension

        if x is None:
            self.x = None
        else:
            self.x = utils.as_numpy(x).reshape(-1)

        # Things we'll set later
        self.box_low = None
        self.box_high = None
        self.l2_radius = None
        self.linf_radius = None

        # Original box constraints to be kept separate from those generated
        # by upper bounds.
        self.original_box_low = None
        self.original_box_high = None
        self.unmodified_bounds_low = None
        self.unmodified_bounds_high = None

    ###########################################################################
    #                                                                         #
    #                   FORWARD FACING METHODS                                #
    #                                                                         #
    ###########################################################################
    def set_original_hyperbox_bound(self, lo, hi):
        """ Sets the original hyperbox bounds which don't ever get modified """

        # Standard hyperbox setup
        lo = self._number_to_arr(lo)
        hi = self._number_to_arr(hi)
        self.set_hyperbox_bound(lo, hi)

        # And then do the original things
        self.original_box_low = lo
        self.original_box_high = hi
        self.unmodified_bounds_high = np.ones(self.dimension, dtype=np.bool)
        self.unmodified_bounds_low = np.ones(self.dimension, dtype=np.bool)



    def set_hyperbox_bound(self, lo, hi):
        self._add_box_constraint(lo, hi)

    def set_upper_bound(self, bound, lp_norm):
        {'l_inf': self.set_l_inf_upper_bound,
         'l_2': self.set_l_2_upper_bound}[lp_norm](bound)


    def set_l_inf_upper_bound(self, bound):
        assert self.x is not None
        if self.linf_radius is not None:
            self.linf_radius = min([bound, self.linf_radius])
        else:
            self.linf_radius = bound

        self._add_box_constraint(self.x - bound, self.x + bound)

    def set_l_2_upper_bound(self, bound):
        assert self.x is not None
        self.l2_radius = bound
        # also update box constraints if we can
        self._add_box_constraint(self.x - bound, self.x + bound)


    def feasible_facets(self, A, b, indices_to_check=None):
        """ Given numpy arrays A, b (corresponding to polytope Ax <= b)
            we want to know which constraints of the form <a_i, x> = b_i
            are feasible within the specified domain
        ARGS:
            A : numpy.ndarray (M x self.dimension) - constraint matrix
            b : numpy.ndarray (M) - constants
            indices_to_check : list of indices (out of M) to check (in the case
                               that we don't want to check them all)
        RETURNS:
            SET of indices that are viable under the l2 and l-inf box.
            Not everything in this list is feasible, but everything that is
            rejected is INFEASIBLE
        """
        A, b, map_fxn = self._idx_map_helper(A, b, indices_to_check)

        l_inf_set = self._linf_box_feasible_facets(A, b)
        l_2_set = self._l2_ball_feasible_facets(A, b)
        both_set = l_inf_set.intersection(l_2_set)
        return set(map_fxn(i) for i in both_set)


    def minimal_facet_projections(self, A, b, indices_to_check=None):
        """ Given numpy arrays A, b (corresponding to polytope Ax <= b)
            we want to know which constraints of the form <a_i, x> = b_i
            have minimal projections that fall within the specified l_2, l_inf
            bounds
        ARGS:
            A : numpy.ndarray (M x self.dimension) - constraint matrix
            b : numpy.ndarray (M) - constants
            indices_to_check : list of indices (out of M) to check (in the case
                               that we don't want to check them all)
        RETURNS:
            SET of indices that are viable under the l2 and l-inf box.
            Not everything in this list is feasible, but everything that is
            rejected is INFEASIBLE
        """
        self._compute_linf_radius()
        A, b, map_fxn = self._idx_map_helper(A, b, indices_to_check)

        l_inf_set = self._minimal_facet_projection_helper(A, b, 'l_inf')
        l_2_set = self._minimal_facet_projection_helper(A, b, 'l_2')
        both_set = l_inf_set.intersection(l_2_set)
        return set(map_fxn(i) for i in both_set)



    def original_box_constraints(self):
        """ Returns two np arrays for the hyperplane constraints that are in
            both the original constraints and the hyperbox low/hi bounds
        """
        eps = 1e-8
        As, bs = [], []
        if self.box_low is not None:
            As.append(-1 * np.eye(self.dimension)[self.unmodified_bounds_low])
            bs.append((-1 * self.box_low[self.unmodified_bounds_low]))
        if self.box_high is not None:
            As.append(np.eye(self.dimension)[self.unmodified_bounds_high])
            bs.append((self.box_high[self.unmodified_bounds_high]))

        if As != []:
            return np.vstack(As), np.hstack(bs)
        else:
            return None, None


    def box_constraints(self):
        """ Returns two np arrays for the hyperplane constraints if we're
            box bounded.
        RETURNS (A, b) for
            - A is a (2N, N) numpy array
            - B is a (2N,) numpy array

        VERTICAL ORDER IS ALWAYS LOWER_BOUNDS -> UPPER_BOUNDS
        """
        As, bs = [], []

        if self.box_low is not None:
            As.append(-1 * np.eye(self.dimension))
            bs.append(-1 * self.box_low)
        if self.box_high is not None:
            As.append(np.eye(self.dimension))
            bs.append(self.box_high)

        if As != []:
            return np.vstack(As), np.hstack(bs)
        else:
            return None, None


    def nonredundant_box_constraints(self, A, b, tight_idx):
        """ Computes an index of redundant box constraints given a system
            Ax <= b, where the i'th row is tight (i.e. <a_i, x> = b_i)

            This is just a fast check to see which of the box constraint
            hyperplanes don't intersect the hyperplane <a_i, x> = b_i
        RETURNS:
            (A,b ) where A is an (M, n) array for M <= 2n
                     and b is an (M,) array

        OLD DOCUMENTATION: ...
        Let a, b be the tight constraints (<a, x> = b)
        and let L, U be the box constraints (vectors of size n)

        Then a necessary (but not sufficient) condition for feasibility of this
        facet is that
            - <a+, U> + <a-, L> := b_u >= b  (and)
            - <a+, L> + <a-, U> := b_l <= b
        Now the hyperplane <a, x> = b intersects hyperplane (x_i = c) iff
            b_u - (a_i+ * u_i + a_i- * l_i) + a_i * c>= b  (and)
            b_l - (a_i+ * l_i + a_i- * u_i) + a_i * c>= b
        """

        # First check inputs/state makes sense
        assert self.box_low is not None
        assert self.box_high is not None
        assert isinstance(tight_idx, int)
        a, b = A[tight_idx].reshape(-1), b[tight_idx]

        # Next separate a into its positive and negative components
        a_plus = np.maximum(a, 0, a.copy())
        a_minus = a - a_plus

        # compute upper bounds and lower bounds for ALL indices
        a_plus_u = a_plus * self.box_high
        a_plus_l = a_plus * self.box_low
        a_minus_u = a_minus * self.box_high
        a_minus_l = a_minus * self.box_low

        b_u = np.sum(a_plus_u + a_minus_l) # this better be >= b
        b_l = np.sum(a_plus_l + a_minus_u) # this better be <= b

        # compute upper/lower bounds as vectors lacking the i'th component
        b_u_lacking_i = b_u - a_plus_u - a_minus_l
        b_l_lacking_i = b_l - a_plus_l - a_minus_u

        # Compute upper bound feasibilities
        uppers_u = b_u_lacking_i + a * self.box_high  >= b
        uppers_l = b_l_lacking_i + a * self.box_high <= b
        uppers = np.logical_and(uppers_u, uppers_l)

        # Compute lower bound feasibilities
        lowers_u = b_u_lacking_i + a * self.box_low >= b
        lowers_l = b_l_lacking_i + a * self.box_low <= b
        lowers = np.logical_and(lowers_u, lowers_l)

        # Boolean selector array is lower -> upper
        selector = np.hstack((lowers, uppers))

        box_constraint_A, box_constraint_b = self.box_constraints()

        return box_constraint_A[selector, :], box_constraint_b[selector]


    def box_to_tensor(self):
        """ If box bounds are not None, returns a tensor version of these bounds
            which is useful for interval propagation
        """
        if self.box_low is None or self.box_high is None:
            return None
        else:
            stacked = np.hstack([self.box_low.reshape(-1, 1),
                                 self.box_high.reshape(-1, 1)])
            return torch.Tensor(stacked)


    def current_upper_bound(self, lp_norm):
        """ Accessor method for current upper bound on each norm """
        return {'l_2': self.l2_radius,
                'l_inf': self.linf_radius}[lp_norm]


    def contains(self, y):
        """ Given a numpy array y (of shape (self.dimension,)), checks to see
            if y is valid in the domain
        """

        assert isinstance(y, np.ndarray)
        y = y.reshape(-1)

        checks = []

        # Box checks
        if self.box_low is not None:
            checks.append(all(y >= self.box_low))
        if self.box_high is not None:
            checks.append((all(y <= self.box_high)))

        # Linf checks
        if self.linf_radius is not None:
            checks.append(abs(y - self.x).max() <= self.linf_radius)

        # L2 checks
        if self.l2_radius is not None:
            checks.append(np.linalg.norm(y - self.x, 2) <= self.l2_radius)

        return all(checks)


    ###########################################################################
    #                                                                         #
    #    FEASIBILITY / MINIMAL PROJECTION HELPERS                             #
    #                                                                         #
    ###########################################################################

    @classmethod
    def _idx_map_helper(cls, A, b, indices_to_check=None):
        if indices_to_check is None:
            identity_fxn = lambda i : i
            return A, b, identity_fxn
        else:
            indices_to_check = list(sorted(indices_to_check))
            A = A[indices_to_check, :]
            b = b[indices_to_check]
            idx_map = {i: el for i, el in enumerate(indices_to_check)}
            map_fxn = lambda i: idx_map[i]
            return A, b, map_fxn


    def _linf_box_feasible_facets(self, A, b):
        """ Same args as self.feasible_facets """
        m = A.shape[0]
        if self.box_low is None or self.box_high is None:
            return set(range(m))

        A_plus = np.maximum(A, 0)
        A_minus = np.minimum(A, 0)

        upper_check = (A_plus.dot(self.box_high) +
                       A_minus.dot(self.box_high)) >= b
        lower_check = (A_plus.dot(self.box_low) +
                       A_minus.dot(self.box_high)) <= b
        total_check = np.logical_and(upper_check, lower_check)

        return {i for i in range(m) if total_check[i]} # <-- this is a set


    def _l2_ball_feasible_facets(self, A, b):
        """ Same args as self.feasible_facets """
        return set(range(A.shape[0])) # NOT IMPLEMENTED


    def _minimal_facet_projection_helper(self, A, b, lp):
        upper_bound = {'l_2': self.l2_radius, 'l_inf': self.linf_radius}[lp]
        if upper_bound is None:
            return set(range(A.shape[0]))
        dual_norm = {'l_2': None, 'l_inf': 1}[lp]

        duals = np.linalg.norm(A, ord=dual_norm, axis=1)

        under_upper_bound = np.divide(b - A.dot(self.x), duals) <= upper_bound
        return set(i for i, el in enumerate(under_upper_bound) if el)



    ###########################################################################
    #                                                                         #
    #                   HELPERS FOR BOX BOUNDS                                #
    #                                                                         #
    ###########################################################################



    def _number_to_arr(self, number_val):
        """ Converts float to array of dimensi
        """
        assert isinstance(number_val, numbers.Real)
        return np.ones(self.dimension) * number_val


    def _compute_new_lohi(self, lo_or_hi, vals):
        """ Takes an array to replace new lows or highs
        ARGS:
            lo_or_hi : string - 'lo' or 'hi'
            vals: np.array of dimension self.dimension - new bounds to be
                                                         considered
        RETURNS:
            - array that's elementwise max or min, depending on lo_or_hi
            - boolean numpy array with the things that have changed
        """
        eps = 1e-8
        assert lo_or_hi in ['lo', 'hi']
        assert isinstance(vals, np.ndarray) and vals.shape == (self.dimension,)
        comp = {'lo': np.maximum,'hi': np.minimum}[lo_or_hi]
        current = {'lo': self.box_low, 'hi': self.box_high}[lo_or_hi]
        output = comp(vals, current)
        unchanged = abs(current - output) < eps
        return output, unchanged

    def _add_box_constraint(self, lo, hi):
        """ Adds a box constraint.
        ARGS:
            lo: float or np.array(self.dimension) - defines the coordinate-wise
                lowerbounds
            hi: float or np.array(self.dimension) - defines the coordinate-wise
                upperbounds
        RETURNS:
            None
        """

        # Make sure these are arrays
        if isinstance(lo, numbers.Real):
            lo = self._number_to_arr(lo)
        if isinstance(hi, numbers.Real):
            hi = self._number_to_arr(hi)

        # Set the lows and highs if they're not already set
        set_lo, set_hi = False, False
        if self.box_low is None:
            set_lo = True
            self.box_low = lo
        if self.box_high is None:
            set_hi = True
            self.box_high = hi

        # Otherwise, set the lows and highs by taking elementwise max/min
        if not set_lo:
            self.box_low, low_unchanged = self._compute_new_lohi('lo', lo)
            np.logical_and(self.unmodified_bounds_low, low_unchanged,
                           self.unmodified_bounds_low)
        if not set_hi:
            self.box_high, high_unchanged = self._compute_new_lohi('hi', hi)
            np.logical_and(self.unmodified_bounds_high, high_unchanged,
                           self.unmodified_bounds_high)

        return

    def _compute_linf_radius(self):
        """ Modifies the self.linf_radius based on the box bounds """
        if self.box_high is None and self.box_low is None:
            return None

        linf_radius = max([np.abs(self.box_high - self.x).max(),
                           np.abs(self.x - self.box_low).max()])
        if self.linf_radius is None:
            self.linf_radius = linf_radius
        else:
            self.linf_radius = min(linf_radius, self.linf_radius)
        return self.linf_radius






