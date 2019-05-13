import numpy as np
import torch
import matplotlib.pyplot as plt
from bitstring import BitArray
import polytope as ptope
import matplotlib.patches as patches
import pulp as plp


##########################################################################
#                                                                        #
#                   NEURAL CONFIG UTILITIES                              #
#                                                                        #
##########################################################################

def config_hamming_distance(config1, config2):
    """ Given two configs (as a list of floattensors, where all elements are
        0.0 or 1.0) computes the hamming distance between them
    """

    hamming_dist = 0
    for comp1, comp2 in zip(config1, config2):
        uneqs = comp1.type(torch.uint8) != comp2.type(torch.uint8)
        hamming_dist += uneqs.sum().numpy()
    return hamming_dist

def string_hamming_distance(str1, str2):
    assert len(str1) == len(str2)
    dist = sum(c1 != c2 for c1, c2 in zip(str1, str2))
    return dist

def hamming_indices(str1, str2):
    assert len(str1) == len(str2)
    # list = [c1 != c2 for c1, c2 in zip(str1, str2)]
    # index = list.index(True)
    indices = [index for index, (c1, c2) in enumerate(zip(str1, str2)) if c1 != c2]

    return indices

def cat_config(conf):
    """ Takes a list of float or uint8 tensors and flattens them into np.ndarray
    """
    return torch.cat([_.cpu().type(torch.uint8).detach() for _ in conf]).numpy()


def flatten_config(config):
    """ Takes a list of floatTensors where each element is either 1 or 0
        and converts into a string of 1s and 0s.
        Is just a binary representation of the neuron config
    """
    return ''.join(str(_) for _ in cat_config(config))


def index_to_config_coord(config, index):
    """ Given an index of the flattened array, returns the 2d index of where
        this corresponds to the configs
    """
    config_shapes = [_.numel() for _ in config]
    assert index < sum(config_shapes)

    for i, config_len in enumerate(config_shapes):
        if index > config_len - 1:
            index -= config_len
        else:
            return (i, index)


##############################################################################
#                                                                            #
#                       MATRIX OPERATIONS/ EQUALITY CHECKS                   #
#                                                                            #
##############################################################################

global_tolerance = 1e-6


def split_tensor_pos(tensor):
    """ Splits tensor into positive and negative terms """
    pos_tensor = tensor.masked_fill(tensor < 0, 0.0)
    neg_tensor = tensor - pos_tensor

    return pos_tensor, neg_tensor


def comparison_form(A, b, tolerance=global_tolerance):
    """ Given polytope Ax<= b
        Convert each constraint into a_i^Tx <= +-1, 0
        If b_i=0, normalize a_i
        Then sort rows of A lexicographically

    A is a 2d numpy array of shape (m,n)
    b is a 1d numpy array of shape (m)
    """
    raise DeprecationWarning("DON'T DO THIS OUTSIDE OF BATCH ")
    m, n = A.shape
    # First scale all constraints to have b = +-1, 0
    b_abs = np.abs(b)
    rows_to_scale = (b_abs > tolerance).astype(int)
    rows_to_normalize = 1 - rows_to_scale
    scale_factor = np.ones(m) - rows_to_scale + b_abs

    b_scaled = (b / scale_factor)
    a_scaled = A / scale_factor[:, None]

    rows_to_scale = 1

    # Only do the row normalization if you have to
    if np.sum(rows_to_normalize) > 0:
        row_norms = np.linalg.norm(a_scaled, axis=1)
        norm_scale_factor = (np.ones(m) - rows_to_normalize +
                             row_norms * rows_to_normalize)
        a_scaled = a_scaled / norm_scale_factor[:, None]


    # Sort scaled version
    sort_indices = np.lexsort(a_scaled.T)
    sorted_a = a_scaled[sort_indices]
    sorted_b = b_scaled[sort_indices]

    return sorted_a, sorted_b


def fuzzy_equal(x, y, tolerance=global_tolerance):
    """ Fuzzy float equality check. Returns true if x,y are within tolerance
        x, y are scalars
    """
    return abs(x - y) < tolerance


def fuzzy_vector_equal(x_vec, y_vec, tolerance=global_tolerance):
    """ Same as above, but for vectors.
        x_vec, y_vec are 1d numpy arrays
     """
    return all(abs(el) < tolerance for el in x_vec - y_vec)

def fuzzy_vector_equal_plus(x_vec, y_vec, tolerance=global_tolerance):
    """ Same as above, but for vectors.
        x_vec, y_vec are 1d numpy arrays
     """
    bools = [abs(el) < tolerance for el in x_vec - y_vec]
    return all(bools), bools

def is_same_hyperplane_nocomp(a1, b1, a2, b2, tolerance=global_tolerance):
    """ Check same hyperplane when not comparison form """

    # Check that neither a is zero
    a1_zero = fuzzy_equal(np.linalg.norm(a1), 0, tolerance=tolerance)
    a2_zero = fuzzy_equal(np.linalg.norm(a1), 0, tolerance=tolerance)
    b1_zero = fuzzy_equal(b1, 0.0, tolerance=tolerance)
    b2_zero = fuzzy_equal(b2, 0.0, tolerance=tolerance)

    # If exactly one is zero, then they can't be equal
    if (a1_zero != a2_zero) or (b1_zero != b2_zero):
        return False

    # Then find if there's a ratio between the two
    first_nonzero_idx = None
    for i, el in enumerate(a1):
        if not fuzzy_equal(el , 0.0, tolerance):
            first_nonzero_idx = i
            break
    two_one_ratio = a2[first_nonzero_idx] / a1[first_nonzero_idx]

    # If this ratio is zero, return False
    if fuzzy_equal(two_one_ratio, 0.0, tolerance=tolerance):
        return False

    # If the vectors aren't parallel, return False
    if not fuzzy_vector_equal(two_one_ratio * a1, a2, tolerance=tolerance):
        return False

    # If the biases aren't equal, return false, o.w. return True
    return fuzzy_equal(two_one_ratio * b1, b2, tolerance=tolerance)



def is_same_hyperplane(a1, b1, a2, b2, tolerance=global_tolerance):
    """ Given two hyperplanes of the form <a1, x> =b1, <a2, x> =b2
        this returns true if the two define the same hyperplane.

        Only works if these two are in 'comparison form'
    """
    # assert that we're in comparison form
    # --- check hyperplane 1 first
    for (a, b) in [(a1, b1), (a2, b2)]:
        if abs(b) < tolerance: # b ~ 0, then ||a|| ~ 1
            assert fuzzy_equal(np.linalg.norm(a), 1, tolerance=tolerance)
        else:
            # otherwise abs(b) ~ 1
            assert fuzzy_equal(abs(b), 1.0, tolerance=tolerance)

    # First check that b1, b2 are either +-1, or 0
    if not fuzzy_equal(abs(b1), abs(b2), tolerance=tolerance):
        return False

    # if b's are zero, then vectors need to be equal up to -1 factor
    if fuzzy_equal(b1, 0, tolerance=tolerance):
        return (fuzzy_vector_equal(a1, a2, tolerance=tolerance) or
                fuzzy_vector_equal(a1, -a2, tolerance=tolerance))


    # if b's are different signs, then a1/b1 ~ a2/b2
    return fuzzy_vector_equal(a1 / b1, a2 / b2, tolerance=tolerance)


def is_same_tight_constraint(a1, b1, a2, b2, tolerance=global_tolerance):
    """ Given tight constraint of the form <a1, x> <=b1, <a2, x> <=b2
        this returns true if the two define the same tight constraint.

        Only works if these two are in 'comparison form'
    """
    # assert that we're in comparison form
    # --- check hyperplane 1 first
    for (a, b) in [(a1, b1), (a2, b2)]:
        if abs(b) < tolerance: # b ~ 0, then ||a|| ~ 1
            assert fuzzy_equal(np.linalg.norm(a), 1, tolerance=tolerance)
        else:
            # otherwise abs(b) ~ 1
            assert fuzzy_equal(abs(b), 1.0, tolerance=tolerance)

    # First check that b1, b2 are either +-1, or 0
    if not fuzzy_equal(abs(b1), abs(b2), tolerance=tolerance):
        return False

    # if b's are zero, then vectors need to be equal up to -1 factor
    if fuzzy_equal(b1, 0, tolerance=tolerance):
        return (fuzzy_vector_equal(a1, a2, tolerance=tolerance) or
                fuzzy_vector_equal(a1, -a2, tolerance=tolerance))


    # check if a1 approx = a2 and b1 approx = b2
    return fuzzy_vector_equal(a1, a2, tolerance=tolerance) and fuzzy_equal(b1, b2, tolerance=tolerance)

##########################################################################
#                                                                        #
#                     Geometric Utilities                                #
#                                                                        #
##########################################################################

from mosek.fusion import *


def geometric_mean(M, x, t):
    '''
    Models the convex set

      S = { (x, t) \in R^n x R | x >= 0, t <= (x1 * x2 * ... * xn)^(1/n) }

    using three-dimensional power cones
    '''

    n = int(x.getSize())
    if n==1:
      M.constraint(Expr.sub(t, x), Domain.lessThan(0.0))
    else:
      t2 = M.variable()
      M.constraint(Var.hstack(t2, x.index(n-1), t), Domain.inPPowerCone(1-1.0/n))
      geometric_mean(M, x.slice(0,n-1), t2)



def det_rootn(M, t, n):
    '''
     Purpose: Models the hypograph of the n-th power of the
     determinant of a positive definite matrix. See [1,2] for more details.

       The convex set (a hypograph)

       C = { (X, t) \in S^n_+ x R |  t <= det(X)^{1/n} },

       can be modeled as the intersection of a semidefinite cone

       [ X, Z; Z^T Diag(Z) ] >= 0

       and a number of rotated quadratic cones and affine hyperplanes,

       t <= (Z11*Z22*...*Znn)^{1/n}  (see geometric_mean).
    '''

    # Setup variables
    Y = M.variable(Domain.inPSDCone(2 * n))

    # Setup Y = [X, Z; Z^T , diag(Z)]
    X   = Y.slice([0, 0], [n, n])
    Z   = Y.slice([0, n], [n, 2 * n])
    DZ  = Y.slice([n, n], [2 * n, 2 * n])

    # Z is lower-triangular
    M.constraint(Z.pick([[i,j] for i in range(n) for j in range(i+1,n)]), Domain.equalsTo(0.0))
    # DZ = Diag(Z)
    M.constraint(Expr.sub(DZ, Expr.mulElm(Z, Matrix.eye(n))), Domain.equalsTo(0.0))

    # t^n <= (Z11*Z22*...*Znn)
    geometric_mean(M, DZ.diag(), t)

    # Return an n x n PSD variable which satisfies t <= det(X)^(1/n)
    return X


def MVIE_ellipse(A, b):
    '''
      The inner ellipsoidal approximation to a polytope

         S = { x \in R^n | Ax < b }.

      maximizes the volume of the inscribed ellipsoid,

         { x | x = C*u + d, || u ||_2 <= 1 }.

      The volume is proportional to det(C)^(1/n), so the
      problem can be solved as

        maximize         t
        subject to       t       <= det(C)^(1/n)
                    || C*ai ||_2 <= bi - ai^T * d,  i=1,...,m
                    C is PSD

      which is equivalent to a mixed conic quadratic and semidefinite
      programming problem.
    '''

    A = A.tolist()
    b = b.tolist()
    try:
        with Model("lownerjohn_inner") as M:
            # M.setLogHandler(sys.stdout)   # output of solver
            m, n = len(A), len(A[0])

            # Setup variables
            t = M.variable("t", 1, Domain.greaterThan(0.0))
            C = det_rootn(M, t, n)
            d = M.variable("d", n, Domain.unbounded())

            # (b-Ad, AC) generate cones
            M.constraint("qc", Expr.hstack(Expr.sub(b, Expr.mul(A, d)), Expr.mul(A, C)),
                         Domain.inQCone())

            # Objective: Maximize t
            M.objective(ObjectiveSense.Maximize, t)

            M.solve()
            C, d = C.level(), d.level()
            C = [C[i:i + n] for i in range(0, n * n, n)]
            return C, d
    except:
        return None, None




def ray_shoot(x_0, d, ptope, facets, facet_bools=None):
    ''' Given a list of facets, an initial point 'x_0', and direction 'd', method finds the first
        constraint hit in that direction. 'Facet_bools' decides if that facet is considered in the
        list of potential facets.

        Modifies:   ptope.redundant '''

    if facet_bools is None:
        facet_bools = [True for _ in range(0, len(facets))]

    # Compute projection onto each considered facet in the direction d
    distances = []
    considered_indices = []
    for index, facet_bool in enumerate(facet_bools):
        if facet_bool:
            considered_indices.append(index)
            face = facets[index]
            dist = (face.b_eq[0] - np.inner(face.a_eq[0], x_0)) / np.inner(face.a_eq[0], d)
            if dist >= 0:
                distances.append(dist)
            else:
                distances.append(np.inf)


    # Sort distances, the closest is non-redundant
    tight_index = np.argsort(distances)[0]

    ptope.redundant[considered_indices[tight_index]] = False

    return x_0 + distances[tight_index]*d

##########################################################################
#                                                                        #
#                     Optimization Utilities                             #
#                                                                        #
##########################################################################

def gurobi_LP(A_ub, b_ub, a_eq, b_eq, c, bounds=None, options=None):
    ''' Solves Linear Program given as a minimization of:
        min    <c,x>
        s.t.   (A_ub)^T*x <= b_ub
               (a_eq)^T*x = b_eq
                bounds[i][0] <= x_i <= bounds[i][1]     for all i

        Returns:
                solved: True if correctly solved False o.w.
                opt_model: pulp object class of optimization prob
        '''
    m, n = np.shape(A_ub)
    m2, n2 = np.shape(a_eq)

    # Setup Optimization Model
    opt_model = plp.LpProblem(name="LP program")
    x_vars = [plp.LpVariable(cat=plp.LpContinuous,
                             lowBound=bounds[j][0], upBound=bounds[j][1],
                             name="x_{0}".format(j)) for j in range(0, n)]

    # Set Objective
    objective = plp.lpDot(x_vars, c)
    opt_model.sense = plp.LpMinimize
    opt_model.setObjective(objective)

    # Less than equal constraints
    for i in range(0, m):
        opt_model.addConstraint(plp.LpConstraint(
            e=plp.lpDot(A_ub[i], x_vars),
            sense=plp.LpConstraintLE,
            rhs=b_ub[i],
            name="constraint_{0}".format(i)))

    # Equality Constraints
    for i in range(0, m2):
        opt_model.addConstraint(plp.LpConstraint(
            e=plp.lpDot(a_eq[i], x_vars),
            sense=plp.LpConstraintEQ,
            rhs=b_eq[i],
            name="eq_constraint_{0}".format(i)))

    print(opt_model)

    plp.GUROBI_CMD(msg=0).solve(opt_model)
    solved = (opt_model.status == 1)    # 1 if solved

    return solved, opt_model


##########################################################################
#                                                                        #
#                     Geometric Utilities                                #
#                                                                        #
##########################################################################

from mosek.fusion import Expr, Domain, Matrix, Var, Model, ObjectiveSense


def geometric_mean(M, x, t):
    '''
    Models the convex set

      S = { (x, t) \in R^n x R | x >= 0, t <= (x1 * x2 * ... * xn)^(1/n) }

    using three-dimensional power cones
    '''
    try:
        n = int(x.getSize())
    except:
        n = x.size()

    if n==1:
      M.constraint(Expr.sub(t, x), Domain.lessThan(0.0))
    else:
      t2 = M.variable()
      M.constraint(Var.hstack(t2, x.index(n-1), t), Domain.inPPowerCone(1-1.0/n))
      geometric_mean(M, x.slice(0,n-1), t2)



def det_rootn(M, t, n):
    '''
     Purpose: Models the hypograph of the n-th power of the
     determinant of a positive definite matrix. See [1,2] for more details.

       The convex set (a hypograph)

       C = { (X, t) \in S^n_+ x R |  t <= det(X)^{1/n} },

       can be modeled as the intersection of a semidefinite cone

       [ X, Z; Z^T Diag(Z) ] >= 0

       and a number of rotated quadratic cones and affine hyperplanes,

       t <= (Z11*Z22*...*Znn)^{1/n}  (see geometric_mean).
    '''

    # Setup variables
    Y = M.variable(Domain.inPSDCone(2 * n))

    # Setup Y = [X, Z; Z^T , diag(Z)]
    X   = Y.slice([0, 0], [n, n])
    Z   = Y.slice([0, n], [n, 2 * n])
    DZ  = Y.slice([n, n], [2 * n, 2 * n])

    # Z is lower-triangular
    M.constraint(Z.pick([[i,j] for i in range(n) for j in range(i+1,n)]), Domain.equalsTo(0.0))
    # DZ = Diag(Z)
    M.constraint(Expr.sub(DZ, Expr.mulElm(Z, Matrix.eye(n))), Domain.equalsTo(0.0))

    # t^n <= (Z11*Z22*...*Znn)
    geometric_mean(M, DZ.diag(), t)

    # Return an n x n PSD variable which satisfies t <= det(X)^(1/n)
    return X


def MVIE_ellipse(A, b):
    '''
      The inner ellipsoidal approximation to a polytope

         S = { x \in R^n | Ax < b }.

      maximizes the volume of the inscribed ellipsoid,

         { x | x = C*u + d, || u ||_2 <= 1 }.

      The volume is proportional to det(C)^(1/n), so the
      problem can be solved as

        maximize         t
        subject to       t       <= det(C)^(1/n)
                    || C*ai ||_2 <= bi - ai^T * d,  i=1,...,m
                    C is PSD

      which is equivalent to a mixed conic quadratic and semidefinite
      programming problem.
    '''

    A = A.tolist()
    b = b.tolist()
    with Model("lownerjohn_inner") as M:
        # M.setLogHandler(sys.stdout)   # output of solver
        m, n = len(A), len(A[0])

        # Setup variables
        t = M.variable("t", 1, Domain.greaterThan(0.0))
        C = det_rootn(M, t, n)
        d = M.variable("d", n, Domain.unbounded())

        # (b-Ad, AC) generate cones
        M.constraint("qc", Expr.hstack(Expr.sub(b, Expr.mul(A, d)), Expr.mul(A, C)),
                     Domain.inQCone())

        # Objective: Maximize t
        M.objective(ObjectiveSense.Maximize, t)

        M.solve()
        C, d = C.level(), d.level()
        C = [C[i:i + n] for i in range(0, n * n, n)]
        return C, d


##########################################################################
#                                                                        #
#                         Plotting Utilities                             #
#                                                                        #
##########################################################################

def plot_polytopes_2d(poly_list, colors=None, alpha=1.0,
                   xylim=5, ax=plt.axes(), linestyle='dashed', linewidth=0):
    """Plots a list of polytopes which exist in R^2.
    """
    if colors == None:
        colors = [np.random.rand(3) for _ in range(0, len(poly_list))]

    if(np.size(xylim)==1):
        xlim = [0, xylim]
        ylim = [0, xylim]
    else:
        xlim = xylim
        ylim = xylim

    for poly, color in zip(poly_list, colors):
        P = Polytope_2(poly.ub_A, poly.ub_b)
        V = ptope.extreme(P)

        if V is not None:
            P.plot(ax, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        else:
            # Polytope may be unbounded, thus add additional constraints x in [-xylim, xylim]
            # and y in [-xylim, xylim]
            new_ub_A = np.vstack((poly.ub_A, [[1,0],[-1,0],[0,1],[0,-1]]))
            new_ub_b = np.hstack((poly.ub_b, [xlim[1], -1*xlim[0], ylim[1], -1*ylim[0]]))
            P2 = Polytope_2(new_ub_A, new_ub_b)
            V2 = ptope.extreme(P2)
            if V2 is not None:
                P2.plot(ax, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
                print('an unbounded polytope was plotted imperfectly')
            else:
                print('polytope not plotted')

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])


def plot_facets_2d(facet_list, alpha=1.0,
                   xylim=5, ax=plt.axes(), linestyle='solid', linewidth=3, color='black'):
    """Plots a list of facets which exist as line segments in R^2
    """
    if(np.size(xylim)==1):
        xlim = [0, xylim]
        ylim = [0, xylim]
    else:
        xlim = xylim
        ylim = xylim

    for facet in facet_list:
        P = Polytope_2(facet.ub_A, facet.ub_b)
        vertices = ptope.extreme(P)

        facet_vertices = []

        if vertices is not None and np.shape(vertices)[0] > 1:
            for vertex in vertices:
                equal = fuzzy_equal(np.dot(facet.a_eq[0], vertex), facet.b_eq[0])
                if equal:
                    facet_vertices.append(vertex)
            x1 = facet_vertices[0][0]; x2 = facet_vertices[1][0]
            y1 = facet_vertices[0][1]; y2 = facet_vertices[1][1]
            x = [x1, x2]; y = [y1, y2]

            ax.plot(x, y, c=color, linestyle=linestyle, linewidth=linewidth)

        else:
            new_ub_A = np.vstack((facet.ub_A, [[1, 0], [-1, 0], [0, 1], [0, -1]]))
            new_ub_b = np.hstack((facet.ub_b, [xlim[1], -1 * xlim[0], ylim[1], -1 * ylim[0]]))
            P2 = Polytope_2(new_ub_A, new_ub_b)
            V2 = ptope.extreme(P2)
            if V2 is not None:
                P2.plot(ax, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
                print('an unbounded facet was plotted imperfectly')
            else:
                print('facet not plotted')

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

def plot_linf_norm(x_0, t, linewidth=1, edgecolor='black', ax=None):
    """Plots linf norm ball of size t centered at x_0 (only in R^2)
    """
    x_0 = as_numpy(x_0).reshape(2)
    rect = patches.Rectangle((x_0[0]-t, x_0[1]-t), 2*t, 2*t, linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
    ax.add_patch(rect)

def plot_l2_norm(x_0, t, linewidth=1, edgecolor='black', ax=plt.axes()):
    """Plots l2 norm ball of size t centered at x_0 (only in R^2)
    """
    x_0 = as_numpy(x_0).reshape(2)
    circle = plt.Circle(x_0, t, color=edgecolor, fill=False)
    ax.add_artist(circle)


def plot_hyperplanes(ub_A, ub_b, styles=None, ax=plt.axes()):
    ''' Plots all hyperplanes defined by each constraint of ub_A and ub_b'''

    if styles is None:
        styles = ['-' for _ in range(0, np.shape(ub_A)[0])]

    for a, b, style in zip(ub_A, ub_b, styles):
        m = -a[0]/a[1]
        intercept = b/a[1]
        plot_line(m, intercept, style, ax)


def get_spaced_colors(n):
    """Given number, n, returns n colors which are visually well distributed
    """
    max_value = 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16) / 255.0, int(i[2:4], 16) / 255.0, int(i[4:], 16) / 255.0, 1) for i in colors]


def get_color_dictionary(list):
    """Creates a dictionary of evenly spaced colors, keys are elements in provided lists
    """
    n = len(list)
    colors = get_spaced_colors(n)
    color_dict = {}

    for element, color in zip(list, colors):
        color_dict[element] = color

    return color_dict

def plot_line(slope, intercept, style, ax=plt.axes()):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, style, c='black')

def plot_ellipse(P, c, ax=plt.axes()):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = P[0][0] * np.cos(theta) + P[0][1] * np.sin(theta) + c[0]
    y = P[1][0] * np.cos(theta) + P[1][1] * np.sin(theta) + c[1]
    ax.plot(x, y)

# ------------------------------------
# Polytope class from PyPi
# ------------------------------------
""" Modified class is used for plotting polytopes and utilizing other functions
    from 'polytope' library
"""

class Polytope_2(ptope.Polytope):
    def __init__(
            self, A=np.array([]), b=np.array([]), minrep=False,
            chebR=0, chebX=None, fulldim=None,
            volume=None, vertices=None, normalize=True):
        super(Polytope_2, self).__init__(A, b, minrep, chebR, chebX,
                                            fulldim, volume, vertices, normalize)

    def plot(self, ax=None, color=None,
             hatch=None, alpha=1.0, linestyle='dashed', linewidth=3):
        if self.dim != 2:
            raise Exception("Cannot plot polytopes of dimension larger than 2")
        ax = _newax(ax)
        if not ptope.is_fulldim(self):
            print("Cannot plot empty polytope")
            return None
        if color is None:
            color = np.random.rand(3)
        poly = _get_patch(
            self, facecolor=color, hatch=hatch,
            alpha=alpha, linestyle=linestyle, linewidth=linewidth,
            edgecolor='black')
        ax.add_patch(poly)
        return ax

# --------------------------------------------------------------------------------------
# Ugly Plotting Code
# --------------------------------------------------------------------------------------

def binarize_relu_configs(relu_configs):
    """ Takes a list of relu configs and turns them into one long binary string
        (each element i of relu_configs is assumed to be an array of relu acts at layer i)
    """
    long_code = [element.data.numpy() for code in relu_configs for element in code]
    bin_code = np.asarray(long_code).astype(int)
    return bin_code

def get_unique_relu_configs(network, xylim, numpts):
    #TODO: repetition in what is returned
    """ Samples within a square of size (xylim[0]) x (xylim[1]), and returns the unique
        ReLu activations. Total number of samples is numpts^2

        Returns: unique relu_configs    =>  (list of arrays of unique ReLu acts)
                 xs                     =>  (points sampled)
                 num_activations        =>  (list of unique numbers associated with Relu acts)
    """

    if(np.size(xylim)==1):
        xlim = [0, xylim]
        ylim = [0, xylim]
    else:
        xlim = xylim
        ylim = xylim

    num_activations = []
    relu_configs_list = []
    xs = []

    for x in np.linspace(xlim[0], xlim[1], numpts):
        for y in np.linspace(ylim[0], ylim[1], numpts):
            pt_0 = torch.Tensor([x, y]).type(torch.float32)
            relu_configs = network.relu_config(pt_0, False)
            bin_code = binarize_relu_configs(relu_configs)
            bin_list = np.ndarray.tolist(bin_code)
            num = BitArray(bin_list).uint

            flag = False
            for previous_num in num_activations:
                if num == previous_num:
                    flag = True
                    continue

            if flag == False:
                num_activations.append(num)
                relu_configs_list.append(relu_configs)

            xs.append(pt_0.data.numpy())

    return relu_configs_list, num_activations, xs, num_activations



def plot_network_polytopes_sloppy(network, xylim, numpts, legend_flag=False):
    """ Roughly plots polytopes in 2d for given network. Samples within a square of size
        (xylim) x (xylim), and plots an identifying color for each unique ReLu configuration.
        Total number of samples is numpts^2
    """
    _, unique_bin_acts, xs, num_activations = get_unique_relu_configs(network, xylim, numpts)
    color_dict = get_color_dictionary(unique_bin_acts)
    plt.figure(figsize=(10, 10))

    for unique_act in unique_bin_acts:
        indices = [index for index, element in enumerate(num_activations) if element == unique_act]
        x_pts_to_plot = [xs[index][0] for index in indices]
        y_pts_to_plot = [xs[index][1] for index in indices]
        colors = [color_dict[unique_act] for _ in range(0, len(x_pts_to_plot))]
        plt.scatter(x_pts_to_plot, y_pts_to_plot, label=str(unique_act), c=colors)

    print('num_unique_activations', len(unique_bin_acts))
    if (legend_flag):
        plt.legend()

def expand_xylim(alpha, xlim, ylim=None):
    if ylim is None:
        ylim = xlim
    expander = lambda lim, alpha: [elem - np.power(-1, i)*alpha*np.abs(lim[1]-lim[0]) for i, elem in enumerate(lim)]

    return expander(xlim, alpha), expander(ylim, alpha)

##########################################################################
#                                                                        #
#                               SAFETY DANCE                             #
#                                                                        #
##########################################################################

def as_numpy(tensor_or_array):
    """ If given a tensor or numpy array returns that object cast numpy array
    """

    if isinstance(tensor_or_array, torch.Tensor):
        tensor_or_array = tensor_or_array.cpu().detach().numpy()
    return tensor_or_array

##########################################################################
#                                                                        #
#                            Misc. Utilities                             #
#                                                                        #
##########################################################################

def star_arg(fxn):
    """ Maps function taking multiple arguments to function taking a single
        tuple of args
    """
    def star_fxn(args, fxn=fxn):
        return fxn(*args)
    return star_fxn


def _newax(ax=None):
    """Add subplot to current figure and return axes."""
    from matplotlib import pyplot as plt
    if ax is not None:
        return ax
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return ax

def _get_patch(poly1, **kwargs):
    """Return matplotlib patch for given Polytope.

    Example::

    > # Plot Polytope objects poly1 and poly2 in the same plot
    > import matplotlib.pyplot as plt
    > fig = plt.figure()
    > ax = fig.add_subplot(111)
    > p1 = _get_patch(poly1, color="blue")
    > p2 = _get_patch(poly2, color="yellow")
    > ax.add_patch(p1)
    > ax.add_patch(p2)
    > ax.set_xlim(xl, xu) # Optional: set axis max/min
    > ax.set_ylim(yl, yu)
    > plt.show()

    @type poly1: L{Polytope}
    @param kwargs: any keyword arguments valid for
        matplotlib.patches.Polygon
    """
    import matplotlib as mpl
    V = ptope.extreme(poly1)

    if (V is not None):
        rc, xc = ptope.cheby_ball(poly1)
        x = V[:, 1] - xc[1]
        y = V[:, 0] - xc[0]
        mult = np.sqrt(x**2 + y**2)
        x = x / mult
        angle = np.arccos(x)
        corr = np.ones(y.size) - 2 * (y < 0)
        angle = angle * corr
        ind = np.argsort(angle)
        # create patch
        patch = mpl.patches.Polygon(V[ind, :], True, **kwargs)
        patch.set_zorder(0)

    else:
        patch = mpl.patches.Polygon([], True, **kwargs)
        patch.set_zorder(0)
    return patch