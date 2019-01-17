import numpy as np
import torch


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
        hamming_dist += uneqs.sum()
    return hamming_dist


def flatten_config(config):
    """ Takes a list of floatTensors where each element is either 1 or 0
        and converts into a string of 1s and 0s.
        Is just a binary representation of the neuron config
    """
    cat_config = torch.cat([_.cpu().type(torch.uint8).detach() for _ in config])
    return ''.join(str(_) for _ in cat_config.numpy())



##############################################################################
#                                                                            #
#                       MATRIX OPERATIONS/ EQUALITY CHECKS                   #
#                                                                            #
##############################################################################


def comparison_form(A, b, tolerance=1e-8):
    """ Given polytope Ax<= b
        Convert each constraint into a_i^Tx <= +-1, 0
        If b_i=0, normalize a_i
        Then sort rows of A lexicographically

    A is a 2d numpy array of shape (m,n)
    b is a 1d numpy array of shape (m)
    """
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


def fuzzy_equal(x, y, tolerance=1e-8):
    """ Fuzzy float equality check. Returns true if x,y are within tolerance
        x, y are scalars
    """
    return abs(x - y) < tolerance


def fuzzy_vector_equal(x_vec, y_vec, tolerance=1e-8):
    """ Same as above, but for vectors.
        x_vec, y_vec are 1d numpy arrays
     """
    return all(abs(el) < tolerance for el in x_vec - y_vec)

def is_same_hyperplane(a1, b1, a2, b2, tolerance=1e-8):
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


