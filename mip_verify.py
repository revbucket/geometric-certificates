""" Quicker and even dirtier code to compute the MIP verification for
    the min-dist problem
"""

import numpy as np
import gurobipy as gb
import utilities as utils
import full_lp
from domains import Domain
import time


##############################################################################
#                                                                            #
#                       MAIN SOLVER METHOD                                   #
#                                                                            #
##############################################################################


def mip_solve_linf(network, x, radius=None, problem_type='min_dist',
              lp_norm='l_inf', box_bounds=None):
    """ Computes the decision problem for MIP :
    - first computes the LP for each neuron to get pre-relu actviations
    - then loops through all logits to compute decisions
    """

    dom = Domain(x.numel(), x)
    if box_bounds is not None:
        dom.set_original_hyperbox_bound(*box_bounds)
    else:
        dom.set_original_hyperbox_bound(0.0, 1.0)

    assert problem_type in ['decision_problem', 'min_dist']
    if problem_type == 'decision_problem':
        assert radius is not None
        dom.set_upper_bound(radius, lp_norm)

    # Build domain and shrink if only doing a decision problem

    start = time.time()
    pre_relu_bounds = full_lp.compute_full_lp_bounds(network, dom,
                                                     compute_logit_bounds=True)

    print("COMPUTED FULL-LP BOUNDS IN %.03f seconds" % (time.time() - start))
    true_label = network(x).max(1)[1].item()
    num_logits = network(x).numel()
    solved_models = []

    model = build_mip_model(network, x, dom, pre_relu_bounds,
                            true_label, problem_type, radius, lp_norm)

    model.optimize()

    if model.Status == 3:
        print("INFEASIBLE!")

    return model




def build_mip_model(network, x, domain, pre_relu_bounds, true_label,
                    problem_type, radius, lp_norm):
    """
    ARGS:
        network : plnn.PLNN - network we wish to compute bounds on
        x : Tensor or numpy of the point we want to verify
        domain : domain.Domain - domain restricting the input domain
        pre_relu_bounds : list of np arrays of shape [#relu x 2] -
                          holds the upper/lower bounds for each pre_relu
                          (and the logits)
        true_label : int - what the model predicts for x
        problem_type: 'min_dist' or 'decision_problem'
        radius:  float - l_inf ball that we are 'deciding' on for
                 'decision_problem' variant
    """



    ##########################################################################
    #   Step 1: setup things we'll need throughout                           #
    ##########################################################################

    num_pre_relu_layers = len(network.fcs) - 1
    # - build model, add variables and box constraints
    model = gb.Model()
    # model.setParam('OutputFlag', False) # -- uncomment to suppress gurobi logs
    x_np = utils.as_numpy(x).reshape(-1)



    assert domain.box_low is not None
    assert domain.box_high is not None

    box_bounds = zip(domain.box_low, domain.box_high)
    x_namer = build_var_namer('x')
    x_vars = [model.addVar(lb=low, ub=high, name= x_namer(i))
                for i, (low, high) in enumerate(box_bounds)]
    var_dict = {'x': x_vars}

    # if l_2, and the radius is not None, add those constraints as well
    l2_norm = gb.quicksum((x_vars[i] - x_np[i]) * (x_vars[i] - x_np[i])
                          for i in range(len(x_vars)))
    model.addConstr(l2_norm <= radius ** 2)

    model.update()


    ##########################################################################
    #   Step 2: Now add layers iteratively                                   #
    ##########################################################################

    # all layers except the last final layer
    for i, fc_layer in enumerate(network.fcs[:-1]):
        # add linear layer
        if i == 0:
            input_name = 'x'
        else:
            input_name = 'fc_%s_post' % i

        pre_relu_name = 'fc_%s_pre' % (i + 1)
        post_relu_name = 'fc_%s_post' % (i + 1)
        relu_name = 'relu_%s' % (i + 1)
        add_linear_layer_mip(network, i, model, var_dict, input_name,
                             pre_relu_name)
        add_relu_layer_mip(network, i, model, var_dict, pre_relu_name,
                           pre_relu_bounds[i], post_relu_name, relu_name)

    # add the final fully connected layer
    output_var_name = 'logits'

    add_linear_layer_mip(network, len(network.fcs) - 1, model, var_dict,
                         post_relu_name, output_var_name)

    ##########################################################################
    #   Step 3: Add the 'adversarial' constraint and objective               #
    ##########################################################################
    add_adversarial_constraint(model, var_dict[output_var_name], true_label,
                               pre_relu_bounds[-1])


    if lp_norm == 'l_inf':
        add_l_inf_obj(model, x_np, var_dict['x'], problem_type)
    else:
        add_l_2_obj(model, x_np, var_dict['x'], problem_type)

    model.update()
    return model


######################################################################
#                                                                    #
#                           HELPER FUNCTIONS                         #
#              (builds layers, objective, adversarial constraint)    #
######################################################################



def add_linear_layer_mip(network, layer_no, model, var_dict, var_input_key,
                         var_output_key):
    """ Method to add the variables and constraints to handle a linear layer
    """
    fc_layer = network.fcs[layer_no]

    fc_weight = utils.as_numpy(fc_layer.weight)
    if fc_layer.bias is not None:
        fc_bias = utils.as_numpy(fc_layer.bias)
    else:
        fc_bias = np.zeros(fc_layer.out_features)

    input_vars = var_dict[var_input_key]
    relu = lambda el: max([el, 0.0])

    # add the variables and constraints for the pre-relu layer

    var_namer = build_var_namer(var_output_key)
    pre_relu_vars = [model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY,
                                  name=var_namer(i))
                     for i in range(fc_layer.out_features)]
    var_dict[var_output_key] = pre_relu_vars
    model.addConstrs((pre_relu_vars[i] ==\
                      gb.LinExpr(fc_weight[i], input_vars) + fc_bias[i])
                     for i in range(fc_layer.out_features))
    model.update()

    return


def add_relu_layer_mip(network, layer_no, model, var_dict, var_input_key,
                       input_bounds, post_relu_var_names,
                       relu_config_var_names):
    """ Method to add the variables and constraints to handle a ReLU layer
    """

    post_relu_vars = []
    relu_vars = []
    post_relu_namer = build_var_namer(post_relu_var_names)
    relu_namer = build_var_namer(relu_config_var_names)
    #input bounds are the pre-relu bound
    for i, (low, high) in enumerate(input_bounds):
        post_relu_name = post_relu_namer(i)
        relu_name = relu_namer(i)
        if high <= 0:
            # If always off, don't add an integral constraint
            post_relu_vars.append(model.addVar(lb=0.0, ub=0.0,
                                               name=post_relu_name))
        else:

            pre_relu = var_dict[var_input_key][i]
            post_relu_vars.append(model.addVar(lb=low, ub=high,
                                               name=post_relu_name))
            post_relu = post_relu_vars[-1]
            if low >= 0:
                # If always on, enforce equality
                model.addConstr(post_relu == pre_relu)
            else:
                # If unstable, add tightest possible relu constraints
                relu_var = model.addVar(lb=0.0, ub=1.0, vtype=gb.GRB.BINARY,
                                        name=relu_name)
                relu_vars.append(relu_var)

                # y <= x - l(1 - a)
                model.addConstr(post_relu <= pre_relu - low * (1 - relu_var))

                # y >= x
                model.addConstr(post_relu >= pre_relu)

                # y <= u * a
                model.addConstr(post_relu <= high * relu_var)

                # y >= 0
                model.addConstr(post_relu >= 0)
    model.update()
    var_dict[post_relu_var_names] = post_relu_vars
    var_dict[relu_config_var_names] = relu_vars

    return


def add_adversarial_constraint(model, logit_vars, true_label, logit_bounds):
    """ Adds the adversarial constraint to the model
        Two cases here:
            1) only two valid logits could be maximal, so
    """

    if len(logit_vars) == 2:
        model.addConstr(logit_vars[true_label] <= logit_vars[1 - true_label])

    # First collect all potential max labels that aren't the true label
    highest_low = max(logit_bounds[:, 0])
    target_labels = []
    for i in range(len(logit_vars)):
        this_high = logit_bounds[i][1]
        if (i == true_label) or (this_high <= highest_low):
            continue
        target_labels.append(i)


    if len(target_labels) == 1:
        # Trivial case
        model.addConstr(logit_vars[true_label] <= logit_vars[target_labels[0]])
        return


    ##########################################################################
    #   If multiple target labels, we have to add a max layer                #
    ##########################################################################

    # Generate a max logit variable (which is greater than all target logits)
    max_logit_var = model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY,
                                 name='max_logit')
    for i in target_labels:
        model.addConstr(max_logit_var >= logit_vars[i])

    # And max logit integer variables (where onnly 1 can be on at a time)
    max_logit_ints = {i: model.addVar(lb=0.0, ub=1.0, vtype=gb.GRB.BINARY,
                                      name='is_max_logit[%s]' % i)
                      for i in target_labels}

    model.addConstr(gb.quicksum(list(max_logit_ints.values())) == 1)


    # Add upper bound constraints on max's
    for i in target_labels:
        high_max_not_i = max(_[1] for j, _ in enumerate(logit_bounds)
                             if (j != i) and (j in target_labels))
        rhs = (1 - max_logit_ints[i]) * (high_max_not_i - logit_bounds[i][0])
        model.addConstr(max_logit_var <= rhs)


    # Finally add in the adversarial constraint
    model.addConstr(logit_vars[true_label] <= max_logit_var)
    model.update()



def add_l_inf_obj(model, x_np, x_vars, problem_type):
    """ Adds objective to minimize the l_inf distance from the original input x
    ARGS:
        x_np: numpy vector for the original fixed point we compute robustness for
        x_vars : list of variables representing input to the MIP
    """

    if problem_type == 'decision_problem':
        model.setObjective(0, gb.GRB.MINIMIZE)
    elif problem_type == 'min_dist':
        # min t  such that |x_var-x_np|_i <= t
        t_var = model.addVar(lb=0, ub=gb.GRB.INFINITY, name='t')
        for coord, val in enumerate(x_np):
            model.addConstr(t_var >= x_vars[coord] - val)
            model.addConstr(t_var >= val - x_vars[coord])
        model.setObjective(t_var, gb.GRB.MINIMIZE)

    model.update()


def add_l_2_obj(model, x_np, x_vars, problem_type):
    """ Adds the constraint for the l2 norm case """
    if problem_type == 'decision_problem':
        model.setObjective(0, gb.GRB.MINIMIZE)
    elif problem_type == 'min_dist':
        t_var = model.addVar(lb=0, ub=gb.GRB.INFINITY, name='t')
        l2_norm = gb.quicksum((x_vars[i] - x_np[i]) * (x_vars[i] - x_np[i])
                              for i in range(len(x_vars)))
        model.addConstr(l2_norm <= t_var)
        model.setObjective(t_var, gb.GRB.MINIMIZE)
    model.update()


###############################################################################
#                                                                             #
#                               SILLY UTILITIES                               #
#                                                                             #
###############################################################################

def build_var_namer(pfx):
    return lambda i: '%s[%s]' % (pfx, i)
