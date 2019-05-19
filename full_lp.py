""" Quick and dirty code to compute the full Linear program bounds for each
    neuron (min and max of each pre-ReLU activation)
"""
import numpy as np
import gurobipy as gb
import utilities as utils

def build_var_namer(pfx):
    return lambda i: '%s[%s]' % (pfx, i)

def compute_full_lp_bounds(network, domain, compute_logit_bounds=False):
    """
    ARGS:
        network : plnn.PLNN - network we wish to compute bounds on
        domain  : domain.Domain - domain restricting the input domain
    RETURNS:
        ????
    """

    ##########################################################################
    #   Step 1: Setup things we'll need throughout                           #
    ##########################################################################


    num_pre_relu_layers = len(network.fcs) - 1
    # - build model, add variables and box constraints
    model = gb.Model()
    model.setParam('OutputFlag', False)

    assert domain.box_low is not None
    assert domain.box_high is not None

    box_bounds = zip(domain.box_low, domain.box_high)

    x_namer = build_var_namer('x')
    var_dict = {'x': [model.addVar(lb=low, ub=high, name= x_namer(i))
                                   for i, (low, high) in enumerate(box_bounds)]
               }
    model.update()
    # - Set up bounds to propagate
    all_pre_relu_bounds = []

    ##########################################################################
    #   Step 2: Handle first layer separately  (this can be done without LP) #
    ##########################################################################

    # -- compute the pre-relu bounds after the first fully connected layer (IA)
    fc1 = network.fcs[0]
    fc1_weight = utils.as_numpy(fc1.weight)
    fc1_weight_pos, fc1_weight_neg = utils.split_tensor_pos(fc1.weight)
    fc1_weight_pos = utils.as_numpy(fc1_weight_pos)
    fc1_weight_neg = utils.as_numpy(fc1_weight_neg)

    if fc1.bias is not None:
        fc1_bias = utils.as_numpy(fc1.bias)
    else:
        fc1_bias = np.zeros(fc1.out_features)

    layer_1_high = (fc1_weight_pos.dot(domain.box_high) +
                    fc1_weight_neg.dot(domain.box_low) + fc1_bias)

    layer_1_low = (fc1_weight_neg.dot(domain.box_high) +
                   fc1_weight_pos.dot(domain.box_low) + fc1_bias)
    pre_relu_bounds = np.hstack((layer_1_low.reshape((-1, 1)),
                                 layer_1_high.reshape((-1, 1))))
    all_pre_relu_bounds.append(pre_relu_bounds)

    # -- add the new pre_relu variables and their equality constraints
    pre_namer = build_var_namer('fc1_pre')
    var_dict['fc1_pre'] = [model.addVar(lb=low, ub=high, name=pre_namer(i))
                           for i, (low, high) in enumerate(pre_relu_bounds)]
    model.addConstrs((var_dict['fc1_pre'][i] ==\
                      gb.LinExpr(fc1_weight[i], var_dict['x']) + fc1_bias[i])
                     for i in range(fc1.out_features))

    # -- add the variables and constraints for the first ReLU layer
    add_relu_layer_vars_constrs(network, model, var_dict, 'fc1_pre',
                                pre_relu_bounds, 'fc1_post')



    ##########################################################################
    #   Step 3: Handle each of the remaining layers                          #
    ##########################################################################

    # Repeat this procedure for all fully connected layers
    for layer_no in range(1, num_pre_relu_layers):
        old_post_key = 'fc%s_post' % layer_no
        new_pre_key = 'fc%s_pre' % (layer_no + 1)
        new_post_key = 'fc%s_post' % (layer_no + 1)
        old_pre_relu_bounds = all_pre_relu_bounds[-1]
        new_pre_relu_bounds = add_linear_layer_vars_constrs(network, layer_no,
                                                            model, var_dict,
                                                            old_post_key,
                                                            old_pre_relu_bounds,
                                                            new_pre_key)
        all_pre_relu_bounds.append(new_pre_relu_bounds)
        add_relu_layer_vars_constrs(network, model, var_dict, new_pre_key,
                                    new_pre_relu_bounds, new_post_key)



    ##########################################################################
    #   Step 4: Return the output bounds                                     #
    ##########################################################################

    # If desired, compute the logit bounds too
    if compute_logit_bounds:
        layer_no = num_pre_relu_layers

        post_key = 'fc%s_post' % (layer_no)
        print('-' * 40)
        print(len(network.fcs))
        print(var_dict[post_key])
        logit_bounds = add_linear_layer_vars_constrs(network, layer_no, model,
                                                     var_dict, post_key,
                                                     all_pre_relu_bounds[-1],
                                                    'logits',
                                                    compute_bounds=True)
        all_pre_relu_bounds.append(logit_bounds)

    return all_pre_relu_bounds


def add_linear_layer_vars_constrs(network, layer_no, model, var_dict,
                                  var_input_key, input_bounds, var_output_key,
                                  compute_bounds=True):
    """ Method to add the variables and constraints to handle a linear layer
    ARGS:
        network : PLNN object
        layer_no : int of which fully connected layer we care about
        model: gurobi model we're building up
        var_dict : dict - values point to lists of gurobi Variables
        var_input_key : string - key pointing to the inputs to this linear layer
                                 (as variables in var_dict)
        input_bounds : numpy array of shape (in_features, 2): bounds for the
                       previous layer's pre-relu inputs.
                --NOTE: we're given the prev.pre-relu bounds, but the
                        prev.post-relus will be the inputs to the linear layer
                        hence the bounds for variables will be
                       (max(low, 0), high)

        var_output_key : string - key for where the output variables will be
                                  stored (as variables in var_dict)
        compute_bounds : bool - if True, we compute the pre-relu bounds for
                                each out_feature
    RETURNS:
        None, if compute_bounds is False,
        the pre_relu bounds as a numpy array of shape (out_features, 2)
    """

    # Read the parameters from the network
    fc_layer = network.fcs[layer_no]
    print("LAYER", layer_no)
    print(fc_layer.in_features, fc_layer.out_features)
    fc_weight = utils.as_numpy(fc_layer.weight)
    if fc_layer.bias is not None:
        fc_bias = utils.as_numpy(fc_layer.bias)
    else:
        fc_bias = np.zeros(fc_layer.out_features)

    input_vars = var_dict[var_input_key]
    relu = lambda el: max([el, 0.0])

    # Make new variables with equality constraints to handle the linear layer
    var_namer = build_var_namer(var_output_key)
    pre_relu_vars = [model.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY,
                                  name=var_namer(i))
                for i in range(fc_layer.out_features)]
    var_dict[var_output_key] = pre_relu_vars
    model.addConstrs((pre_relu_vars[i] ==\
                      gb.LinExpr(fc_weight[i], input_vars) + fc_bias[i])
                     for i in range(fc_layer.out_features))
    model.update()

    # If we don't want to compute bounds, just return
    if not compute_bounds:
        return None


    # Otherwise setup and solve all of the linear programs
    new_pre_relus = np.zeros((fc_layer.out_features, 2))
    for i, var in enumerate(pre_relu_vars):
        for j, obj in enumerate([gb.GRB.MINIMIZE, gb.GRB.MAXIMIZE]):
            model.setObjective(var, obj)
            model.update()
            model.optimize()
            if model.Status != 2:
                print("WHAT HAPPENED TO MY OPTIMIZATION???")
            new_pre_relus[i][j] = model.getObjective().getValue()
    return new_pre_relus


def add_relu_layer_vars_constrs(network, model, var_dict,
                                var_input_key, input_bounds, var_output_key):
    """ Method to add the variables and constraints to handle a linear layer
    ARGS:
        network : PLNN object
        model: gurobi model we're building up
        var_dict : dict - values point to lists of gurobi Variables
        var_input_key : string - key pointing to the inputs to this reLu layer
                                 (as variables in var_dict)
        input_bounds : numpy array of shape (out_features, 2): bounds for the
                       inputs to this relu layer
        var_output_key : string - key for where the output variables will be
                                  stored (as variables in var_dict)
    RETURNS:
        None
    """
    post_relu_vars = []
    for i, (low, high) in enumerate(input_bounds):
        var_name = '%s[%s]' % (var_output_key, i)
        if high <= 0:
            post_relu_vars.append(model.addVar(lb=0.0, ub=0.0, name=var_name))
        else:
            # Handle the not-always-off cases
            pre_relu = var_dict[var_input_key][i]
            post_relu_vars.append(model.addVar(lb=low, ub=high, name=var_name))
            post_relu = post_relu_vars[-1]
            if low >= 0:
                # Equality constraint in always-on-case
                model.addConstr(post_relu == pre_relu)
            else:
                # Convex upper envelope in unstable case
                model.addConstr(post_relu >= pre_relu)
                interval_width = (high - low)
                slope = high / interval_width
                intercept = -low * slope
                model.addConstr(post_relu <= slope * pre_relu + intercept)
    var_dict[var_output_key] = post_relu_vars
    model.update()



