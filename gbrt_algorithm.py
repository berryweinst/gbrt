from trees_data_structures import *
import numpy as np
import pandas as pd
from time import time


def get_percentiles(data, num_thresholds):
    percent_step = 1.0 / (num_thresholds+1)
    percentiles = np.arange(percent_step, 1.0, percent_step, dtype=float)
    percentiles_values = data.quantile(percentiles, 'nearest').unique()
    return percentiles_values


def get_optimal_partition(data, label_name, params):
    x = data.drop(label_name, axis=1)
    y = data[label_name]

    min_so_far = np.inf
    best_col_name, best_split_value = None, None

    for col_name in x.columns:
        # col_unique_values = x[col_name].unique()
        col_unique_values = get_percentiles(x[col_name], params.num_thresholds)
        for unique_value in col_unique_values:
            y_left = y[x[col_name] <= unique_value]
            y_right = y[x[col_name] > unique_value]

            loss = sum((y_left - y_left.mean())**2) + sum((y_right - y_right.mean())**2)

            if loss < min_so_far:
                min_so_far = loss
                best_col_name = col_name
                best_split_value = unique_value
    return best_col_name, best_split_value


def cart(data, label_name, params):
    tree = RegressionTree()

    tree_levels_list = {0: [(data, tree.get_root())]}

    for depth in range(params.max_depth):
        tree_levels_list[depth+1] = []
        for node_data, node_reference in tree_levels_list[depth]:  # for each depth, iterate over all nodes and split as necessary
            col_name, split_value = get_optimal_partition(node_data, label_name, params)
            left_node_data = node_data[node_data[col_name] <= split_value]
            right_node_data = node_data[node_data[col_name] > split_value]

            # checks minimum node size violation
            if len(left_node_data.index) > params.min_node_size and len(right_node_data.index) > params.min_node_size:
                # define split parameters
                node_reference.split(col_name, split_value)

                # append descendants to the next depth
                tree_levels_list[depth+1].append((left_node_data, node_reference.left_descendant))
                tree_levels_list[depth+1].append((right_node_data, node_reference.right_descendant))
            else:
                node_reference.set_const(node_data[label_name])

    # set all nodes in max depth to nodes
    for node_data, node_reference in tree_levels_list[params.max_depth]:
        node_reference.set_const(node_data[label_name])

    return tree


def gbrt(train_data, test_data, label_name, params):
    start_time = time()
    tree_ensemble = RegressionTreeEnsemble()

    y_train = train_data[label_name].copy()
    y_test = test_data[label_name]

    y_train_ensemble_pred = pd.Series(data=np.zeros_like(y_train), index=y_train.index)
    y_test_ensemble_pred = pd.Series(data=np.zeros_like(y_test), index=y_test.index)
    logs = {'trees': [], 'train_loss': [], 'test_loss': []}
    for m in range(params.num_trees):

        # Update learning by rate decay factor
        if m % params.lr_update == 0 and m != 0:
            params.weight_decay *= params.lr_step
            if params.verbose:
                print("Updating leraning rate to new value of %.3f" % (params.lr_update))


        grad = -(y_train - y_train_ensemble_pred)
        train_data[label_name] = grad
        sub_data = train_data.sample(frac=params.sub_samp)
        tree = cart(sub_data, label_name, params)

        if params.verbose == 2:
            tree.root.print_sub_tree()

        y_train_tree_pred = train_data.apply(lambda xi: tree.evaluate(xi[:]), axis=1)
        y_test_tree_pred = test_data.apply(lambda xi: tree.evaluate(xi[:]), axis=1)

        weight = sum(-grad * y_train_tree_pred) / sum(y_train_tree_pred ** 2) * params.weight_decay
        tree_ensemble.add_tree(tree, weight)

        # evaluate train and test sets
        y_train_ensemble_pred += weight * y_train_tree_pred
        y_test_ensemble_pred += weight * y_test_tree_pred

        train_mean_loss = np.mean((y_train - y_train_ensemble_pred) ** 2)
        test_mean_loss = np.mean((y_test - y_test_ensemble_pred) ** 2)

        if params.verbose >= 1:
            print('Add tree number {}'.format(m+1))
            print('Train mean loss is: {}'.format(train_mean_loss))
            print('Test mean loss is: {}'.format(test_mean_loss))

        logs['trees'].append(m)
        logs['train_loss'].append(train_mean_loss)
        logs['test_loss'].append(test_mean_loss)

    logs['training_time'] = time() - start_time
    logs['params'] = params

    train_data[label_name] = y_train

    return tree_ensemble, logs




