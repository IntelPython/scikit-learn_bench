# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    float_or_int, parse_args, time_mean_min, load_data, print_output,
    rmse_score
)
from cuml.ensemble import RandomForestRegressor

parser = argparse.ArgumentParser(description='cuml random forest '
                                             'regression benchmark')

parser.add_argument('--criterion', type=str, default='mse',
                    choices=('mse', 'mae'),
                    help='The function to measure the quality of a split')
parser.add_argument('--split-algorithm', type=str, default='hist',
                    choices=('hist', 'global_quantile'),
                    help='The algorithm to determine how '
                         'nodes are split in the tree')
parser.add_argument('--num-trees', type=int, default=100,
                    help='Number of trees in the forest')
parser.add_argument('--max-features', type=float_or_int, default=None,
                    help='Upper bound on features used at each split')
parser.add_argument('--max-depth', type=int, default=None,
                    help='Upper bound on depth of constructed trees')
parser.add_argument('--min-samples-split', type=float_or_int, default=2,
                    help='Minimum samples number for node splitting')
parser.add_argument('--max-leaf-nodes', type=int, default=-1,
                    help='Maximum leaf nodes per tree')
parser.add_argument('--min-impurity-decrease', type=float, default=0.,
                    help='Needed impurity decrease for node splitting')
parser.add_argument('--no-bootstrap', dest='bootstrap', default=True,
                    action='store_false', help="Don't control bootstraping")

params = parse_args(parser, loop_types=('fit', 'predict'))

# Load and convert data
X_train, X_test, y_train, y_test = load_data(params)

if params.criterion == 'mse':
    params.criterion = 2
else:
    params.criterion = 3

if params.split_algorithm == 'hist':
    params.split_algorithm = 0
else:
    params.split_algorithm = 1

# Create our random forest regressor
def fit(X, y):
    global regr
    regr = RandomForestRegressor(split_criterion=params.criterion,
                                 split_algo=params.split_algorithm,
                                 n_estimators=params.num_trees,
                                 max_depth=params.max_depth,
                                 max_features=params.max_features,
                                 min_rows_per_node=params.min_samples_split,
                                 max_leaves=params.max_leaf_nodes,
                                 min_impurity_decrease=params.min_impurity_decrease,
                                 bootstrap=params.bootstrap)
    return regr.fit(X, y)


def predict(X):
    return regr.predict(X, predict_model='GPU')


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'num_trees', 'time')

fit_time, _ = time_mean_min(fit, X_train, y_train,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

y_pred = predict(X_train)
train_rmse = rmse_score(y_pred, y_train)

predict_time, y_pred = time_mean_min(predict, X_test,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)
test_rmse = rmse_score(y_pred, y_test)

print_output(library='cuml', algorithm='decision_forest_regression',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['df_regr.fit', 'df_regr.predict'],
             times=[fit_time, predict_time], accuracy_type='rmse',
             accuracies=[train_rmse, test_rmse], data=[X_train, X_test],
             alg_instance=regr)
