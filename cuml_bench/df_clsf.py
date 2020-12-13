# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    float_or_int, parse_args, measure_function_time, load_data, print_output,
    accuracy_score
)
import cuml
from cuml.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser(description='cuml random forest '
                                             'classification benchmark')

parser.add_argument('--criterion', type=str, default='gini',
                    choices=('gini', 'entropy'),
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

params = parse_args(parser)

# Load and convert data
X_train, X_test, y_train, y_test = load_data(params, int_label=True)

if params.criterion == 'gini':
    params.criterion = 0
else:
    params.criterion = 1

if params.split_algorithm == 'hist':
    params.split_algorithm = 0
else:
    params.split_algorithm = 1

params.n_classes = y_train[y_train.columns[0]].nunique()


def fit(X, y):
    global clf
    clf = RandomForestClassifier(split_criterion=params.criterion,
                                 split_algo=params.split_algorithm,
                                 n_estimators=params.num_trees,
                                 max_depth=params.max_depth,
                                 max_features=params.max_features,
                                 min_rows_per_node=params.min_samples_split,
                                 max_leaves=params.max_leaf_nodes,
                                 min_impurity_decrease=params.min_impurity_decrease,
                                 bootstrap=params.bootstrap)
    return clf.fit(X, y)


def predict(X):
    prediction_args = {'predict_model': 'GPU'}
    if int(cuml.__version__.split('.')[1]) <= 14:
        prediction_args.update({'num_classes': params.n_classes})
    return clf.predict(X, **prediction_args)


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'num_trees', 'n_classes', 'accuracy', 'time')

fit_time, _ = measure_function_time(fit, X_train, y_train, params=params)
y_pred = predict(X_train)
train_acc = 100 * accuracy_score(y_pred, y_train)

predict_time, y_pred = measure_function_time(predict, X_test, params=params)
test_acc = 100 * accuracy_score(y_pred, y_test)

print_output(library='cuml', algorithm='decision_forest_classification',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['df_clsf.fit', 'df_clsf.predict'],
             times=[fit_time, predict_time], accuracy_type='accuracy[%]',
             accuracies=[train_acc, test_acc], data=[X_train, X_test],
             alg_instance=clf)
