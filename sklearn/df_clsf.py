# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    float_or_int, parse_args, measure_function_time, load_data, print_output,
    accuracy_score
)
import numpy as np

parser = argparse.ArgumentParser(description='scikit-learn random forest '
                                             'classification benchmark')

parser.add_argument('--criterion', type=str, default='gini',
                    choices=('gini', 'entropy'),
                    help='The function to measure the quality of a split')
parser.add_argument('--num-trees', type=int, default=100,
                    help='Number of trees in the forest')
parser.add_argument('--max-features', type=float_or_int, default=None,
                    help='Upper bound on features used at each split')
parser.add_argument('--max-depth', type=int, default=None,
                    help='Upper bound on depth of constructed trees')
parser.add_argument('--min-samples-split', type=float_or_int, default=2,
                    help='Minimum samples number for node splitting')
parser.add_argument('--max-leaf-nodes', type=int, default=None,
                    help='Maximum leaf nodes per tree')
parser.add_argument('--min-impurity-decrease', type=float, default=0.,
                    help='Needed impurity decrease for node splitting')
parser.add_argument('--no-bootstrap', dest='bootstrap', default=True,
                    action='store_false', help="Don't control bootstraping")

parser.add_argument('--use-sklearn-class', action='store_true',
                    help='Force use of '
                         'sklearn.ensemble.RandomForestClassifier')
params = parse_args(parser)

# Get some RandomForestClassifier
if params.use_sklearn_class:
    from sklearn.ensemble import RandomForestClassifier
else:
    try:
        from daal4py.sklearn.ensemble import RandomForestClassifier
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

# Load and convert data
X_train, X_test, y_train, y_test = load_data(params)

# Create our random forest classifier
clf = RandomForestClassifier(criterion=params.criterion,
                             n_estimators=params.num_trees,
                             max_depth=params.max_depth,
                             max_features=params.max_features,
                             min_samples_split=params.min_samples_split,
                             max_leaf_nodes=params.max_leaf_nodes,
                             min_impurity_decrease=params.min_impurity_decrease,
                             bootstrap=params.bootstrap,
                             random_state=params.seed)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'num_trees', 'n_classes', 'accuracy', 'time')
params.n_classes = len(np.unique(y_train))

fit_time, _ = measure_function_time(clf.fit, X_train, y_train, params=params)
y_pred = clf.predict(X_train)
train_acc = 100 * accuracy_score(y_pred, y_train)

predict_time, y_pred = measure_function_time(
    clf.predict, X_test, params=params)
test_acc = 100 * accuracy_score(y_pred, y_test)

print_output(library='sklearn', algorithm='decision_forest_classification',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['df_clsf.fit', 'df_clsf.predict'],
             times=[fit_time, predict_time], accuracy_type='accuracy[%]',
             accuracies=[train_acc, test_acc], data=[X_train, X_test],
             alg_instance=clf)
