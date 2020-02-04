# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, load_data, print_output
)
import numpy as np
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='scikit-learn random forest '
                                             'classification benchmark')

parser.add_argument('--num-trees', type=int, default=100,
                    help='Number of trees in the forest')
parser.add_argument('--max-features',   type=int, default=None,
                    help='Upper bound on features used at each split')
parser.add_argument('--max-depth',   type=int, default=None,
                    help='Upper bound on depth of constructed trees')

parser.add_argument('--use-sklearn-class', action='store_true',
                    help='Force use of '
                         'sklearn.ensemble.RandomForestClassifier')
params = parse_args(parser, loop_types=('fit', 'predict'))

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
clf = RandomForestClassifier(n_estimators=params.num_trees,
                             max_depth=params.max_depth,
                             max_features=params.max_features,
                             random_state=params.seed)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'num_trees', 'n_classes', 'accuracy', 'time')
params.n_classes = len(np.unique(y_train))

fit_time, _ = time_mean_min(clf.fit, X_train, y_train,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)
y_pred = clf.predict(X_train)
train_acc = 100 * accuracy_score(y_pred, y_train)

predict_time, y_pred = time_mean_min(clf.predict, X_test,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)
test_acc = 100 * accuracy_score(y_pred, y_test)

print_output(library='sklearn', algorithm='decision_forest_classification',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['df_clsf.fit', 'df_clsf.predict'],
             times=[fit_time, predict_time], accuracy_type='accuracy[%]',
             accuracies=[train_acc, test_acc], data=[X_train, X_test],
             alg_instance=clf)
