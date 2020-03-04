# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, load_data, print_output, rmse_score
)
from cuml import Ridge

parser = argparse.ArgumentParser(description='cuML ridge regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--solver', default='eig', choices=('eig', 'cd', 'svd'),
                    help='Solver used for training')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Regularization strength')
params = parse_args(parser, size=(1000000, 50), loop_types=('fit', 'predict'))

# Load data
X_train, X_test, y_train, y_test = load_data(
    params, generated_data=['X_train', 'y_train'])

# Create our regression object
regr = Ridge(fit_intercept=params.fit_intercept, alpha=params.alpha,
             solver=params.solver)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

# Time fit
fit_time, _ = time_mean_min(regr.fit, X_train, y_train,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

# Time predict
predict_time, yp = time_mean_min(regr.predict, X_test,
                                 outer_loops=params.predict_outer_loops,
                                 inner_loops=params.predict_inner_loops,
                                 goal_outer_loops=params.predict_goal,
                                 time_limit=params.predict_time_limit,
                                 verbose=params.verbose)

test_rmse = rmse_score(yp, y_test)
yp = regr.predict(X_train)
train_rmse = rmse_score(yp, y_train)

print_output(library='cuml', algorithm='ridge_regression',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['Ridge.fit', 'Ridge.predict'],
             times=[fit_time, predict_time], accuracy_type='rmse',
             accuracies=[train_rmse, test_rmse], data=[X_train, X_test],
             alg_instance=regr)
