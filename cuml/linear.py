# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, load_data, print_output, rmse_score
)
from cuml import LinearRegression

parser = argparse.ArgumentParser(description='cuML linear regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--solver', default='eig', choices=('eig', 'svd'),
                    help='Solver used for training')
params = parse_args(parser, prefix='cuml', size=(1000000, 50))

# Load data
X_train, X_test, y_train, y_test = load_data(
    params, generated_data=['X_train', 'y_train'])

# Create our regression object
regr = LinearRegression(fit_intercept=params.fit_intercept,
                        algorithm=params.solver)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

# Time fit
fit_time, _ = measure_function_time(regr.fit, X_train, y_train, params=params)

# Time predict
predict_time, yp = measure_function_time(regr.predict, X_test, params=params)

test_rmse = rmse_score(yp, y_test)
yp = regr.predict(X_train)
train_rmse = rmse_score(yp, y_train)

print_output(library='cuml', algorithm='linear_regression',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['Linear.fit', 'Linear.predict'],
             times=[fit_time, predict_time], accuracy_type='rmse',
             accuracies=[train_rmse, test_rmse], data=[X_train, X_test],
             alg_instance=regr)
