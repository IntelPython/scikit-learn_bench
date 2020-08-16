# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, load_data, print_output, rmse_score
)
from cuml.linear_model import ElasticNet

parser = argparse.ArgumentParser(description='scikit-learn elastic-net regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--alpha', dest='alpha', type=float, default=1.0,
                    help='Regularization parameter')
parser.add_argument('--maxiter', type=int, default=1000,
                    help='Maximum iterations for the iterative solver')
parser.add_argument('--l1_ratio', dest='l1_ratio', type=float, default=0.5,
                    help='Regularization parameter')
parser.add_argument('--tol', type=float, default=0.0,
                    help='Tolerance for solver.')
params = parse_args(parser)

# Load data
X_train, X_test, y_train, y_test = load_data(params)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Create our regression object
regr = ElasticNet(fit_intercept=params.fit_intercept, l1_ratio=params.l1_ratio, alpha=params.alpha,
                        tol=params.tol, max_iter=params.maxiter)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

# Time fit
fit_time, _ = measure_function_time(regr.fit, X_train, y_train, params=params)

print('y_train.shape: ', y_train.shape)
print('X_train.shape: ', X_train.shape)
# print('iter: ', regr.n_iter_)

# Time predict
predict_time, pred_train = measure_function_time(regr.predict, X_train, params=params)

train_rmse = rmse_score(pred_train, y_train)
pred_test = regr.predict(X_test)
test_rmse = rmse_score(pred_test, y_test)

print_output(library='cuml', algorithm='elastic-net',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['ElasticNet.fit', 'ElasticNet.predict'],
             times=[fit_time, predict_time], accuracy_type='rmse',
             accuracies=[train_rmse, test_rmse], data=[X_train, X_test],
             alg_instance=regr)
