# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, load_data, print_output, accuracy_score
)
from cuml import LogisticRegression

parser = argparse.ArgumentParser(description='cuML logistic '
                                             'regression benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept',
                    action='store_false', default=True,
                    help="Don't fit intercept")
parser.add_argument('--solver', default='qn', choices=('qn', 'owl'),
                    help='Solver to use.')
parser.add_argument('--linesearch-max-iter', type=int, default=50,
                    help='Maximum iterations per solver outer iteration')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum iterations for the iterative solver')
parser.add_argument('-C', dest='C', type=float, default=1.0,
                    help='Regularization parameter')
parser.add_argument('--tol', type=float, default=1e-10,
                    help='Tolerance for solver. Default is 1e-10.')
params = parse_args(parser)

# Load generated data
X_train, X_test, y_train, y_test = load_data(params)

params.n_classes = y_train[y_train.columns[0]].nunique()

# Create our classifier object
clf = LogisticRegression(penalty='l2', C=params.C,
                         linesearch_max_iter=params.linesearch_max_iter,
                         fit_intercept=params.fit_intercept,
                         verbose=params.verbose, tol=params.tol,
                         max_iter=params.maxiter, solver=params.solver)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'solver', 'C', 'multiclass', 'n_classes', 'accuracy', 'time')

# Time fit and predict
fit_time, _ = measure_function_time(clf.fit, X_train, y_train, params=params)
y_pred = clf.predict(X_train)
train_acc = 100 * accuracy_score(y_pred, y_train)

predict_time, y_pred = measure_function_time(
    clf.predict, X_test, params=params)
test_acc = 100 * accuracy_score(y_pred, y_test)

print_output(library='cuml', algorithm='logistic_regression',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['LogReg.fit', 'LogReg.predict'],
             times=[fit_time, predict_time], accuracy_type='accuracy[%]',
             accuracies=[train_acc, test_acc], data=[X_train, X_test],
             alg_instance=clf)

if params.verbose:
    print()
    print(f'@ Number of iterations: {clf.n_iter_}')
    print('@ fit coefficients:')
    print(f'@ {clf.coef_.tolist()}')
    print('@ fit intercept:')
    print(f'@ {clf.intercept_.tolist()}')
