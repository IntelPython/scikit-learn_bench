# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, output_csv, load_data, gen_basic_dict,
    accuracy_score
)
from cuml import LogisticRegression

parser = argparse.ArgumentParser(description='scikit-learn logistic '
                                             'regression benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept',
                    action='store_false', default=True,
                    help="Don't fit intercept")
parser.add_argument('--solver', default='lbfgs',
                    choices=('lbfgs', 'qn', 'owl'),
                    help='Solver to use.')
parser.add_argument('--linesearch-max-iter', type=int, default=50,
                    help='Maximum iterations per solver outer iteration')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum iterations for the iterative solver')
parser.add_argument('-C', dest='C', type=float, default=1.0,
                    help='Regularization parameter')
parser.add_argument('--tol', type=float, default=None,
                    help='Tolerance for solver. If solver == "newton-cg", '
                         'then the default is 1e-3. Otherwise, the default '
                         'is 1e-10.')
params = parse_args(parser, loop_types=('fit', 'predict'))

# Load generated data
X_train, X_test, y_train, y_test = load_data(params)

params.n_classes = y_train[y_train.columns[0]].nunique()

if not params.tol:
    params.tol = 1e-3 if params.solver == 'newton-cg' else 1e-10

# Create our classifier object
clf = LogisticRegression(penalty='l2', C=params.C,
                         linesearch_max_iter=params.linesearch_max_iter,
                         fit_intercept=params.fit_intercept,
                         verbose=params.verbose, tol=params.tol,
                         max_iter=params.maxiter, solver=params.solver)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'solver', 'C', 'multiclass', 'n_classes', 'accuracy', 'time')

# Time fit and predict
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

if params.output_format == 'csv':
    output_csv(columns, params, functions=['LogReg.fit', 'LogReg.predict'],
               times=[fit_time, predict_time], accuracies=[None, test_acc])
    if params.verbose:
        print()
        print('@ Number of iterations: {}'.format(clf.n_iter_))
        print('@ fit coefficients:')
        print('@ {}'.format(clf.coef_.tolist()))
        print('@ fit intercept:')
        print('@ {}'.format(clf.intercept_.tolist()))

elif params.output_format == 'json':
    import json

    result = gen_basic_dict('cuml', 'logistic_regression',
                            'training', params, X_train, clf)
    result['input_data'].update({'classes': params.n_classes})
    result.update({
        'time[s]': fit_time,
        'accuracy[%]': train_acc
    })
    print(json.dumps(result, indent=4))

    result = gen_basic_dict('cuml', 'logistic_regression',
                            'prediction', params, X_test, clf)
    result['input_data'].update({'classes': params.n_classes})
    result.update({
        'time[s]': predict_time,
        'accuracy[%]': test_acc
    })
    print(json.dumps(result, indent=4))
