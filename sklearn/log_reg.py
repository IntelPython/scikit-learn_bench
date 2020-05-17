# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from bench import (measure_function_time, parse_args, load_data, print_output,
                  run_with_context, patch_sklearn)


def main():
    import argparse
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    parser = argparse.ArgumentParser(description='scikit-learn logistic '
                                                 'regression benchmark')
    parser.add_argument('--no-fit-intercept', dest='fit_intercept',
                        action='store_false', default=True,
                        help="Don't fit intercept")
    parser.add_argument('--multiclass', default='auto',
                        choices=('auto', 'ovr', 'multinomial'),
                        help='How to treat multi class data. '
                             '"auto" picks "ovr" for binary classification, and '
                             '"multinomial" otherwise.')
    parser.add_argument('--solver', default='lbfgs',
                        choices=('lbfgs', 'newton-cg', 'saga'),
                        help='Solver to use.')
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

    params.n_classes = len(np.unique(y_train))

    if params.multiclass == 'auto':
        params.multiclass = 'ovr' if params.n_classes == 2 else 'multinomial'

    if not params.tol:
        params.tol = 1e-3 if params.solver == 'newton-cg' else 1e-10

    # Create our classifier object
    clf = LogisticRegression(penalty='l2', C=params.C, n_jobs=params.n_jobs,
                             fit_intercept=params.fit_intercept,
                             verbose=params.verbose,
                             tol=params.tol, max_iter=params.maxiter,
                             solver=params.solver, multi_class=params.multiclass)

    columns = ('batch', 'arch', 'prefix', 'function', 'patch_sklearn', 'device', 'threads', 'dtype', 'size',
               'solver', 'C', 'multiclass', 'n_classes', 'accuracy', 'time')

    # Time fit and predict
    fit_time, _ = measure_function_time(clf.fit, X_train, y_train, params=params)
    y_pred = clf.predict(X_train)
    train_acc = 100 * accuracy_score(y_pred, y_train)

    predict_time, y_pred = measure_function_time(
        clf.predict, X_test, params=params)
    test_acc = 100 * accuracy_score(y_pred, y_test)

    print_output(library='sklearn', algorithm='logistic_regression',
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

if __name__ == "__main__":
    patch_sklearn()
    run_with_context(main)
