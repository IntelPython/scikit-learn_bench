# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, \
                  size_str, accuracy_score
import numpy as np
import daal4py
from daal4py import math_logistic, math_softmax
from daal4py.sklearn.utils import getFPType, make2d
import scipy.optimize

_logistic_loss = daal4py.optimization_solver_logistic_loss
_cross_entropy_loss = daal4py.optimization_solver_cross_entropy_loss


def _results_to_compute(value=True, gradient=True, hessian=False):

    results_to_compute = []
    if value:
        results_to_compute.append('value')
    if gradient:
        results_to_compute.append('gradient')
    if hessian:
        results_to_compute.append('hessian')

    return '|'.join(results_to_compute)


class Loss:

    def __init__(self, X, y, beta, hess=False, fit_intercept=True):
        self.compute_hess = hess
        self.n = X.shape[0]
        self.fptype = getFPType(X)
        self.fit_intercept = fit_intercept
        self.X = make2d(X)
        self.y = make2d(y)

        self.last_beta = beta.copy()

        self.func = None
        self.grad = None
        self.hess = None

    def compute(self, beta):
        # Don't compute if we have already cached func, grad, hess
        if self.func is not None and np.array_equal(beta, self.last_beta):
            return

        result = self.algo.compute(self.X, self.y, make2d(beta))
        np.copyto(self.last_beta, beta)
        self.func = result.valueIdx[0, 0] * self.n
        self.grad = result.gradientIdx.ravel() * self.n
        if self.compute_hess:
            self.hess = result.hessianIdx * self.n

    def get_value(self, arg):
        self.compute(arg)
        return self.func

    def get_grad(self, arg):
        self.compute(arg)
        return self.grad

    def get_hess(self, arg):
        if not self.compute_hess:
            raise ValueError('You asked for Hessian but compute_hess=False')
        self.compute(arg)
        return self.hess


class LogisticLoss(Loss):

    def __init__(self, *args, l1=0.0, l2=0.0, **kwargs):

        super().__init__(*args, **kwargs)

        self.algo = _logistic_loss(
            numberOfTerms=self.n,
            fptype=self.fptype,
            method='defaultDense',
            interceptFlag=self.fit_intercept,
            penaltyL1=l1 / self.n,
            penaltyL2=l2 / self.n,
            resultsToCompute=_results_to_compute(hessian=self.compute_hess)
        )


class CrossEntropyLoss(Loss):

    def __init__(self, n_classes, *args, l1=0.0, l2=0.0, **kwargs):

        super().__init__(*args, **kwargs)

        self.algo = _cross_entropy_loss(
            nClasses=n_classes,
            numberOfTerms=self.n,
            fptype=self.fptype,
            method='defaultDense',
            interceptFlag=self.fit_intercept,
            penaltyL1=l1 / self.n,
            penaltyL2=l2 / self.n,
            resultsToCompute=_results_to_compute(hessian=self.compute_hess)
        )


def test_fit(X, y, penalty='l2', C=1.0, fit_intercept=True, tol=1e-4,
             max_iter=100, solver='lbfgs', verbose=0):

    if penalty == 'l2':
        l2 = 0.5 / C
    else:
        l2 = 0.0

    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    compute_hessian = (solver == 'newton-cg')

    if n_classes == 2:
        # Use the standard logistic regression formulation
        multi_class = 'ovr'
    else:
        # Use the multinomial logistic regression formulation
        multi_class = 'multinomial'

    if multi_class == 'ovr':
        beta = np.zeros(n_features + 1, dtype='f8')
        loss_obj = LogisticLoss(X, y, beta, fit_intercept=fit_intercept, l2=l2,
                                hess=compute_hessian)
    else:
        beta = np.zeros((n_classes, n_features + 1), dtype='f8')
        beta = beta.ravel()
        loss_obj = CrossEntropyLoss(n_classes, X, y, beta,
                                    hess=compute_hessian,
                                    fit_intercept=fit_intercept, l2=l2)

    if solver == 'lbfgs':
        result = scipy.optimize.minimize(loss_obj.get_value, beta,
                                         method='L-BFGS-B',
                                         jac=loss_obj.get_grad,
                                         options=dict(disp=verbose, gtol=tol))
    else:
        result = scipy.optimize.minimize(loss_obj.get_value, beta,
                                         method='Newton-CG',
                                         jac=loss_obj.get_grad,
                                         hess=loss_obj.get_hess,
                                         options=dict(disp=verbose))

    beta = result.x
    beta_n_classes = n_classes if n_classes > 2 else 1
    beta = beta.reshape((beta_n_classes, n_features + 1))

    return beta[:, 1:], beta[:, 0], result, multi_class


def test_predict(X, beta, intercept=0, multi_class='ovr'):

    fptype = getFPType(X)

    scores = np.dot(X, beta.T) + intercept
    if multi_class == 'ovr':
        # use binary logistic regressions and normalize
        logistic = math_logistic(fptype=fptype, method='defaultDense')
        prob = logistic.compute(scores).value
        if prob.shape[1] == 1:
            return np.c_[1 - prob, prob]
        else:
            return prob / prob.sum(axis=1)[:, np.newaxis]
    else:
        # use softmax of exponentiated scores
        if scores.shape[1] == 1:
            scores = np.c_[-scores, scores]
        softmax = math_softmax(fptype=fptype, method='defaultDense')
        return softmax.compute(scores).value


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='daal4py logistic '
                                                 'regression benchmark')
    parser.add_argument('-x', '--filex', '--fileX',
                        type=argparse.FileType('r'),
                        help='Input file with features, in NPY format')
    parser.add_argument('-y', '--filey', '--fileY',
                        type=argparse.FileType('r'),
                        help='Input file with labels, in NPY format')
    parser.add_argument('--no-fit-intercept', dest='fit_intercept',
                        action='store_false', default=True,
                        help="Don't fit intercept")
    parser.add_argument('--solver', default='lbfgs',
                        choices=('lbfgs', 'newton-cg'),
                        help='Solver to use.')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='Maximum iterations setting for the '
                             'iterative solver of choice')
    parser.add_argument('-C', dest='C', type=float, default=1.0,
                        help='Regularization parameter')
    parser.add_argument('--tol', type=float, default=None,
                        help="Tolerance for solver. If solver == 'newton-cg', "
                             "then the default is 1e-3. Otherwise the default "
                             "is 1e-10.")
    params = parse_args(parser, loop_types=('fit', 'predict'),
                        prefix='daal4py')

    # Load generated data
    X = np.load(params.filex.name)
    y = np.load(params.filey.name)

    params.n_classes = len(np.unique(y))
    if not params.tol:
        params.tol = 1e-3 if params.solver == 'newton-cg' else 1e-10

    columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype',
               'size', 'solver', 'C', 'multiclass', 'n_classes', 'accuracy',
               'time')
    params.size = size_str(X.shape)
    params.dtype = X.dtype

    print_header(columns, params)

    # Time fit and predict
    fit_time, res = time_mean_min(test_fit, X, y, penalty='l2', C=params.C,
                                  fit_intercept=params.fit_intercept,
                                  tol=params.tol,
                                  max_iter=params.maxiter,
                                  solver=params.solver,
                                  outer_loops=params.fit_outer_loops,
                                  inner_loops=params.fit_inner_loops,
                                  goal_outer_loops=params.fit_goal,
                                  time_limit=params.fit_time_limit,
                                  verbose=params.verbose)

    beta, intercept, solver_result, params.multiclass = res
    print_row(columns, params, function='LogReg.fit', time=fit_time)

    predict_time, yp = time_mean_min(test_predict, X, beta,
                                     intercept=intercept,
                                     multi_class=params.multiclass,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)
    y_pred = np.argmax(yp, axis=1)
    acc = 100 * accuracy_score(y_pred, y)
    print_row(columns, params, function='LogReg.predict', time=predict_time,
              accuracy=acc)

    if params.verbose:
        print()
        print("@ Number of iterations: {}".format(solver_result.nit))
        print("@ fit coefficients:")
        print("@ {}".format(beta.tolist()))
        print("@ fit intercept:")
        print("@ {}".format(intercept.tolist()))
