# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


import numpy as np
from bench import prepare_benchmark
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

    def compute(self, beta):
        result = self.algo.compute(self.X, self.y, make2d(beta))
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
    import argparse

    def getArguments(argParser):
        argParser.add_argument('--prefix', type=str, default='daal4py',
                               help="Identifier of the bench being executed")
        argParser.add_argument('--fileX', type=argparse.FileType('r'),
                               help="Input file with features")
        argParser.add_argument('--fileY', type=argparse.FileType('r'),
                               help="Input file with labels")
        argParser.add_argument('--intercept', action="store_true",
                               help="Whether to fit intercept")
        argParser.add_argument('--solver', default='lbfgs',
                               choices=['lbfgs', 'newton-cg'],
                               help="Solver to use.")
        argParser.add_argument('--maxiter',      type=int, default=100,
                               help="Maximum iterations setting for the "
                                    "iterative solver of choice")
        argParser.add_argument('--fit-repetitions', dest="fit_inner_reps",
                               type=int, default=1,
                               help="Count of operations whose execution time "
                                    "is being clocked, average time reported")
        argParser.add_argument('--fit-samples',  dest="fit_outer_reps",
                               type=int, default=5,
                               help="Count of repetitions of time "
                                    "measurements to collect statistics ")
        argParser.add_argument('--verbose',  action="store_const",
                               const=1, default=0,
                               help="Whether to print additional information.")
        argParser.add_argument('--header',  action="store_true",
                               help="Whether to print header.")
        argParser.add_argument('--predict-repetitions',
                               dest="predict_inner_reps", type=int, default=50,
                               help="Count of operations whose execution time "
                               "is being clocked, average time reported")
        argParser.add_argument('--predict-samples',  dest="predict_outer_reps",
                               type=int, default=5,
                               help="Count of repetitions of time "
                                    "measurements to collect statistics ")
        argParser.add_argument('--num-threads', type=int, dest="num_threads",
                               default=0,
                               help="Number of threads for DAAL to use")

        args = argParser.parse_args()
        return args

    argParser = argparse.ArgumentParser(description="daal4py logistic "
                                                    "regression benchmark")

    args = getArguments(argParser)

    num_threads, daal_version = prepare_benchmark(args)

    import timeit

    X = np.load(args.fileX.name)
    y = np.load(args.fileY.name)

    if args.verbose:
        print("@ {", end='')
        print(" FIT_SAMPLES : {0}, FIT_REPETITIONS : {1},"
              "  PREDICT_SAMPLES: {2}, PREDICT_REPETITIONS: {3}".format(
                  args.fit_outer_reps, args.fit_inner_reps,
                  args.predict_outer_reps, args.predict_inner_reps
              ), end='')
        print("}")

    C = 1.0
    tol = 1e-3 if args.solver == 'newton-cg' else 1e-10
    fit_intercept = args.intercept

    if args.verbose:
        print("@ {", end='')
        print("'fit_intercept' : {0}, 'C' : {1}, 'max_iter' : {2}, "
              "'tol' : {3}, 'solver' : {4}".format(
                  fit_intercept, C, args.maxiter, tol, args.solver
              ), end='')
        print("}")

    fit_times = []
    n_iters = []
    for outer_it in range(args.fit_outer_reps):
        t0 = timeit.default_timer()
        for _ in range(args.fit_inner_reps):
            w, w0, r, mc = test_fit(X, y, penalty='l2', C=C,
                                    verbose=args.verbose,
                                    fit_intercept=fit_intercept, tol=tol,
                                    max_iter=args.maxiter, solver=args.solver)
        t1 = timeit.default_timer()
        fit_times.append((t1 - t0) / args.fit_inner_reps)
        n_iters.append(r.nit)

    predict_times = []
    for outer_it in range(args.predict_outer_reps):

        t0 = timeit.default_timer()
        for _ in range(args.predict_inner_reps):
            y_proba = test_predict(X, w, intercept=w0, multi_class=mc)
            y_pred = np.argmax(y_proba, axis=1)
        t1 = timeit.default_timer()
        predict_times.append((t1 - t0) / args.predict_inner_reps)

    acc = np.mean(abs(y_pred - y) < 0.5)

    def num_classes(c):
        if c.shape[0] == 1:
            return 2
        else:
            return c.shape[0]

    if args.header:
        print("prefix_ID,function,solver,threads,rows,features,fit,predict,"
              "accuracy,classes")
    print(",".join((
        args.prefix,
        'log_reg',
        args.solver,
        "Serial" if num_threads == 1 else "Threaded",
        str(X.shape[0]),
        str(X.shape[1]),
        "{0:.3f}".format(min(fit_times)),
        "{0:.3f}".format(min(predict_times)),
        "{0:.4f}".format(100*acc),
        str(num_classes(w))
    )))

    if args.verbose:
        print("")
        print("@ Median of {0} runs of .fit averaging over {1} executions is "
              "{2:3.3f}".format(args.fit_outer_reps, args.fit_inner_reps,
                                np.percentile(fit_times, 50)))
        print("@ Median of {0} runs of .predict averaging over {1} executions "
              "is {2:3.3f}".format(args.predict_outer_reps,
                                   args.predict_inner_reps,
                                   np.percentile(predict_times, 50)))
        print("")
        print("@ Number of iterations: {}".format(r.nit))
        print("@ fit coefficients:")
        print("@ {}".format(w.tolist()))
        print("@ fit intercept")
        print("@ {}".format(w0.tolist()))
