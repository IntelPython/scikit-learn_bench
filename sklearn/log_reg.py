# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


import numpy as np
from bench import set_daal_num_threads, sklearn_set_no_input_check


if __name__ == '__main__':
    import argparse

    def valid_solver(s):
        if s in ['lbfgs', 'newton-cg', 'saga']:
            return s
        raise ValueError("Invalid solver choice")


    def valid_multiclass(s):
        if s in ['ovr', 'multinomial']:
            return s
        raise ValueError("Invalid multiclass choice")


    def getArguments(argParser):
        argParser.add_argument('--prefix',   type=str, default='python',
                               help="Identifier of the bench being executed")
        argParser.add_argument('--fileX',        type=argparse.FileType('r'),
                               help="Input file with features")
        argParser.add_argument('--fileY',        type=argparse.FileType('r'),
                               help="Input file with labels")
        argParser.add_argument('--intercept',    action="store_true",
                               help="Whether to fit intercept")
        argParser.add_argument('--multiclass',   type=valid_multiclass, default='ovr',
                               help="How to treat multi class data. Valid choices are 'ovr' or 'multinomial'")
        argParser.add_argument('--solver',       type=valid_solver, default='lbfgs',
                               help="Solver to use. Valid choices are 'lbfgs', 'newton-cg' or 'saga'.")
        argParser.add_argument('--maxiter',      type=int, default=100,
                               help="Maximum iterations setting for the iterative solver of choice")
        argParser.add_argument('--fit-repetitions', dest="fit_inner_reps", type=int, default=1,
                               help="Count of operations whose execution time is being clocked, average time reported")
        argParser.add_argument('--fit-samples',  dest="fit_outer_reps", type=int, default=5,
                               help="Count of repetitions of time measurements to collect statistics ")
        argParser.add_argument('--verbose',  action="store_true",
                               help="Whether to print additional information.")
        argParser.add_argument('--header',  action="store_true",
                               help="Whether to print header.")
        argParser.add_argument('--predict-repetitions', dest="predict_inner_reps", type=int, default=50,
                               help="Count of operations whose execution time is being clocked, average time reported")
        argParser.add_argument('--predict-samples',  dest="predict_outer_reps", type=int, default=5,
                               help="Count of repetitions of time measurements to collect statistics ")
        argParser.add_argument('--num-threads', type=int, dest="num_threads", default=0,
                               help="Number of threads for DAAL to use")

        args = argParser.parse_args()

        return args


    argParser = argparse.ArgumentParser(prog="logistic_bench.py",
                                        description="Execute Logistic Regression",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = getArguments(argParser)

    num_threads = args.num_threads
    set_daal_num_threads(num_threads)

    import sklearn
    from sklearn.linear_model import LogisticRegression
    import sklearn.linear_model.logistic

    sklearn_set_no_input_check()

    import timeit


    X = np.load(args.fileX.name)
    y = np.load(args.fileY.name)

    if args.verbose:
       print("@ {", end='')
       print(" FIT_SAMPLES : {0}, FIT_REPETITIONS : {1}, PREDICT_SAMPLES: {2}, PREDICT_REPETITIONS: {3}".format(
          args.fit_outer_reps, args.fit_inner_reps, args.predict_outer_reps, args.predict_inner_reps
       ), end='')
       print("}")

    C = 1.0
    tol = 1e-3 if args.solver == 'newton-cg' else 1e-10
    fit_intercept = True

    if args.verbose:
       print("@ {", end='')
       print("'fit_intercept' : {0}, 'C' : {1}, 'max_iter' : {2}, 'tol' : {3}, 'solver' : {4}, 'multi_class' : {5}".format(
          fit_intercept, C, args.maxiter, tol, args.solver, args.multiclass
       ), end='')
       print("}")

    fit_times = []
    n_iters = []
    for outer_it in range(args.fit_outer_reps):
        clf = LogisticRegression(penalty='l2', C=C, fit_intercept=fit_intercept, tol = tol,
                                 max_iter=args.maxiter, solver=args.solver, multi_class=args.multiclass)
        t0 = timeit.default_timer()
        for _ in range(args.fit_inner_reps):
            clf.fit(X, y)
        t1 = timeit.default_timer()
        fit_times.append((t1 - t0) / args.fit_inner_reps)
        n_iters.append(clf.n_iter_)


    predict_times = []
    for outer_it in range(args.predict_outer_reps):

        t0 = timeit.default_timer()
        for _ in range(args.predict_inner_reps):
            y_pred = clf.predict(X)
        t1 = timeit.default_timer()
        predict_times.append((t1 - t0) / args.predict_inner_reps)


    acc = sklearn.metrics.accuracy_score(y_pred, y)

    num_classes = lambda c: 2 if c.shape[0] == 1 else c.shape[0]
    if args.header:
       print("prefix_ID,function,solver,threads,rows,features,fit,predict,accuracy,classes")
    print(",".join((
       args.prefix,
       'log_reg',
       args.solver,
       "Serial" if num_threads==1 else "Threaded",
       str(X.shape[0]),
       str(X.shape[1]),
       "{0:.3f}".format(min(fit_times)),
       "{0:.3f}".format(min(predict_times)),
       "{0:.4f}".format(100*acc),
       str(num_classes(clf.coef_))
    )))

    if args.verbose:
        print("")
        print("@ Median of {0} runs of .fit averaging over {1} executions is {2:3.3f}".format(args.fit_outer_reps, args.fit_inner_reps, np.percentile(fit_times, 50)))
        print("@ Median of {0} runs of .predict averaging over {1} executions is {2:3.3f}".format(args.predict_outer_reps, args.predict_inner_reps, np.percentile(predict_times, 50)))
        print("")
        print("@ Number of iterations: {}".format(clf.n_iter_))
        print("@ fit coefficients:")
        print("@ {}".format(clf.coef_.tolist()))
        print("@ fit intercept")
        print("@ {}".format(clf.intercept_.tolist()))
