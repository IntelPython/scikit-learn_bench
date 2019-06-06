# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


import numpy as np
import bench
from daal4py import decision_forest_classification_training
from daal4py import decision_forest_classification_prediction
from daal4py import engines_mt2203
from daal4py.sklearn.utils import getFPType


def df_clsf_fit(X, y, n_classes, n_trees=100, seed=12345,
                n_features_per_node=0, max_depth=0, min_impurity=0,
                bootstrap=True, verbose=False):

    fptype = getFPType(X)

    features_per_node = X.shape[1]
    if n_features_per_node > 0 and n_features_per_node < features_per_node:
        features_per_node = n_features_per_node

    engine = engines_mt2203(seed=seed, fptype=fptype)

    algorithm = decision_forest_classification_training(
            nClasses=n_classes,
            fptype=fptype,
            method='defaultDense',
            nTrees=n_trees,
            observationsPerTreeFraction=1.,
            featuresPerNode=features_per_node,
            maxTreeDepth=max_depth,
            minObservationsInLeafNode=1,
            engine=engine,
            impurityThreshold=min_impurity,
            varImportance='MDI',
            resultsToCompute='',
            memorySavingMode=False,
            bootstrap=bootstrap
    )

    df_clsf_result = algorithm.compute(X, y)

    return df_clsf_result


def df_clsf_predict(X, training_result, n_classes, verbose=False):

    algorithm = decision_forest_classification_prediction(
            nClasses=n_classes,
            fptype='float', # we give float here specifically to match sklearn
    )

    result = algorithm.compute(X, training_result.model)

    return result.prediction


if __name__ == '__main__':
    import argparse

    def getArguments(argParser):
        argParser.add_argument('--prefix',   type=str, default='daal4py',
                               help="Identifier of the bench being executed")
        argParser.add_argument('--fileX',        type=argparse.FileType('r'),
                               help="Input file with features")
        argParser.add_argument('--fileY',        type=argparse.FileType('r'),
                               help="Input file with labels")
        argParser.add_argument('--num-trees',   type=int, default=100,
                               help="Number of trees in decision forest")
        argParser.add_argument('--max-features',   type=int, default=0,
                               help="Max features used to build trees")
        argParser.add_argument('--max-depth',   type=int, default=0,
                               help="Maximal depth of trees constructed")

        argParser.add_argument('--use-sklearn-class', action='store_true',
                               help="Force use of sklearn.ensemble.RandomForestClassifier")
        argParser.add_argument('--seed', type=int, default=12345,
                               help="Seed to pass as random_state to the class")

        argParser.add_argument('--fit-repetitions', dest="fit_inner_reps", type=int, default=1,
                               help="Count of operations whose execution time is being clocked, average time reported")
        argParser.add_argument('--fit-samples',  dest="fit_outer_reps", type=int, default=5,
                               help="Count of repetitions of time measurements to collect statistics ")
        argParser.add_argument('--predict-repetitions', dest="predict_inner_reps", type=int, default=50,
                               help="Count of operations whose execution time is being clocked, average time reported")
        argParser.add_argument('--predict-samples',  dest="predict_outer_reps", type=int, default=5,
                               help="Count of repetitions of time measurements to collect statistics ")

        argParser.add_argument('--verbose',  action="store_true",
                               help="Whether to print additional information.")
        argParser.add_argument('--header',  action="store_true",
                               help="Whether to print header.")
        argParser.add_argument('--num-threads', type=int, dest="num_threads", default=0,
                               help="Number of threads for DAAL to use")

        args = argParser.parse_args()

        return args


    argParser = argparse.ArgumentParser(prog="df_clsf_bench.py",
                                        description="Execute RandomForest classification",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = getArguments(argParser)
    num_threads, daal_version = bench.prepare_benchmark(args)


    import timeit

    if args.fileX is None or args.fileY is None:
        argParser.error("Please specify data for the algorithm to train on. Use --fileX and --fileY or --generate options.")
    else:
        X = np.load(args.fileX.name)
        y = np.load(args.fileY.name)[:,np.newaxis]

    if args.verbose:
       print("@ {", end='')
       print(" FIT_SAMPLES : {0}, FIT_REPETITIONS : {1}, PREDICT_SAMPLES: {2}, PREDICT_REPETITIONS: {3}".format(
          args.fit_outer_reps, args.fit_inner_reps, args.predict_outer_reps, args.predict_inner_reps
       ), end='')
       print("}")

    if args.verbose:
       print("@ {", end='')
       print("'n_estimators': {0}, 'max_depth': {1}, 'max_features': {2}, 'random_state': {3}".format(
          args.num_trees, args.max_depth, args.max_features, args.seed
       ), end='')
       print("}")

    n_classes = np.max(y) - np.min(y) + 1

    fit_times = []
    for outer_it in range(args.fit_outer_reps):
        t0 = timeit.default_timer()
        for _ in range(args.fit_inner_reps):
            result = df_clsf_fit(X, y, n_classes, n_trees=args.num_trees,
                                 seed=args.seed,
                                 n_features_per_node=args.max_features,
                                 max_depth=args.max_depth, verbose=args.verbose)
        t1 = timeit.default_timer()
        fit_times.append((t1 - t0) / args.fit_inner_reps)


    predict_times = []
    for outer_it in range(args.predict_outer_reps):

        t0 = timeit.default_timer()
        for _ in range(args.predict_inner_reps):
            y_pred = df_clsf_predict(X, result, n_classes,
                                     verbose=args.verbose)
        t1 = timeit.default_timer()
        predict_times.append((t1 - t0) / args.predict_inner_reps)


    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y, y_pred)


    num_classes = np.unique(y).shape[0]
    if args.header:
       print("prefix_ID,function,threads,rows,features,fit,predict,accuracy,classes")
    print(",".join((
       args.prefix,
       'df_clsf',
       str(num_threads),
       str(X.shape[0]),
       str(X.shape[1]),
       "{0:.3f}".format(min(fit_times)),
       "{0:.3f}".format(min(predict_times)),
       "{0:.4f}".format(100*acc),
       str(num_classes)
    )))

    if args.verbose:
        print("")
        print("@ Median of {0} runs of .fit averaging over {1} executions is {2:3.3f}".format(args.fit_outer_reps, args.fit_inner_reps, np.percentile(fit_times, 50)))
        print("@ Median of {0} runs of .predict averaging over {1} executions is {2:3.3f}".format(args.predict_outer_reps, args.predict_inner_reps, np.percentile(predict_times, 50)))
