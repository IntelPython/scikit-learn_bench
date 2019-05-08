# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function

import os
import sys
import argparse
from timeit import default_timer as time

import numpy as np
import sklearn
import sklearn.svm as svm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import bench

def _decoy_check_X_y(X, y, *args,  **kwargs):
    return X, y

def _decoy_check_array(X, *args, **kwargs):
    return X

sklearn.utils.validation.check_X_y = _decoy_check_X_y
sklearn.utils.validation.check_array = _decoy_check_array


def getOptimalCacheSize(numFeatures):
    byte_size = np.empty(0, dtype=np.double).itemsize
    optimal_cache_size_bytes = numFeatures * numFeatures * byte_size
    eight_gb = byte_size * 1024 * 1024 * 1024
    cache_size_bytes = eight_gb if optimal_cache_size_bytes > eight_gb else optimal_cache_size_bytes
    return cache_size_bytes


def _bench(meta_info, X_train, y_train, fit_samples, fit_repetitions, predict_samples, predict_repetitions, svc_params_dict):

    fit_times = []
    for it in range(fit_samples):
        start = time()
        for __ in range(fit_repetitions):
            clf = svm.SVC(**svc_params_dict)
            clf.fit(X_train, y_train)
        stop = time()
        fit_times.append(stop-start)


    predict_times = []
    for it in range(predict_samples):
        start = time()
        for __ in range(predict_repetitions):
            res = clf.predict(X_train)
        stop = time()
        predict_times.append(stop-start)

    print("{meta_info},{fit_t:0.6g},{pred_t:0.6g},{acc:0.3f},{sv_len},{cl}".format(
        meta_info = meta_info,
        fit_t = min(fit_times) / fit_repetitions,
        pred_t = min(predict_times) / predict_repetitions,
        acc = 100 * accuracy_score(y_train, res),
        sv_len = clf.support_.shape[0],
        cl = clf.n_support_.shape[0]
    ))


def getArguments(argParser):
    argParser.add_argument('--fileX',               type=argparse.FileType('r'),  help="file with features to train/test on")
    argParser.add_argument('--fileY',               type=argparse.FileType('r'),  help="file with labels to train on")
    #
    argParser.add_argument('--fit-samples',         type=int, default=3,  help="Timing samples for fitting")
    argParser.add_argument('--predict-samples',     type=int, default=5,  help="Timing sample for predict")
    argParser.add_argument('--fit-repetitions',     type=int, default=10, help="Number of fit calls to time for each sample")
    argParser.add_argument('--predict-repetitions', type=int, default=10, help="Number of predict calls to time for each sample")
    #
    argParser.add_argument('--num-threads', '--core-number',         type=int, default=0, help="core numbers")
    argParser.add_argument('--verbose','-v',        action='store_true', help="Whether to be verbose or terse")
    argParser.add_argument('--header',              action='store_true', help="Whether to be print header")
    argParser.add_argument('--prefix',              type=str,  default=sys.executable, help="Prefix to report in the output")
    args = argParser.parse_args()
    #
    return args


def main():
    argParser = argparse.ArgumentParser(prog="svm-linear.py",
                                    description="sklearn two-class SVC benchmark for linear kernel",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = getArguments(argParser)
    numThreads, daal_version = bench.prepare_benchmark(args)

    # This is so the .csv file can go directly into excel. However, it requries adding timing
    # in sklearn/daal4sklearn/svm.py around pydaal compute and post-processing
    if args.verbose:
        print('@ {fit_samples: ' + str(args.fit_samples) +
              ', fit_repetitions: ' + str(args.fit_repetitions) +
              ', predict_samples: ' + str(args.predict_samples) +
              ', predict_repetitions: ' + str(args.predict_repetitions) +
              ', pyDAAL: ' + str(daal_version) + '}', file=sys.stderr)

    # Load data and cast to float64
    X_train = np.load(args.fileX.name).astype('f8')
    y_train = np.load(args.fileY.name).astype('f8')

    v, f = X_train.shape
    cache_size_bytes = getOptimalCacheSize(X_train.shape[0])
    cache_size_mb = cache_size_bytes / 1024**2
    meta_info = ",".join([args.prefix, 'SVM', str(numThreads), str(v), str(f), str(int(cache_size_mb))])

    svc_params_dict = {'C' : 0.01, 'kernel': 'linear', 'max_iter' : 2000, 'cache_size': cache_size_mb, 'tol': 1e-16, 'shrinking': True}

    if args.verbose:
        print("@ {}".format(svc_params_dict), file=sys.stderr)

    if args.header:
        print('prefix_ID,function,threads,rows,features,cache-size-MB,fit,predict,accuracy,sv-len,classes')
    _bench(meta_info, X_train, y_train,
           args.fit_samples, args.fit_repetitions, args.predict_samples, args.predict_repetitions, svc_params_dict)


if __name__ == '__main__':
    main()
