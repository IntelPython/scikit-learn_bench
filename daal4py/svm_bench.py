# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function

import os
import sys
import argparse
from timeit import default_timer as time
from bench import prepare_benchmark

import numpy as np
from daal4py import daalinit, svm_training, svm_prediction, kernel_function_linear
from daal4py import multi_class_classifier_training, multi_class_classifier_prediction
accuracy_score = lambda y, yp: np.mean(y == yp)


def getOptimalCacheSize(numFeatures):
    byte_size = np.empty(0, dtype=np.double).itemsize
    optimal_cache_size_bytes = numFeatures * numFeatures * byte_size
    eight_gb = byte_size * 1024 * 1024 * 1024
    cache_size_bytes = eight_gb if optimal_cache_size_bytes > eight_gb else optimal_cache_size_bytes
    return cache_size_bytes


# Methods to extract coefficients
def group_indices_by_class(num_classes, sv_ind_by_clf, labels):
    sv_ind_counters = np.zeros(num_classes, dtype=np.intp)

    num_of_sv_per_class = np.bincount(labels[np.hstack(sv_ind_by_clf)])
    sv_ind_by_class = [np.empty(n, dtype=np.int32) for n in num_of_sv_per_class]

    for indices_per_clf in sv_ind_by_clf:
        for sv_index in indices_per_clf:
            sv_label = labels[sv_index]
            i = sv_ind_counters[sv_label]
            sv_ind_by_class[sv_label][i] = sv_index
            sv_ind_counters[sv_label] += 1

    return sv_ind_by_class


def map_sv_to_columns_in_dual_coef_matrix(sv_ind_by_class):
    from collections import defaultdict
    sv_ind_mapping = defaultdict(lambda: -1)
    p = 0
    for indices_per_class in sv_ind_by_class:
        indices_per_class.sort()
        for sv_index in indices_per_class:
            if sv_ind_mapping[sv_index] == -1:
                sv_ind_mapping[sv_index] = p
                p += 1
    return sv_ind_mapping


def map_to_lexicographic(n):
    """Returns permutation of reverse lexicographics to lexicographics orders for pairs of n consecutive integer indexes"""
    from itertools import (combinations, count)
    two_class_order_gen = ((j, i) for i in range(n) for j in range(i))
    reverse_lookup = { key:val for key,val in zip(two_class_order_gen, count(0))}
    perm_iter = (reverse_lookup[pair] for pair in combinations(range(n), 2))
    return np.fromiter(perm_iter, dtype=np.intp)


def permute_list(li, perm):
    "Rearrange `li` according to `perm`"
    return [ li[i] for i in perm ]


def extract_dual_coef(num_classes, sv_ind_by_clf, sv_coef_by_clf, labels):
    "Construct dual coefficients array in SKLearn peculiar layout, as well corresponding support vector indexes"
    sv_ind_by_class = group_indices_by_class(num_classes, sv_ind_by_clf, labels)
    sv_ind_mapping = map_sv_to_columns_in_dual_coef_matrix(sv_ind_by_class)

    num_unique_sv = len(sv_ind_mapping)
    dc_dt = sv_coef_by_clf[0].dtype

    dual_coef = np.zeros((num_classes - 1, num_unique_sv), dtype=dc_dt)
    support_ = np.empty((num_unique_sv,), dtype=np.int32)

    p = 0
    for i in range(0, num_classes):
        for j in range(i + 1, num_classes):
            sv_ind_i_vs_j = sv_ind_by_clf[p]
            sv_coef_i_vs_j = sv_coef_by_clf[p]
            p += 1

            for k, sv_index in enumerate(sv_ind_i_vs_j):
                label = labels[sv_index]
                col_index = sv_ind_mapping[sv_index]
                if j == label:
                    row_index = i
                else:
                    row_index = j - 1
                dual_coef[row_index, col_index] = sv_coef_i_vs_j[k]
                support_[col_index] = sv_index

    return dual_coef, support_


def construct_dual_coefs(model, num_classes, X, y):

    if num_classes == 2:
        # binary
        two_class_sv_ind_ = model.SupportIndices
        two_class_sv_ind_ = two_class_sv_ind_.ravel()

        # support indexes need permutation to arrange them into the same layout as that of Scikit-Learn
        tmp = np.empty(two_class_sv_ind_.shape, dtype=np.dtype([('label', y.dtype), ('ind', two_class_sv_ind_.dtype)]))
        tmp['label'][:] = y[two_class_sv_ind_].ravel()
        tmp['ind'][:] = two_class_sv_ind_
        perm = np.argsort(tmp, order=['label', 'ind'])
        del tmp

        support_ = two_class_sv_ind_[perm]
        support_vectors_ = X[support_]

        dual_coef_ = model.ClassificationCoefficients.T
        dual_coef_ = dual_coef_[:, perm]
        intercept_ = np.array([model.Bias])

    else:
        # multi-class
        intercepts = []
        coefs = []
        num_models = model.NumberOfTwoClassClassifierModels
        sv_ind_by_clf = []
        label_indexes = []

        model_id = 0
        for i1 in range(num_classes):
            label_indexes.append(np.where( y == i1 )[0])
            for i2 in range(i1):
                svm_model = model.TwoClassClassifierModel(model_id)

                # Indices correspond to input features with label i1 followed by input features with label i2
                two_class_sv_ind_ = svm_model.SupportIndices
                # Map these indexes to indexes of the training data
                sv_ind = np.take(np.hstack((label_indexes[i1], label_indexes[i2])), two_class_sv_ind_.ravel())
                sv_ind_by_clf.append(sv_ind)

                # svs_ = getArrayFromNumericTable(svm_model.getSupportVectors())
                # assert np.array_equal(svs_, X[sv_ind])

                intercepts.append(-svm_model.Bias)
                coefs.append(-svm_model.ClassificationCoefficients)
                model_id += 1

        # permute solutions to lexicographic ordering
        to_lex_perm = map_to_lexicographic(num_classes)
        sv_ind_by_clf = permute_list(sv_ind_by_clf, to_lex_perm)
        sv_coef_by_clf = permute_list(coefs, to_lex_perm)
        intercepts = permute_list(intercepts, to_lex_perm)

        dual_coef_, support_ = extract_dual_coef(
            num_classes,    # number of classes
            sv_ind_by_clf,  # support vector indexes by two-class classifiers
            sv_coef_by_clf, # classification coefficients by two-class classifiers
            y.squeeze().astype(np.intp, copy=False)   # integer labels
        )
        support_vectors_ = X[support_]
        intercept_ = np.array(intercepts)

    return support_


def bench(meta_info, X_train, y_train, fit_samples, fit_repetitions,
          predict_samples, predict_repetitions, classes, cache_size,
          accuracy_threshold=1e-16, max_iterations=2000):

    kf = kernel_function_linear(fptype='double')

    if classes == 2:
        y_train[y_train == 0] = -1
    else:
        y_train[y_train == -1] = 0

    fit_times = []
    for it in range(fit_samples):
        start = time()
        for __ in range(fit_repetitions):
            svm_train = svm_training(
                    fptype='double',
                    C=0.01,
                    maxIterations=max_iterations,
                    tau=1e-12,
                    cacheSize=cache_size,
                    accuracyThreshold=accuracy_threshold,
                    doShrinking=True,
                    kernel=kf
            )

            if classes == 2:
                clf = svm_train
            else:
                clf = multi_class_classifier_training(
                        nClasses=classes,
                        fptype='double',
                        accuracyThreshold=accuracy_threshold,
                        method='oneAgainstOne',
                        maxIterations=max_iterations,
                        training=svm_train
                )

            training_result = clf.compute(X_train, y_train)

            support = construct_dual_coefs(training_result.model, classes, X_train, y_train)
            indices = y_train.take(support, axis=0)
            if classes == 2:
                n_support_ = np.array([np.sum(indices == -1), np.sum(indices == 1)], dtype=np.int32)
            else:
                n_support_ = np.array([np.sum(indices == c) for c in [-1] + list(range(1, classes))], dtype=np.int32)
        stop = time()
        fit_times.append(stop-start)


    predict_times = []
    for it in range(predict_samples):
        svm_predict = svm_prediction(
                fptype='double',
                method='defaultDense',
                kernel=kf
        )
        if classes == 2:
            prdct = svm_predict
        else:
            prdct = multi_class_classifier_prediction(
                    nClasses=classes,
                    fptype='double',
                    maxIterations=max_iterations,
                    accuracyThreshold=accuracy_threshold,
                    pmethod='voteBased',
                    tmethod='oneAgainstOne',
                    prediction=svm_predict
            )

        start = time()
        for __ in range(predict_repetitions):
            res = prdct.compute(X_train, training_result.model)
        stop = time()
        predict_times.append(stop-start)

    if classes == 2:
        y_predict = np.greater(res.prediction.ravel(), 0)
        y_train = np.greater(y_train, 0)
    else:
        y_predict = res.prediction.ravel()

    print("{meta_info},{fit_t:0.6g},{pred_t:0.6g},{acc:0.3f},{sv_len},{cl}".format(
        meta_info = meta_info,
        fit_t = min(fit_times) / fit_repetitions,
        pred_t = min(predict_times) / predict_repetitions,
        acc = 100 * accuracy_score(y_train.ravel(), y_predict),
        sv_len = support.shape[0],
        cl = n_support_.shape[0]
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
    argParser.add_argument('--num-threads', '--core-number', type=int, default=0, help="core numbers")
    argParser.add_argument('--verbose','-v',        action='store_true', help="Whether to be verbose or terse")
    argParser.add_argument('--header',              action='store_true', help="Whether to be print header")
    argParser.add_argument('--prefix',              type=str,  default=sys.executable, help="Prefix to report in the output")
    args = argParser.parse_args()
    #
    return args


def main():
    argParser = argparse.ArgumentParser(prog="svm-linear.py",
                                    description="SVC benchmark for linear kernel",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = getArguments(argParser)
    num_threads, daal_version = prepare_benchmark(args)

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
    n_classes = np.unique(y_train).size
    y_train[y_train == 0] = -1
    y_train = y_train[:,np.newaxis]

    v, f = X_train.shape
    cache_size = getOptimalCacheSize(X_train.shape[0])
    meta_info = ",".join([args.prefix, 'SVM', str(num_threads), str(v), str(f), str(int(cache_size))])

    svc_params_dict = {
            'C' : 0.01,
            'maxIterations' : 2000,
            'tau': 1e-12,
            'cacheSize': cache_size,
            'accuracyThreshold': 1e-16,
            'doShrinking': True,
    }

    if args.verbose:
        print("@ {}".format(svc_params_dict), file=sys.stderr)

    if args.header:
        print('prefix_ID,function,threads,rows,features,cache-size-MB,fit,predict,accuracy,sv_len,classes')
    bench(meta_info, X_train, y_train, args.fit_samples,
          args.fit_repetitions, args.predict_samples,
          args.predict_repetitions, n_classes, cache_size, accuracy_threshold=1e-16,
          max_iterations=2000)


if __name__ == '__main__':
    main()
