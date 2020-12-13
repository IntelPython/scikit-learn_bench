# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, load_data, print_output, accuracy_score,
    getFPType
)
import numpy as np
from daal4py import (
    svm_training, svm_prediction, kernel_function_linear, kernel_function_rbf,
    multi_class_classifier_training, multi_class_classifier_prediction
)


def get_optimal_cache_size(n_rows, dtype=np.double, max_cache=64):
    '''
    Get an optimal cache size for daal4py.svm_training.

    Parameters
    ----------
    n_rows : int
        Number of rows in the dataset
    dtype : dtype-like, optional (default np.double)
        dtype to use for computing cache size
    max_cache : int, optional (default 64)
        maximum cache size, in gigabytes
    '''

    byte_size = np.empty(0, dtype=dtype).itemsize
    optimal_cache_size_bytes = byte_size * (n_rows ** 2)
    one_gb = 2 ** 30
    max_cache_bytes = max_cache * one_gb
    if optimal_cache_size_bytes > max_cache_bytes:
        return max_cache_bytes
    else:
        return optimal_cache_size_bytes


# Methods to extract coefficients
def group_indices_by_class(num_classes, sv_ind_by_clf, labels):
    sv_ind_counters = np.zeros(num_classes, dtype=np.intp)

    num_sv_per_class = np.bincount(labels[np.hstack(sv_ind_by_clf)])
    sv_ind_by_class = [np.empty(n, dtype=np.int32) for n in num_sv_per_class]

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
    '''
    Returns permutation of reverse lexicographics to lexicographics orders
    for pairs of n consecutive integer indexes
    '''

    from itertools import (combinations, count)
    two_class_order_gen = ((j, i) for i in range(n) for j in range(i))
    reverse_lookup = {k: v for k, v in zip(two_class_order_gen, count(0))}
    perm_iter = (reverse_lookup[pair] for pair in combinations(range(n), 2))
    return np.fromiter(perm_iter, dtype=np.intp)


def permute_list(li, perm):
    'Rearrange `li` according to `perm`'
    return [li[i] for i in perm]


def extract_dual_coef(num_classes, sv_ind_by_clf, sv_coef_by_clf, labels):
    '''
    Construct dual coefficients array in SKLearn peculiar layout, as well
    as corresponding support vector indexes
    '''

    sv_ind_by_class = group_indices_by_class(num_classes, sv_ind_by_clf,
                                             labels)
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

        # support indexes need permutation to arrange them into the same layout
        # as that of Scikit-Learn
        tmp = np.empty(two_class_sv_ind_.shape,
                       dtype=np.dtype([('label', y.dtype),
                                       ('ind', two_class_sv_ind_.dtype)]))
        tmp['label'][:] = y[two_class_sv_ind_].ravel()
        tmp['ind'][:] = two_class_sv_ind_
        perm = np.argsort(tmp, order=['label', 'ind'])
        del tmp

        support_ = two_class_sv_ind_[perm]
        dual_coef_ = model.ClassificationCoefficients.T
        dual_coef_ = dual_coef_[:, perm]

    else:
        # multi-class
        intercepts = []
        coefs = []
        # num_models = model.NumberOfTwoClassClassifierModels
        sv_ind_by_clf = []
        label_indexes = []

        model_id = 0
        for i1 in range(num_classes):
            label_indexes.append(np.where(y == i1)[0])
            for i2 in range(i1):
                svm_model = model.TwoClassClassifierModel(model_id)

                # Indices correspond to input features with label i1 followed
                # by input features with label i2
                two_class_sv_ind_ = svm_model.SupportIndices

                # Map these indexes to indexes of the training data
                sv_ind = np.take(np.hstack((label_indexes[i1],
                                            label_indexes[i2])),
                                 two_class_sv_ind_.ravel())
                sv_ind_by_clf.append(sv_ind)

                intercepts.append(-svm_model.Bias)
                coefs.append(-svm_model.ClassificationCoefficients)
                model_id += 1

        # permute solutions to lexicographic ordering
        to_lex_perm = map_to_lexicographic(num_classes)
        sv_ind_by_clf = permute_list(sv_ind_by_clf, to_lex_perm)
        sv_coef_by_clf = permute_list(coefs, to_lex_perm)
        intercepts = permute_list(intercepts, to_lex_perm)

        dual_coef_, support_ = extract_dual_coef(
            num_classes,     # number of classes
            sv_ind_by_clf,   # support vector indexes by two-class classifiers
            sv_coef_by_clf,  # classification coeffs by two-class classifiers
            y.squeeze().astype(np.intp, copy=False)   # integer labels
        )

    return support_


def daal_kernel(name, fptype, gamma=1.0):

    if name == 'linear':
        return kernel_function_linear(fptype=fptype)
    else:
        sigma = np.sqrt(0.5 / gamma)
        return kernel_function_rbf(fptype=fptype, sigma=sigma)


def test_fit(X, y, params):

    fptype = getFPType(X)
    kf = daal_kernel(params.kernel, fptype, gamma=params.gamma)

    svm_train = svm_training(
            method='thunder',
            fptype=fptype,
            C=params.C,
            maxIterations=params.maxiter,
            tau=params.tau,
            cacheSize=params.cache_size_bytes,
            accuracyThreshold=params.tol,
            doShrinking=params.shrinking,
            kernel=kf
    )

    if params.n_classes == 2:
        clf = svm_train
    else:
        clf = multi_class_classifier_training(
                fptype=fptype,
                nClasses=params.n_classes,
                accuracyThreshold=params.tol,
                method='oneAgainstOne',
                maxIterations=params.maxiter,
                training=svm_train
        )

    training_result = clf.compute(X, y)

    support = construct_dual_coefs(training_result.model, params.n_classes,
                                   X, y)
    indices = y.take(support, axis=0)
    if params.n_classes == 2:
        n_support_ = np.array([np.sum(indices == -1),
                               np.sum(indices == 1)], dtype=np.int32)
    else:
        n_support_ = np.array([np.sum(indices == c) for c in
                               [-1] + list(range(1, params.n_classes))],
                              dtype=np.int32)

    return training_result, support, indices, n_support_


def test_predict(X, training_result, params):

    fptype = getFPType(X)
    kf = daal_kernel(params.kernel, fptype, gamma=params.gamma)

    svm_predict = svm_prediction(
            fptype=fptype,
            method='defaultDense',
            kernel=kf
    )
    if params.n_classes == 2:
        prdct = svm_predict
    else:
        prdct = multi_class_classifier_prediction(
                nClasses=params.n_classes,
                fptype=fptype,
                maxIterations=params.maxiter,
                accuracyThreshold=params.tol,
                pmethod='voteBased',
                tmethod='oneAgainstOne',
                prediction=svm_predict
        )

    res = prdct.compute(X, training_result.model)

    if params.n_classes == 2:
        y_predict = np.greater(res.prediction.ravel(), 0)
    else:
        y_predict = res.prediction.ravel()

    return y_predict


def main():
    parser = argparse.ArgumentParser(description='daal4py SVC benchmark with '
                                                 'linear kernel')
    parser.add_argument('-C', dest='C', type=float, default=1.0,
                        help='SVM regularization parameter')
    parser.add_argument('--kernel', choices=('linear', 'rbf'),
                        default='linear', help='SVM kernel function')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Parameter for kernel="rbf"')
    parser.add_argument('--maxiter', type=int, default=100000,
                        help='Maximum iterations for the iterative solver. ')
    parser.add_argument('--max-cache-size', type=int, default=8,
                        help='Maximum cache size, in gigabytes, for SVM.')
    parser.add_argument('--tau', type=float, default=1e-12,
                        help='Tau parameter for working set selection scheme')
    parser.add_argument('--tol', type=float, default=1e-3,
                        help='Tolerance')
    parser.add_argument('--no-shrinking', action='store_false', default=True,
                        dest='shrinking',
                        help="Don't use shrinking heuristic")
    params = parse_args(parser, prefix='daal4py')

    # Load data
    X_train, X_test, y_train, y_test = load_data(
        params, add_dtype=True, label_2d=True)

    if params.gamma is None:
        params.gamma = 1 / X_train.shape[1]

    cache_size_bytes = get_optimal_cache_size(X_train.shape[0],
                                              max_cache=params.max_cache_size)
    params.cache_size_mb = cache_size_bytes / 2**20
    params.cache_size_bytes = cache_size_bytes
    params.n_classes = np.unique(y_train).size

    columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype',
               'size', 'kernel', 'cache_size_mb', 'C', 'sv_len', 'n_classes',
               'accuracy', 'time')

    # Time fit and predict
    fit_time, res = measure_function_time(
        test_fit, X_train, y_train, params, params=params)
    res, support, indices, n_support = res
    params.sv_len = support.shape[0]

    yp = test_predict(X_train, res, params)
    train_acc = 100 * accuracy_score(yp, y_train)

    predict_time, yp = measure_function_time(
        test_predict, X_test, res, params, params=params)

    test_acc = 100 * accuracy_score(yp, y_train)

    print_output(library='daal4py', algorithm='svc',
                 stages=['training', 'prediction'], columns=columns,
                 params=params, functions=['SVM.fit', 'SVM.predict'],
                 times=[fit_time, predict_time], accuracy_type='accuracy[%]',
                 accuracies=[train_acc, test_acc], data=[X_train, X_test])


if __name__ == '__main__':
    main()
