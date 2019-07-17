/*
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#define DAAL_DATA_TYPE double
#include "CLI11.hpp"
#include "common.hpp"
#include "daal.h"
#include "npyfile.h"

namespace dam = daal::algorithms::multi_class_classifier;

struct svm_params {
    double C;
    double tol;
    double tau;
    int max_iter;
};

void print_svm_params(svm_params p) {
    std::clog << "@ { C: " << p.C << ", tol: " << p.tol << ", tau: " << p.tau
              << ", max_iter: " << p.max_iter << "}" << std::endl;
}

size_t get_optimal_cache_size(size_t n) {
    return sizeof(double) * n * n;
}

std::vector<int> lexicographic_permutation(int n_cl) {
    std::vector<int> perm;

    int *mat = new int[n_cl * n_cl];
    for (int i1 = 0, k = 0; i1 < n_cl; i1++) {
        mat[i1 * (n_cl + 1)] = 0;
        for (int i2 = 0; i2 < i1; i2++, k++) {
            mat[i1 * n_cl + i2] = k;
            mat[i2 * n_cl + i1] = 0;
        }
    }

    for (int i1 = 0; i1 < n_cl; i1++) {
        for (int i2 = i1 + 1; i2 < n_cl; i2++) {
            perm.push_back(mat[i2 * n_cl + i1]);
        }
    }
    delete[] mat;

    return perm;
}

#if 1
#define ROUND(x) round(x)
#else
#define ROUND(x) x
#endif

std::vector<std::vector<int>>
group_indices_by_class(int n_classes, double *labels,
                       std::vector<std::vector<int>> sv_ind_by_clf) {
    int max_lbl = -1;
    std::vector<std::vector<int>> sv_ind_by_class;
    sv_ind_by_class.resize(n_classes);
    for (std::vector<std::vector<int>>::iterator it = sv_ind_by_clf.begin();
         it != sv_ind_by_clf.end(); ++it) {
        std::vector<int> v = *it;
        for (std::vector<int>::iterator it2 = v.begin(); it2 != v.end();
             ++it2) {
            int idx = *it2;
            int lbl = static_cast<int>(ROUND(labels[idx]));
            sv_ind_by_class[lbl].push_back(idx);

            if (lbl > max_lbl)
                max_lbl = lbl;
        }
    }
    if (max_lbl + 1 < n_classes) {
        sv_ind_by_class.resize(max_lbl + 1);
    }

    return sv_ind_by_class;
}

#define IS_IN_MAP(_map, _key) ((_map).count(_key))

std::map<int, int> map_sv_to_columns_on_dual_coef_matrix(
    std::vector<std::vector<int>> sv_ind_by_class) {
    std::map<int, int> sv_ind_mapping;
    int p = 0;
    for (auto indices_per_class : sv_ind_by_class) {
        std::sort(indices_per_class.begin(), indices_per_class.end());
        for (auto sv_index : indices_per_class) {
            if (!IS_IN_MAP(sv_ind_mapping, sv_index)) {
                sv_ind_mapping[sv_index] = p;
                ++p;
            }
        }
    }

    return sv_ind_mapping;
}

template <typename T>
void permute_vector(std::vector<T> &v, const std::vector<int> &perm) {
    std::vector<T> v_permuted;
    v_permuted.reserve(v.size());
    for (auto idx : perm)
        v_permuted.push_back(v[idx]);
    std::swap(v, v_permuted);
}

size_t construct_dual_coefs(dam::training::ResultPtr training_result,
                            int n_classes, dm::NumericTablePtr Y_nt, int n_rows,
                            double *dual_coef_ptr, bool verbose) {
    dm::BlockDescriptor<double> blockY;
    Y_nt->getBlockOfRows(0, n_rows, dm::readOnly, blockY);
    double *Y_data_ptr = blockY.getBlockPtr();

    dam::ModelPtr multi_svm_model =
        training_result->get(da::classifier::training::model);

    int num_models = multi_svm_model->getNumberOfTwoClassClassifierModels();

    std::vector<double> intercepts(num_models);
    std::vector<std::vector<double>> coefs(num_models);

    std::vector<std::vector<int>> sv_ind_by_clf(num_models);
    std::vector<std::vector<int>> label_indexes(n_classes);

    for (int i1 = 0, model_id = 0; i1 < n_classes; i1++) {
        for (int j = 0; j < n_rows; j++)
            if (Y_data_ptr[j] == i1)
                label_indexes[i1].push_back(j);

        int idx_len = label_indexes[i1].size();

        for (int i2 = 0; i2 < i1; i2++, model_id++) {
            auto classifier_model =
                multi_svm_model->getTwoClassClassifierModel(model_id);
            da::svm::ModelPtr bin_svm_model =
                ds::dynamicPointerCast<da::svm::Model>(classifier_model);

            dm::NumericTablePtr sv_indx = bin_svm_model->getSupportIndices();

            /* sv_ind = np.take(np.hstack((label_indexes[i1],
               label_indexes[i2])), two_class_sv_ind_.ravel())
               sv_ind_by_clf.append(sv_ind) */

            dm::BlockDescriptor<int> block_sv_indx;
            int two_class_classifier_sv_len = sv_indx->getNumberOfRows();
            sv_indx->getBlockOfRows(0, two_class_classifier_sv_len,
                                    dm::readOnly, block_sv_indx);
            int *sv_ind_data_ptr = block_sv_indx.getBlockPtr();

            for (int j = 0; j < two_class_classifier_sv_len; j++) {
                int sv_idx = sv_ind_data_ptr[j];
                sv_ind_by_clf[model_id].push_back(
                    (sv_idx < idx_len) ? label_indexes[i1][sv_idx]
                                       : label_indexes[i2][sv_idx - idx_len]);
            }
            sv_indx->releaseBlockOfRows(block_sv_indx);

            auto bias = bin_svm_model->getBias();
            intercepts.push_back(-bias);

            auto sv_coefs = bin_svm_model->getClassificationCoefficients();
            dm::BlockDescriptor<double> block_sv_coefs;
            sv_coefs->getBlockOfRows(0, two_class_classifier_sv_len,
                                     dm::readOnly, block_sv_coefs);
            double *sv_coefs_ptr = block_sv_coefs.getBlockPtr();

            for (int q = 0; q < two_class_classifier_sv_len; q++) {
                coefs[model_id].push_back(sv_coefs_ptr[q]);
            }
            sv_coefs->releaseBlockOfRows(block_sv_coefs);
        }
    }
    Y_nt->releaseBlockOfRows(blockY);

    std::vector<int> perm = lexicographic_permutation(n_classes);

    assert(perm.size() == n_classes * (n_classes - 1) / 2);

    permute_vector(sv_ind_by_clf, perm);
    permute_vector(intercepts, perm);
    permute_vector(coefs, perm);

    std::vector<std::vector<int>> sv_ind_by_class =
        group_indices_by_class(n_classes, Y_data_ptr, sv_ind_by_clf);
    auto mp = map_sv_to_columns_on_dual_coef_matrix(sv_ind_by_class);

    size_t num_unique_sv = mp.size();
    dual_coef_ptr = new double[(n_classes - 1) * num_unique_sv];
    std::vector<int> support_(num_unique_sv);
    int p = 0;
    for (int i = 0; i < n_classes; i++) {
        for (int j = i + 1; j < n_classes; j++, p++) {
            std::vector<int> sv_ind_i_vs_j = sv_ind_by_clf[p];
            std::vector<double> sv_coef_i_vs_j = coefs[p];

            int k = 0;
            for (auto sv_index : sv_ind_i_vs_j) {
                int label = static_cast<int>(round(Y_data_ptr[sv_index]));
                int col_index = mp[sv_index];
                int row_index = (j == label) ? i : j - 1;
                dual_coef_ptr[row_index * (num_unique_sv) + col_index] =
                    sv_coef_i_vs_j[k];
                support_[col_index] = sv_index;
            }
        }
    }

    return num_unique_sv;
}

template <typename dtype = double>
std::tuple<da::classifier::training::ResultPtr, unsigned long>
svm_fit(svm_params svc_params, dm::NumericTablePtr Xt, dm::NumericTablePtr Yt,
        int n_classes, bool verbose) {

    ds::SharedPtr<da::svm::training::Batch<dtype>> training_algo_ptr(
        new da::svm::training::Batch<dtype>());

    /* Parameters for the SVM kernel function */
    ds::SharedPtr<da::kernel_function::linear::Batch<dtype>> kernel_ptr(
        new da::kernel_function::linear::Batch<dtype>());
    size_t n_features = Xt->getNumberOfColumns();
    size_t n_samples = Xt->getNumberOfRows();

    training_algo_ptr->parameter.C = svc_params.C;
    training_algo_ptr->parameter.kernel = kernel_ptr;
    training_algo_ptr->parameter.cacheSize = get_optimal_cache_size(n_samples);
    training_algo_ptr->parameter.accuracyThreshold = svc_params.tol;
    training_algo_ptr->parameter.tau = svc_params.tau;
    training_algo_ptr->parameter.maxIterations = svc_params.max_iter;
    training_algo_ptr->parameter.doShrinking = true;

    ds::SharedPtr<da::classifier::training::Batch> algorithm;

    if (n_classes > 2) {

        if (verbose) {
            std::clog << "@ Using DAAL multi_class_classifier training"
                      << std::endl;
        }

        ds::SharedPtr<dam::training::Batch<dtype>> mc_algorithm(
            new dam::training::Batch<dtype>(n_classes));
        mc_algorithm->parameter.training = training_algo_ptr;
        mc_algorithm->parameter.maxIterations = svc_params.max_iter;
        mc_algorithm->parameter.accuracyThreshold = svc_params.tol;

        algorithm = mc_algorithm;

    } else {
        algorithm = training_algo_ptr;
    }

    /* Pass a training data set and dependent values to the algorithm */
    algorithm->getInput()->set(da::classifier::training::data, Xt);
    algorithm->getInput()->set(da::classifier::training::labels, Yt);

    if (verbose) {
        print_svm_params(svc_params);
    }

    /* Build the SVM model */
    algorithm->compute();
    auto training_result = algorithm->getResult();

    // for multi_class: allocates memory for dual coefficients
    double *dual_coefs_ptr = NULL;
    size_t sv_len;
    auto mc_training_result =
        ds::dynamicPointerCast<dam::training::Result>(training_result);
    if (mc_training_result) {
        sv_len = construct_dual_coefs(mc_training_result, n_classes, Yt,
                                      n_samples, dual_coefs_ptr, verbose);
    } else {
        auto svm_training_result =
            ds::dynamicPointerCast<da::svm::training::Result>(training_result);
        assert(svm_training_result);
        auto svm_model = ds::dynamicPointerCast<da::svm::Model>(
            svm_training_result->get(da::classifier::training::model));
        assert(svm_model);
        auto sv_idx = svm_model->getSupportIndices();
        sv_len = sv_idx->getNumberOfRows();
    }

    if (dual_coefs_ptr) {
        delete[] dual_coefs_ptr;
        dual_coefs_ptr = NULL;
    }

    return std::make_tuple(training_result, sv_len);
}

template <typename dtype = double>
dm::NumericTablePtr
svm_predict(svm_params svc_params, da::classifier::training::ResultPtr result,
            dm::NumericTablePtr X_nt, int n_classes, bool verbose) {

    ds::SharedPtr<da::kernel_function::linear::Batch<dtype>> kernel_ptr(
        new da::kernel_function::linear::Batch<dtype>());

    ds::SharedPtr<da::svm::prediction::Batch<dtype>>
        two_class_pred_batch_algo_ptr(new da::svm::prediction::Batch<dtype>());
    two_class_pred_batch_algo_ptr->parameter.kernel = kernel_ptr;

    ds::SharedPtr<da::classifier::prediction::Batch> algorithm;

    // Was our result from a multi_class_classifier?
    auto mc_training_result =
        ds::dynamicPointerCast<dam::training::Result>(result);
    if (mc_training_result) {

        if (verbose) {
            std::clog << "@ Using DAAL multi_class_classifier prediction"
                      << std::endl;
        }

        ds::SharedPtr<dam::prediction::Batch<dtype, dam::prediction::voteBased>>
            mc_algorithm(
                new dam::prediction::Batch<dtype, dam::prediction::voteBased>(
                    n_classes));

        mc_algorithm->parameter.prediction = two_class_pred_batch_algo_ptr;

        algorithm = mc_algorithm;

    } else {
        algorithm = two_class_pred_batch_algo_ptr;
    }

    algorithm->getInput()->set(da::classifier::prediction::data, X_nt);
    algorithm->getInput()->set(da::classifier::prediction::model,
                               result->get(da::classifier::training::model));

    algorithm->compute();

    return algorithm->getResult()->get(da::classifier::prediction::prediction);
}

void bench(size_t threadNum, const std::string &X_fname,
           const std::string &y_fname, int fit_samples, int fit_repetitions,
           int predict_samples, int predict_repetitions, bool verbose,
           bool header) {

    /* Set the maximum number of threads to be used by the library */
    if (threadNum != 0)
        daal::services::Environment::getInstance()->setNumberOfThreads(
            threadNum);

    size_t daal_thread_num =
        daal::services::Environment::getInstance()->getNumberOfThreads();
    /* Load data */
    struct npyarr *arrX = load_npy(X_fname.c_str());
    struct npyarr *arrY = load_npy(y_fname.c_str());
    if (!arrX || !arrY) {
        std::cerr << "Failed to load input arrays" << std::endl;
        std::exit(1);
        return;
    }
    if (arrX->shape_len != 2) {
        std::cerr << "Expected 2 dimensions for X, found " << arrX->shape_len
                  << std::endl;
        std::exit(1);
        return;
    }
    if (arrY->shape_len != 1) {
        std::cerr << "Expected 1 dimension for y, found " << arrY->shape_len
                  << std::endl;
        std::exit(1);
        return;
    }

    /* Create numeric tables */
    dm::NumericTablePtr X_nt = dm::HomogenNumericTable<double>::create(
        (double *) arrX->data, arrX->shape[1], arrX->shape[0]);
    dm::NumericTablePtr Y_nt = dm::HomogenNumericTable<int64_t>::create(
        (int64_t *) arrY->data, 1, arrY->shape[0]);

    size_t n_rows = Y_nt->getNumberOfRows();

    int n_classes = count_classes(Y_nt);

    if (n_classes == 2) {
        // DAAL wants labels in {-1, 1} instead of {0, 1}
        int64_t *y_data = (int64_t *) arrY->data;
        for (int i = 0; i < arrY->shape[0]; i++) {
            if (y_data[i] == 0)
                y_data[i] = -1;
        }
    }

    svm_params svm_problem;
    svm_problem.C = 0.01;
    svm_problem.tol = 1e-16;
    svm_problem.tau = 1e-12;
    svm_problem.max_iter = 2000;

    size_t sv_len = 0;
    std::vector<std::chrono::duration<double>> fit_times;
    da::classifier::training::ResultPtr training_result;
    for (int i = 0; i < fit_samples; i++) {
        auto start = std::chrono::system_clock::now();

        for (int j = 0; j < fit_repetitions; j++) {
            std::tie(training_result, sv_len) = svm_fit(
                svm_problem, X_nt, Y_nt, n_classes, verbose && (!i) && (!j));
        }

        auto finish = std::chrono::system_clock::now();
        fit_times.push_back(finish - start);
    }

    std::vector<std::chrono::duration<double>> predict_times;
    dm::NumericTablePtr Yp_nt;
    for (int i = 0; i < predict_samples; i++) {
        auto start = std::chrono::system_clock::now();
        for (int j = 0; j < predict_repetitions; j++) {
            Yp_nt = svm_predict(svm_problem, training_result, X_nt, n_classes,
                                verbose && (!i) && (!j));
        }
        auto finish = std::chrono::system_clock::now();
        predict_times.push_back(finish - start);
    }

    double accuracy = accuracy_score(Y_nt, Yp_nt) * 100.00;

    if (header) {
        std::cout << "prefix_ID,function,threads,rows,features,cache-size-MB,"
                  << "fit,predict,accuracy,sv-len,classes" << std::endl;
    }

    std::cout << "Native-C";
    std::cout << ","
              << "SVM";
    std::cout << "," << daal_thread_num;
    std::cout << "," << X_nt->getNumberOfRows() << ","
              << X_nt->getNumberOfColumns();
    std::cout << ","
              << get_optimal_cache_size(X_nt->getNumberOfRows()) /
                     (1024 * 1024);
    std::cout << ","
              << std::min_element(fit_times.begin(), fit_times.end())->count() /
                     fit_repetitions;
    std::cout << ","
              << std::min_element(predict_times.begin(), predict_times.end())
                         ->count() /
                     predict_repetitions;
    std::cout << "," << accuracy;
    std::cout << "," << sv_len;
    std::cout << "," << n_classes;
    std::cout << std::endl;
}

int main(int argc, char **argv) {

    CLI::App app("Native benchmark code for Intel(R) DAAL SVM classifier");

    std::string xfn = "./data/mX.csv";
    app.add_option("--fileX", xfn, "Feature file name")
        ->required()
        ->check(CLI::ExistingFile);

    std::string yfn = "./data/mY.csv";
    app.add_option("--fileY", yfn, "Labels file name")
        ->required()
        ->check(CLI::ExistingFile);

    int fit_samples = 3, fit_repetitions = 1, predict_samples = 5,
        predict_repetitions = 1;
    app.add_option("--fit-samples", fit_samples,
                   "Number of samples to collect for time "
                   "of execution of repeated fit calls",
                   true);
    app.add_option("--fit-repetitions", fit_repetitions,
                   "Number of repeated fit calls to time", true);
    app.add_option("--predict-samples", predict_samples,
                   "Number of samples to collect for time of "
                   "execution of repeated predict calls",
                   true);
    app.add_option("--predict-repetitions", predict_repetitions,
                   "Number of repeated predict calls to time", true);

    int num_threads = 0;
    app.add_option("-n,--num-threads", num_threads,
                   "Number of threads for DAAL to use", true);

    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Whether to be verbose or terse");

    bool header = false;
    app.add_flag("--header", header, "Whether to output header");

    CLI11_PARSE(app, argc, argv);

    assert(num_threads >= 0);
    assert(fit_samples > 0);
    assert(fit_repetitions > 0);
    assert(predict_samples > 0);
    assert(predict_repetitions > 0);

    if (verbose) {
        std::clog << "@ {FIT_SAMPLES: " << fit_samples
                  << ", FIT_REPETITIONS: " << fit_repetitions
                  << ", PREDICT_SAMPLES: " << predict_samples
                  << ", PREDICT_REPETITIONS: " << predict_repetitions << "}"
                  << std::endl;
    }

    bench(num_threads, xfn, yfn, fit_samples, fit_repetitions, predict_samples,
          predict_repetitions, verbose, header);

    return 0;
}
