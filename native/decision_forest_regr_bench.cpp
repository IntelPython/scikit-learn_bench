/*
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>

#define DAAL_DATA_TYPE double
#include "daal.h"
#include "CLI11.hpp"
#include "npyfile.h"

namespace dm=daal::data_management;
namespace ds=daal::services;
namespace da=daal::algorithms;
namespace df=daal::algorithms::decision_forest;
namespace dfr=daal::algorithms::decision_forest::regression;

using namespace daal;
using namespace da;

void print_numeric_table(dm::NumericTablePtr, std::string);

dfr::training::ResultPtr
df_regression_fit(
    size_t nTrees,
    size_t seed,
    size_t n_features_per_node,
    size_t max_depth,
    double min_impurity,
    bool bootstrap,
    dm::NumericTablePtr Xt,
    dm::NumericTablePtr Yt,
    bool verbose)
{
    size_t n_samples = Yt->getNumberOfRows();
    size_t n_features = Xt->getNumberOfColumns();

    size_t fpn = n_features;
    if (n_features_per_node > 0 && n_features_per_node <= n_features) {
        fpn = n_features_per_node;
    }

    dfr::training::Batch<double> df_reg_alg;
    df_reg_alg.parameter.nTrees = nTrees;
    df_reg_alg.parameter.varImportance = df::training::MDI;
    df_reg_alg.parameter.observationsPerTreeFraction = 1.0;
    df_reg_alg.parameter.maxTreeDepth = max_depth;
    df_reg_alg.parameter.featuresPerNode = fpn;
    df_reg_alg.parameter.minObservationsInLeafNode = 1;
    df_reg_alg.parameter.impurityThreshold = min_impurity;
    df_reg_alg.parameter.bootstrap = bootstrap;
    df_reg_alg.parameter.memorySavingMode = false;
    df_reg_alg.parameter.engine
        = da::engines::mt2203::Batch<double>::create(seed);

    if (verbose) {
        std::cout << "@ {'nTrees': " << nTrees
                  << ", 'variable_importance': " << "MDI"
                  << ", 'features_per_node': " << fpn
                  << ", 'max_depth': " << max_depth 
                  << ", 'min_impurity': " << min_impurity
                  << ", 'seed': " << seed
                  << ", 'bootstrap': " << (bootstrap ? "True" : "False")
                  << "}" << std::endl;
    }

    df_reg_alg.input.set(dfr::training::data, Xt);
    df_reg_alg.input.set(dfr::training::dependentVariable, Yt);

    df_reg_alg.compute();

    dfr::training::ResultPtr result_ptr = df_reg_alg.getResult();

    return result_ptr;
}

dm::NumericTablePtr
df_regression_predict(
    dfr::training::ResultPtr training_result_ptr,
    dm::NumericTablePtr Xt,
    bool verbose
    )
{
    // We explicitly specify float here to match sklearn.
    dfr::prediction::Batch<float> pred_alg;
    pred_alg.input.set(dfr::prediction::data, Xt);
    pred_alg.input.set(dfr::prediction::model,
               training_result_ptr->get(dfr::training::model));

    pred_alg.compute();

    dfr::prediction::ResultPtr pred_res = pred_alg.getResult();
    dm::NumericTablePtr Y_pred_t = pred_res->get(dfr::prediction::prediction);

    return Y_pred_t;
}

double
explained_variance_score(
    dm::NumericTablePtr Y_nt,
    dm::NumericTablePtr Yp_nt,
    size_t n_rows)
{
    // http://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
    dm::BlockDescriptor<double> blockY;
    dm::BlockDescriptor<double> blockYp;

    double mean_y = 0.0, mean_ypy = 0.0;
    double vy = 0.0, vypy = 0.0;

    Yp_nt->getBlockOfRows(0, n_rows, dm::readOnly, blockYp);
    Y_nt->getBlockOfRows(0, n_rows, dm::readOnly, blockY);
    double *Y_data_ptr = blockY.getBlockPtr();
    double *Yp_data_ptr = blockYp.getBlockPtr();

    for (size_t i = 0; i < n_rows; i++) {
        double y = Y_data_ptr[i], yp = Yp_data_ptr[i];
        double dy = y - mean_y;
        double dypy = (yp - y) - mean_ypy;
        mean_y += dy / (i+1);
        mean_ypy += dypy / (i+1);
        vy += dy*(y - mean_y);
        vypy += dypy*((yp - y) - mean_ypy);
    }
    Yp_nt->releaseBlockOfRows(blockYp);
    Y_nt->releaseBlockOfRows(blockY);

    return (vy - vypy) / vy;
}


void
print_numeric_table(dm::NumericTablePtr X_nt, std::string label)
{
    size_t n_cols = X_nt->getNumberOfColumns();
    size_t n_rows = X_nt->getNumberOfRows();

    dm::BlockDescriptor<double> blockX;
    X_nt->getBlockOfRows(0, n_rows, dm::readOnly, blockX);

    double *x = blockX.getBlockPtr();
    std::cout << "@ " << label << ":" << std::endl;
    std::cout << std::setprecision(18) << std::scientific;
    for (size_t i_outer=0; i_outer < n_rows; i_outer++) {
    std::cout << "@ ";
    for(size_t i_inner=0; i_inner < n_cols; i_inner++) {
        std::cout << x[i_inner + i_outer * n_cols] << ", ";
    }
    std::cout << std::endl;
    }

    X_nt->releaseBlockOfRows(blockX);
}


void
bench(
    size_t threadNum,
    const std::string& X_fname,
    const std::string& y_fname,
    int fit_samples,
    int fit_repetitions,
    int predict_samples,
    int predict_repetitions,
    bool verbose,
    bool header,
    size_t n_trees,
    size_t seed,
    size_t n_features_per_node,
    size_t max_depth,
    double min_impurity,
    bool bootstrap)
{
    /* Set the maximum number of threads to be used by the library */
    if (threadNum != 0)
        ds::Environment::getInstance()->setNumberOfThreads(threadNum);

    threadNum = ds::Environment::getInstance()->getNumberOfThreads();

    /* Load data */
    struct npyarr *arrX = load_npy(X_fname.c_str());
    struct npyarr *arrY = load_npy(y_fname.c_str());
    if (!arrX || !arrY) {
        std::cerr << "Failed to load input arrays" << std::endl;
        std::exit(1);
        return;
    }
    if (arrX->shape_len != 2) {
        std::cerr << "Expected 2 dimensions for X, found "
            << arrX->shape_len << std::endl;
        std::exit(1);
        return;
    }
    if (arrY->shape_len != 1) {
        std::cerr << "Expected 1 dimension for y, found "
            << arrY->shape_len << std::endl;
        std::exit(1);
        return;
    }

    /* Create numeric tables */
    dm::NumericTablePtr X_nt = dm::HomogenNumericTable<double>::create(
            (double *) arrX->data, arrX->shape[1], arrX->shape[0]);
    dm::NumericTablePtr Y_nt = dm::HomogenNumericTable<double>::create(
            (double *) arrY->data, 1, arrY->shape[0]);

    std::vector<std::chrono::duration<double> > fit_times;
    dfr::training::ResultPtr training_result;
    for(int i = 0; i < fit_samples; i++) {
        auto start = std::chrono::system_clock::now();
    for(int j=0; j < fit_repetitions; j++) {
        training_result = df_regression_fit(
        n_trees, seed, n_features_per_node, max_depth, min_impurity, bootstrap,
        X_nt, Y_nt, verbose && (!i) && (!j)
        );
    }
        auto finish = std::chrono::system_clock::now();
        fit_times.push_back(finish - start);
    }

    std::vector<std::chrono::duration<double> > predict_times;
    dm::NumericTablePtr Yp_nt;
    for(int i=0; i < predict_samples; i++) {
        auto start = std::chrono::system_clock::now();
        for(int j=0; j < predict_repetitions; j++) {
        Yp_nt = df_regression_predict(
        training_result,
        X_nt, verbose && (!i) && (!j));
    }
        auto finish = std::chrono::system_clock::now();
        predict_times.push_back(finish - start);

    }

    size_t n_rows = Y_nt->getNumberOfRows();
    double accuracy = explained_variance_score(Y_nt, Yp_nt, n_rows);

    if (header) {
    std::cout <<
        ""  << "prefix_ID"     <<
        "," << "function"      <<
        "," << "threads"       <<
        "," << "rows"          <<
        "," << "features"      <<
        "," << "fit"           <<
        "," << "predict"       <<
        "," << "accuracy"      << std::endl;
    }

    std::cout << "Native-C,df_regr," << threadNum;
    std::cout << "," << X_nt->getNumberOfRows()
              << "," << X_nt->getNumberOfColumns();
    std::cout << "," << std::min_element(fit_times.begin(),
            fit_times.end())->count() / fit_repetitions;
    std::cout << "," << std::min_element(predict_times.begin(),
            predict_times.end())->count() / predict_repetitions;
    std::cout << "," << (100 * accuracy);
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    CLI::App app("Native benchmark code for Intel(R) DAAL "
                 "random forest regressor");

    std::string xfn = "./data/mX.csv";
    app.add_option("--fileX", xfn, "Feature file name")
        ->required()->check(CLI::ExistingFile);

    std::string yfn = "./data/mY.csv";
    app.add_option("--fileY", yfn, "Labels file name")
        ->required()->check(CLI::ExistingFile);

    int fit_samples = 3, fit_repetitions = 1,
        predict_samples = 5, predict_repetitions = 50;
    app.add_option("--fit-samples", fit_samples,
                   "Number of samples to collect for time of execution of "
                   "repeated fit calls", true);
    app.add_option("--fit-repetitions", fit_repetitions,
                   "Number of repeated fit calls to time", true);
    app.add_option("--predict-samples", predict_samples,
                   "Number of samples to collect for time of execution of "
                   "repeated predict calls", true);
    app.add_option("--predict-repetitions", predict_repetitions,
                   "Number of repeated predict calls to time", true);

    int num_threads = 0;
    app.add_option("-n,--num-threads", num_threads,
                   "Number of threads for DAAL to use", true);

    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Whether to be verbose or terse");

    bool header = false;
    app.add_flag("--header", header, "Whether to output header");

    bool no_bootstrap = false;
    app.add_flag("--no-bootstrap", no_bootstrap,
                 "Do not use bootstrap samples to build trees");

    size_t num_trees = 100;
    app.add_option("--num-trees", num_trees,
                   "Number of trees in decision forest", true);

    size_t max_depth = 0;
    app.add_option("--max-depth", max_depth,
                   "Maximal depth of trees in the forest. "
                   "Zero means depth is not limited.", true);

    size_t num_features_per_node = 0;
    app.add_option("--features-per-node", num_features_per_node,
                   "Number of features per node", true);

    size_t seed = 12345;
    app.add_option("--seed", seed, "Seed for the MT2203 RNG", true);


    CLI11_PARSE(app, argc, argv);

    assert(num_threads >= 0);
    assert(fit_samples > 0);
    assert(fit_repetitions > 0);
    assert(predict_samples > 0);
    assert(predict_repetitions > 0);

    if (verbose) {
    std::clog <<
        "@ {FIT_SAMPLES: "        << fit_samples <<
        ", FIT_REPETITIONS: "     << fit_repetitions <<
        ", PREDICT_SAMPLES: "     << predict_samples <<
        ", PREDICT_REPETITIONS: " << predict_repetitions <<
        "}" << std::endl;
    }

    bench(num_threads, xfn, yfn, fit_samples, fit_repetitions, predict_samples,
          predict_repetitions, verbose, header, num_trees, seed,
          num_features_per_node, max_depth, 0., !no_bootstrap);

    return 0;
}
