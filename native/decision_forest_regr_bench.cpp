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
#include "common.hpp"

namespace dm=daal::data_management;
namespace ds=daal::services;
namespace da=daal::algorithms;
namespace df=daal::algorithms::decision_forest;
namespace dfr=daal::algorithms::decision_forest::regression;

using namespace daal;
using namespace da;

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


int main(int argc, char** argv) {
    CLI::App app("Native benchmark code for Intel(R) DAAL "
                 "random forest regressor");

    std::string batch, arch, prefix;
    int num_threads;
    bool header, verbose;
    add_common_args(app, batch, arch, prefix, num_threads, header, verbose);

    std::string xfn = "./data/mX.csv";
    app.add_option("-x,--fileX", xfn, "Feature file name")
        ->required()
        ->check(CLI::ExistingFile);

    std::string yfn = "./data/mY.csv";
    app.add_option("-y,--fileY", yfn, "Labels file name")
        ->required()
        ->check(CLI::ExistingFile);

    struct timing_options fit_opts = {100, 100, 10., 10};
    add_timing_args(app, "fit", fit_opts);

    struct timing_options predict_opts = {10, 100, 10., 10};
    add_timing_args(app, "predict", predict_opts);

    bool no_bootstrap = false;
    app.add_flag("--no-bootstrap", no_bootstrap,
                 "Do not use bootstrap samples to build trees");

    size_t n_trees = 100;
    app.add_option("--num-trees", n_trees,
                   "Number of trees in decision forest", true);

    size_t n_features_per_node = 0;
    app.add_option("--features-per-node", n_features_per_node,
                   "Number of features per node", true);

    size_t max_depth = 0;
    app.add_option("--max-depth", max_depth,
                   "Maximal depth of trees in the forest. "
                   "Zero means depth is not limited.", true);

    size_t seed = 12345;
    app.add_option("--seed", seed, "Seed for the MT2203 RNG", true);

    double min_impurity = 0.;

    CLI11_PARSE(app, argc, argv);

    bool bootstrap = !no_bootstrap;

    /* Load data */
    struct npyarr *arrX = load_npy(xfn.c_str());
    struct npyarr *arrY = load_npy(yfn.c_str());
    if (!arrX || !arrY) {
        std::cerr << "Failed to load input arrays" << std::endl;
        return EXIT_FAILURE;
    }
    if (arrX->shape_len != 2) {
        std::cerr << "Expected 2 dimensions for X, found "
            << arrX->shape_len << std::endl;
        return EXIT_FAILURE;
    }
    if (arrY->shape_len != 1) {
        std::cerr << "Expected 1 dimension for y, found "
            << arrY->shape_len << std::endl;
        return EXIT_FAILURE;
    }

    /* Create numeric tables */
    dm::NumericTablePtr X_nt = dm::HomogenNumericTable<double>::create(
            (double *) arrX->data, arrX->shape[1], arrX->shape[0]);
    dm::NumericTablePtr Y_nt = dm::HomogenNumericTable<double>::create(
            (double *) arrY->data, 1, arrY->shape[0]);

    size_t n_rows = Y_nt->getNumberOfRows();
    size_t n_features = X_nt->getNumberOfColumns();
    std::ostringstream string_size_stream;
    string_size_stream << n_rows << 'x' << n_features;
    std::string stringSize = string_size_stream.str();

    // Set DAAL threads
    int daal_threads = set_threads(num_threads);

    // Prepare header and metadata info
    std::string header_string = "batch,arch,prefix,threads,size,"
                                "n_trees,n_features_per_node,max_depth,"
                                "min_impurity,bootstrap,function,accuracy,"
                                "time";
    std::ostringstream meta_info_stream;
    meta_info_stream
        << batch << ','
        << arch << ','
        << prefix << ','
        << daal_threads << ','
        << stringSize << ','
        << n_trees << ','
        << n_features_per_node << ','
        << max_depth << ','
        << min_impurity << ','
        << bootstrap << ',';
    std::string meta_info = meta_info_stream.str();

    // Actually time benchmarks
    
    double time;
    bool verbose_fit = verbose;
    dfr::training::ResultPtr training_result;
    std::tie(time, training_result) = time_min<dfr::training::ResultPtr> ([&] {
            auto r = df_regression_fit(n_trees, seed, n_features_per_node,
                                       max_depth, min_impurity, bootstrap,
                                       X_nt, Y_nt, verbose_fit);
            verbose_fit = false;
            return r;
        }, fit_opts, verbose);

    if (header) {
        std::cout << header_string << std::endl;
    }
    std::cout << meta_info << "df_regr.fit,," << time << std::endl;

    dm::NumericTablePtr Yp_nt;
    std::tie(time, Yp_nt) = time_min<dm::NumericTablePtr> ([&] {
            return df_regression_predict(training_result, X_nt, verbose);
        }, predict_opts, verbose);

    double accuracy = explained_variance_score(Y_nt, Yp_nt, n_rows);
    std::cout << meta_info << "df_regr.predict," << accuracy << ','
        << time << std::endl;

    return EXIT_SUCCESS;
}
