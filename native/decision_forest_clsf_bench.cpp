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
namespace dfc=daal::algorithms::decision_forest::classification;

using namespace daal;
using namespace da;


dfc::training::ResultPtr
df_classification_fit(
    int nClasses,
    size_t nTrees,
    size_t seed,
    size_t n_features_per_node,
    size_t max_depth,
    double min_impurity,
    bool bootsrap,
    dm::NumericTablePtr Xt,
    dm::NumericTablePtr Yt,
    bool verbose)
{
    size_t n_samples = Yt->getNumberOfRows();
    size_t n_features = Xt->getNumberOfColumns();

    size_t fpn = (n_features_per_node > 0 && n_features_per_node <= n_features) ? \
	n_features_per_node : n_features;

    dfc::training::Batch<double> df_clsf_alg(nClasses);
    df_clsf_alg.parameter.nTrees = nTrees;
    df_clsf_alg.parameter.varImportance = df::training::MDI;
    df_clsf_alg.parameter.observationsPerTreeFraction = 1.0;
    df_clsf_alg.parameter.maxTreeDepth = max_depth;
    df_clsf_alg.parameter.featuresPerNode = fpn;
    df_clsf_alg.parameter.minObservationsInLeafNode = 1;
    df_clsf_alg.parameter.impurityThreshold = min_impurity;
    df_clsf_alg.parameter.bootstrap = bootsrap;
    df_clsf_alg.parameter.engine = da::engines::mt2203::Batch<double>::create(seed);

    if (verbose) {
	std::cout << "@ {'nTrees': " << nTrees <<
                      ", 'variable_importance': " << "MDI"<<
                      ", 'features_per_node': " <<  fpn <<
	              ", 'max_depth': " << max_depth << 
                      ", 'min_impurity': " << min_impurity <<
                      ", 'seed': " << seed <<
	              ", 'bootstrap': " << (bootsrap ? "True" : "False") <<
                      "}" <<  std::endl;
    }

    df_clsf_alg.input.set(da::classifier::training::data, Xt);
    df_clsf_alg.input.set(da::classifier::training::labels, Yt);

    df_clsf_alg.compute();

    dfc::training::ResultPtr result_ptr = df_clsf_alg.getResult();

    return result_ptr;
}

dm::NumericTablePtr
df_classification_predict(
    int nClasses,
    dfc::training::ResultPtr training_result_ptr,
    dm::NumericTablePtr Xt,
    bool verbose
    )
{
    // We explicitly specify float here to match sklearn.
    dfc::prediction::Batch<float> pred_alg(nClasses);
    pred_alg.input.set(da::classifier::prediction::data, Xt);
    pred_alg.input.set(da::classifier::prediction::model,
		       training_result_ptr->get(da::classifier::training::model));

    pred_alg.compute();

    da::classifier::prediction::ResultPtr pred_res = pred_alg.getResult();
    dm::NumericTablePtr Y_pred_t = pred_res->get(da::classifier::prediction::prediction);

    return Y_pred_t;
}


int main(int argc, char** argv) {
    CLI::App app("Native benchmark code for Intel(R) DAAL random forest classifier");

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
    app.add_option("--seed", seed, "Number of features per node", true);

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
    dm::NumericTablePtr Y_nt = dm::HomogenNumericTable<int64_t>::create(
            (int64_t *) arrY->data, 1, arrY->shape[0]);

    size_t n_rows = Y_nt->getNumberOfRows();
    size_t n_features = X_nt->getNumberOfColumns();
    std::ostringstream string_size_stream;
    string_size_stream << n_rows << 'x' << n_features;
    std::string stringSize = string_size_stream.str();

    // Set DAAL threads
    int daal_threads = set_threads(num_threads);

    int n_classes = count_classes(Y_nt);

    // Prepare header and metadata info
    std::string header_string = "batch,arch,prefix,threads,size,classes,"
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
        << n_classes << ','
        << n_trees << ','
        << n_features_per_node << ','
        << max_depth << ','
        << min_impurity << ','
        << bootstrap << ',';
    std::string meta_info = meta_info_stream.str();

    // Actually time benchmarks

    double time;
    bool verbose_fit = verbose;
    dfc::training::ResultPtr training_result;
    std::tie(time, training_result) = time_min<dfc::training::ResultPtr> ([&] {
            auto r = df_classification_fit(n_classes, n_trees, seed,
                                           n_features_per_node, max_depth,
                                           min_impurity, bootstrap, X_nt, Y_nt,
                                           verbose_fit);
            verbose_fit = false;
            return r;
        }, fit_opts, verbose);

    if (header) {
        std::cout << header_string << std::endl;
    }
    std::cout << meta_info << "df_clsf.fit,," << time << std::endl;

    dm::NumericTablePtr Yp_nt;
    std::tie(time, Yp_nt) = time_min<dm::NumericTablePtr> ([&] {
            return df_classification_predict(n_classes, training_result,
                                             X_nt, verbose);
        }, predict_opts, verbose);

    double accuracy = accuracy_score(Y_nt, Yp_nt) * 100.;
    std::cout << meta_info << "df_clsf.predict," << accuracy << ','
        << time << std::endl;

    return EXIT_SUCCESS;
}
