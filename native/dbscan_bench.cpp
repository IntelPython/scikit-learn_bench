/*
 * Copyright (C) 2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>  

#define DAAL_DATA_TYPE double
#include "common.hpp"
#include "CLI11.hpp"
#include "daal.h"
#include "npyfile.h"


da::dbscan::ResultPtr
dbscan_test(dm::NumericTablePtr X_nt, double eps, int min_samples) {

    da::dbscan::Batch<double> algorithm(eps, min_samples);
    algorithm.input.set(da::dbscan::data, X_nt);
    algorithm.compute();

    return algorithm.getResult();

}


int main(int argc, char *argv[]) {

    CLI::App app("Native benchmark for Intel(R) DAAL DBSCAN clustering");

    std::string batch, arch, prefix;
    int num_threads;
    bool header, verbose;
    add_common_args(app, batch, arch, prefix, num_threads, header, verbose);

    struct timing_options timing_opts = {100, 100, 10., 10};
    add_timing_args(app, "", timing_opts);

    std::string filex, filei;
    app.add_option("-x,--filex,--fileX", filex,
                   "Feature file name")
        ->required()->check(CLI::ExistingFile);

    double eps = 10.;
    app.add_option("-e,--eps,--epsilon", eps,
                   "Radius of neighborhood of a point");

    int min_samples = 5;
    app.add_option("-m,--min-samples", min_samples,
                   "The minimum number of samples required in a neighborhood "
                   "to consider a point a core point");

    CLI11_PARSE(app, argc, argv);

    // Set DAAL thread count
    int daal_threads = set_threads(num_threads);

    // Load data
    struct npyarr *arrX = load_npy(filex.c_str());
    if (!arrX) {
        std::cerr << "Failed to load input array" << std::endl;
        return EXIT_FAILURE;
    }
    if (arrX->shape_len != 2) {
        std::cerr << "Expected 2 dimensions for X, found "
            << arrX->shape_len << std::endl;
        return EXIT_FAILURE;
    }

    // Infer data size from loaded arrays
    std::ostringstream stringSizeStream;
    stringSizeStream << arrX->shape[0] << 'x' << arrX->shape[1];
    std::string stringSize = stringSizeStream.str();

    // Create numeric tables from input data
    dm::NumericTablePtr X_nt = make_table((double *) arrX->data,
                                          arrX->shape[0],
                                          arrX->shape[1]);

    // Prepare meta-info
    std::string header_string = "Batch,Arch,Prefix,Threads,Size,Function,"
                                "Clusters,Time";
    std::ostringstream meta_info_stream;
    meta_info_stream
        << batch << ','
        << arch << ','
        << prefix << ','
        << daal_threads << ','
        << stringSize << ',';
    std::string meta_info = meta_info_stream.str();

    // Actually time benches
    double time;
    da::dbscan::ResultPtr dbscan_result;
    std::tie(time, dbscan_result) = time_min<da::dbscan::ResultPtr> ([=] {
                return dbscan_test(X_nt, eps, min_samples);
            }, timing_opts, verbose);

    // Get number of clusters found
    dm::NumericTablePtr n_clusters_nt
        = dbscan_result->get(da::dbscan::nClusters);
    dm::BlockDescriptor<int> n_clusters_block;
    n_clusters_nt->getBlockOfRows(0, 1, dm::readOnly, n_clusters_block);
    int n_clusters = n_clusters_block.getBlockPtr()[0];
    n_clusters_nt->releaseBlockOfRows(n_clusters_block);

    std::cout << meta_info << "DBSCAN," << n_clusters << time << std::endl;

    return 0;
}

