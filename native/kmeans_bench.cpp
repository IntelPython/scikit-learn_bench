/*
 * Copyright (C) 2017-2019 Intel Corporation
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


const size_t max_iters = 100;

da::kmeans::ResultPtr
kmeans_fit_test(dm::NumericTablePtr X_nt, dm::NumericTablePtr X_init_nt, double tol) {

    dm::NumericTablePtr seeding_centroids = X_init_nt;

    int n_clusters = seeding_centroids->getNumberOfRows();
    da::kmeans::Batch<double> algorithm(n_clusters, max_iters);
    algorithm.input.set(da::kmeans::data, X_nt);
    algorithm.input.set(da::kmeans::inputCentroids, seeding_centroids);
    algorithm.parameter.maxIterations = max_iters;
    algorithm.parameter.accuracyThreshold = tol;
    algorithm.compute();

    da::kmeans::ResultPtr kmeans_result = algorithm.getResult();

    kmeans_result->get(da::kmeans::assignments);
    kmeans_result->get(da::kmeans::centroids  );
    kmeans_result->get(da::kmeans::goalFunction);

    dm::NumericTablePtr nIterationsNumericTable = algorithm.getResult()->get(da::kmeans::nIterations);
    dm::BlockDescriptor<int> blockNI;
    nIterationsNumericTable->getBlockOfRows(0, 1, dm::readOnly, blockNI);
    int *niPtr = blockNI.getBlockPtr();
    int actual_iters = niPtr[0];
    nIterationsNumericTable->releaseBlockOfRows(blockNI);
    
    if(actual_iters != max_iters) {
	std::cout << std::endl << "@ WARNING: Number of actual iterations " << actual_iters << " is less than max_iters of " << max_iters << " " << std::endl;
	std::cout << "@ Tolerance: " << tol << std::endl;
    }

    return kmeans_result;

}


dm::NumericTablePtr
kmeans_predict_test(dm::NumericTablePtr X_nt, dm::NumericTablePtr X_init_nt) {

    dm::NumericTablePtr seeding_centroids = X_init_nt;

    int n_clusters = seeding_centroids->getNumberOfRows();
	da::kmeans::Batch<double> algorithm(n_clusters, max_iters);
	algorithm.input.set(da::kmeans::data, X_nt);
	algorithm.input.set(da::kmeans::inputCentroids, seeding_centroids);
    algorithm.parameter.assignFlag = 1;
	algorithm.parameter.maxIterations = max_iters;
	algorithm.parameter.accuracyThreshold = 0.0;
	algorithm.compute();

	return algorithm.getResult()->get(da::kmeans::assignments);

}


int main(int argc, char *argv[]) {

    CLI::App app("Native benchmark for Intel(R) DAAL KMeans clustering");

    std::string batch, arch, prefix;
    int num_threads;
    bool header, verbose;
    add_common_args(app, batch, arch, prefix, num_threads, header, verbose);

    int fit_samples = 1;
    app.add_option("--fit-samples", fit_samples,
                   "Number of samples to report (fit)");

    int fit_reps = 10;
    app.add_option("--fit-reps", fit_reps,
                   "Number of repetitions in each sample (fit)");

    int predict_samples = 1;
    app.add_option("--predict-samples", predict_samples,
                   "Number of samples to report (predict)");

    int predict_reps = 10;
    app.add_option("--predict-reps", predict_samples,
                   "Number of repetitions in each sample (predict)");

    std::string filex, filei, filet;
    app.add_option("-x,--filex,--fileX", filex,
                   "Feature file name")
        ->required()->check(CLI::ExistingFile);
    app.add_option("-i,--filei,--fileI", filei,
                   "Initial cluster centers file name")
        ->required()->check(CLI::ExistingFile);
    app.add_option("-t,--filet,--fileT", filet,
                   "Absolute threshold file name")
        ->required()->check(CLI::ExistingFile);

    int data_multiplier = 100;
    app.add_option("-m,--data-multiplier", data_multiplier, "Data multiplier");

    CLI11_PARSE(app, argc, argv);

    // Set DAAL thread count
    int daal_threads = set_threads(num_threads);

    // Load data
    struct npyarr *arrX = load_npy(filex.c_str());
    struct npyarr *arrX_init = load_npy(filei.c_str());
    struct npyarr *arrX_tol = load_npy(filet.c_str());
    if (!arrX || !arrX_init || !arrX_tol) {
        std::cerr << "Failed to load input arrays" << std::endl;
        return EXIT_FAILURE;
    }
    if (arrX->shape_len != 2) {
        std::cerr << "Expected 2 dimensions for X, found "
            << arrX->shape_len << std::endl;
        return EXIT_FAILURE;
    }
    if (arrX_init->shape_len != 2) {
        std::cerr << "Expected 2 dimensions for X_init, found "
            << arrX_init->shape_len << std::endl;
        return EXIT_FAILURE;
    }
    if (arrX_tol->shape_len != 0) {
        std::cerr << "Expected 0 dimensions for X_tol, found "
            << arrX_tol->shape_len << std::endl;
        return EXIT_FAILURE;
    }
    double tol = ((double *) arrX_tol->data)[0];

    // Infer data size from loaded arrays
    std::ostringstream stringSizeStream;
    stringSizeStream << arrX->shape[0] << 'x' << arrX->shape[1];
    std::string stringSize = stringSizeStream.str();

    // Create numeric tables from input data
    dm::NumericTablePtr X_nt = make_table(
            (double *) arrX->data, arrX->shape[0], arrX->shape[1]);
    dm::NumericTablePtr X_init_nt = make_table(
            (double *) arrX_init->data, arrX_init->shape[0], arrX_init->shape[1]);

    // Apply data multiplier for KMeans prediction
	double* X_mult = (double*) daal::services::daal_malloc(
            X_nt->getNumberOfColumns() * X_nt->getNumberOfRows() *
            data_multiplier * sizeof(double));
    
	for (int i = 0; i < data_multiplier; i++) {
		for (int j = 0; j < X_nt->getNumberOfColumns() * X_nt->getNumberOfRows(); j++) {
			X_mult[i * X_nt->getNumberOfColumns() * X_nt->getNumberOfRows() + j] = ((double *) arrX->data)[j];
		}
	}

    dm::NumericTablePtr X_mult_nt = make_table(
            (double *) X_mult, arrX->shape[0], arrX->shape[1]);

    // Prepare meta-info
    std::string header_string = "Batch,Arch,Prefix,Threads,Size,Function,Time";
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
    da::kmeans::ResultPtr kmeans_result;
    for (int i = 0; i < fit_samples; i++) {
        std::tie(time, kmeans_result) = time_min<da::kmeans::ResultPtr> ([=] {
                    return kmeans_fit_test(X_nt, X_init_nt, tol);
                }, fit_reps);
        std::cout << meta_info << "KMeans.fit," << time << std::endl;
    }

    dm::NumericTablePtr predict_result;
    for (int i = 0; i < predict_samples; i++) {
        std::tie(time, predict_result) = time_min<dm::NumericTablePtr> ([=] {
                    return kmeans_predict_test(X_mult_nt, X_init_nt);
                }, predict_reps);
        std::cout << meta_info << "KMeans.predict," << time << std::endl;
    }

    return 0;
}

