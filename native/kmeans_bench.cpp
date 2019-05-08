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
#include "daal.h"
#include "npyfile.h"

namespace dm=daal::data_management;
namespace ds=daal::services;
namespace da=daal::algorithms;


#define REPS 10


template<typename T>
ds::SharedPtr<dm::HomogenNumericTable<T> > makeTable(T* data, size_t rows, size_t cols)
{
    return ds::SharedPtr<dm::HomogenNumericTable<T> >(new dm::HomogenNumericTable<T>(data, cols, rows));
}


std::vector<std::pair<int, int> > init_problem_sizes(int size0, int size1) 
{
    std::vector<std::pair<int, int> > ret_val;
    ret_val.push_back({size0, size1});
    return ret_val; 
}


const int n_clusters = 10;
da::kmeans::init::Batch<double, da::kmeans::init::plusPlusDense> init(n_clusters);

const size_t max_iters = 100;

void kmeans_fit_test(dm::NumericTablePtr &X_nt, dm::NumericTablePtr &X_init_nt, double tol)
{
    dm::NumericTablePtr &seeding_centroids = X_init_nt;

    int n_clusters = seeding_centroids->getNumberOfRows();
    da::kmeans::Batch<double> algorithm(n_clusters, max_iters);
    algorithm.input.set(da::kmeans::data, X_nt);
    algorithm.input.set(da::kmeans::inputCentroids, seeding_centroids);
    algorithm.parameter.maxIterations = max_iters;
    algorithm.parameter.accuracyThreshold = tol;
    algorithm.compute();

    algorithm.getResult()->get(da::kmeans::assignments);
    algorithm.getResult()->get(da::kmeans::centroids  );
    algorithm.getResult()->get(da::kmeans::goalFunction);

    dm::NumericTablePtr nIterationsNumericTable = algorithm.getResult()->get(da::kmeans::nIterations);
    dm::BlockDescriptor<int> blockNI;
    nIterationsNumericTable->getBlockOfRows(0, 1, dm::readOnly, blockNI);
    int *niPtr = blockNI.getBlockPtr();
    int actual_iters = niPtr[0];
    nIterationsNumericTable->releaseBlockOfRows(blockNI);
    
    if(actual_iters != max_iters) {
	std::cout << std::endl << "WARNING: Number of actual iterations " << actual_iters << " is less than max_iters of " << max_iters << " " << std::endl;
	std::cout << "Tolerance: " << tol << std::endl;
    }
}


void kmeans_predict_test(dm::NumericTablePtr& X_nt, dm::NumericTablePtr &X_init_nt)
{
    dm::NumericTablePtr &seeding_centroids = X_init_nt;

    int n_clusters = seeding_centroids->getNumberOfRows();
	da::kmeans::Batch<double> algorithm(n_clusters, max_iters);
	algorithm.input.set(da::kmeans::data, X_nt);
	algorithm.input.set(da::kmeans::inputCentroids, seeding_centroids);
    algorithm.parameter.assignFlag = 1;
	algorithm.parameter.maxIterations = max_iters;
	algorithm.parameter.accuracyThreshold = 0.0;
	algorithm.compute();

	algorithm.getResult()->get(da::kmeans::assignments);
}


std::pair<double, double> bench(size_t threadNum, const std::string& fname,
                                double data_multiplier)
{
    /* Set the maximum number of threads to be used by the library */
    if (threadNum != 0)
        daal::services::Environment::getInstance()->setNumberOfThreads(threadNum);

    /* Load data */
    struct npyarr *arrX = load_npy((fname + ".npy").c_str());
    struct npyarr *arrX_init = load_npy((fname + ".init.npy").c_str());
    struct npyarr *arrX_tol = load_npy((fname + ".tol.npy").c_str());
    if (!arrX || !arrX_init || !arrX_tol) {
        std::cerr << "Failed to load input arrays" << std::endl;
        std::exit(1);
    }
    if (arrX->shape_len != 2) {
        std::cerr << "Expected 2 dimensions for X, found "
            << arrX->shape_len << std::endl;
        std::exit(1);
    }
    if (arrX_init->shape_len != 2) {
        std::cerr << "Expected 2 dimension for X_init, found "
            << arrX_init->shape_len << std::endl;
        std::exit(1);
    }
    if (arrX_tol->shape_len != 0) {
        std::cerr << "Expected 0 dimensions for X_tol, found "
            << arrX_tol->shape_len << std::endl;
        std::exit(1);
    }
    double tol = ((double *) arrX_tol->data)[0];

    /* Create numeric tables */
    dm::NumericTablePtr X_nt = dm::HomogenNumericTable<double>::create(
            (double *) arrX->data, arrX->shape[1], arrX->shape[0]);
    dm::NumericTablePtr X_init_nt = dm::HomogenNumericTable<double>::create(
            (double *) arrX_init->data, arrX_init->shape[1], arrX_init->shape[0]);

    /* Apply data multiplier for KMeans prediction */
	double* X_mult = (double*) daal::services::daal_malloc(
            X_nt->getNumberOfColumns() * X_nt->getNumberOfRows() *
            data_multiplier * sizeof(double));
    
	for (int i = 0; i < data_multiplier; i++) {
		for (int j = 0; j < X_nt->getNumberOfColumns() * X_nt->getNumberOfRows(); j++) {
			X_mult[i * X_nt->getNumberOfColumns() * X_nt->getNumberOfRows() + j] = ((double *) arrX->data)[j];
		}
	}

    dm::NumericTablePtr X_mult_nt = dm::HomogenNumericTable<double>::create(
            (double *) X_mult, arrX->shape[1], arrX->shape[0]);

    /* Actually time benches */

    std::vector<std::chrono::duration<double>> times_fit, times_predict;
    for(int i = 0; i < REPS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kmeans_fit_test(X_nt, X_init_nt, tol);
        auto finish = std::chrono::high_resolution_clock::now();
        times_fit.push_back(finish - start);

        start = std::chrono::high_resolution_clock::now();
        kmeans_predict_test(X_mult_nt, X_init_nt);
        finish = std::chrono::high_resolution_clock::now();
        times_predict.push_back(finish - start);
    }

    return std::make_pair(std::min_element(times_fit.begin(), times_fit.end())->count(),
                          std::min_element(times_predict.begin(), times_predict.end())->count());
}


int main(int args, char **argsv)
{
    if (args != 10) {
        fprintf(stderr, "usage: %s BATCH ARCH PREFIX FUNC CORES DTYPE SIZE1xSIZE2 INPUT MULTIPLIER\n", argsv[0]);
        exit(1);
    }

    std::string fname = argsv[8];
    if (fname.back() == '/') {
        // We got a dir in which kmeans data should be found
        fname += "kmeans_";
        fname += argsv[7];
    }

    double data_multiplier = std::stod(argsv[9]);
    const size_t nThreads = atoi(argsv[5]);
    std::pair<double, double> results = bench(nThreads, fname, data_multiplier);

    for(int f=1; f < args; f++) {
        if (f==5)
            std::cout << ((atoi(argsv[f]) == 1) ? "Serial," : "Threaded,"); 
        else if (f == 4)
            std::cout << "KMeans.fit,";
        else
            std::cout << argsv[f] << ",";
    }
    std::cout << results.first << std::endl;

    for(int f=1; f < args; f++) {
        if (f==5)
            std::cout << ((atoi(argsv[f]) == 1) ? "Serial," : "Threaded,"); 
        else if (f == 4)
            std::cout << "KMeans.predict,";
        else
            std::cout << argsv[f] << ",";
    }
    std::cout << results.second << std::endl;
    return 0;
}

