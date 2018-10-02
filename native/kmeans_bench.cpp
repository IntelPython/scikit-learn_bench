/*
 * Copyright (C) 2017-2018 Intel Corporation
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

void kmean_test(dm::FileDataSource<dm::CSVFeatureManager>& dataSource, dm::FileDataSource<dm::CSVFeatureManager>& initSource, double tol)
{
    dm::NumericTablePtr seeding_centroids = initSource.getNumericTable();

    da::kmeans::Batch<double> algorithm(n_clusters, max_iters);
    algorithm.input.set(da::kmeans::data, dataSource.getNumericTable());
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


void bench(size_t threadNum, const std::string& fname)
{
    /* Set the maximum number of threads to be used by the library */
    if (threadNum != 0)
        daal::services::Environment::getInstance()->setNumberOfThreads(threadNum);
    dm::FileDataSource<dm::CSVFeatureManager> dataSource(fname, dm::DataSource::doAllocateNumericTable,
                                                         dm::DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();
    std::string fname_tol = fname + ".tol";
    std::ifstream ifs(fname_tol);
    double tol;
    if (ifs.good()) ifs >> tol;
    ifs.close();

    std::string fname_init = fname + ".init";
    dm::FileDataSource<dm::CSVFeatureManager> initSource(fname_init, dm::DataSource::doAllocateNumericTable,
                                                         dm::DataSource::doDictionaryFromContext);
    initSource.loadDataBlock();

    std::vector<std::chrono::duration<double> > times;
    for(int i = 0; i < REPS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kmean_test(dataSource, initSource, tol);
        auto finish = std::chrono::high_resolution_clock::now();
        times.push_back(finish - start);
    }
    std::cout << " " << std::min_element(times.begin(), times.end())->count() << std::endl;
}


int main(int args, char **argsv)
{
    if (args != 9) {
        fprintf(stderr, "usage: %s BATCH ARCH PREFIX FUNC CORES DTYPE SIZE1xSIZE2 INPUT\n", argsv[0]);
        exit(1);
    }

    for(int f=1; f < args; f++) {
        if (f==5)
            std::cout << ((atoi(argsv[f]) == 1) ? "Serial," : "Threaded,"); 
        else
            std::cout << argsv[f] << ",";
    }

    std::string fname = argsv[8];
    if (fname.back() == '/') {
        // We got a dir in which kmeans data should be found
        fname += "kmeans_";
        fname += argsv[7];
        fname += ".csv";
    }

    const size_t nThreads = atoi(argsv[5]);
    bench(nThreads, fname);
    return 0;
}

