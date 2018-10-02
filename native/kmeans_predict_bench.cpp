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

namespace dm = daal::data_management;
namespace ds = daal::services;
namespace da = daal::algorithms;

#define REPS 10
const int n_clusters = 10;
const size_t max_iters = 0;

void kmean_test(dm::NumericTablePtr& multiData, dm::FileDataSource<dm::CSVFeatureManager>& initSource)
{
	dm::NumericTablePtr seeding_centroids = initSource.getNumericTable();

	da::kmeans::Batch<double> algorithm(n_clusters, max_iters);
	algorithm.input.set(da::kmeans::data, multiData);
	algorithm.input.set(da::kmeans::inputCentroids, seeding_centroids);
    algorithm.parameter.assignFlag = 1;
	algorithm.parameter.maxIterations = max_iters;
	algorithm.parameter.accuracyThreshold = 0.0;
	algorithm.compute();

	algorithm.getResult()->get(da::kmeans::assignments);
}


void bench(size_t threadNum, const std::string& fname, size_t data_multiplier)
{
	/* Set the maximum number of threads to be used by the library */
	if (threadNum != 0)
		daal::services::Environment::getInstance()->setNumberOfThreads(threadNum);
	dm::FileDataSource<dm::CSVFeatureManager> dataSource(fname, dm::DataSource::doAllocateNumericTable,
		dm::DataSource::doDictionaryFromContext);
	dataSource.loadDataBlock();
	
	dm::NumericTablePtr nt = dataSource.getNumericTable();
	double* data = (double*)daal::services::daal_malloc(nt->getNumberOfColumns() * nt->getNumberOfRows() * data_multiplier * sizeof(double));
	dm::BlockDescriptor<double> bd;
	nt->getBlockOfRows(0, nt->getNumberOfRows(), dm::readOnly, bd);
	double* ptr = bd.getBlockPtr();
	for (int i = 0; i < data_multiplier; i++) {
		for (int j = 0; j < nt->getNumberOfColumns() * nt->getNumberOfRows(); j++) {
			data[i * nt->getNumberOfColumns() * nt->getNumberOfRows() + j] = ptr[j];
		}
	}
	nt->releaseBlockOfRows(bd);
	dm::NumericTablePtr multiData(new dm::HomogenNumericTable<double>(data, nt->getNumberOfColumns(), nt->getNumberOfRows() * data_multiplier));
	
	std::string fname_init = fname + ".init";
	dm::FileDataSource<dm::CSVFeatureManager> initSource(fname_init, dm::DataSource::doAllocateNumericTable,
		dm::DataSource::doDictionaryFromContext);
	initSource.loadDataBlock();

	std::vector<std::chrono::duration<double> > times;
	for (int i = 0; i < REPS; i++) {
		auto start = std::chrono::high_resolution_clock::now();
		kmean_test(multiData, initSource);
		auto finish = std::chrono::high_resolution_clock::now();
		times.push_back(finish - start);
	}
	std::cout << " " << std::min_element(times.begin(), times.end())->count() << std::endl;
	daal::services::daal_free(data);
}


int main(int args, char **argsv)
{
    if (args != 10) {
        fprintf(stderr, "usage: %s BATCH ARCH PREFIX FUNC CORES DTYPE SIZE1xSIZE2 INPUT MULTIPLIER\n", argsv[0]);
        exit(1);
    }

	for (int f = 1; f < args; f++) {
		if (f == 5)
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
	const size_t data_multiplier = atoi(argsv[9]);
   	bench(nThreads, fname, data_multiplier);
	return 0;
}

