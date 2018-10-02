/*
 * Copyright (C) 2018 Intel Corporation
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
#include "CLI/CLI.hpp"
#include "common_svm.hpp"

namespace dm=daal::data_management;
namespace ds=daal::services;
namespace da=daal::algorithms;

using namespace daal;
using namespace da;


da::svm::training::ResultPtr
two_class_linear_svm_fit(
    svm_linear_kernel_parameters svc_params,
    dm::NumericTablePtr Xt,
    dm::NumericTablePtr Yt,
    bool verbose)
{
    da::svm::training::Batch<double> algorithm;

    /* Parameters for the SVM kernel function */
    ds::SharedPtr<kernel_function::linear::Batch<double> > kernel(new kernel_function::linear::Batch<double>());
    size_t nFeatures = Xt->getNumberOfColumns();
    size_t nSamples = Xt->getNumberOfRows();

    algorithm.parameter.C = svc_params.C;
    algorithm.parameter.kernel = kernel;
    algorithm.parameter.cacheSize = getOptimalCacheSize(nSamples);
    algorithm.parameter.accuracyThreshold = svc_params.tol;
    algorithm.parameter.tau = svc_params.tau;
    algorithm.parameter.maxIterations = svc_params.max_iter;
    algorithm.parameter.doShrinking = true;

    if (verbose) {
	print_svm_linear_kernel_parameters(svc_params);
    }

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(da::classifier::training::data, Xt);
    algorithm.input.set(da::classifier::training::labels, Yt);

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    da::svm::training::ResultPtr trainingResult = algorithm.getResult();

    return trainingResult;
}

dm::NumericTablePtr two_class_linear_svm_predict(
    svm_linear_kernel_parameters svc_params,
    da::svm::training::ResultPtr trainingResult,
    dm::NumericTablePtr X_nt,
    bool verbose)
{
    svm::prediction::Batch<double> prediction_algorithm;

    services::SharedPtr<kernel_function::linear::Batch<double> > kernel(new kernel_function::linear::Batch<double>());
    prediction_algorithm.parameter.kernel = kernel;

    prediction_algorithm.input.set(classifier::prediction::data, X_nt);
    prediction_algorithm.input.set(classifier::prediction::model,
                         trainingResult->get(classifier::training::model));

    prediction_algorithm.compute();

    da::classifier::prediction::ResultPtr res = prediction_algorithm.getResult();

    return res->get(da::classifier::prediction::prediction);
}

size_t
count_same_labels(
    dm::NumericTablePtr Y_nt,
    dm::NumericTablePtr Yp_nt,
    size_t n_rows) 
{
    size_t equal_counter = 0;
    dm::BlockDescriptor<double> blockY;
    dm::BlockDescriptor<double> blockYp;
    Yp_nt->getBlockOfRows(0, n_rows, dm::readOnly, blockYp);
    Y_nt->getBlockOfRows(0, n_rows, dm::readOnly, blockY);
    double *Y_data_ptr = blockY.getBlockPtr();
    double *Yp_data_ptr = blockYp.getBlockPtr();

    for(size_t i = 0; i < n_rows; i++) {
	equal_counter += ((Y_data_ptr[i] >= 0.0) ^ (Yp_data_ptr[i] >= 0.0)) ? 0 : 1;
    }
    Yp_nt->releaseBlockOfRows(blockYp);
    Y_nt->releaseBlockOfRows(blockY);

    return equal_counter;
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
    bool header)
{
    /* Set the maximum number of threads to be used by the library */
    if (threadNum != 0)
        daal::services::Environment::getInstance()->setNumberOfThreads(threadNum);

    size_t daal_thread_num = daal::services::Environment::getInstance()->getNumberOfThreads();

    dm::FileDataSource<dm::CSVFeatureManager> XdataSource(X_fname, dm::DataSource::doAllocateNumericTable,
                                                          dm::DataSource::doDictionaryFromContext);
    XdataSource.loadDataBlock();
    dm::NumericTablePtr X_nt = XdataSource.getNumericTable();

    dm::FileDataSource<dm::CSVFeatureManager> YdataSource(y_fname, dm::DataSource::doAllocateNumericTable,
                                                          dm::DataSource::doDictionaryFromContext);
    YdataSource.loadDataBlock();
    dm::NumericTablePtr Y_nt = YdataSource.getNumericTable();


    svm_linear_kernel_parameters svm_problem;
    svm_problem.C = 0.01;
    svm_problem.tol = 1e-16;
    svm_problem.tau = svm_problem.tol * (1e-3);
    svm_problem.max_iter = 2000;

    std::vector<std::chrono::duration<double> > fit_times;
    svm::training::ResultPtr training_result;
    for(int i = 0; i < fit_samples; i++) {
        auto start = std::chrono::system_clock::now();
	for(int j=0; j < fit_repetitions; j++) {
	    training_result = two_class_linear_svm_fit(svm_problem, X_nt, Y_nt, verbose && (!i) && (!j));
	}
        auto finish = std::chrono::system_clock::now();
        fit_times.push_back(finish - start);
    }

    svm::ModelPtr svm_model = ds::dynamicPointerCast<svm::Model>(training_result->get(classifier::training::model));
    auto sv_idx = svm_model->getSupportIndices();
    size_t sv_len = sv_idx->getNumberOfRows();
  
    std::vector<std::chrono::duration<double> > predict_times;
    dm::NumericTablePtr Yp_nt;
    for(int i=0; i < predict_samples; i++) {
        auto start = std::chrono::system_clock::now();
        for(int j=0; j < predict_repetitions; j++) {
	    Yp_nt = two_class_linear_svm_predict(svm_problem, training_result, X_nt, verbose && (!i) && (!j));
	}
        auto finish = std::chrono::system_clock::now();
        predict_times.push_back(finish - start);

    }

    size_t n_rows = Y_nt->getNumberOfRows();
    size_t equal_counter = count_same_labels(Y_nt, Yp_nt, n_rows);
    double accuracy = ((double) equal_counter / (double) n_rows) * 100.00;

    if (header) {
	std::cout << 
            ""  << "prefix_ID"     <<
	    "," << "threads"       <<
	    "," << "rows"          <<
	    "," << "features"      <<
	    "," << "cache-size-MB" <<
	    "," << "fit"           <<
	    "," << "predict"       <<
	    "," << "accuracy"      <<
	    "," << "sv-len"        <<
	    "," << "classes"       << std::endl;
    }

    std::cout << "Native-C," << daal_thread_num;
    std::cout << "," << X_nt->getNumberOfRows() << 
	         "," << X_nt->getNumberOfColumns();
    std::cout << "," << getOptimalCacheSize(X_nt->getNumberOfRows()) / (1024*1024);
    std::cout << "," << std::min_element(fit_times.begin(), fit_times.end())->count() / fit_repetitions;
    std::cout << "," << std::min_element(predict_times.begin(), predict_times.end())->count() / predict_repetitions;
    std::cout << "," << accuracy;
    std::cout << "," << sv_len;
    std::cout << "," << 2; // number of classes
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    CLI::App app("Native benchmark code for Intel(R) DAAL two-class SVM classifier");

    std::string xfn = "./data/mX.csv";
    CLI::Option *optX = app.add_option("--fileX", xfn, "Feature file name")->required()->check(CLI::ExistingFile);

    std::string yfn = "./data/mY.csv";
    CLI::Option *optY = app.add_option("--fileY", yfn, "Labels file name")->required()->check(CLI::ExistingFile);

    int fit_samples = 3, fit_repetitions = 1, predict_samples = 5, predict_repetitions = 50;
    CLI::Option *optFitS = app.add_option("--fit-samples", fit_samples, "Number of samples to collect for time of execution of repeated fit calls", true);
    CLI::Option *optFitR = app.add_option("--fit-repetitions", fit_repetitions, "Number of repeated fit calls to time", true);
    CLI::Option *optPredS = app.add_option("--predict-samples", predict_samples, "Number of samples to collect for time of execution of repeated predict calls", true);
    CLI::Option *optPredR = app.add_option("--predict-repetitions", predict_repetitions, "Number of repeated predict calls to time", true);

    int num_threads = 0;
    CLI::Option *optNumThreads = app.add_option("-n,--num-threads", num_threads, "Number of threads for DAAL to use", true);

    bool verbose = false;
    CLI::Option *optVerbose = app.add_flag("-v,--verbose", verbose, "Whether to be verbose or terse");

    bool header = false;
    CLI::Option *optHeader = app.add_flag("--header", header, "Whether to output header");

    CLI11_PARSE(app, argc, argv);

    assert(num_threads >= 0);
    assert(fit_samples > 0);
    assert(fit_repetitions > 0);
    assert(predict_samples > 0);
    assert(predict_repetitions > 0);

    if (verbose) {
	std::clog << 
	    "@ {FIT_SAMPLES: "        << fit_samples <<
	    ", FIT_REPETIONS: "       << fit_repetitions <<
	    ", PREDICT_SAMPLES: "     << predict_samples <<
	    ", PREDICT_REPETITIONS: " << predict_repetitions << 
	    "}" << std::endl;
    }

    bench(num_threads, xfn, yfn, fit_samples, fit_repetitions, predict_samples, predict_repetitions, verbose, header);

    return 0;
}
