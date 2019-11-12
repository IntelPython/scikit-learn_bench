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
#include "CLI11.hpp"
#include "common.hpp"
#include "daal.h"
#include "mkl.h"
#include "npyfile.h"
#include "lbfgsb/lbfgsb_daal.h"

namespace dm=daal::data_management;
namespace ds=daal::services;
namespace da=daal::algorithms;
namespace dl=daal::algorithms::logistic_regression;

using namespace daal;
using namespace da;

void print_numeric_table(dm::NumericTablePtr, std::string);

dl::training::ResultPtr 
logistic_regression_fit(
    int nClasses,
    bool fit_intercept,
    double C,
    size_t max_iter,
    double tol, 
    dm::NumericTablePtr Xt,
    dm::NumericTablePtr Yt,
    bool verbose)
{
    size_t n_samples = Yt->getNumberOfRows();

    ds::SharedPtr<lbfgsb::Batch> lbfgsSolver(new lbfgsb::Batch());

    lbfgsSolver->parameter.nIterations = max_iter;
    lbfgsSolver->parameter.accuracyThreshold = tol;
    lbfgsSolver->parameter.iprint = (verbose) ? 1 : -1;
    lbfgsSolver->parameter.funcScaling = n_samples;
    lbfgsSolver->parameter.gradScaling = n_samples;


    dl::training::Batch<double> log_reg_alg(nClasses);
    log_reg_alg.parameter().interceptFlag = fit_intercept;
    log_reg_alg.parameter().penaltyL1 = 0.;
    log_reg_alg.parameter().penaltyL2 = 0.5 / C / n_samples;

    log_reg_alg.parameter().optimizationSolver = lbfgsSolver;

    if (verbose) {
	std::cout << "@ {'fit_intercept': " << fit_intercept << 
                    ", 'C': " << C << 
                    ", 'max_iter': " <<  max_iter << 
                    ", 'tol': " << tol << 
                    "}" <<  std::endl;
    }

    log_reg_alg.input.set(da::classifier::training::data, Xt);
    log_reg_alg.input.set(da::classifier::training::labels, Yt);

    log_reg_alg.compute();

    dl::training::ResultPtr result_ptr = log_reg_alg.getResult();

    if(verbose) {
	print_numeric_table(
	    lbfgsSolver->getResult()->get(da::optimization_solver::iterative_solver::nIterations),
	    "Number of iterations");
	print_numeric_table(
	    result_ptr->get(da::classifier::training::model)->getBeta(),
	    "Fitted coefficients"
	    );
    }

    return result_ptr;
}

dm::NumericTablePtr 
logistic_regression_predict(
    int nClasses,
    dl::training::ResultPtr training_result_ptr,
    dm::NumericTablePtr Xt, 
    bool verbose
    )
{
    dl::prediction::Batch<double> pred_alg(nClasses);
    pred_alg.input.set(da::classifier::prediction::data, Xt);
    pred_alg.input.set(da::classifier::prediction::model, 
		       training_result_ptr->get(da::classifier::training::model));

    pred_alg.compute();

    da::classifier::prediction::ResultPtr pred_res = pred_alg.getResult();
    dm::NumericTablePtr Y_pred_t = pred_res->get(da::classifier::prediction::prediction);

    return Y_pred_t;
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
	equal_counter += (fabs(Y_data_ptr[i] - Yp_data_ptr[i]) < 0.5) ? 1 : 0;
    }
    Yp_nt->releaseBlockOfRows(blockYp);
    Y_nt->releaseBlockOfRows(blockY);

    return equal_counter;
}

int
find_nClasses(dm::NumericTablePtr Y_nt)
{
    /* compute min and max labels with DAAL */
    da::low_order_moments::Batch<double> algorithm;
    algorithm.input.set(da::low_order_moments::data, Y_nt);
    algorithm.compute();
    da::low_order_moments::ResultPtr res = algorithm.getResult();
    dm::NumericTablePtr min_nt = res->get(da::low_order_moments::minimum);
    dm::NumericTablePtr max_nt = res->get(da::low_order_moments::maximum);
    int min, max;
    dm::BlockDescriptor<> block;
    min_nt->getBlockOfRows(0, 1, dm::readOnly, block);
    min = block.getBlockPtr()[0];
    max_nt->getBlockOfRows(0, 1, dm::readOnly, block);
    max = block.getBlockPtr()[0];
    return 1 + max - min;
}

void
print_numeric_table(dm::NumericTablePtr X_nt, std::string label)
{
    size_t n_cols = X_nt->getNumberOfColumns();
    size_t n_rows = X_nt->getNumberOfRows();

    dm::BlockDescriptor<double> blockX;
    X_nt->getBlockOfRows(0, n_rows, dm::readOnly, blockX);

    std::streamsize prec = std::cout.precision();

    double *x = blockX.getBlockPtr();
    std::cout << "@ " << label << ":" << std::endl;
    std::cout << std::setprecision(18) << std::scientific;
    for (size_t i_outer=0; i_outer < n_rows; i_outer++) {
	std::cout << "@ ";
	for(size_t i_inner=0; i_inner < n_cols; i_inner++) {
	    std::cout << x[i_inner + i_outer * n_cols] << ", ";
	}
	std::cout << std::setprecision(prec) << std::defaultfloat << std::endl;
    }

    X_nt->releaseBlockOfRows(blockX);
}

int main(int argc, char** argv) {

    CLI::App app("Native benchmark code for Intel(R) DAAL logistic regression classifier");

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

    double C = 1.0;
    app.add_option("-C,--C", C, "Slack parameter")
        ->check(CLI::PositiveNumber);

    double tol = 1e-10;
    app.add_option("--tol", tol, "Tolerance")
        ->check(CLI::PositiveNumber);

    size_t max_iter = 1000;
    app.add_option("--maxiter", max_iter,
                   "Maximum iterations for the iterative solver")
        ->check(CLI::PositiveNumber);

    // TODO add configurable fit_intercept parameter

    CLI11_PARSE(app, argc, argv);

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

    // Set DAAL and MKL threads
    int daal_threads = set_threads(num_threads);
    mkl_set_threading_layer(MKL_THREADING_TBB);

    int n_classes = count_classes(Y_nt);
    bool fit_intercept = true;

    // Prepare header and metadata info
    std::string header_string = "batch,arch,prefix,threads,size,classes,"
                                "solver,tol,maxiter,C,function,accuracy,time";
    std::ostringstream meta_info_stream;
    meta_info_stream
        << batch << ','
        << arch << ','
        << prefix << ','
        << daal_threads << ','
        << stringSize << ','
        << n_classes << ','
        << "lbfgs" << ','
        << tol << ','
        << max_iter << ','
        << C << ',';
    std::string meta_info = meta_info_stream.str();

    // Actually time benchmarks

    double time;
    bool verbose_fit = true;
    dl::training::ResultPtr training_result;
    std::tie(time, training_result) = time_min<dl::training::ResultPtr> ([&] {
            auto r = logistic_regression_fit(n_classes, fit_intercept, C,
                                             max_iter, tol, X_nt, Y_nt,
                                             verbose_fit);
            verbose_fit = false;
            return r;
        }, fit_opts, verbose);

    if (header) {
        std::cout << header_string << std::endl;
    }
    std::cout << meta_info << "LogReg.fit,," << time << std::endl;

    dm::NumericTablePtr Yp_nt;
    std::tie(time, Yp_nt) = time_min<dm::NumericTablePtr> ([&] {
            return logistic_regression_predict(n_classes, training_result,
                                               X_nt, verbose);
            }, predict_opts, verbose);

    double accuracy = accuracy_score(Y_nt, Yp_nt) * 100.;
    std::cout << meta_info << "LogReg.predict," << accuracy << ','
        << time << std::endl;

    return EXIT_SUCCESS;
}
