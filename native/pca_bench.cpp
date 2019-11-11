/*
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <chrono>  

#include "common.hpp"
#include "daal.h"


namespace dn = daal::algorithms::normalization;


std::pair<da::pca::ResultPtr, dm::NumericTablePtr>
pca_fit_daal(double *X, size_t rows, size_t cols, size_t n_components) {

    // Find number of components to use in DAAL
    if (n_components < 1) {
        n_components = std::min(cols, rows);
    }

    da::pca::Batch<double, da::pca::svdDense> pca_algorithm;

    pca_algorithm.input.set(da::pca::data, make_table(X, rows, cols));
    pca_algorithm.parameter.resultsToCompute = 
        da::pca::mean | da::pca::variance | da::pca::eigenvalue;
    pca_algorithm.parameter.isDeterministic = true;
    pca_algorithm.parameter.nComponents = n_components;

    // We must explicitly create zscore_algorithm here to make
    // DAAL only center, and not scale.
    // We do two things differently here:
    // 1. The zscore Batch object is in the heap because DAAL SharedPtr likes to
    //    free it afterwards.
    // 2. In order to change zscore parameters, we call its parameter() method,
    //    which returns a Parameter object like the one PCA provides directly.
    auto zscore_algorithm = new dn::zscore::Batch<double, dn::zscore::defaultDense>;
    zscore_algorithm->parameter().doScale = false;
    ds::SharedPtr<dn::zscore::Batch<double, dn::zscore::defaultDense>> zscore_ptr {zscore_algorithm};

    pca_algorithm.parameter.normalization = zscore_ptr;

    pca_algorithm.compute();
    da::pca::ResultPtr pca_result = pca_algorithm.getResult();

    // Compute singular values
    dm::NumericTablePtr eigenvalues = pca_result->get(da::pca::eigenvalues);
    dm::BlockDescriptor<double> block;
    eigenvalues->getBlockOfRows(0, eigenvalues->getNumberOfRows(), dm::readOnly, block);
    double *eigenvalues_arr = block.getBlockPtr();

    size_t s_diag_size = eigenvalues->getNumberOfRows() * eigenvalues->getNumberOfColumns();
    double *singular_values_arr = new double[s_diag_size];

    for (int i = 0; i < s_diag_size; i++) {
        singular_values_arr[i] = std::sqrt((rows - 1) * eigenvalues_arr[i]);
    }

    eigenvalues->releaseBlockOfRows(block);

    dm::NumericTablePtr singular_values = make_table(
            singular_values_arr,
            eigenvalues->getNumberOfRows(),
            eigenvalues->getNumberOfColumns());
    
    return std::make_pair(pca_result, singular_values);

}


da::pca::transform::ResultPtr pca_transform_daal(
        da::pca::ResultPtr pca_result,
        double *X, size_t rows, size_t cols, int n_components,
        bool whiten, bool scale_eigenvalues) {

    da::pca::transform::Batch<double> transform_algorithm;
    dm::NumericTablePtr pca_eigvals = pca_result->get(da::pca::eigenvalues);
    double *new_eigvals;
    bool need_to_free_pca_eigvals = false;
    
    // sklearn scales eigenvalues before whitening operation...
    if (scale_eigenvalues) {
        size_t eigrows = pca_eigvals->getNumberOfRows();
        size_t eigcols = pca_eigvals->getNumberOfColumns();
        size_t arrsize = eigrows * eigcols;
        new_eigvals = new double[arrsize];
        need_to_free_pca_eigvals = true;
        if (whiten) {
            dm::BlockDescriptor<double> block;
            pca_eigvals->getBlockOfRows(0, pca_eigvals->getNumberOfRows(), dm::readOnly, block);
            double *eigvals = block.getBlockPtr();

            for (int i = 0; i < arrsize; i++) {
                new_eigvals[i] = (rows - 1) * eigvals[i];
            }

            pca_eigvals->releaseBlockOfRows(block);
        } else {
            for (int i = 0; i < arrsize; i++) {
                new_eigvals[i] = rows - 1;
            }
        }
        pca_eigvals = make_table(new_eigvals, eigrows, eigcols);
    }

    // only pass means and eigenvalues to pca transform
    dm::KeyValueDataCollection *new_map_p = new dm::KeyValueDataCollection;
    dm::KeyValueDataCollectionPtr new_map {new_map_p};
    dm::KeyValueDataCollectionPtr old_map = pca_result->get(da::pca::dataForTransform);
    (*new_map)[da::pca::mean] = (*old_map)[da::pca::mean];
    (*new_map)[da::pca::eigenvalue] = pca_eigvals;
    
    // time to call DAAL algorithm.
    transform_algorithm.input.set(da::pca::transform::data,
                                  make_table(X, rows, cols));
    transform_algorithm.input.set(da::pca::transform::eigenvectors,
                                  pca_result->get(da::pca::eigenvectors));
    transform_algorithm.input.set(da::pca::transform::dataForTransform,
                                  new_map);
    transform_algorithm.parameter.nComponents = n_components;

    transform_algorithm.compute();

    if (need_to_free_pca_eigvals) delete new_eigvals;

    return transform_algorithm.getResult();

}


/*
 * Equivalent to sklearn.util.extmath.svd_flip with u_based_decision=True.
 */
void svd_flip(dm::NumericTablePtr U, dm::NumericTablePtr V) {

    int u_rows = U->getNumberOfRows();
    int u_cols = V->getNumberOfColumns();
    int v_rows = V->getNumberOfRows();
    int v_cols = V->getNumberOfColumns();

    double *u, *v;
    dm::BlockDescriptor<double> ublock, vblock;
    U->getBlockOfRows(0, u_rows, dm::readWrite, ublock);
    u = ublock.getBlockPtr();
    V->getBlockOfRows(0, v_rows, dm::readWrite, vblock);
    v = vblock.getBlockPtr();
    
    // Need to allocate a sign array of ints here.
    // Then we need TWO loops. The first one gets signs.,
    // and the second one actually scales.
    // Just store 1 or -1 and multiply.
    // Use the sign function on the component we find as max
    // std::sign <- might be sgn, std::abs
    bool *flip = new bool[u_cols];

    // Find signs.
    // for each column in u...
    for (int i = 0; i < u_cols; i++) {
        // find the maximum absolute value...
        double absmax = 0.0;
        for (int j = 0; j < u_rows; j++) {
            double curr = u[j*u_cols + i];
            if (std::abs(curr) > std::abs(absmax)) {
                absmax = curr;
            }
        }
        flip[i] = (absmax < 0);
    }

    // Apply sign flipping
    for (int i = 0; i < u_cols; i++) {

        // now, scale this column of u and same-indexed ROW of v
        // by the sign of absmax.
        if (flip[i]) {
// #pragma vector
            for (int j = 0; j < u_rows; j++) {
                u[j*u_cols + i] = -u[j*u_cols + i];
            }
            
// #pragma vector
            for (int j = 0; j < v_cols; j++) {
                v[j + i*v_rows] = -v[j + i*v_rows];
            }
        }
    }

    U->releaseBlockOfRows(ublock);
    V->releaseBlockOfRows(vblock);

}


/**
 * equivalent to _fit_full_daal.
 *
 * Returns U, S, V in the SVD.
 */
std::tuple<da::pca::ResultPtr, dm::NumericTablePtr, dm::NumericTablePtr, dm::NumericTablePtr>
pca_fit_full_daal(double *X, size_t rows, size_t cols, size_t n_components) {

    // Run full decomposition...
    size_t full_n_components = std::min(rows, cols);
    da::pca::ResultPtr pca_result;
    dm::NumericTablePtr singular_values;
    std::tie(pca_result, singular_values) = pca_fit_daal(X, rows, cols, full_n_components);

    da::pca::transform::ResultPtr transform_result;
    transform_result = pca_transform_daal(pca_result, X, rows, cols, full_n_components, true, true);

    dm::NumericTablePtr U = transform_result->get(da::pca::transform::transformedData);
    dm::NumericTablePtr V = pca_result->get(da::pca::eigenvectors);

    // Flip signs to make largest row values positive for each column in U.
    svd_flip(U, V);

    // Need to take subcomponents of the eigenvalues and eigenvectors here.
    dm::NumericTablePtr orig_eigvals = pca_result->get(da::pca::eigenvalues);
    dm::NumericTablePtr eigvals = copy_submatrix<double>(orig_eigvals, 0, 1, 0, n_components);
    dm::NumericTablePtr eigvecs = copy_submatrix<double>(V, 0, n_components, 0, cols);

    pca_result->set(da::pca::eigenvalues, eigvals);
    pca_result->set(da::pca::eigenvectors, eigvecs);

    return std::make_tuple(pca_result, U, singular_values, V);

}


/*
 * Function to time for native equivalent to sklearn PCA.fit.
 *
 * Parameters
 * ----------
 * X : double *
 *     input matrix
 * rows : size_t
 *     number of rows in input matrix
 * cols : size_t
 *     number of columns in input matrix
 * svd_solver : char
 *     svd solver to use
 *     'a' (auto) = automatically pick
 *     'f' (full) = run full SVD
 *     'k' (arpack) = not implemented
 *     'r' (randomized) = not implemented
 *     'd' (daal) = use daal solver
 * n_components : size_t
 *     number of components to retain
 *
 * Returns
 * -------
 * pca_result, U, S, V
 *     U, S, V for full fit, pca_result for daal fit
 */
std::tuple<da::pca::ResultPtr, dm::NumericTablePtr, dm::NumericTablePtr, dm::NumericTablePtr>
pca_fit_test(double *X, size_t rows, size_t cols,
             char svd_solver, size_t n_components) {

    // Skip input validation that sklearn does (we disable it in sklearn benchesa)
    // n_components is given, don't need to worry about it being None...

    if (svd_solver == 'a') {
        // Automatically picking SVD solver using same logic as sklearn.
        // TODO n_components = 'mle'?
        if (std::max(rows, cols) <= 500) {
            svd_solver = 'f';
        } else if (n_components >= 1 && n_components < std::min(rows, cols) * 8 / 10) {
            svd_solver = 'r';
        } else {
            svd_solver = 'f';
        }
    }

    da::pca::ResultPtr pca_result;
    dm::NumericTablePtr U, S, V;
    U = S = V = make_table(X, 0, 0);
    switch (svd_solver) {
        case 'd':
            std::tie(pca_result, S) = pca_fit_daal(X, rows, cols, n_components);
            break;
        case 'f':
            std::tie(pca_result, U, S, V) = pca_fit_full_daal(X, rows, cols, n_components);
            break;
        default:
            std::cerr << "Unsupported svd_solver='" << svd_solver << '\''
                << std::endl;
            std::exit(1);
    }

    return std::make_tuple(pca_result, U, S, V);

}


da::pca::transform::ResultPtr
pca_transform_test(da::pca::ResultPtr pca_result,
                   double *X, size_t rows, size_t cols,
                   int n_components) {

    return pca_transform_daal(pca_result, X, rows, cols, n_components, false, false);

}


int main(int argc, char *argv[]) {

    CLI::App app("Native benchmark for Intel(R) DAAL principal component analysis");

    std::string batch, arch, prefix;
    int num_threads;
    bool header, verbose;
    add_common_args(app, batch, arch, prefix, num_threads, header, verbose);

    std::string stringSize = "1000000x50";
    app.add_option("-s,--size", stringSize, "Problem size");

    struct timing_options fit_opts = {100, 100, 10., 10};
    add_timing_args(app, "fit", fit_opts);

    struct timing_options transform_opts = {10, 100, 10., 10};
    add_timing_args(app, "transform", transform_opts);

    int n_components = -1;
    app.add_option("--n-components", n_components,
                   "Number of components to get from PCA");

    bool write_results = false;
    app.add_flag("--write-results", write_results,
                 "Write arrays to file");

    std::string svd_solver = "daal";
    app.add_option("--svd-solver", svd_solver,
                   "Method to use for computing PCA");

    CLI11_PARSE(app, argc, argv);

    std::vector<int> size;
    parse_size(stringSize, size);
    check_dims(size, 2);
    int daal_threads = set_threads(num_threads);

    // Find n_components
    if (n_components == -1) {
        n_components = std::min(size[1], (2 + std::min(size[0], size[1])) / 3);
    }

    std::string header_string = "Batch,Arch,Prefix,Threads,Size,n_components,Function,Time";
    std::ostringstream meta_info_stream;
    meta_info_stream
        << batch << ','
        << arch << ','
        << prefix << ','
        << daal_threads << ','
        << stringSize << ','
        << n_components << ',';
    std::string meta_info = meta_info_stream.str();

    if (header)
        std::cout << header_string << std::endl;

    // Actual bench here
    double *X = gen_random(size[0] * size[1]);
    double *Xp = gen_random(size[0] * size[1]);

    double time;
    da::pca::ResultPtr pca_result;
    dm::NumericTablePtr U, S, V;

    // PCA fit also *might* return U, S, V...
    std::tuple<da::pca::ResultPtr, dm::NumericTablePtr,
               dm::NumericTablePtr, dm::NumericTablePtr> fit_results;

    // Get time and PCA results, including U, S, V.
    // N.B.: we use DAAL solver here
    std::tie(time, fit_results)
        = time_min<std::tuple<da::pca::ResultPtr, dm::NumericTablePtr,
                   dm::NumericTablePtr, dm::NumericTablePtr>> ([=] {
                return pca_fit_test(X, size[0], size[1], svd_solver[0], n_components);
            }, fit_opts, verbose);

    // Extract PCA results and U, S, V from tuple.
    std::tie(pca_result, U, S, V) = fit_results;
    std::cout << meta_info << "PCA.fit," << time << std::endl;

    da::pca::transform::ResultPtr transform_result;
    std::tie(time, transform_result)
        = time_min<da::pca::transform::ResultPtr> ([=] {
                return pca_transform_test(pca_result, Xp, size[0], size[1], n_components);
            }, transform_opts, verbose);
    std::cout << meta_info << "PCA.transform," << time << std::endl;

    if (write_results) {
        write_table<double>(make_table(X, size[0], size[1]), "<f8", "pca_X.npy");
        write_table<double>(make_table(Xp, size[0], size[1]), "<f8", "pca_Xp.npy");
        write_table<double>(pca_result->get(da::pca::eigenvalues), "<f8", "pca_eigvals.npy");
        write_table<double>(pca_result->get(da::pca::eigenvectors), "<f8", "pca_eigvecs.npy");
        write_table<double>(transform_result->get(da::pca::transform::transformedData), "<f8", "pca_transformed.npy");
    }
    return 0;

}
