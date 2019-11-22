/*
 * Copyright (C) 2017-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>

#include <daal.h>
#include "CLI11.hpp"
#include "common.hpp"

dm::NumericTablePtr correlation_test(double *X, size_t rows, size_t cols) {

    da::correlation_distance::Batch<double> algorithm;
    algorithm.input.set(da::correlation_distance::data, make_table(X, rows, cols));
    algorithm.compute();
    return algorithm.getResult()->get(da::correlation_distance::correlationDistance);

}


dm::NumericTablePtr cosine_test(double *X, size_t rows, size_t cols) {

    da::cosine_distance::Batch<double> algorithm;
    algorithm.input.set(da::cosine_distance::data, make_table(X, rows, cols));
    algorithm.compute();
    return algorithm.getResult()->get(da::cosine_distance::cosineDistance);

}


int main(int argc, char *argv[]) {

    CLI::App app("Native benchmark for Intel(R) DAAL correlation and cosine distances");

    std::string batch, arch, prefix;
    int num_threads;
    bool header, verbose;
    add_common_args(app, batch, arch, prefix, num_threads, header, verbose);

    std::string stringSize = "1000x150000";
    app.add_option("-s,--size", stringSize, "Problem size");

    struct timing_options timing_opts = {100, 100, 10., 10};
    add_timing_args(app, "", timing_opts);

    CLI11_PARSE(app, argc, argv);

    std::vector<int> size;
    parse_size(stringSize, size);
    check_dims(size, 2);
    int daal_threads = set_threads(num_threads);

    std::string header_string = "Batch,Arch,Prefix,Threads,Size,Function,Time";
    std::ostringstream meta_info_stream;
    meta_info_stream
        << batch << ','
        << arch << ','
        << prefix << ','
        << daal_threads << ','
        << stringSize << ',';
    std::string meta_info = meta_info_stream.str();

    if (header)
        std::cout << header_string << std::endl;

    // Actual bench here
    double *X = gen_random(size[0] * size[1]);
    double time;
    dm::NumericTablePtr result;

    std::tie(time, result) = time_min<dm::NumericTablePtr> ([=] {
                return correlation_test(X, size[0], size[1]);
            }, timing_opts, verbose);
    std::cout << meta_info << "Correlation," << time << std::endl;

    std::tie(time, result) = time_min<dm::NumericTablePtr> ([=] {
                return cosine_test(X, size[0], size[1]);
            }, timing_opts, verbose);
    std::cout << meta_info << "Cosine," << time << std::endl;
    return 0;

}
