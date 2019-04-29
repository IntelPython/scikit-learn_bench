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

void correlation_test(double *X, size_t rows, size_t cols) {

    da::correlation_distance::Batch<double> algorithm;
    algorithm.input.set(da::correlation_distance::data, make_table(X, rows, cols));
    algorithm.compute();
    algorithm.getResult()->get(da::correlation_distance::correlationDistance);

}


void cosine_test(double *X, size_t rows, size_t cols) {

    da::cosine_distance::Batch<double> algorithm;
    algorithm.input.set(da::cosine_distance::data, make_table(X, rows, cols));
    algorithm.compute();
    algorithm.getResult()->get(da::cosine_distance::cosineDistance);

}


int main(int argc, char *argv[]) {

    CLI::App app("Native benchmark for Intel(R) DAAL correlation and cosine distances");

    std::string batch, arch, prefix;
    int num_threads;
    bool header, verbose;
    add_common_args(app, batch, arch, prefix, num_threads, header, verbose);

    std::string stringSize = "1000x150000";
    app.add_option("-s,--size", stringSize, "Problem size");

    int samples = 1;
    app.add_option("--samples", samples, "Number of samples to report");

    int reps = 10;
    app.add_option("-r,--reps", samples, "Number of repetitions in each sample");


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
        << num_threads << ','
        << stringSize << ',';
    std::string meta_info = meta_info_stream.str();

    if (header)
        std::cout << header_string << std::endl;

    // Actual bench here
    double *X = gen_random(size[0] * size[1]);
    double time;

    for (int i = 0; i < samples; i++) {
        time = time_min([=] { correlation_test(X, size[0], size[1]); }, reps);
        std::cout << meta_info << "Correlation," << time << std::endl;

        time = time_min([=] { cosine_test(X, size[0], size[1]); }, reps);
        std::cout << meta_info << "Cosine," << time << std::endl;
    }
    return 0;

}
