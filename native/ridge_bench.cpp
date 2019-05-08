/*
 * Copyright (C) 2017-2019 Intel Corporation
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

namespace dar=da::ridge_regression;


dar::training::ResultPtr
linear_fit_test(double *X, double *y, size_t rows, size_t cols) {

    dar::training::Batch<double> training_algorithm;
    training_algorithm.input.set(dar::training::data, make_table(X, rows, cols));
    training_algorithm.input.set(dar::training::dependentVariables, make_table(y, rows, cols));
    training_algorithm.compute();
    return training_algorithm.getResult();

}


dm::NumericTablePtr
linear_predict_test(dar::training::ResultPtr training_result,
                    double *X, size_t rows, size_t cols) {

    dar::prediction::Batch<double> predict_algorithm;
    predict_algorithm.input.set(dar::prediction::data, make_table(X, rows, cols));
    predict_algorithm.input.set(dar::prediction::model, training_result->get(dar::training::model));
    predict_algorithm.compute();
    return predict_algorithm.getResult()->get(dar::prediction::prediction);

}


int main(int argc, char *argv[]) {

    CLI::App app("Native benchmark for Intel(R) DAAL ridge regression");

    std::string batch, arch, prefix;
    int num_threads;
    bool header, verbose;
    add_common_args(app, batch, arch, prefix, num_threads, header, verbose);

    std::string stringSize = "1000000x50";
    app.add_option("-s,--size", stringSize, "Problem size");

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
    app.add_option("--predict-reps", predict_reps,
                   "Number of repetitions in each sample (predict)");

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
    double *Xp = gen_random(size[0] * size[1]);
    double *y = gen_random(size[0] * size[1]);

    double time;
    dar::training::ResultPtr training_result;
    for (int i = 0; i < fit_samples; i++) {
        auto pair = time_min<dar::training::ResultPtr> ([=] {
                    return linear_fit_test(X, y, size[0], size[1]);
                }, fit_reps);
        time = pair.first;
        training_result = pair.second;
        std::cout << meta_info << "Ridge.fit," << time << std::endl;
    }

    dm::NumericTablePtr predict_result;
    for (int i = 0; i < predict_samples; i++) {
        auto pair = time_min<dm::NumericTablePtr> ([=] {
                    return linear_predict_test(training_result, Xp, size[0], size[1]);
                }, predict_reps);
        time = pair.first;
        predict_result = pair.second;
        std::cout << meta_info << "Ridge.predict," << time << std::endl;
    }
    return 0;

}
