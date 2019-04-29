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

namespace dal=da::linear_regression;


ds::SharedPtr<dal::training::Result> trainingResult;
dal::training::Batch<double> training_algorithm;
void linear_fit_test(double *X, double *y, size_t rows, size_t cols) {

    training_algorithm.input.set(dal::training::data, make_table(X, rows, cols));
    training_algorithm.input.set(dal::training::dependentVariables, make_table(y, rows, cols));
    training_algorithm.compute();
    trainingResult = training_algorithm.getResult();

}


dal::prediction::Batch<double> predict_algorithm;
void linear_predict_test(double *X, size_t rows, size_t cols) {

    predict_algorithm.input.set(dal::prediction::data, make_table(X, rows, cols));
    predict_algorithm.input.set(dal::prediction::model, trainingResult->get(dal::training::model));
    predict_algorithm.compute();
    predict_algorithm.getResult()->get(dal::prediction::prediction);

}


int main(int argc, char *argv[]) {

    CLI::App app("Native benchmark for Intel(R) DAAL linear regression");

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
    app.add_option("--predict-reps", predict_samples,
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
    for (int i = 0; i < fit_samples; i++) {
        time = time_min([=] { linear_fit_test(X, y, size[0], size[1]); }, fit_reps);
        std::cout << meta_info << "Linear.fit," << time << std::endl;
    }

    for (int i = 0; i < predict_samples; i++) {
        time = time_min([=] { linear_predict_test(X, size[0], size[1]); }, predict_reps);
        std::cout << meta_info << "Linear.predict," << time << std::endl;
    }
    return 0;

}
