/*
 * Copyright (C) 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

struct svm_linear_kernel_parameters {
    double C;
    double tol;
    double tau;
    int max_iter;
};

size_t getOptimalCacheSize(size_t n) {
    size_t cache_size = sizeof(double) * n * n;
    return cache_size;
}

void
print_svm_linear_kernel_parameters(svm_linear_kernel_parameters alg)
{
    std::clog << "@ { C: " << alg.C <<
	", tol: " << alg.tol <<
        ", tau: " << alg.tau <<
        ", max_iter: " << alg.max_iter <<
        "}" << std::endl;
}
