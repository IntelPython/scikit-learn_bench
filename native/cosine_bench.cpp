/*
 * Copyright (C) 2017-2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <chrono>  

#include <daal.h>

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


void cosine_test(double* X, size_t rows, size_t cols)
{
    da::cosine_distance::Batch<double> algorithm;
    algorithm.input.set(da::cosine_distance::data, makeTable(X, rows, cols));
    algorithm.compute();
    algorithm.getResult()->get(da::cosine_distance::cosineDistance);
}


void bench(size_t threadNum, int size0, int size1)
{
    /* Set the maximum number of threads to be used by the library */
    if (threadNum != 0)
        daal::services::Environment::getInstance()->setNumberOfThreads(threadNum);

    std::vector<std::pair<int, int> > problem_sizes = init_problem_sizes(size0, size1);
    std::vector<std::pair<int, int> >::iterator it; 
    for (it = problem_sizes.begin(); it != problem_sizes.end(); it++) {
        size_t size = it->first * it->second;
        double* X = new double[size];
        for(size_t i = 0; i < size; i++)
            X[i] = (double)rand() / RAND_MAX;
        std::vector<std::chrono::duration<double> > times;
        for(int i = 0; i < REPS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            cosine_test(X, it->first, it->second);
            auto finish = std::chrono::high_resolution_clock::now();
            times.push_back(finish - start);
        }
        std::cout << std::min_element(times.begin(), times.end())->count() << std::endl;
        delete[] X;
    }
}


int main(int args, char **argsv)
{
    if (args != 8) {
        fprintf(stderr, "usage: %s BATCH ARCH PREFIX FUNC CORES DTYPE SIZE1xSIZE2\n", argsv[0]);
        exit(1);
    }

    for(int f=1; f < args; f++) {
        if (f==5)
            std::cout << ((atoi(argsv[f]) == 1) ? "Serial," : "Threaded,"); 
        else
            std::cout << argsv[f] << ",";
    }

    std::vector<int> arraySize;
    std::stringstream tmp(argsv[7]);
    int i;

    while (tmp >> i)
    {
        arraySize.push_back(i);

        if (tmp.peek() == 'x')
            tmp.ignore();
    }

    const size_t nThreads = atoi(argsv[5]);
    int size0 = arraySize.at(0);
    int size1 = arraySize.at(1);
    bench(nThreads, size0, size1);
    return 0;
}
