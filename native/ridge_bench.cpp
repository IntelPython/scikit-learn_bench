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

#include "daal.h"

namespace dm=daal::data_management;
namespace ds=daal::services;
namespace da=daal::algorithms;
namespace dar=da::ridge_regression;

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

ds::SharedPtr<dar::training::Result> trainingResult;
dar::training::Batch<double> training_algorithm;
void linear_fit_test(double* X, double* Y, size_t rows, size_t cols)
{
    training_algorithm.input.set(dar::training::data, makeTable(X, rows, cols));
    training_algorithm.input.set(dar::training::dependentVariables, makeTable(Y, rows, cols));
    training_algorithm.compute();
    trainingResult = training_algorithm.getResult();
}

dar::prediction::Batch<double> predict_algorithm;
void linear_predict_test(double* X, size_t rows, size_t cols)
{
    predict_algorithm.input.set(dar::prediction::data, makeTable(X, rows, cols));
    predict_algorithm.input.set(dar::prediction::model, trainingResult->get(dar::training::model));
    predict_algorithm.compute();
    predict_algorithm.getResult()->get(dar::prediction::prediction);
}

std::pair<double, double> bench(size_t threadNum, int size0, int size1)
{
    /* Set the maximum number of threads to be used by the library */
    if (threadNum != 0)
        daal::services::Environment::getInstance()->setNumberOfThreads(threadNum);

    std::vector<std::pair<int, int> > problem_sizes = init_problem_sizes(size0, size1);
    std::vector<std::pair<int, int> >::iterator it;
    for (it = problem_sizes.begin(); it != problem_sizes.end(); it++) {
        size_t size = it->first * it->second;
        double* X = new double[size];
        double* Xp = new double[size];
        double* Y = new double[size];
        for(size_t i = 0; i < size; i++) {
            X[i] = (double)rand() / RAND_MAX;
            Xp[i] = (double)rand() / RAND_MAX;
            Y[i] = (double)rand() / RAND_MAX;
        }
        std::vector<std::chrono::duration<double> > times_fit;
        std::vector<std::chrono::duration<double> > times_predict;
        for(int i = 0; i < REPS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            linear_fit_test(X, Y, it->first, it->second);
            auto finish = std::chrono::high_resolution_clock::now();
            times_fit.push_back(finish - start);

            start = std::chrono::high_resolution_clock::now();
            linear_predict_test(Xp, it->first, it->second);
            finish = std::chrono::high_resolution_clock::now();
            times_predict.push_back(finish - start);
        }
        delete[] X;
        delete[] Xp;
        delete[] Y;

        return std::make_pair(std::min_element(times_fit.begin(), times_fit.end())->count(),
                              std::min_element(times_predict.begin(), times_predict.end())->count());
    }
}


int main(int args, char **argsv)
{
    if (args != 8) {
        fprintf(stderr, "usage: %s BATCH ARCH PREFIX FUNC CORES DTYPE SIZE1xSIZE2\n", argsv[0]);
        exit(1);
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
    std::pair<double, double> results = bench(nThreads, size0, size1);

    for(int f=1; f < args; f++) {
        if (f==5)
            std::cout << (atoi(argsv[f]) == 1 ? "Serial," : "Threaded,"); 
        else if (f == 4)
            std::cout << "Ridge.fit,"; 
        else 
            std::cout << argsv[f] << ",";
    }
    std::cout << results.first << std::endl;


    for(int f=1; f < args; f++) {
        if (f==5)
            std::cout << (atoi(argsv[f]) == 1 ? "Serial," : "Threaded,"); 
        else if (f == 4)
            std::cout << "Ridge.prediction,"; 
        else 
            std::cout << argsv[f] << ",";
    }
    std::cout << results.second << std::endl;
    return 0;
}
