#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include "CLI11.hpp"
#include "daal.h"
#include "npyfile.h"

namespace dm = daal::data_management;
namespace ds = daal::services;
namespace da = daal::algorithms;

/*
 * Parse a size argument as given to the command-line options.
 * This consists of each element of the size, delimited by
 * a 'x' or ','.
 *
 * Parameters
 * ----------
 * in : std::string
 *     String to parse
 * size : std::vector<int> &size
 *     Vector to which we should output size values
 */
void parse_size(std::string in, std::vector<int> &size) {

    std::stringstream tmp(in);
    int i;

    while (tmp >> i) {
        size.push_back(i);
        char c = tmp.peek();
        if (c == 'x' || c == ',')
            tmp.ignore();
    }

}


/*
 * Terminate the program with an error message if the number of dimensions
 * in the given size vector does not match the expected number of dimensions.
 *
 * Parameters
 * ----------
 * size : std::vector<int> &size
 *     Vector to check
 * expected : int
 *     Number of dimensions we expect
 */
void check_dims(std::vector<int> &size, int expected) {

    if (size.size() != expected) {
        std::cerr << "error: expected " << expected << " dimensions in size "
                  << "but got " << size.size();
        std::exit(1);
    }

}


/*
 * Try to set DAAL's number of threads to the given number of threads,
 * and returns the actual number of threads DAAL will use.
 *
 * Parameters
 * ----------
 * num_threads : int
 *     Number of threads we ask DAAL to use
 *
 * Returns
 * -------
 * int
 *     Number of threads DAAL says it will use
 */
int set_threads(int num_threads) {

    if (num_threads > 0)
        ds::Environment::getInstance()->setNumberOfThreads(num_threads);

    return ds::Environment::getInstance()->getNumberOfThreads();

}


/*
 * Time the given function for the specified number of repetitions,
 * returning a pair of a vector of durations and the LAST result.
 */
template <typename T>
std::pair<std::vector<std::chrono::duration<double>>, T>
time_vec(std::function<T()> func, int reps) {

    std::vector<std::chrono::duration<double>> vec;
    T result;
    for (int i = 0; i < reps; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        result = func();
        auto t1 = std::chrono::high_resolution_clock::now();
        vec.push_back(t1 - t0);
    }
    return std::make_pair(vec, result);

}


/*
 * Time the given function for the specified number of repetitions,
 * returning a pair of the minimum duration and the LAST result.
 */
template <typename T>
std::pair<double, T> time_min(std::function<T()> func, int reps) {

    auto pair = time_vec(func, reps);
    auto times = pair.first;
    double time = std::min_element(times.begin(), times.end())->count();

    return std::make_pair(time, pair.second);

}


/*
 * Create a DAAL HomogenNumericTable from an array in memory.
 */
template <typename T>
dm::NumericTablePtr make_table(T *data, size_t rows, size_t cols) {

    return dm::HomogenNumericTable<T>::create(data, cols, rows);

}


/*
 * Write a NumericTable to npy format.
 */
template <typename T>
void write_table(dm::NumericTablePtr table,
                 const std::string descr, const std::string fn) {

    dm::BlockDescriptor<T> block;
    table->getBlockOfRows(0, table->getNumberOfRows(), dm::readOnly, block);
    T *data = block.getBlockPtr();

    size_t shape[] = {table->getNumberOfRows(), table->getNumberOfColumns()};

    char *c_descr = new char[descr.size() + 1];
    strcpy(c_descr, descr.c_str());
    const struct npyarr arr {
        .descr = c_descr,
        .fortran_order = false,
        .shape_len = 2,
        .shape = shape,
        .data = data
    };

    save_npy(&arr, fn.c_str(), sizeof(T));

    table->releaseBlockOfRows(block);

}


/*
 * Create a new HomogenNumericTable from a submatrix of an existing
 * HomogenNumericTable.
 */
template <typename T>
dm::NumericTablePtr
copy_submatrix(dm::NumericTablePtr src,
               size_t row, size_t col, size_t rows, size_t cols) {

    dm::BlockDescriptor<T> srcblock;
    src->getBlockOfRows(0, src->getNumberOfRows(), dm::readOnly, srcblock);
    T *srcdata = srcblock.getBlockPtr();

    dm::NumericTablePtr dest = dm::HomogenNumericTable<T>::create(
            cols, rows, dm::NumericTable::doAllocate);
    dm::BlockDescriptor<T> destblock;
    dest->getBlockOfRows(0, dest->getNumberOfRows(), dm::readWrite, destblock);
    T *destdata = destblock.getBlockPtr();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            destdata[i*cols + j] = srcdata[(i+col)*cols + j+row];
        }
    }

    src->releaseBlockOfRows(srcblock);
    dest->releaseBlockOfRows(destblock);

    return dest;

}


/*
 * Generate an array of random numbers.
 */
double *gen_random(size_t n) {

    double *x = new double[n];
    for (size_t i = 0; i < n; i++)
        x[i] = (double) rand() / RAND_MAX;
    return x;

}


void add_common_args(CLI::App &app,
                     std::string &batch, std::string &arch,
                     std::string &prefix, int &num_threads,
                     bool &header, bool &verbose) {

    batch = "?";
    app.add_option("-b,--batch", batch, "Batch ID, for bookkeeping");

    arch = "?";
    app.add_option("-a,--arch", arch, "Machine architecture, for bookkeeping");

    prefix = "Native-C";
    app.add_option("-p,--prefix", prefix, "Prefix string, for bookkeeping");

    num_threads = 0;
    app.add_option("-n,--num-threads", num_threads, "Number of threads for DAAL to use", true);

    header = false;
    app.add_flag("--header", header, "Output CSV header");

    verbose = false;
    app.add_flag("--verbose", header, "Output extra debug messages");

}


