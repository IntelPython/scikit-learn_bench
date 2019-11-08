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

struct timing_options {
    int inner_loops; // Maximum number of inner loops
    int outer_loops; // Maximum number of outer loops
    double time_limit; // Time limit goal
    int goal_outer_loops; // Number of outer loops to aim for
};


void add_timing_args(CLI::App &app, const std::string &loop,
                     struct timing_options &opts) {

    std::string loop_dash;

    if (loop == "") {
        loop_dash = "";
    } else {
        loop_dash = loop + "-";
    }

    app.add_option("--" + loop_dash + "inner-loops", opts.inner_loops,
                   "Maximum inner loop iterations to run " + loop +
                   " (we take the mean over inner loop iterations)");
    app.add_option("--" + loop_dash + "outer-loops", opts.outer_loops,
                   "Maximum outer loop iterations to run " + loop +
                   " (we take the min over outer loop iterations)");
    app.add_option("--" + loop_dash + "time-limit", opts.time_limit,
                   "Target time to spend to benchmark " + loop);
    app.add_option("--goal-" + loop_dash + "outer-loops", opts.goal_outer_loops,
                   "Number of outer loops to aim for " + loop +
                   " while automatically picking number of inner loops. If "
                   "zero, do not automatically decide number of inner loops.");
}


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
time_vec(std::function<T()> func, int inner_loops, int outer_loops,
         double time_limit, int goal_outer_loops, bool verbose) {

    std::vector<std::chrono::duration<double>> vec;
    double total_time = 0.;

    T result;

    // Execute warm-up iterations to determine optimal inner_loops
    bool warmup = (goal_outer_loops > 0);
    double warmup_time = 0.;
    std::chrono::duration<double> last_warmup;

    if (warmup) {
        for (int i = 0; i < inner_loops; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            result = func();
            auto t1 = std::chrono::high_resolution_clock::now();

            last_warmup = t1 - t0;
            warmup_time += last_warmup.count();
            if (warmup_time > time_limit / 10) {
                break;
            }
        }

        inner_loops = time_limit / last_warmup.count() / goal_outer_loops;
        if (inner_loops < 1) {
            inner_loops = 1;
        }

        if (verbose) {
            std::cout << "@ Optimal inner loops = " << inner_loops << std::endl;
        }
    }

    if (last_warmup.count() > time_limit) {
        // If we took too much time in warm-up, just use those numbers
        if (verbose) {
            std::cout << "@ A single warmup iteration took "
                      << last_warmup.count()
                      << "s > " << time_limit << "s - not performing any "
                      << "more timings" << std::endl;
        }

        outer_loops = 1;
        inner_loops = 1;
        vec.push_back(last_warmup);
    } else {
        // Otherwise, actually take the timing
        for (int i = 0; i < outer_loops; i++) {

            std::chrono::duration<double> delta;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < inner_loops; j++) {
                result = func();
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            delta = t1 - t0;

            vec.push_back(delta / inner_loops);
            total_time += delta.count();

            if (time_limit > 0 && total_time > time_limit) {
                if (verbose) {
                    std::cout << "@ TT=" << total_time << "s exceeding "
                              << time_limit << "s after iteration "
                              << i + 1 << std::endl;
                }
                break;
            }
        }
    }

    if (verbose) {
        std::cout << "@ Mean times [s]" << std::endl;
        for (int i = 0; i < vec.size(); i++) {
            std::cout << "@ times[" << i << "] = " << vec[i].count()
                      << std::endl;
        }
    }

    return std::make_pair(vec, result);

}


/*
 * Time the given function for the specified number of repetitions,
 * returning a pair of the minimum duration and the LAST result.
 */
template <typename T>
std::pair<double, T>
time_min(std::function<T()> func, int inner_loops, int outer_loops,
         double time_limit, int goal_outer_loops, bool verbose) {

    auto pair = time_vec(func, inner_loops, outer_loops, time_limit,
                         goal_outer_loops, verbose);
    auto times = pair.first;
    double time = std::min_element(times.begin(), times.end())->count();

    return std::make_pair(time, pair.second);

}


/*
 * Time the given function for the specified number of repetitions,
 * returning a pair of the minimum duration and the LAST result.
 */
template <typename T>
std::pair<double, T>
time_min(std::function<T()> func, struct timing_options &o, bool verbose) {

    return time_min(func, o.inner_loops, o.outer_loops, o.time_limit,
                    o.goal_outer_loops, verbose);

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

    size_t shape[2] = {table->getNumberOfRows(), table->getNumberOfColumns()};

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
 *
 * equivalent in python:
 *   return src[row_start:row_end, col_start:col_end].copy()
 */
template <typename T>
dm::NumericTablePtr
copy_submatrix(dm::NumericTablePtr src,
               size_t row_start, size_t row_end,
               size_t col_start, size_t col_end) {

    dm::BlockDescriptor<T> srcblock;
    src->getBlockOfRows(0, src->getNumberOfRows(), dm::readOnly, srcblock);
    T *srcdata = srcblock.getBlockPtr();

    size_t cols = col_end - col_start;
    size_t rows = row_end - row_start;

    dm::NumericTablePtr dest = dm::HomogenNumericTable<T>::create(
            cols, rows, dm::NumericTable::doAllocate);
    dm::BlockDescriptor<T> destblock;
    dest->getBlockOfRows(0, dest->getNumberOfRows(), dm::readWrite, destblock);
    T *destdata = destblock.getBlockPtr();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            destdata[i*cols + j] = srcdata[(i+col_start)*cols + j+row_start];
        }
    }

    src->releaseBlockOfRows(srcblock);
    dest->releaseBlockOfRows(destblock);

    return dest;

}


int count_classes(dm::NumericTablePtr y) {

    /* compute min and max labels with DAAL */
    da::low_order_moments::Batch<double> algorithm;
    algorithm.input.set(da::low_order_moments::data, y);
    algorithm.compute();
    da::low_order_moments::ResultPtr res = algorithm.getResult();
    dm::NumericTablePtr min_nt = res->get(da::low_order_moments::minimum);
    dm::NumericTablePtr max_nt = res->get(da::low_order_moments::maximum);

    int min, max;
    dm::BlockDescriptor<double> block;
    min_nt->getBlockOfRows(0, 1, dm::readOnly, block);
    min = block.getBlockPtr()[0];
    max_nt->getBlockOfRows(0, 1, dm::readOnly, block);
    max = block.getBlockPtr()[0];
    return 1 + max - min;

}


size_t count_same_labels(dm::NumericTablePtr y1, dm::NumericTablePtr y2,
                         double tol = 1e-6) {

    size_t equal_counter = 0;
    size_t n_rows = std::min(y1->getNumberOfRows(), y2->getNumberOfRows());
    dm::BlockDescriptor<double> block_y1, block_y2;
    y1->getBlockOfRows(0, n_rows, dm::readOnly, block_y1);
    y2->getBlockOfRows(0, n_rows, dm::readOnly, block_y2);

    double *ptr1 = block_y1.getBlockPtr();
    double *ptr2 = block_y2.getBlockPtr();

    for (size_t i = 0; i < n_rows; i++) {
        if (abs(ptr1[i] - ptr2[i]) < tol) {
            equal_counter++;
        }
    }
    y1->releaseBlockOfRows(block_y1);
    y2->releaseBlockOfRows(block_y2);

    return equal_counter;

}


double accuracy_score(dm::NumericTablePtr y1, dm::NumericTablePtr y2,
                      double tol = 1e-6) {

    double n_rows = std::min(y1->getNumberOfRows(), y2->getNumberOfRows());
    return (double) count_same_labels(y1, y2) / n_rows;

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
    app.add_flag("--verbose", verbose, "Output extra debug messages");

}


