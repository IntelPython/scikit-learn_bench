# scikit-learn_bench

Benchmark for optimizations to scikit-learn in the Intel(R) Distribution for
Python*. See benchmark results [here](https://intelpython.github.io/scikit-learn_bench).

## Prerequisites
- python and scikit-learn to run python versions
- `icc` and `daal` to compile and run native benchmarks

## Automatically build and run
- Run `make`. This will generate data, compile benchmarks, and run them.
  - To run only scikit-learn benchmarks, use `make sklearn`.
  - To run only native benchmarks, use `make native`.
  - To run only daal4py benchmarks, use `make daal4py`.
  - To run a specific implementation of a specific benchmark,
    directly request the corresponding file: `make output/<impl>/<bench>.out`.
  - If you have activated a conda environment, the build will use daal from
    the conda environment, if available.
