# scikit-learn_bench

Benchmark for optimizations to scikit-learn in the Intel(R) Distribution for
Python*

## Prerequisites
- python and scikit-learn to run python versions
- `icc`, `ifort`, `mkl`, `daal` to compile and run native benchmarks

## Automatically build and run
- Run `make`. This will generate data, compile benchmarks, and run them.
  - To run only python benchmarks, use `make python`.
  - To run only native benchmarks, use `make native`.
  - If you have activated a conda environment, the build will use daal from
    the conda environment, if available.
