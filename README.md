# scikit-learn_bench

Benchmark for optimizations to scikit-learn in the Intel(R) Distribution for
Python*. See benchmark results [here](https://intelpython.github.io/scikit-learn_bench).

## Prerequisites
- python and scikit-learn to run python versions
- pandas when using its DataFrame as input data format
- `icc`, `ifort`, `mkl`, `daal` to compile and run native benchmarks

## How to create conda environment for benchmarking
`conda create -n skl_bench -c intel python=3.7 scikit-learn pandas`

## Running Python benchmarks with runner script
`python runner.py --config config_example.json [--output-format json --verbose]`

## Legacy automatic building and running
- Run `make`. This will generate data, compile benchmarks, and run them.
  - To run only scikit-learn benchmarks, use `make sklearn`.
  - To run only native benchmarks, use `make native`.
  - To run only daal4py benchmarks, use `make daal4py`.
  - To run a specific implementation of a specific benchmark,
    directly request the corresponding file: `make output/<impl>/<bench>.out`.
  - If you have activated a conda environment, the build will use daal from
    the conda environment, if available.
