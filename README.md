# scikit-learn_bench

Benchmark for optimizations to scikit-learn in the Intel Distribution for
Python*

## Prerequisites
- python and scikit-learn to run python versions
- `icc` and `daal` to compile and run native benchmarks

## Automatically build and run
- Run `make`. This will generate data, compile benchmarks, and run them.
  - To run only python benchmarks, use `make python`.
  - To run only native benchmarks, use `make native`.
  - If you have activated a conda environment, the build will use daal from
    the conda environment, if available.

## Manually

### Build

- `git submodule init && git submodule update` (for native versions only)
- To build native versions, run `make -C native`.
- Prepare data for KMeans benchmarks by running
  `mkdir -p data && python python/kmeans_data.py --size 1000000x50 --fname data/kmeans_1000000x50.csv --clusters 10`
  - Size can be adjusted. Example sizes are `500000x5`, `500000x25`, `1000000x50`.
- Prepare data for SVM benchmarks by running
  `python python/svm_data.py -v 10000 -f 1000`.
  - Number of vectors can be specified with `-v`, and number of features
    can be specified with `-f`. Data will by default go in `data/two` and `data/multi`.
- Prepare data for native benchmarks by running `native/svm_native_data.sh`.

### Run
- All benchmarks must be given the number of threads to run. If this value
  is `-1`, then the number of processing threads will equal the number of
  CPUs available on the system. Otherwise, the benchmark will use the given
  number of threads.
- For KMeans benchmarks, an input directory must be specified.
  The slash must be included at the end of the directory name.
- For all benchmarks, the `batch`, `hostname`, and `env_name`
  arguments are only for bookkeeping purposes and can be replaced with
  placeholders.
- The KMeans predict benchmark has a multiplier argument. An example value
  is 100.

#### Python benchmarks
- Python benchmarks are located in the `python` directory
  `python python/<benchmark>.py <args...>`
- The following benchmarks are available:
  - `distances`: benchmark pairwise distances using `cosine` and `correlation`
    metrics
  - `ridge`: benchmark ridge regression fit and prediction
  - `linear`: benchmark linear regression fit and prediction
  - `kmeans`: benchmark KMeans fit
  - `kmeans_predict`: benchmark KMeans predict
  - `svm_bench`: benchmark two- and multi-class SVM
- A size must be passed in the form `--size M,N` for all benchmarks except SVM
- The number of threads to run must be passed in the form `--core-number T`
  for all benchmarks.
- For KMeans benchmarks, the input directory must be passed in the form
  `--input INPUT_DIR`.
- For SVM benchmarks, the input files must be passed in the form
  `--fileX FILE_X --fileY FILE_Y`.
- For the KMeans predict benchmark, the multiplier must be passed in the form
  `--data-multiplier X`.

#### Native benchmarks
- Binaries are located in the `native/bin` directory.
- Sizes must be specified in `MxN` form.
- The following benchmarks are available:
  - `cosine <batch> <hostname> <env_name> cosine <threads> double <size>`:
    benchmark pairwise distances using `cosine` metric
  - `correlation <batch> <hostname> <env_name> cosine <threads> double <size>`:
    benchmark pairwise distances using `correlation` metric
  - `ridge <batch> <hostname> <env_name> cosine <threads> double <size>`:
    benchmark ridge regression fit and prediction
  - `linear <batch> <hostname> <env_name> cosine <threads> double <size>`:
    benchmark linear regression fit and prediction
  - `kmeans <batch> <hostname> <env_name> cosine <threads> double <size> <input_dir>`:
    benchmark KMeans fit, finding pregenerated input files in `input_dir`
  - `kmeans_predict <batch> <hostname> <env_name> cosine <threads> double <size> <input_dir> <multiplier>`
    benchmark KMeans fit, finding pregenerated input files in `input_dir`.
    A possible value for `multiplier` is 100.
  - `{two,multi}_class_svm --fileX <feature-file> --fileY <label-file> --num-threads <threads>`:
    benchmark two/multi class SVM, using pregenerated feature and label file and using the given
    number of threads
