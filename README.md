
# scikit-learn_bench

This repository contains benchmarks for various implementations of machine learning algorithms.
See benchmark results [here](https://intelpython.github.io/scikit-learn_bench).

## Table of content

* [Prerequisites](#prerequisites)
* [How to create conda environment for benchmarking](#how-to-create-conda-environment-for-benchmarking)
* [Running Python benchmarks with runner script](#running-python-benchmarks-with-runner-script)
* [Supported algorithms](#supported-algorithms)
* [Algorithms parameters](#algorithms-parameters)
* [Config JSON Schema](#config-json-schema)
* [Legacy automatic building and running](#legacy-automatic-building-and-running)

## Prerequisites
- `python` and `scikit-learn` to run python versions
- pandas when using its DataFrame as input data format
- `icc`, `ifort`, `mkl`, `daal` to compile and run native benchmarks

## How to create conda environment for benchmarking

 You must create a suitable conda environment for each framework you want to test.
Below is a list of supported frameworks and a link to instructions how to create appropriate conda environment.

* [**scikit-learn**](https://github.com/PivovarA/scikit-learn_bench/blob/master/sklearn/README.md#how-to-create-conda-environment-for-benchmarking)
* [**daal4py**](https://github.com/PivovarA/scikit-learn_bench/blob/master/daal4py/README.md#how-to-create-conda-environment-for-benchmarking)
* [**cuml**](https://github.com/PivovarA/scikit-learn_bench/blob/master/cuml/README.md#how-to-create-conda-environment-for-benchmarking)
* [**xgboost**](https://github.com/PivovarA/scikit-learn_bench/tree/master/xgboost/README.md#how-to-create-conda-environment-for-benchmarking)

## Running Python benchmarks with runner script

Run `python runner.py --config configs/config_example.json [--output-format json --verbose]` to launch benchmarks.

runner options:
* ``config`` : path to configuration file
* ``dummy-run`` : run configuration parser and datasets generation without benchmarks running
* ``verbose`` : print additional information during benchmarks running
* ``output-format``: *json* or *csv*. Output type of benchmarks to use with their runner

Benchmarks currently support the following frameworks:
* **scikit-learn**
* **daal4py**
* **cuml**
* **xgboost**

To select frameworks just specify them in the configuration file in *general* in *lib*.

The configuration of benchmarks allows you to select the frameworks to run, select datasets for measurements and configure the parameters of the algorithms.

 You can configure benchmarks by editing a config file. Check  [config.json schema](#config-json-schema) for more details.

## Benchmark supported algorithms

| algorithm  | benchmark name | sklearn | daal4py | cuml | xgboost |
|---|---|---|---|---|---|
|**[DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)**|dbscan|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)**|df_clfs|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)**|df_regr|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[pairwise_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)**|distances|:white_check_mark:|:white_check_mark:|:x:|:x:|
|**[KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)**|kmeans|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)**|knn_clsf|:white_check_mark:|:x:|:white_check_mark:|:x:|
|**[LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)**|linear|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**|log_reg|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)**|pca|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)**|ridge|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)**|svm|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)**|train_test_split|:white_check_mark:|:x:|:white_check_mark:|:x:|
|**[GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)**|gbt|:x:|:x:|:x:|:white_check_mark:|
|**[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)**|gbt|:x:|:x:|:x:|:white_check_mark:|

##  Algorithms parameters

Benchmarks can be launched for each algorithm separately.
To do this just go to the directory with the benchmark

`cd <framework>`

and run the following command

`python <benchmark_file> --dataset-name <path to the dataset> <other algorithm parameters>`

The list of supported parameters for each algorithm you can find here:

* [**scikit-learn**](https://github.com/PivovarA/scikit-learn_bench/blob/master/sklearn/README.md#algorithms-parameters)
* [**daal4py**](https://github.com/PivovarA/scikit-learn_bench/blob/master/daal4py/README.md#algorithms-parameters)
* [**cuml**](https://github.com/PivovarA/scikit-learn_bench/blob/master/cuml/README.md#algorithms-parameters)
* [**xgboost**](https://github.com/PivovarA/scikit-learn_bench/tree/master/xgboost/README.md#algorithms-parameters)

##  Config JSON Schema

Benchmarks are configured by editing the config.json file.
In the benchmarks settings can be configured: some algorithms parameters, datasets, a list of frameworks to use, and the usage of some environment variables.
Check [config tab](https://github.com/PivovarA/scikit-learn_bench/blob/master/configs/README.md) for more details.

## Legacy automatic building and running
- Run `make`. This will generate data, compile benchmarks, and run them.
  - To run only scikit-learn benchmarks, use `make sklearn`.
  - To run only native benchmarks, use `make native`.
  - To run only daal4py benchmarks, use `make daal4py`.
  - To run a specific implementation of a specific benchmark,
    directly request the corresponding file: `make output/<impl>/<bench>.out`.
  - If you have activated a conda environment, the build will use daal from
    the conda environment, if available.
