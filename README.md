
# scikit-learn_bench

**scikit-learn_bench** benchmarks various implementations of machine learning algorithms across data analytics frameworks.  Scikit-learn_bench can be extended to add new frameworks and algorithms.  It currently support the [scikit-learn](https://scikit-learn.org/), [DAAL4PY](https://intelpython.github.io/daal4py/), [cuML](https://github.com/rapidsai/cuml), and [XGBoost](https://github.com/dmlc/xgboost) frameworks for commonly used [machine learning algorithms](#supported-algorithms).

See benchmark results [here](https://intelpython.github.io/scikit-learn_bench).


## Table of content

* [Prerequisites](#prerequisites)
* [How to create conda environment for benchmarking](#how-to-create-conda-environment-for-benchmarking)
* [How to enable daal4py patching for scikit-learn benchmarks](#how-to-enable-daal4py-patching-for-scikit-learn-benchmarks)
* [Running Python benchmarks with runner script](#running-python-benchmarks-with-runner-script)
* [Supported algorithms](#supported-algorithms)
* [Algorithms parameters](#algorithms-parameters)
* [Legacy automatic building and running](#legacy-automatic-building-and-running)

## Prerequisites
- `python` and `scikit-learn` to run python versions
- pandas when using its DataFrame as input data format
- machine learning frameworks, that you want to test. Check [this item](#how-to-create-conda-environment-for-benchmarking) to get additional information how to set environment.

## How to create conda environment for benchmarking

Create a suitable conda environment for each framework to test. Each item in the list below links to instructions to create an appropriate conda environment for the framework.

* [**scikit-learn**](https://github.com/IntelPython/scikit-learn_bench/blob/master/sklearn/README.md#how-to-create-conda-environment-for-benchmarking)
* [**daal4py**](https://github.com/IntelPython/scikit-learn_bench/blob/master/daal4py/README.md#how-to-create-conda-environment-for-benchmarking)
* [**cuml**](https://github.com/IntelPython/scikit-learn_bench/blob/master/cuml/README.md#how-to-create-conda-environment-for-benchmarking)
* [**xgboost**](https://github.com/IntelPython/scikit-learn_bench/tree/master/xgboost/README.md#how-to-create-conda-environment-for-benchmarking)

## How to enable daal4py patching for scikit-learn benchmarks
Set specific environment variable `export FORCE_DAAL4PY_SKLEARN=YES`

## Running Python benchmarks with runner script

Run `python runner.py --configs configs/config_example.json [--output-format json --verbose]` to launch benchmarks.

runner options:
* ``configs`` : configuration files paths
* ``dummy-run`` : run configuration parser and datasets generation without benchmarks running
* ``verbose`` : print additional information during benchmarks running
* ``output-format``: *json* or *csv*. Output type of benchmarks to use with their runner

Benchmarks currently support the following frameworks:
* **scikit-learn**
* **daal4py**
* **cuml**
* **xgboost**

The configuration of benchmarks allows you to select the frameworks to run, select datasets for measurements and configure the parameters of the algorithms.

 You can configure benchmarks by editing a config file. Check  [config.json schema](https://github.com/IntelPython/scikit-learn_bench/blob/master/configs/README.md) for more details.

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

You can launch benchmarks for each algorithm separately.
To do this, go to the directory with the benchmark:

    cd <framework>

Run the following command:

    python <benchmark_file> --dataset-name <path to the dataset> <other algorithm parameters>

The list of supported parameters for each algorithm you can find here:

* [**scikit-learn**](https://github.com/IntelPython/scikit-learn_bench/blob/master/sklearn/README.md#algorithms-parameters)
* [**daal4py**](https://github.com/IntelPython/scikit-learn_bench/blob/master/daal4py/README.md#algorithms-parameters)
* [**cuml**](https://github.com/IntelPython/scikit-learn_bench/blob/master/cuml/README.md#algorithms-parameters)
* [**xgboost**](https://github.com/IntelPython/scikit-learn_bench/tree/master/xgboost/README.md#algorithms-parameters)
