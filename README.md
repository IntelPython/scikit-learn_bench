
# Machine Learning Benchmarks

[![Build Status](https://dev.azure.com/daal/scikit-learn_bench/_apis/build/status/IntelPython.scikit-learn_bench?branchName=master)](https://dev.azure.com/daal/scikit-learn_bench/_build/latest?definitionId=8&branchName=master)

**Machine Learning Benchmarks** contains implementations of machine learning algorithms
across data analytics frameworks.  Scikit-learn_bench can be extended to add new frameworks
and algorithms. It currently support the [scikit-learn](https://scikit-learn.org/),
[DAAL4PY](https://intelpython.github.io/daal4py/), [cuML](https://github.com/rapidsai/cuml),
and [XGBoost](https://github.com/dmlc/xgboost) frameworks for commonly used
[machine learning algorithms](#supported-algorithms).

## Follow us on Medium

We publish blogs on Medium, so [follow us](https://medium.com/intel-analytics-software/tagged/machine-learning) to learn tips and tricks for more efficient data analysis. Here are our latest blogs:

- [Intel Gives Scikit-Learn the Performance Boost Data Scientists Need](https://medium.com/intel-analytics-software/intel-gives-scikit-learn-the-performance-boost-data-scientists-need-42eb47c80b18)
- [From Hours to Minutes: 600x Faster SVM](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)
- [Improve the Performance of XGBoost and LightGBM Inference](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Kaggle Challenges Using Intel AI Analytics Toolkit](https://medium.com/intel-analytics-software/accelerate-kaggle-challenges-using-intel-ai-analytics-toolkit-beb148f66d5a)
- [Accelerate Your scikit-learn Applications](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Optimizing XGBoost Training Performance](https://medium.com/intel-analytics-software/new-optimizations-for-cpu-in-xgboost-1-1-81144ea21115)
- [Accelerate Linear Models for Machine Learning](https://medium.com/intel-analytics-software/accelerating-linear-models-for-machine-learning-5a75ff50a0fe)
- [Accelerate K-Means Clustering](https://medium.com/intel-analytics-software/accelerate-k-means-clustering-6385088788a1)
- [Fast Gradient Boosting Tree Inference](https://medium.com/intel-analytics-software/fast-gradient-boosting-tree-inference-for-intel-xeon-processors-35756f174f55)

## Table of content

- [How to create conda environment for benchmarking](#how-to-create-conda-environment-for-benchmarking)
- [Running Python benchmarks with runner script](#running-python-benchmarks-with-runner-script)
- [Benchmark supported algorithms](#benchmark-supported-algorithms)
- [Intel(R) Extension for Scikit-learn* support](#intelr-extension-for-scikit-learn-support)
- [Algorithms parameters](#algorithms-parameters)

## How to create conda environment for benchmarking

Create a suitable conda environment for each framework to test. Each item in the list below links to instructions to create an appropriate conda environment for the framework.

- [**scikit-learn**](sklearn_bench#how-to-create-conda-environment-for-benchmarking)

```bash
pip install -r sklearn_bench/requirements.txt
# or
conda install -c intel scikit-learn scikit-learn-intelex pandas tqdm
```

- [**daal4py**](daal4py_bench#how-to-create-conda-environment-for-benchmarking)

```bash
conda install -c conda-forge scikit-learn daal4py pandas tqdm
```

- [**cuml**](cuml_bench#how-to-create-conda-environment-for-benchmarking)

```bash
conda install -c rapidsai -c conda-forge cuml pandas cudf tqdm
```

- [**xgboost**](xgboost_bench#how-to-create-conda-environment-for-benchmarking)

```bash
pip install -r xgboost_bench/requirements.txt
# or
conda install -c conda-forge xgboost scikit-learn pandas tqdm
```

## Running Python benchmarks with runner script

Run `python runner.py --configs configs/config_example.json [--output-file result.json --verbose INFO --report]` to launch benchmarks.

Options:

- ``--configs``: specify the path to a configuration file.
- ``--no-intel-optimized``: use Scikit-learn without [Intel(R) Extension for Scikit-learn*](#intelr-extension-for-scikit-learn-support). Now available for [scikit-learn benchmarks](https://github.com/IntelPython/scikit-learn_bench/tree/master/sklearn_bench). By default, the runner uses Intel(R) Extension for Scikit-learn.
- ``--output-file``: output file name for the benchmark result. The default name is `result.json`
- ``--report``: create an Excel report based on benchmark results. The `openpyxl` library is required.
- ``--dummy-run``: run configuration parser and dataset generation without benchmarks running.
- ``--verbose``: *WARNING*, *INFO*, *DEBUG*. print additional information during benchmarks running. Default is *INFO*.

|   Level   |  Description  |
|-----------|---------------|
| *DEBUG*   | etailed information, typically of interest only when diagnosing problems. Usually at this level the logging output is so low level that it’s not useful to users who are not familiar with the software’s internals. |
| *INFO*    | Confirmation that things are working as expected. |
| *WARNING* | An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected. |

Benchmarks currently support the following frameworks:

- **scikit-learn**
- **daal4py**
- **cuml**
- **xgboost**

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

## Intel(R) Extension for Scikit-learn support

When you run scikit-learn benchmarks on CPU, [Intel(R) Extension for Scikit-learn](https://github.com/intel/scikit-learn-intelex) is used by default. Use the ``--no-intel-optimized`` option to run the benchmarks without the extension.

The following benchmarks have a GPU support:

- dbscan
- kmeans
- linear
- log_reg

You may use the [configuration file for these benchmarks](https://github.com/IntelPython/scikit-learn_bench/blob/master/configs/skl_xpu_config.json) to run them on both CPU and GPU.

## Algorithms parameters

You can launch benchmarks for each algorithm separately.
To do this, go to the directory with the benchmark:

```bash
cd <framework>
```

Run the following command:

```bash
python <benchmark_file> --dataset-name <path to the dataset> <other algorithm parameters>
```

The list of supported parameters for each algorithm you can find here:

- [**scikit-learn**](sklearn_bench#algorithms-parameters)
- [**daal4py**](daal4py_bench#algorithms-parameters)
- [**cuml**](cuml_bench#algorithms-parameters)
- [**xgboost**](xgboost_bench#algorithms-parameters)
