# scikit-learn_bench

This repository contains benchmarks for various implementations of machine learning algorithms.
[Here](https://intelpython.github.io/scikit-learn_bench) you can find benchmark results for optimizations to scikit-learn in the Intel&reg; Distribution for Python*.

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
`conda create -n skl_bench -c intel python=3.7 scikit-learn pandas`

## Running Python benchmarks with runner script

Rub `python runner.py --config config_example.json [--output-format json --verbose]` to launch benchmarks.

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

## Supported algorithms

| algorithm  | benchmark name | sklearn | daal4py | cuml | xgboost |
|---|---|---|---|---|---|
|**[DBSCAN]([https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html))**|dbscan|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[RandomForestClassifier]([https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))**|df_clfs|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[RandomForestRegressor]([https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html))**|df_regr|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[pairwise_distances]([https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html))**|distances|:white_check_mark:|:white_check_mark:|:x:|:x:|
|**[KMeans]([https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html))**|kmeans|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[KNeighborsClassifier]([https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html))**|knn_clsf|:white_check_mark:|:x:|:white_check_mark:|:x:|
|**[LinearRegression]([https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html))**|linear|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[LogisticRegression]([https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))**|log_reg|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[PCA]([https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html))**|pca|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[Ridge]([https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html))**|ridge|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[SVC]([https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))**|svm|:white_check_mark:|:white_check_mark:|:white_check_mark:|:x:|
|**[train_test_split]([https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html))**|train_test_split|:white_check_mark:|:x:|:white_check_mark:|:x:|
|**[GradientBoostingClassifier]([https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html))**|gbt|:x:|:x:|:x:|:white_check_mark:|
|**[GradientBoostingRegressor]([https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html))**|gbt|:x:|:x:|:x:|:white_check_mark:|

##  Algorithms parameters

Benchmarks can be launched for each algorithm separately. The list of supported parameters for each algorithm provided below.

#### General
| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
|num-threads|int|-1| Number of threads to use|
|arch|str|?|achine architecture, for bookkeeping|
|batch|str|?|Batch ID, for bookkeeping|
|prefix|str|sklearn|Prefix string, for bookkeeping|
|header|action|False|Output CSV header|
|verbose|action|False|Output extra debug messages|
|data-format|str|numpy|Data formats: *numpy*, *pandas* or *cudf*|
|data-order|str|C|Data order: C (row-major, default) or F (column-major)|
|dtype|np.dtype|np.float64|Data type: *float64* (default) or *float32*|
|check-finiteness|action|False|Check finiteness in sklearn input check(disabled by default)|
|output-format|str|csv|Output format: *csv* (default) or *json*'|
|time-method|str|mean_min|Method used for time mesurements|
|box-filter-measurements|int|100|Maximum number of measurements in box filter|
|inner-loops|int|100|Maximum inner loop iterations. (we take the mean over inner iterations)|
|outer-loops|int|100|Maximum outer loop iterations. (we take the min over outer iterations)|
|time-limit|float|10|Target time to spend to benchmark|
|goal-outer-loops|int|10|Number of outer loops to aim while automatically picking number of inner loops. If zero, do not automatically decide number of inner loops.|
|seed|int|12345|Seed to pass as random_state|
|dataset-name|str|None|Dataset name|


#### DBSCAN
| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| epsilon | float | 10 | Radius of neighborhood of a point|
| min_samples | int | 5 | The minimum number of samples required in a 'neighborhood to consider a point a core point |

#### RandomForestClassifier

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| criterion | str | gini | *gini* or *entropy*. The function to measure the quality of a split |
| num-trees | int | 100 | Number of trees in the forest |
| max-features | float_or_int | None | Upper bound on features used at each split |
| max-depth | int | None | Upper bound on depth of constructed trees |
| min-samples-split | float_or_int | 2 | Minimum samples number for node splitting |
| max-leaf-nodes | int | None | Maximum leaf nodes per tree |
| min-impurity-decrease | float | 0 | Needed impurity decrease for node splitting |
| no-bootstrap | store_false | True | Don't control bootstraping |
| use-sklearn-class | store_true |  | Force use of sklearn.ensemble.RandomForestClassifier |

#### RandomForestRegressor

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| criterion | str | gini | *gini* or *entropy*. The function to measure the quality of a split |
| num-trees | int | 100 | Number of trees in the forest |
| max-features | float_or_int | None | Upper bound on features used at each split |
| max-depth | int | None | Upper bound on depth of constructed trees |
| min-samples-split | float_or_int | 2 | Minimum samples number for node splitting |
| max-leaf-nodes | int | None | Maximum leaf nodes per tree |
| min-impurity-decrease | float | 0 | Needed impurity decrease for node splitting |
| no-bootstrap | action | True | Don't control bootstraping |
| use-sklearn-class | action |  | Force use of sklearn.ensemble.RandomForestClassifier |

#### pairwise_distances

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| metric | str | cosine | *cosine* or *correlation* Metric to test for pairwise distances |

#### KMeans

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| init | str |  | Initial clusters |
| tol | float | 0 | Absolute threshold |
| maxiter | inte | 100 | Maximum number of iterations |
| n-clusters | int |  | Number of clusters |

#### KNeighborsClassifier

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| n-neighbors | int | 5 | Number of neighbors to use |
| weights | str | uniform | Weight function used in prediction |
| method | str | brute | Algorithm used to compute the nearest neighbors |
| metric | str | euclidean | Distance metric to use |

#### LinearRegression

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| no-fit-intercept | action | True | Don't fit intercept (assume data already centered) |

#### LogisticRegression

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| no-fit-intercept | action | True | Don't fit intercept|
| multiclass | str | auto | *auto*, *ovr* or *multinomial*. How to treat multi class data|
| solver | str | lbfgs | *lbfgs*, *newton-cg* or *saga*. Solver to use|
| maxiter | int | 100 | Maximum iterations for the iterative solver |
| C | float | 1.0 | Regularization parameter |
| tol | float | None | Tolerance for solver |

#### PCA

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| svd-solver | str | daal | *daal*, *full*. SVD solver to use |
| n-components | int | None | Number of components to find |
| whiten | action | False | Perform whitening |

#### Ridge

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| no-fit-intercept | action | True | Don't fit intercept (assume data already centered) |
| solver | str | auto | Solver used for training |
| alpha | float | 1.0 | Regularization strength |

#### SVC

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| C | float | 0.01 | SVM slack parameter |
| kernel | str | linear | *linear* or *rbf*. SVM kernel function |
| gamma | float | None | Parameter for kernel="rbf" |
| maxiter | int | 2000 | Maximum iterations for the iterative solver |
| max-cache-size | int | 64 | Maximum cache size for SVM. |
| tol | float | 1e-16 | Tolerance passed to sklearn.svm.SVC |
| no-shrinking | action | True | Don't use shrinking heuristic |

#### train_test_split

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| train-size | float | 0.75 | Size of training subset |
| test-size | float | 0.25 | Size of testing subset |
| do-not-shuffle | action | False | Do not perform data shuffle before splitting |
| include-y | action | False | Include label (Y) in splitting |
| rng | str | None | *MT19937*, *SFMT19937*, *MT2203*, *R250*, *WH*, *MCG31*, *MCG59*, *MRG32K3A*, *PHILOX4X32X10*, *NONDETERM* or None. Random numbers generator for shuffling.(only for IDP scikit-learn)|


##  Config JSON Schema

Benchmarks are configured by editing the config.json file.
In the benchmarks settings can be configured: some algorithms parameters, datasets, a list of frameworks to use, and the usage of some environment variables.
Below is a list and description of all fields in the configuration file.

###  Root Config Object
| Field Name  | Type | Description |
| ----- | ---- |------------ |
|omp_env| array[string] | for xgboost only. Specify an environment variable to set the number of omp threads |
|common| [Common Object](#common-object)| **REQUIRED** common benchmarks setting: frameworks and input data settings |
|cases| array[[Case Object](#case-object)] | **REQUIRED**  list of algorithms, their parameters and training data |

###  Common Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
|lib| array[string] | **REQUIRED** list of test frameworks. It can be *sklearn*, *daal4py*, *cuml* or *xgboost* |
|data-format| array[string] | **REQUIRED** input data format. Data formats: *numpy*, *pandas* or *cudf* |
|data-order| array[string] | **REQUIRED**  input data order. Data order: *C* (row-major, default) or *F* (column-major) |
|dtype| array[string] | **REQUIRED**  input data type. Data type: *float64* (default) or *float32* |
|check-finitness| array[] | Check finiteness in sklearn input check(disabled by default) |

###  Case Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
|lib| array[string] | **REQUIRED** list of test frameworks. It can be *sklearn*, *daal4py*, *cuml* or *xgboost*|
|algorithm| string | **REQUIRED** benchmark name |
|dataset| array[[Dataset Object](#dataset-object)] | **REQUIRED**  input data specifications. |
|benchmark parameters| array[Any] | **REQUIRED** algorithm parameters. a list of supported parameters can be found here |

###  Dataset Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
|source| string | **REQUIRED** data source. It can be *synthetic* or *csv* |
|type| string | **REQUIRED**  for synthetic data only. The type of task for which the dataset is generated. It can be *classification*, *blobs* or *regression* |
|n_classes| int | for *synthetic* data and for *classification* type only. The number of classes (or labels) of the classification problem |
|n_clusters| int | for *synthetic* data and for *blobs* type only. The number of centers to generate |
|n_features| int | **REQUIRED**  For *synthetic* data only. The number of features to generate |
|name| string | Name of dataset |
|training| [Training Object](#training-object) | **REQUIRED** algorithm parameters. a list of supported parameters can be found here |
|testing| [Testing Object](#testing-object) | **REQUIRED** algorithm parameters. a list of supported parameters can be found here |

###  Training Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
| n_samples | int | the total number of training points |
| x | str | path to training samples |
| y | str | path to training labels |

###  Testing Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
| n_samples | int | the total number of testing points |
| x | str | path to testing samples |
| y | str | path to testing labels |

## Legacy automatic building and running
- Run `make`. This will generate data, compile benchmarks, and run them.
  - To run only scikit-learn benchmarks, use `make sklearn`.
  - To run only native benchmarks, use `make native`.
  - To run only daal4py benchmarks, use `make daal4py`.
  - To run a specific implementation of a specific benchmark,
    directly request the corresponding file: `make output/<impl>/<bench>.out`.
  - If you have activated a conda environment, the build will use daal from
    the conda environment, if available.
