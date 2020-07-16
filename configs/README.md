##  Config JSON Schema

Configure benchmarks by editing the `config.json` file.
You can configure some algorithm parameters, datasets, a list of frameworks to use, and the usage of some environment variables.
Refer to the tables below for descriptions of all fields in the configuration file.

- [Root Config Object](#root-config-object)
- [Common Object](#common-object)
- [Case Object](#case-object)
- [Dataset Object](#dataset-object)
- [Training Object](#training-object)
- [Testing Object](#testing-object)

###  Root Config Object
| Field Name  | Type | Description |
| ----- | ---- |------------ |
|omp_env| array[string] | For xgboost only. Specify an environment variable to set the number of omp threads |
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
|n_classes| int | For *synthetic* data and for *classification* type only. The number of classes (or labels) of the classification problem |
|n_clusters| int | For *synthetic* data and for *blobs* type only. The number of centers to generate |
|n_features| int | **REQUIRED**  For *synthetic* data only. The number of features to generate |
|name| string | Name of dataset |
|training| [Training Object](#training-object) | **REQUIRED** algorithm parameters. a list of supported parameters can be found here |
|testing| [Testing Object](#testing-object) | **REQUIRED** algorithm parameters. a list of supported parameters can be found here |

###  Training Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
| n_samples | int | The total number of the training points |
| x | str | The path to the training samples |
| y | str | The path to the training labels |

###  Testing Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
| n_samples | int | The total number of the testing points |
| x | str | The path to the testing samples |
| y | str | The path to the testing labels |
