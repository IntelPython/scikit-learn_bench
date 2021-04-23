# Config JSON Schema

Configure benchmarks by editing the `config.json` file.
You can configure some algorithm parameters, datasets, a list of frameworks to use, and the usage of some environment variables.
Refer to the tables below for descriptions of all fields in the configuration file.

- [Root Config Object](#root-config-object)
- [Common Object](#common-object)
- [Case Object](#case-object)
- [Dataset Object](#dataset-object)
- [Training Object](#training-object)
- [Testing Object](#testing-object)

## Root Config Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
|common| [Common Object](#common-object)| **REQUIRED** common benchmarks setting: frameworks and input data settings |
|cases| List[[Case Object](#case-object)] | **REQUIRED**  list of algorithms, their parameters and training data |

## Common Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
|data-format| Union[str, List[str]] | **REQUIRED** Input data format: *numpy*, *pandas*, or *cudf*. |
|data-order| Union[str, List[str]] | **REQUIRED**  Input data order: *C* (row-major, default) or *F* (column-major). |
|dtype| Union[str, List[str]] | **REQUIRED**  Input data type: *float64* (default) or *float32*. |
|check-finitness| List[] | Check finiteness during scikit-learn input check (disabled by default). |
|device| array[string] | For scikit-learn only. The list of devices to run the benchmarks on.<br/>It can be *None* (default, run on CPU without sycl context) or one of the types of sycl devices: *cpu*, *gpu*, *host*.<br/>Refer to [SYCL specification](https://www.khronos.org/files/sycl/sycl-2020-reference-guide.pdf) for details.|

## Case Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
|lib| Union[str, List[str]] | **REQUIRED** A test framework or a list of frameworks. Must be from [*sklearn*, *daal4py*, *cuml*, *xgboost*]. |
|algorithm| string | **REQUIRED** Benchmark file name. |
|dataset| List[[Dataset Object](#dataset-object)] | **REQUIRED**  Input data specifications. |
|**specific algorithm parameters**| Union[int, float, str, List[int], List[float], List[str]] | Other algorithm-specific parameters. The list of supported parameters can be found here. |

### **Important:** feel free to move any parameter from **cases** to **common** section since this parameter is common for all cases

## Dataset Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
|source| string | **REQUIRED** Data source: *synthetic*, *csv*, or *npy*. |
|type| string | **REQUIRED for synthetic data**. The type of task for which the dataset is generated: *classification*, *blobs*, or *regression*. |
|n_classes| int | For *synthetic* data and for *classification* type only. The number of classes (or labels) of the classification problem |
|n_clusters| int | For *synthetic* data and for *blobs* type only. The number of centers to generate |
|n_features| int | **REQUIRED for *synthetic* data**. The number of features to generate. |
|name| string | Name of the dataset. |
|training| [Training Object](#training-object) | **REQUIRED** An object with the paths to the training datasets. |
|testing| [Testing Object](#testing-object) | An object with the paths to the testing datasets. If not provided, the training datasets are used. |

## Training Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
| n_samples | int | **REQUIRED** The total number of the training samples |
| x | str | **REQUIRED** The path to the training samples |
| y | str | **REQUIRED** The path to the training labels |

## Testing Object

| Field Name  | Type | Description |
| ----- | ---- |------------ |
| n_samples | int | **REQUIRED** The total number of the testing samples |
| x | str | **REQUIRED** The path to the testing samples |
| y | str | **REQUIRED** The path to the testing labels |
