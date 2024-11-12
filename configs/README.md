# Configs

Benchmarking cases in `scikit-learn_bench` are defined by configuration files and stored in the `configs` directory of the repository.

The configuration file (config) defines:
 - Measurement and profiling parameters
 - Library and algorithm to use
 - Algorithm-specific parameters
 - Data to use as input of the algorithm

Configs are split into subdirectories and files by benchmark scope and algorithm.

# Benchmarking Configs Specification

## Config Structure

Benchmark config files are written in JSON format and have a few reserved keys:
 - `INCLUDE` - Other configuration files whose parameter sets to include
 - `PARAMETERS_SETS` - Benchmark parameters within each set
 - `TEMPLATES` - List different setups with parameters sets template-specific parameters
 - `SETS` - List parameters sets to include in the template

Configs heavily utilize lists of scalar values and dictionaries to avoid duplication of cases.

Formatting specification:
```json
{
    "INCLUDE": [
        "another_config_file_path_0"
        ...
    ]
    "PARAMETERS_SETS": {
        "parameters_set_name_0": Dict or List[Dict] of any JSON-serializable with any level of nesting,
        ...
    },
    "TEMPLATES": {
        "template_name_0": {
            "SETS": ["parameters_set_name_0", ...],
            Dict of any JSON-serializable with any level of nesting overwriting parameter sets
        },
        ...
    }
}
```

Example
```json
{
    "PARAMETERS_SETS": {
        "estimator parameters": {
            "algorithm": {
                "estimator": "LinearRegression",
                "estimator_params": {
                    "fit_intercept": false
                }
            }
        },
        "regression data": {
            "data": [
                { "source": "fetch_openml", "id": 1430 },
                { "dataset": "california_housing" }
            ]
        }
    },
    "TEMPLATES": {
        "linear regression": {
            "SETS": ["estimator parameters", "regression data"],
            "algorithm": {
                "library": ["sklearn", "sklearnex", "cuml"]
            }
        }
    }
}
```

## Common Parameters

Configs have the three highest parameter keys:
 - `bench` - Specifies a workflow of the benchmark, such as parameters of measurement or profiling
 - `algorithm` - Specifies measured entity parameters
 - `data` - Specifies data parameters to use

| Parameter keys | Default value | Choices | Description |
|:---------------|:--------------|:--------|:------------|
|<h3>Benchmark workflow parameters</h3>||||
| `bench`:`taskset` | None |  | Value for `-c` argument of `taskset` utility used over benchmark subcommand. |
| `bench`:`vtune_profiling` | None |  | Analysis type for `collect` argument of Intel(R) VTune* Profiler tool. Linux* OS only. |
| `bench`:`vtune_results_directory` | `_vtune_results` |  | Directory path to store Intel(R) VTune* Profiler results. |
| `bench`:`n_runs` | `10` |  | Number of runs for measured entity. |
| `bench`:`time_limit` | `3600` |  | Time limit in seconds before the benchmark early stop. |
| `bench`:`distributor` | None | None, `mpi` | Library used to handle distributed algorithm. |
| `bench`:`mpi_params` | Empty dict |  | Parameters for `mpirun` command of MPI library. |
|<h3>Data parameters</h3>||||
| `data`:`cache_directory` | `data_cache` |  | Directory path to store cached datasets for fast loading. |
| `data`:`raw_cache_directory` | `data`:`cache_directory` + "raw" |  | Directory path to store downloaded raw datasets. |
| `data`:`dataset` | None |  | Name of dataset to use from implemented dataset loaders. |
| `data`:`source` | None | `fetch_openml`, `make_regression`, `make_classification`, `make_blobs` | Data source to use for loading or synthetic generation. |
| `data`:`id` | None |  | OpenML data id for `fetch_openml` source. |
| `data`:`preprocessing_kwargs`:`replace_nan` | `median` | `median`, `mean` | Value to replace NaNs in preprocessed data. |
| `data`:`preprocessing_kwargs`:`category_encoding` | `ordinal` | `ordinal`, `onehot`, `drop`, `ignore` | How to encode categorical features in preprocessed data. |
| `data`:`preprocessing_kwargs`:`normalize` | False |  | Enables normalization of preprocessed data. |
| `data`:`preprocessing_kwargs`:`force_for_sparse` | True |  | Forces preprocessing for sparse data formats. |
| `data`:`split_kwargs` | Empty `dict` or default split from dataset description |  | Data split parameters for `train_test_split` function. |
| `data`:`format` | `pandas` | `pandas`, `numpy`, `cudf` | Data format to use in benchmark. |
| `data`:`order` | `F` | `C`, `F` | Data order to use in benchmark: contiguous(C) or Fortran. |
| `data`:`dtype` | `float64` |  | Data type to use in benchmark. |
| `data`:`distributed_split` | None | None, `rank_based` | Split type used to distribute data between machines in distributed algorithm. `None` type means usage of all data without split on all machines. `rank_based` type splits the data equally between machines with split sequence based on rank id from MPI. |
|<h3>Algorithm parameters</h3>||||
| `algorithm`:`library` | None |  | Python module containing measured entity (class or function). |
| `algorithm`:`device` | `default` | `default`, `cpu`, `gpu` | Device selected for computation. |

## Benchmark-Specific Parameters

### `Scikit-learn Estimator`

| Parameter keys | Default value | Choices | Description |
|:---------------|:--------------|:--------|:------------|
| `algorithm`:`estimator` | None |  | Name of measured estimator. |
| `algorithm`:`estimator_params` | Empty `dict` |  | Parameters for estimator constructor. |
| `algorithm`:`online_inference_mode` | False |  | Enables online mode for inference methods of estimator (separate call for each sample). |
| `algorithm`:`sklearn_context` | None |  | Parameters for sklearn `config_context` used over estimator. |
| `algorithm`:`sklearnex_context` | None |  | Parameters for sklearnex `config_context` used over estimator. Updated by `sklearn_context` if set. |
| `bench`:`ensure_sklearnex_patching` | True |  | If True, warns about sklearnex patching failures. |

### `Function`

| Parameter keys | Default value | Choices | Description |
|:---------------|:--------------|:--------|:------------|
| `algorithm`:`function` | None |  | Name of measured function. |
| `algorithm`:`args_order` | `x_train\|y_train` | Any in format `{subset_0}\|..\|{subset_n}` | Arguments order for measured function. |
| `algorithm`:`kwargs` | Empty `dict` |  | Named arguments for measured function. |

## Special Value

You can define some parameters as specific from other parameters or properties with `[SPECIAL_VALUE]` prefix in string value:
```json
... "estimator_params": { "n_jobs": "[SPECIAL_VALUE]physical_cpus" } ...
... "generation_kwargs": { "n_informative": "[SPECIAL_VALUE]0.5" } ...
```

List of available special values:

| Parameter keys | Benchmark type[s] | Special value | Description |
|:---------------|:------------------|:--------------|:------------|
| `data`:`dataset` | all | `all_named` | Sets datasets to use as list of all named datasets available in loaders. |
| `data`:`generation_kwargs`:`n_informative` | all | *float* value in [0, 1] range | Sets datasets to use as list of all named datasets available in loaders. |
| `bench`:`taskset` | all | Specification of numa nodes in `numa:{numa_node_0}[\|{numa_node_1}...]` format | Sets CPUs affinity using `taskset` utility. |
| `algorithm`:`estimator_params`:`n_jobs` | sklearn_estimator | `physical_cpus`, `logical_cpus`, or ratio of previous ones in format `{type}_cpus:{ratio}` where `ratio` is float | Sets `n_jobs` parameter to a number of physical/logical CPUs or ratio of them for an estimator. |
| `algorithm`:`estimator_params`:`scale_pos_weight` | sklearn_estimator | `auto` | Sets `scale_pos_weight` parameter to `sum(negative instances) / sum(positive instances)` value for estimator. |
| `algorithm`:`estimator_params`:`n_clusters` | sklearn_estimator | `auto` | Sets `n_clusters` parameter to number of clusters or classes from dataset description for estimator. |
| `algorithm`:`estimator_params`:`eps` | sklearn_estimator | `distances_quantile:{quantile}` format where quantile is *float* value in [0, 1] range | Computes `eps` parameter as quantile value of distances in `x_train` matrix for estimator. |

## Range of Values

You can define some parameters as a range of values with the `[RANGE]` prefix in string value:
```json
... "generation_kwargs": {"n_features": "[RANGE]pow:2:5:6"} ...
```

Supported ranges:

 - `add:start{int}:end{int}:step{int}` - Arithmetic progression (Sequence: start + step * i <= end)
 - `mul:current{int}:end{int}:step{int}` - Geometric progression (Sequence: current * step <= end)
 - `pow:base{int}:start{int}:end{int}[:step{int}=1]` - Powers of base number

## Removal of Values

You can remove specific parameter from subset of cases when stacking parameters sets using `[REMOVE]` parameter value:

```json
... "estimator_params": { "n_jobs": "[REMOVE]" } ...
```

---
[Documentation tree](../README.md#-documentation)
