# Benchmarking config specification

## Config structure

Benchmarking config files are written in JSON format and have few reserved keys:
 - `PARAMETERS_SETS` with benchmark parameters in each set
 - `TEMPLATES` map where each named instance combines parameters sets and template-specific parameters
 - `INCLUDE` list which specifies parameters sets to include in template

Formatting example:
```json
{
    "PARAMETERS_SETS": {
        "parameters_set_name_0": Dict or List[Dict] of any JSON-serializable with any level of nesting,
        ...
    },
    "TEMPLATES": {
        "template_name_0": {
            "INCLUDE": ["parameters_set_name_0", ...],
            Dict of any JSON-serializable with any level of nesting overwriting parameter sets
        },
        ...
    }
}
```

## Common parameters

| Parameter keys | Default value | Choices | Description |
|:---------------|:--------------|:--------|:------------|
| `bench`:`taskset` | None |  | Value for `-c` argument of `taskset` utility used over benchmark subcommand. |
| `bench`:`vtune_profiling` | None |  | Type of Intel(R) VTune(TM) profiling. Supported only for Linux systems. |
| `bench`:`vtune_results_directory` | `vtune_results` |  | Directory path to store Intel(R) VTune(TM) results. |
| `bench`:`n_runs` | `10` |  | Number of runs for measured entity. |
| `bench`:`time_limit` | `3600` |  | Time limit in seconds before benchmark early stopping. |
| `data`:`cache_directory` | `data_cache` |  | Directory path to store cached datasets for fast loading. |
| `data`:`raw_cache_directory` | `data`:`cache_directory` + "raw" |  | Directory path to store downloaded raw datasets. |
| `data`:`dataset` | None |  | Name of dataset to use from implemented dataset loaders. |
| `data`:`source` | None | `fetch_openml`, `make_regression`, `make_classification`, `make_blobs` | Data source to use for loading or synthetic generation. |
| `data`:`id` | None |  | OpenML data id for `fetch_openml` source. |
| `data`:`preprocessing_kwargs`:`replace_nan` | `median` | `median`, `mean` | Value to replace NaNs in preprocessed data. |
| `data`:`preprocessing_kwargs`:`category_encoding` | `ordinal` | `ordinal`, `onehot`, `drop`, `ignore` | How to encode categorical features in preprocessed data. |
| `data`:`preprocessing_kwargs`:`normalize` | False |  | Enables normalization of preprocessed data. |
| `data`:`preprocessing_kwargs`:`forse_for_sparse` | True |  | Forces preprocessing for sparse data formats. |
| `data`:`split_kwargs` | Empty `dict` or default split from dataset description |  | Data split parameters for `train_test_split` function. |
| `data`:`format` | `pandas` | `pandas`, `numpy` | Data format to use in benchmark. |
| `data`:`order` | `F` | `C`, `F` | Data order to use in benchmark: contiguous(C) or Fortran. |
| `data`:`dtype` | `float64` |  | Data type to use in benchmark. |
| `algorithm`:`library` | None |  | Python module containing measured entity (class or function). |
| `algorithm`:`device` | `default` | `default`, `cpu`, `gpu` | Selected device for computation. |

## Benchmark-specific parameters

### `sklearn_estimator`

| Parameter keys | Default value | Choices | Description |
|:---------------|:--------------|:--------|:------------|
| `algorithm`:`estimator` | None |  | Name of measured estimator. |
| `algorithm`:`estimator_params` | Empty `dict` |  | Parameters for estimator constructor. |
| `algorithm`:`online_inference_mode` | False |  | Enables online mode for inference methods of estimator (separate call for each sample). |
| `algorithm`:`sklearn_context` | None |  | Parameters for sklearn's config_context used over estimator. |
| `algorithm`:`sklearnex_context` | None |  | Parameters for sklearnex's config_context used over estimator. Updated by `sklearn_context` if set. |
| `bench`:`ensure_sklearnex_patching` | True |  | If True, warns about sklearnex patching failures. |

### `custom_function`

| Parameter keys | Default value | Choices | Description |
|:---------------|:--------------|:--------|:------------|
| `algorithm`:`function` | None |  | Name of measured function. |
| `algorithm`:`args_order` | `x_train\|y_train` | Any in format `{subset_0}\|..\|{subset_n}` | Arguments order for measured function. |
| `algorithm`:`kwargs` | Empty `dict` |  | Positional arguments for measured function. |

## Special values

Some parameters might be defined as specific from other parameters or properties with `[SPECIAL_VALUE]` prefix in string value:
```json
... "estimator_params": { "n_jobs": "[SPECIAL_VALUE]physical_cpus" } ...
... "generation_kwargs": { "n_informative": "[SPECIAL_VALUE]0.5" } ...
```

List of available special values:

| Parameter keys | Benchmark type[s] | Special value | Description |
|:---------------|:------------------|:--------------|:------------|
| `data`:`dataset` | all | `all_named` | Sets datasets to use as list of all named datasets available in loaders. |
| `data`:`generation_kwargs`:`n_informative` | all | *float* value in [0, 1] range | Sets datasets to use as list of all named datasets available in loaders. |
| `bench`:`taskset` | all | specification of numa nodes in `numa:{numa_node_0}[\|{numa_node_1}...]` format | Sets CPUs affinity using `taskset` utility. |
| `algorithm`:`estimator_params`:`n_jobs` | sklearn_estimator | `physical_cpus`, `logical_cpus` or ratio of previous ones in format `{type}_cpus:{ratio}` where `ratio` is float | Sets `n_jobs` parameter to number of physical/logical cpus or ratio of them for estimator. |
| `algorithm`:`estimator_params`:`scale_pos_weight` | sklearn_estimator | `auto` | Sets `scale_pos_weight` parameter to `sum(negative instances) / sum(positive instances)` value for estimator. |
| `algorithm`:`estimator_params`:`n_clusters` | sklearn_estimator | `auto` | Sets `n_clusters` parameter to number of clusters or classes from dataset description for estimator. |
| `algorithm`:`estimator_params`:`eps` | sklearn_estimator | `distances_quantile:{quantile}` format where quantile is *float* value in [0, 1] range | Computes `eps` parameter as quantile value of distances in `x_train` matrix for estimator. |

## Range of values

Some parameters might be defined as range of values with `[RANGE]` prefix in string value:
```json
... "generation_kwargs": {"n_features": "[RANGE]pow:2:5:6"} ...
```

Supported ranges:

 - `add:start{int}:end{int}:step{int}` - Arithmetic progression (Sequence: start + step * i <= end)
 - `mul:current{int}:end{int}:step{int}` - Geometric progression (Sequence: current * step <= end)
 - `pow:base{int}:start{int}:end{int}[:step{int}=1]` - Powers of base number
