# Data Processing and Storage in Benchmarks

Data handling steps:
1. Load data:
   - If not cached: download/generate dataset and put it in raw and/or usual cache
   - If cached: load from cached files
2. Split data into subsets if requested
3. Convert to requested form (data type, format, order, etc.)

Existing data sources:
 - Synthetic data from sklearn
 - OpenML datasets
 - Custom loaders for named datasets
 - User-provided datasets in compatible format

## Data Caching

There are two levels of caching with corresponding directories: `raw cache` for files downloaded from external sources, and just `cache` for files applicable for fast-loading in benchmarks.

Each dataset has few associated files in usual `cache`: data component files (`x`, `y`, `weights`, etc.) and JSON file with dataset properties (number of classes, clusters, default split arguments).
For example:
```
data_cache/
...
├── mnist.json
├── mnist_x.parq
├── mnist_y.npz
...
```

Cached file formats:
| Format | File extension | Associated Python types | Comment |
| --- | --- | --- | --- |
| [Parquet](https://parquet.apache.org) | `.parq` | pandas.DataFrame |  |
| Numpy uncompressed binary dense data | `.npz` | numpy.ndarray, pandas.Series | Data is stored under `arr_0` name |
| Numpy uncompressed binary CSR data | `.csr.npz` | scipy.sparse.csr_matrix | Data is stored under `data`, `indices` and `indptr` names |

## How to Modify Dataset for Compatibility with Scikit-learn_bench

In order to reuse an existing dataset in scikit-learn_bench, you need to convert its file(s) into compatible format for dataset cache loader.

Cached dataset consist of few files:
- `{dataset name}.json` file which store required and optional dataset information
- `{dataset name}_{data component name}.{data component extension}` files which store dataset components (data, labels, etc.)

Example of `{dataset name}.json`:
```json
{"n_classes": 2, "default_split": {"test_size": 0.2, "random_state": 11}}
```

`n_classes` property in a dataset info file is *required* for classification datasets.

Currently, `x` (data) and `y` (labels) are the only supported and *required* data components.

Scikit-learn_bench-compatible dataset should be stored in `data:cache_directory` (`${PWD}/data_cache` or `{repository root}/data_cache` by default).

You can specify created compatible dataset in config files the same way as datasets explicitly registered in scikit-learn_bench using its name:
```json
{
    "data": {
        "dataset": "{dataset name}"
    }
}
```

---
[Documentation tree](../../README.md#-documentation)
