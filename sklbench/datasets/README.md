# Data Handling in Benchmarks

Data handling steps:
1. Load data:
   - If not cached: download/generate dataset and put it in raw and/or usual cache
   - If cached: load from cached files
2. Split data into subsets if requested
3. Convert to requested form (data type, format, order, etc.)

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
| Format | File extension | Associated Python types |
| --- | --- | --- |
| [Parquet](https://parquet.apache.org) | `.parq` | pandas.DataFrame |
| Numpy uncompressed binary dense data | `.npz` | numpy.ndarray, pandas.Series |
| Numpy uncompressed binary CSR data | `.csr.npz` | scipy.sparse.csr_matrix |

Existing data sources:
 - Synthetic data from sklearn
 - OpenML datasets
 - Custom loaders for named datasets

---
[Documentation tree](../../README.md#-documentation)
