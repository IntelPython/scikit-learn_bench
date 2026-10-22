# Experimental Configs

`daal4py_svd`: tests performance scalability of `daal4py.svd` algorithm

`hdbscan_datasets`: runs `HDBSCAN` over real datasets, synthetic shapes and large tall or wide data, to cover different data situations rather than parameters.

`hdbscan_gpu_boundaries`: compares `HDBSCAN` on CPU and GPU over `n_samples`, `n_features`, dtypes and metrics, up to sizes stock `sklearn` does not finish.

`hdbscan_parameters`: sweeps the `HDBSCAN` parameter space (metrics, cluster selection, density thresholds, stored centers, dtypes and data formats) over the `sklearn` and `sklearnex` implementations.

`hdbscan_scaling`: tests thread, NUMA, `n_samples` and `n_features` scalability of `HDBSCAN`.

`hdbscan_vs_cuml`: compares `sklearnex` `HDBSCAN` on Intel GPU with `cuml` `HDBSCAN` on NVIDIA GPU.

`hdbscan_vs_dbscan`: compares `HDBSCAN` and its `dbscan_clustering` re-cut with `DBSCAN` at the same distance.

`nearest_neighbors`: tests performance of neighbors search implementations from `sklearnex`, `sklearn`, `raft`, `faiss` and `svs`.
