# Experimental Configs

`daal4py_svd`: tests performance scalability of `daal4py.svd` algorithm

`hdbscan_parameters`: sweeps the `HDBSCAN` parameter space (metrics, cluster selection, density thresholds, stored centers, dtypes and data formats) over the `sklearn` and `sklearnex` implementations.

`hdbscan_scaling`: tests thread, NUMA, `n_samples` and `n_features` scalability of `HDBSCAN`.

`nearest_neighbors`: tests performance of neighbors search implementations from `sklearnex`, `sklearn`, `raft`, `faiss` and `svs`.
