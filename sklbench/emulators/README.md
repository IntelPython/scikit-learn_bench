# Emulators

This part of **scikit-learn_bench** contains emulators - sklearn-like estimators wrapping other non-compliant frameworks' APIs.

Emulators are specified in configs using full module path and emulator name, for example:
```json
{ "library": "sklbench.emulators.svs", "estimator": "NearestNeighbors" }
```

## Emulators list

| Library | Emulator name | Supported methods | Wrapped entity |
| --- | --- | --- | --- |
| Faiss | NearestNeighbors | `fit`, `kneighbors` | `FlatL2`, `IVFFlat` and `IVFPQ` index search. Supports both `cpu` and `gpu` devices. |
| RAFT | NearestNeighbors | `fit`, `kneighbors` | `FlatL2`, `IVFFlat`, `IVFPQ` and `CAGRA` index search. |
| SVS | NearestNeighbors | `fit`, `kneighbors` | `Vamana` index search. |

---
[Documentation tree](../../README.md#-documentation)
