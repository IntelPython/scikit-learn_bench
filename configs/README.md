# Configs

Benchmarking cases in `scikit-learn_bench` are defined by configuration files and stored in the `configs` directory of the repository.

The configuration file (config) defines:
 - Measurement and profiling parameters
 - Library and algorithm to use
 - Algorithm-specific parameters
 - Data to use as input of the algorithm

Configs are split into subdirectories and files by benchmark scope and algorithm.

# Benchmarking Config Scopes

| Scope (Folder) | Description    |
|:---------------|:---------------|
| `common` | Defines common parameters for other scopes |
| `experiments` | Configurations for specific performance-profiling experiments |
| `regular` | Configurations used to regularly track performance changes |
| `weekly` | Configurations with high-load cases used to track performance changes at longer intervals |
| `spmd` | Configurations used to track the performance of SPMD algorithms |
| `testing` | Configurations used in testing `scikit-learn_bench` |

# Benchmarking Config Specification

Refer to [`Benchmarking Config Specification`](BENCH-CONFIG-SPEC.md) for the details how to read and write benchmarking configs in `scikit-learn_bench`.

---
[Documentation tree](../README.md#-documentation)
