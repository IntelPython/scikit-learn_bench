# Benchmarks

# TODO: ind bench is python script

```mermaid
flowchart LR
    A["Benchmarking case parameters\n[JSON-formatted string]"] --> C[Individual benchmark]
    B["Benchmarking case filters\n[JSON-formatted string]"] --> C
    C --> D["Raw results with parameters and metrics\n[JSON-formatted string]"]

    classDef inputOutputStyle fill:#44b,color:white,stroke-width:2px,stroke:white;
    classDef benchStyle font-size:x-large
    class A inputOutputStyle
    class B inputOutputStyle
    class D inputOutputStyle
    class C benchStyle
```

## `Scikit-learn estimator` benchmark

Benchmark workflow:
 - Load estimator from specified library by recursive module search
 - Load data with common loader function
 - Assign special values which requires estimator/data to be loaded
 - Get sklearn/sklearnex context, estimator parameters, running parameters
 - Measure required estimator methods
 - Combine metrics and parameters into output

See benchmark-specific config parameters in corresponding [`sklearn_estimator` section ](../../configs/BENCH-CONFIG-SPEC.md#benchmark-specific-parameters) of benchmarking config specification.

## `Function` benchmark

Benchmark workflow:
 - Load function from specified library by recursive module search
 - Load data with common loader function
 - Construct data arguments in specified order
 - Assign special values which requires estimator/data to be loaded
 - Measure function performance metrics

See benchmark-specific config parameters in corresponding [`custom_function` section ](../../configs/BENCH-CONFIG-SPEC.md#custom_function) of benchmarking config specification.

---
[Documentation tree](../../README.md#-documentation-tree)
