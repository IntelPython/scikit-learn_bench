# Benchmarks

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

## `Scikit-learn Estimator`

Benchmark workflow:
 - Load estimator from the specified library by recursive module search
 - Load data with a common loader function
 - Assign special values that require estimator/data to be loaded
 - Get sklearn/sklearnex context, estimator parameters, running parameters
 - Measure required estimator methods
 - Combine metrics and parameters into the output

See [benchmark-specific config parameters](../../configs/README.md#benchmark-specific-parameters).

## `Function`

Benchmark workflow:
 - Load function from the specified library by recursive module search
 - Load data with a common loader function
 - Construct data arguments in specified order
 - Assign special values that require estimator/data to be loaded
 - Measure function performance metrics

See [benchmark-specific config parameters](../../configs/README.md#benchmark-specific-parameters).

---
[Documentation tree](../../README.md#-documentation)
