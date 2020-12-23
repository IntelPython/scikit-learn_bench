## How to create conda environment for benchmarking

    conda create -n bench -c conda-forge python=3.7 xgboost pandas

##  Algorithms parameters

You can launch benchmarks for each algorithm separately. The table below lists all supported parameters for each algorithm.

#### General
| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
|num-threads|int|-1| The number of threads to use|
|arch|str|?|Achine architecture, for bookkeeping|
|batch|str|?|Batch ID, for bookkeeping|
|prefix|str|sklearn|Prefix string, for bookkeeping|
|header|action|False|Output CSV header|
|verbose|action|False|Output extra debug messages|
|data-format|str|numpy|Data formats: *numpy*, *pandas* or *cudf*|
|data-order|str|C|Data order: C (row-major, default) or F (column-major)|
|dtype|np.dtype|np.float64|Data type: *float64* (default) or *float32*|
|check-finiteness|action|False|Check finiteness in sklearn input check(disabled by default)|
|output-format|str|csv|Output format: *csv* (default) or *json*'|
|time-method|str|mean_min|Method used for time mesurements|
|box-filter-measurements|int|100|Maximum number of measurements in box filter|
|inner-loops|int|100|Maximum inner loop iterations. (we take the mean over inner iterations)|
|outer-loops|int|100|Maximum outer loop iterations. (we take the min over outer iterations)|
|time-limit|float|10|Target time to spend to benchmark|
|goal-outer-loops|int|10|The number of outer loops to aim while automatically picking number of inner loops. If zero, do not automatically decide number of inner loops.|
|seed|int|12345|Seed to pass as random_state|
|dataset-name|str|None|Dataset name|

#### GradientBoostingTrees

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| n-estimators | int | 100 | The number of gradient boosted trees |
| learning-rate | float | 0.3 | Step size shrinkage used in update to prevents overfitting|
| min-split-loss | float | 0 | Minimum loss reduction required to make partition on a leaf node |
| max-depth | int | 6 | Maximum depth of a tree |
| min-child-weight | float | 1 | Minimum sum of instance weight needed in a child |
| max-delta-step | float | 0 | Maximum delta step we allow each leaf output to be |
| subsample | float | 1 | Subsample ratio of the training instances |
| colsample-bytree | float | 1 | Subsample ratio of columns when constructing each tree |
| reg-lambda | float | 1 | L2 regularization term on weights |
| reg-alpha | float | 0 | L1 regularization term on weights |
| tree-method | str |  | The tree construction algorithm used in XGBoost |
| scale-pos-weight | float | 1 | Controls a balance of positive and negative weights |
| grow-policy | str | depthwise | Controls a way new nodes are added to the tree |
| max-leaves | int | 0 | Maximum number of nodes to be added |
| max-bin | int | 256 | Maximum number of discrete bins to bucket continuous features |
| objective | str | True | *reg:squarederror*, *binary:logistic*, *multi:softmax* or *multi:softprob*. Control a balance of positive and negative weights |
