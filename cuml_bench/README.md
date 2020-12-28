
## How to create conda environment for benchmarking
`conda create -n bench -c rapidsai -c conda-forge python=3.7 cuml pandas cudf`

##  Algorithms parameters

You can launch benchmarks for each algorithm separately. The tables below list all supported parameters for each algorithm:

- [General](#general)
- [DBSCAN](#dbscan)
- [RandomForestClassifier](#randomforestclassifier)
- [RandomForestRegressor](#randomforestregressor)
- [pairwise_distances](#pairwise_distances)
- [KMeans](#kmeans)
- [KNeighborsClassifier](#kneighborsclassifier)
- [LinearRegression](#linearregression)
- [LogisticRegression](#logisticregression)
- [PCA](#pca)
- [Ridge Regression](#ridge)
- [SVC](#svc)
- [train_test_split](#train_test_split)

#### General
| Parameter Name  | Type | Default Value | Description |
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
|goal-outer-loops|int|10|The number of outer loops to aim while automatically picking number of inner loops. If zero, do not automatically decide number of inner loops|
|seed|int|12345|Seed to pass as random_state|
|dataset-name|str|None|Dataset name|


#### DBSCAN
| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| epsilon | float | 10 | Radius of neighborhood of a point|
| min_samples | int | 5 | The minimum number of samples required in a 'neighborhood to consider a point a core point |

#### RandomForestClassifier

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| criterion | str | gini | *gini* or *entropy*. The function to measure the quality of a split |
|split-algorithm|str|hist|*hist* or *global_quantile*. The algorithm to determine how nodes are split in the tree|
| num-trees | int | 100 | The number of trees in the forest |
| max-features | float_or_int | None | Upper bound on features used at each split |
| max-depth | int | None | Upper bound on depth of constructed trees |
| min-samples-split | float_or_int | 2 | Minimum samples number for node splitting |
| max-leaf-nodes | int | None | Maximum leaf nodes per tree |
| min-impurity-decrease | float | 0 | Needed impurity decrease for node splitting |
| no-bootstrap | store_false | True | Don't control bootstraping |

#### RandomForestRegressor

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| criterion | str | gini | *gini* or *entropy*. The function to measure the quality of a split |
|split-algorithm|str|hist|*hist* or *global_quantile*. The algorithm to determine how nodes are split in the tree|
| num-trees | int | 100 | The number of trees in the forest |
| max-features | float_or_int | None | Upper bound on features used at each split |
| max-depth | int | None | Upper bound on depth of constructed trees |
| min-samples-split | float_or_int | 2 | Minimum samples number for node splitting |
| max-leaf-nodes | int | None | Maximum leaf nodes per tree |
| min-impurity-decrease | float | 0 | Needed impurity decrease for node splitting |
| no-bootstrap | action | True | Don't control bootstraping |

#### KMeans

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| init | str |  | Initial clusters |
| tol | float | 0 | Absolute threshold |
| maxiter | int | 100 | Maximum number of iterations |
| samples-per-batch | int | 32768 | The number of samples per batch |
| n-clusters | int |  | The number of clusters |

#### KNeighborsClassifier

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| n-neighbors | int | 5 | The number of neighbors to use |
| weights | str | uniform | Weight function used in prediction |
| method | str | brute | Algorithm used to compute the nearest neighbors |
| metric | str | euclidean | Distance metric to use |

#### LinearRegression

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| no-fit-intercept | action | True | Don't fit intercept (assume data already centered) |
| solver | str | eig | *eig* or *svd*. Solver used for training |

#### LogisticRegression

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| no-fit-intercept | action | True | Don't fit intercept|
| solver | str | qn | *qn*, *owl*. Solver to use|
| maxiter | int | 100 | Maximum iterations for the iterative solver |
| C | float | 1.0 | Regularization parameter |
| tol | float | None | Tolerance for solver |

#### PCA

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| svd-solver | str | full | *auto*, *full* or *jacobi*. SVD solver to use |
| n-components | int | None | The number of components to find |
| whiten | action | False | Perform whitening |

#### Ridge

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| no-fit-intercept | action | True | Don't fit intercept (assume data already centered) |
| solver | str | eig | *eig*, *cd* or *svd*. Solver used for training |
| alpha | float | 1.0 | Regularization strength |

#### SVC

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| C | float | 0.01 | SVM slack parameter |
| kernel | str | linear | *linear* or *rbf*. SVM kernel function |
| gamma | float | None | Parameter for kernel="rbf" |
| maxiter | int | 2000 | Maximum iterations for the iterative solver |
| max-cache-size | int | 64 | Maximum cache size for SVM. |
| tol | float | 1e-16 | Tolerance passed to sklearn.svm.SVC |
| no-shrinking | action | True | Don't use shrinking heuristic |

#### train_test_split

| parameter Name  | Type | default value | description |
| ----- | ---- |---- |---- |
| train-size | float | 0.75 | Size of training subset |
| test-size | float | 0.25 | Size of testing subset |
| do-not-shuffle | action | False | Do not perform data shuffle before splitting |
