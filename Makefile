# Sizes
DISTANCES_SIZE = 1000x15000
REGRESSION_SIZE = 1000000x50
KMEANS_SAMPLES = 1000000
KMEANS_FEATURES = 50
KMEANS_SIZE = $(KMEANS_SAMPLES)x$(KMEANS_FEATURES)
SVM_SAMPLES = 50000
SVM_FEATURES = 100
SVM_SIZE = $(SVM_SAMPLES)x$(SVM_FEATURES)
LOGREG_SAMPLES = 100000
LOGREG_FEATURES = 100
LOGREG_SIZE = $(LOGREG_SAMPLES)x$(LOGREG_FEATURES)
DFCLF_SAMPLES = 10000
DFCLF_FEATURES = 50
DFCLF_SIZE = $(DFCLF_SAMPLES)x$(DFCLF_FEATURES)
DFREG_SAMPLES = 10000
DFREG_FEATURES = 50
DFREG_SIZE = $(DFREG_SAMPLES)x$(DFREG_FEATURES)

ITERATIONS = 10

# Bookkeeping options
BATCH = $(shell date -Iseconds)
HOST = $(shell hostname)

# This makes the makefile exit on failed benchmarks. We pipe the
# benchmark outputs to "tee", which results in unexpected successes.
SHELL = bash -o pipefail

# Other options
NUM_THREADS = -1
SVM_NUM_THREADS = 0
DFCLF_NUM_THREADS = 0
DFREG_NUM_THREADS = 0
LOGREG_NUM_THREADS = 0
MULTIPLIER = 100
DATA_DIR = data/
DATA_kmeans = data/kmeans_$(KMEANS_SIZE).npy

# Define which benchmarks to run
NATIVE_BENCHMARKS =		distances ridge linear kmeans svm2 svm5 \
						logreg2 logreg5 dfclf2 dfclf5 dfreg pca_daal pca_full
SKLEARN_BENCHMARKS = 	distances ridge linear kmeans svm2 svm5 \
						logreg2 logreg5 dfclf2 dfclf5 dfreg pca_full
DAAL4PY_BENCHMARKS = 	distances ridge linear kmeans svm2 svm5 \
						pca_daal pca_full

# Define native benchmark binary names
NATIVE_distances = distances
NATIVE_ridge = ridge
NATIVE_linear = linear
NATIVE_kmeans = kmeans
NATIVE_svm2 = two_class_svm
NATIVE_svm5 = multi_class_svm
NATIVE_logreg2 = log_reg_lbfgs
NATIVE_logreg5 = log_reg_lbfgs
NATIVE_dfclf2 = decision_forest_clsf
NATIVE_dfclf5 = decision_forest_clsf
NATIVE_dfreg = decision_forest_regr
NATIVE_pca_daal = pca
NATIVE_pca_full = pca

# Define arguments for native benchmarks
ARGS_NATIVE_distances = --batch "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(DISTANCES_SIZE)" --header
ARGS_NATIVE_ridge = 	--batch "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" --header
ARGS_NATIVE_linear = 	--batch "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" --header
ARGS_NATIVE_pca_daal = 	--batch "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" --header \
						--svd-solver daal
ARGS_NATIVE_pca_full = 	--batch "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" --header \
						--svd-solver full
ARGS_NATIVE_kmeans = 	--batch "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--data-multiplier "$(MULTIPLIER)" \
						--filex data/kmeans_$(KMEANS_SIZE).npy \
						--filei data/kmeans_$(KMEANS_SIZE).init.npy \
						--filet data/kmeans_$(KMEANS_SIZE).tol.npy \
						--header
ARGS_NATIVE_svm2 =		--fileX data/two/X-$(SVM_SIZE).npy \
						--fileY data/two/y-$(SVM_SIZE).npy \
						--num-threads $(SVM_NUM_THREADS) --header
ARGS_NATIVE_svm5 = 		--fileX data/multi/X-$(SVM_SIZE).npy \
						--fileY data/multi/y-$(SVM_SIZE).npy \
						--num-threads $(SVM_NUM_THREADS) --header
ARGS_NATIVE_logreg2 =	--fileX data/two/X-$(LOGREG_SIZE).npy \
						--fileY data/two/y-$(LOGREG_SIZE).npy \
						--num-threads $(LOGREG_NUM_THREADS) --header
ARGS_NATIVE_logreg5 =	--fileX data/multi/X-$(LOGREG_SIZE).npy \
						--fileY data/multi/y-$(LOGREG_SIZE).npy \
						--num-threads $(LOGREG_NUM_THREADS) --header
ARGS_NATIVE_dfclf2 = 	--fileX data/two/X-$(DFCLF_SIZE).npy \
						--fileY data/two/y-$(DFCLF_SIZE).npy \
						--num-threads $(DFCLF_NUM_THREADS) --header
ARGS_NATIVE_dfclf5 = 	--fileX data/multi/X-$(DFCLF_SIZE).npy \
						--fileY data/multi/y-$(DFCLF_SIZE).npy \
						--num-threads $(DFCLF_NUM_THREADS) --header
ARGS_NATIVE_dfreg = 	--fileX data/reg/X-$(DFREG_SIZE).npy \
						--fileY data/reg/y-$(DFREG_SIZE).npy \
						--num-threads $(DFREG_NUM_THREADS) --header

SKLEARN_distances = distances
SKLEARN_ridge = ridge
SKLEARN_linear = linear
SKLEARN_pca_full = pca
SKLEARN_pca_daal = pca
SKLEARN_kmeans = kmeans
SKLEARN_svm2 = svm_bench
SKLEARN_svm5 = svm_bench
SKLEARN_logreg2 = log_reg
SKLEARN_logreg5 = log_reg
SKLEARN_dfclf2 = df_clsf
SKLEARN_dfclf5 = df_clsf
SKLEARN_dfreg = df_regr

ARGS_SKLEARN_distances = --batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(DISTANCES_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix sklearn
ARGS_SKLEARN_ridge = 	--batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix sklearn
ARGS_SKLEARN_linear = 	--batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix sklearn
ARGS_SKLEARN_pca_daal = --batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix sklearn \
						--svd-solver daal
ARGS_SKLEARN_pca_full = --batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix sklearn \
						--svd-solver full
ARGS_SKLEARN_kmeans = 	--batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--data-multiplier "$(MULTIPLIER)" \
						--filex data/kmeans_$(KMEANS_SIZE).npy \
						--filei data/kmeans_$(KMEANS_SIZE).init.npy \
						--iteration "$(ITERATIONS)" --prefix sklearn \
						--size "$(KMEANS_SIZE)"
ARGS_SKLEARN_svm2 =		--fileX data/two/X-$(SVM_SIZE).npy \
						--fileY data/two/y-$(SVM_SIZE).npy \
						--num-threads $(SVM_NUM_THREADS) --header
ARGS_SKLEARN_svm5 = 	--fileX data/multi/X-$(SVM_SIZE).npy \
						--fileY data/multi/y-$(SVM_SIZE).npy \
						--num-threads $(SVM_NUM_THREADS) --header
ARGS_SKLEARN_logreg2 =	--fileX data/two/X-$(LOGREG_SIZE).npy \
						--fileY data/two/y-$(LOGREG_SIZE).npy \
						--num-threads $(LOGREG_NUM_THREADS) --header \
						--multiclass ovr
ARGS_SKLEARN_logreg5 =	--fileX data/multi/X-$(LOGREG_SIZE).npy \
						--fileY data/multi/y-$(LOGREG_SIZE).npy \
						--num-threads $(LOGREG_NUM_THREADS) --header \
						--multiclass multinomial
ARGS_SKLEARN_dfclf2 = 	--fileX data/two/X-$(DFCLF_SIZE).npy \
						--fileY data/two/y-$(DFCLF_SIZE).npy \
						--num-threads $(DFCLF_NUM_THREADS) --header
ARGS_SKLEARN_dfclf5 = 	--fileX data/multi/X-$(DFCLF_SIZE).npy \
						--fileY data/multi/y-$(DFCLF_SIZE).npy \
						--num-threads $(DFCLF_NUM_THREADS) --header
ARGS_SKLEARN_dfreg = 	--fileX data/multi/X-$(DFREG_SIZE).npy \
						--fileY data/multi/y-$(DFREG_SIZE).npy \
						--num-threads $(DFREG_NUM_THREADS) --header

DAAL4PY_distances = distances
DAAL4PY_ridge = ridge
DAAL4PY_linear = linear
DAAL4PY_pca_full = pca
DAAL4PY_pca_daal = pca
DAAL4PY_kmeans = kmeans
DAAL4PY_svm2 = svm_bench
DAAL4PY_svm5 = svm_bench
DAAL4PY_logreg2 = log_reg
DAAL4PY_logreg5 = log_reg
DAAL4PY_dfclf2 = df_clsf
DAAL4PY_dfclf5 = df_clsf
DAAL4PY_dfreg = df_regr

ARGS_DAAL4PY_distances = --batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(DISTANCES_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix daal4py
ARGS_DAAL4PY_ridge = 	--batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix daal4py
ARGS_DAAL4PY_linear = 	--batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix daal4py
ARGS_DAAL4PY_pca_daal = --batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix daal4py \
						--svd-solver daal
ARGS_DAAL4PY_pca_full = --batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--size "$(REGRESSION_SIZE)" \
						--iteration "$(ITERATIONS)" --prefix daal4py \
						--svd-solver full
ARGS_DAAL4PY_kmeans = 	--batchID "$(BATCH)" --arch "$(HOST)" \
						--num-threads "$(NUM_THREADS)" \
						--data-multiplier "$(MULTIPLIER)" \
						--filex data/kmeans_$(KMEANS_SIZE).npy \
						--filei data/kmeans_$(KMEANS_SIZE).init.npy \
						--iteration "$(ITERATIONS)" --prefix daal4py \
						--size "$(KMEANS_SIZE)"
ARGS_DAAL4PY_svm2 =		--fileX data/two/X-$(SVM_SIZE).npy \
						--fileY data/two/y-$(SVM_SIZE).npy \
						--num-threads $(SVM_NUM_THREADS) --header
ARGS_DAAL4PY_svm5 = 	--fileX data/multi/X-$(SVM_SIZE).npy \
						--fileY data/multi/y-$(SVM_SIZE).npy \
						--num-threads $(SVM_NUM_THREADS) --header
ARGS_DAAL4PY_logreg2 =	--fileX data/two/X-$(LOGREG_SIZE).npy \
						--fileY data/two/y-$(LOGREG_SIZE).npy \
						--num-threads $(LOGREG_NUM_THREADS) --header \
						--multiclass ovr
ARGS_DAAL4PY_logreg5 =	--fileX data/multi/X-$(LOGREG_SIZE).npy \
						--fileY data/multi/y-$(LOGREG_SIZE).npy \
						--num-threads $(LOGREG_NUM_THREADS) --header \
						--multiclass multinomial
ARGS_DAAL4PY_dfclf2 = 	--fileX data/two/X-$(DFCLF_SIZE).npy \
						--fileY data/two/y-$(DFCLF_SIZE).npy \
						--num-threads $(DFCLF_NUM_THREADS) --header
ARGS_DAAL4PY_dfclf5 = 	--fileX data/multi/X-$(DFCLF_SIZE).npy \
						--fileY data/multi/y-$(DFCLF_SIZE).npy \
						--num-threads $(DFCLF_NUM_THREADS) --header
ARGS_DAAL4PY_dfreg = 	--fileX data/multi/X-$(DFREG_SIZE).npy \
						--fileY data/multi/y-$(DFREG_SIZE).npy \
						--num-threads $(DFREG_NUM_THREADS) --header

comma = ,

ifneq ($(CONDA_PREFIX),)
	LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(CONDA_PREFIX)/lib
    export LD_LIBRARY_PATH
endif

export I_MPI_ROOT

all: native python

python: sklearn daal4py

native/bin/%: native/%.cpp
	git submodule init && git submodule update
	$(MAKE) -C native

output/native/%.out: | DATA_% output/native/
	[ -f native/bin/$(NATIVE_$*) ] || make -C native
	native/bin/$(NATIVE_$*) $(ARGS_NATIVE_$*) | tee $@

output/sklearn/%.out: | DATA_% output/sklearn/
	python sklearn/$(SKLEARN_$*).py $(ARGS_SKLEARN_$*) | tee $@

output/daal4py/%.out: | DATA_% output/daal4py/
	python daal4py/$(DAAL4PY_$*).py $(ARGS_DAAL4PY_$*) | tee $@

output/%/:
	mkdir -p $@

native: $(addsuffix .out,$(addprefix output/native/,$(NATIVE_BENCHMARKS))) data

sklearn: $(addsuffix .out,$(addprefix output/sklearn/,$(SKLEARN_BENCHMARKS))) data

daal4py: $(addsuffix .out,$(addprefix output/daal4py/,$(DAAL4PY_BENCHMARKS))) data


daal4py_: data
	@echo "# Running daal4py benchmarks"
	python daal4py/distances.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(DISTANCES_SIZE)) --iteration $(ITERATIONS)
	python daal4py/ridge.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(REGRESSION_SIZE)) --iteration $(ITERATIONS)
	python daal4py/linear.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(REGRESSION_SIZE)) --iteration $(ITERATIONS)
	python daal4py/kmeans.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(KMEANS_SIZE)) --iteration $(ITERATIONS) \
		-x $(KMEANS_DATA) -i $(basename $(KMEANS_DATA)).init.npy
	python daal4py/svm_bench.py --core-number $(NUM_THREADS) \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header
	python daal4py/svm_bench.py --core-number $(NUM_THREADS) \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header
	python daal4py/df_clsf.py --num-threads $(NUM_THREADS) \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header
	python daal4py/df_clsf.py --num-threads $(NUM_THREADS) \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header

data: $(KMEANS_DATA) svm_data logreg_data df_clf_data

DATA_kmeans: data/kmeans_$(KMEANS_SIZE).npy
DATA_svm2: data/two/X-$(SVM_SIZE).npy
DATA_svm5: data/multi/X-$(SVM_SIZE).npy
DATA_logreg2: data/two/X-$(LOGREG_SIZE).npy
DATA_logreg5: data/multi/X-$(LOGREG_SIZE).npy
DATA_dfclf2: data/two/X-$(DFCLF_SIZE).npy
DATA_dfclf5: data/multi/X-$(DFCLF_SIZE).npy
DATA_dfreg: data/reg/X-$(DFREG_SIZE).npy
DATA_%: ;


data/kmeans_$(KMEANS_SAMPLES)x$(KMEANS_FEATURES).npy: | data/
	python make_datasets.py -f $(KMEANS_FEATURES) -s $(KMEANS_SAMPLES) \
		kmeans -c 10 -x $(basename $@) -i $(basename $@).init \
		-t $(basename $@).tol

data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy: | data/two/
	python make_datasets.py -f $(SVM_FEATURES) -s $(SVM_SAMPLES) \
		classification -c 2 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy: | data/multi/
	python make_datasets.py -f $(SVM_FEATURES) -s $(SVM_SAMPLES) \
		classification -c 5 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/two/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy: | data/two/
	python make_datasets.py -f $(LOGREG_FEATURES) -s $(LOGREG_SAMPLES) \
		classification -c 2 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/multi/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy: | data/multi/
	python make_datasets.py -f $(LOGREG_FEATURES) -s $(LOGREG_SAMPLES) \
		classification -c 5 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/two/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy: | data/two/
	python make_datasets.py -f $(DFCLF_FEATURES) -s $(DFCLF_SAMPLES) \
		classification -c 2 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/multi/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy: | data/multi/
	python make_datasets.py -f $(DFCLF_FEATURES) -s $(DFCLF_SAMPLES) \
		classification -c 5 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/reg/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy: | data/reg/
	python make_datasets.py -f $(DFCLF_FEATURES) -s $(DFCLF_SAMPLES) \
		regression -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/%/:
	mkdir -p $@

clean:
	$(MAKE) -C native clean
	rm -rf data
	rm -rf output

.PRECIOUS: output/sklearn/ output/native/ output/daal4py/
.PHONY: native python sklearn daal4py all clean native_data data kmeans_data svm_data logreg_data df_clf_data
.DELETE_ON_ERROR: ;
