# Sizes
DISTANCES_SIZE = 1000x15000
REGRESSION_SIZE = 1000000x50
KMEANS_SAMPLES = 1000000
KMEANS_FEATURES = 50
KMEANS_SIZE = $(KMEANS_SAMPLES)x$(KMEANS_FEATURES)
SVM_SAMPLES = 50000
SVM_FEATURES = 100
LOGREG_SAMPLES = 100000
LOGREG_FEATURES = 100
DFCLF_SAMPLES = 10000
DFCLF_FEATURES = 50

ITERATIONS = 10

# Bookkeeping options
BATCH = $(shell date -Iseconds)
HOST = $(shell hostname)

# Other options
NUM_THREADS = -1
SVM_NUM_THREADS = 0
DFCLF_NUM_THREADS = 0
LOGREG_NUM_THREADS = 0
MULTIPLIER = 100
DATA_DIR = data/
KMEANS_DATA = data/kmeans_$(KMEANS_SIZE).npy

comma = ,

ifneq ($(CONDA_PREFIX),)
	LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(CONDA_PREFIX)/lib
    export LD_LIBRARY_PATH
endif

export I_MPI_ROOT

all: native python

python: sklearn daal4py

native: data
	git submodule init && git submodule update
	@echo "# Compiling native benchmarks"
	$(MAKE) -C native
	@echo "# Running native benchmarks"
	native/bin/distances --batch "$(BATCH)" --arch "$(HOST)" \
		--num-threads "$(NUM_THREADS)" --size "$(DISTANCES_SIZE)"
	native/bin/ridge --batch "$(BATCH)" --arch "$(HOST)" \
		--num-threads "$(NUM_THREADS)" --size "$(REGRESSION_SIZE)"
	native/bin/linear --batch "$(BATCH)" --arch "$(HOST)" \
		--num-threads "$(NUM_THREADS)" --size "$(REGRESSION_SIZE)"
	native/bin/kmeans --batch "$(BATCH)" --arch "$(HOST)" \
		--num-threads "$(NUM_THREADS)" --data-multiplier "$(MULTIPLIER)" \
		--filex $(KMEANS_DATA) --filei $(basename $(KMEANS_DATA)).init.npy \
		--filet $(basename $(KMEANS_DATA)).tol.npy
	native/bin/two_class_svm \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--num-threads $(SVM_NUM_THREADS) --header
	native/bin/multi_class_svm \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--num-threads $(SVM_NUM_THREADS) --header
	native/bin/log_reg_lbfgs \
		--fileX data/two/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
		--fileY data/two/y-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
		--num-threads $(LOGREG_NUM_THREADS) --header
	native/bin/log_reg_lbfgs \
		--fileX data/multi/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
		--fileY data/multi/y-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
		--num-threads $(LOGREG_NUM_THREADS) --header
	native/bin/decision_forest_clsf \
		--fileX data/two/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
		--fileY data/two/y-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
		--num-threads $(DFCLF_NUM_THREADS) --header
	native/bin/decision_forest_clsf \
		--fileX data/multi/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
		--fileY data/multi/y-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
		--num-threads $(DFCLF_NUM_THREADS) --header

sklearn: data
	@echo "# Running scikit-learn benchmarks"
	python sklearn/distances.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(DISTANCES_SIZE)) --iteration $(ITERATIONS)
	python sklearn/ridge.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(REGRESSION_SIZE)) --iteration $(ITERATIONS)
	python sklearn/linear.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(REGRESSION_SIZE)) --iteration $(ITERATIONS)
	python sklearn/kmeans.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(KMEANS_SIZE)) --iteration $(ITERATIONS) \
		-x $(KMEANS_DATA) -i $(basename $(KMEANS_DATA)).init.npy
	python sklearn/svm_bench.py --core-number $(NUM_THREADS) \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header --prefix python
	python sklearn/svm_bench.py --core-number $(NUM_THREADS) \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header --prefix python
	python sklearn/log_reg.py --num-threads $(NUM_THREADS) \
		--fileX data/two/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
		--fileY data/two/y-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
		--header --multiclass ovr --prefix python
	python sklearn/log_reg.py --num-threads $(NUM_THREADS) \
		--fileX data/multi/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
		--fileY data/multi/y-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
		--header --multiclass multinomial --prefix python
	python sklearn/df_clsf.py --num-threads $(NUM_THREADS) \
		--fileX data/two/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
		--fileY data/two/y-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
		--header --prefix python
	python sklearn/df_clsf.py --num-threads $(NUM_THREADS) \
		--fileX data/multi/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
		--fileY data/multi/y-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
		--header --prefix python
	python sklearn/pca.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(REGRESSION_SIZE)) \
		--iteration $(ITERATIONS) \
		--svd-solver daal
	python sklearn/pca.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(REGRESSION_SIZE)) \
		--iteration $(ITERATIONS) \
		--svd-solver full

daal4py: data
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

data: $(KMEANS_DATA) svm_data logreg_data df_clf_data

$(KMEANS_DATA): | data/
	python make_datasets.py -f $(KMEANS_FEATURES) -s $(KMEANS_SAMPLES) \
		kmeans -c 10 -x $(basename $@) -i $(basename $@).init \
		-t $(basename $@).tol

svm_data: data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
	data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy

logreg_data: data/two/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy \
	data/multi/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy

df_clf_data: data/two/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy \
	data/multi/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy


data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy: | data/
	python make_datasets.py -f $(SVM_FEATURES) -s $(SVM_SAMPLES) \
		classification -c 2 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy: | data/
	python make_datasets.py -f $(SVM_FEATURES) -s $(SVM_SAMPLES) \
		classification -c 5 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/two/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy: | data/
	python make_datasets.py -f $(LOGREG_FEATURES) -s $(LOGREG_SAMPLES) \
		classification -c 2 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/multi/X-$(LOGREG_SAMPLES)x$(LOGREG_FEATURES).npy: | data/
	python make_datasets.py -f $(LOGREG_FEATURES) -s $(LOGREG_SAMPLES) \
		classification -c 5 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/two/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy: | data/
	python make_datasets.py -f $(DFCLF_FEATURES) -s $(DFCLF_SAMPLES) \
		classification -c 2 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/multi/X-$(DFCLF_SAMPLES)x$(DFCLF_FEATURES).npy: | data/
	python make_datasets.py -f $(DFCLF_FEATURES) -s $(DFCLF_SAMPLES) \
		classification -c 5 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/:
	mkdir -p data/
	mkdir -p data/two
	mkdir -p data/multi

clean:
	$(MAKE) -C native clean
	rm -rf data

.PHONY: native python sklearn daal4py all clean native_data data kmeans_data svm_data logreg_data df_clf_data
