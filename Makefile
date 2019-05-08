# Sizes
DISTANCES_SIZE = 1000x15000
REGRESSION_SIZE = 1000000x50
KMEANS_SAMPLES = 1000000
KMEANS_FEATURES = 50
KMEANS_SIZE = $(KMEANS_SAMPLES)x$(KMEANS_FEATURES)
SVM_SAMPLES = 10000
SVM_FEATURES = 1000
ITERATIONS = 10

# Bookkeeping options
BATCH = $(shell date -Iseconds)
HOST = $(shell hostname)

# Other options
NUM_THREADS = -1
SVM_NUM_THREADS = 0
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
	native/bin/cosine $(BATCH) $(HOST) native cosine_distances \
		$(NUM_THREADS) double $(DISTANCES_SIZE)
	native/bin/correlation $(BATCH) $(HOST) native correlation_distances \
		$(NUM_THREADS) double $(DISTANCES_SIZE)
	native/bin/ridge $(BATCH) $(HOST) native ridge \
		$(NUM_THREADS) double $(REGRESSION_SIZE)
	native/bin/linear $(BATCH) $(HOST) native linear \
		$(NUM_THREADS) double $(REGRESSION_SIZE)
	native/bin/kmeans $(BATCH) $(HOST) native kmeans \
		$(NUM_THREADS) double $(KMEANS_SIZE) $(DATA_DIR) $(MULTIPLIER)
	native/bin/two_class_svm \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--num-threads $(SVM_NUM_THREADS) --header
	native/bin/multi_class_svm \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--num-threads $(SVM_NUM_THREADS) --header
	native/bin/log_reg_lbfgs \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--num-threads $(SVM_NUM_THREADS) --header
	native/bin/log_reg_lbfgs \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--num-threads $(SVM_NUM_THREADS) --header
	native/bin/decision_forest_clsf \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--num-threads $(SVM_NUM_THREADS) --header
	native/bin/decision_forest_clsf \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--num-threads $(SVM_NUM_THREADS) --header

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
		--header
	python sklearn/svm_bench.py --core-number $(NUM_THREADS) \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header
	python sklearn/log_reg.py --num-threads $(NUM_THREADS) \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header
	python sklearn/log_reg.py --num-threads $(NUM_THREADS) \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header
	python sklearn/df_clsf.py --num-threads $(NUM_THREADS) \
		--fileX data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header
	python sklearn/df_clsf.py --num-threads $(NUM_THREADS) \
		--fileX data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
		--header

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

data: $(KMEANS_DATA) svm_data

$(KMEANS_DATA): | data/
	python make_datasets.py -f $(KMEANS_FEATURES) -s $(KMEANS_SAMPLES) \
		kmeans -c 10 -x $(basename $@) -i $(basename $@).init \
		-t $(basename $@).tol

svm_data: data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy \
	data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy

data/two/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy: | data/
	python make_datasets.py -f $(SVM_FEATURES) -s $(SVM_SAMPLES) \
		classification -c 2 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/multi/X-$(SVM_SAMPLES)x$(SVM_FEATURES).npy: | data/
	python make_datasets.py -f $(SVM_FEATURES) -s $(SVM_SAMPLES) \
		classification -c 5 -x $@ -y $(dir $@)/$(subst X-,y-,$(notdir $@))

data/:
	mkdir -p data/
	mkdir -p data/two
	mkdir -p data/multi

clean:
	$(MAKE) -C native clean
	rm -rf data

.PHONY: native python sklearn daal4py all clean native_data data kmeans_data svm_data
