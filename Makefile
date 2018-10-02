# Sizes
DISTANCES_SIZE = 1000x15000
REGRESSION_SIZE = 1000000x50
KMEANS_SIZE = $(REGRESSION_SIZE)
SVM_VECTORS = 10000
SVM_FEATURES = 1000
ITERATIONS = ?

# Bookkeeping options
BATCH = $(shell date -Iseconds)
HOST = $(shell hostname)

# Other options
NUM_THREADS = -1
SVM_NUM_THREADS = 0
MULTIPLIER = 100
DATA_DIR = data/
KMEANS_DATA = $(addsuffix .csv,$(addprefix data/kmeans_,$(KMEANS_SIZE))) 

comma = ,

ifneq ($(CONDA_PREFIX),)
    LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib
    export LD_LIBRARY_PATH
endif


all: native python

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
	native/bin/kmeans $(BATCH) $(HOST) native kmeans.fit \
		$(NUM_THREADS) double $(REGRESSION_SIZE) $(DATA_DIR)
	native/bin/kmeans_predict $(BATCH) $(HOST) native kmeans.predict \
		$(NUM_THREADS) double $(REGRESSION_SIZE) $(DATA_DIR) $(MULTIPLIER)
	native/bin/two_class_svm \
		--fileX data/two/X-$(SVM_VECTORS)x$(SVM_FEATURES).npy.csv \
		--fileY data/two/y-$(SVM_VECTORS)x$(SVM_FEATURES).npy.csv \
		--num-threads $(SVM_NUM_THREADS)
	native/bin/multi_class_svm \
		--fileX data/multi/X-$(SVM_VECTORS)x$(SVM_FEATURES).npy.csv \
		--fileY data/multi/y-$(SVM_VECTORS)x$(SVM_FEATURES).npy.csv \
		--num-threads $(SVM_NUM_THREADS)

python: data
	@echo "# Running python benchmarks"
	python python/distances.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(DISTANCES_SIZE)) --iteration $(ITERATIONS)
	python python/ridge.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(REGRESSION_SIZE)) --iteration $(ITERATIONS)
	python python/linear.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(REGRESSION_SIZE)) --iteration $(ITERATIONS)
	python python/kmeans.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(KMEANS_SIZE)) --iteration $(ITERATIONS) \
		--input $(DATA_DIR)
	python python/kmeans_predict.py --batchID $(BATCH) --arch $(HOST) \
		--prefix python --core-number $(NUM_THREADS) \
		--size $(subst x,$(comma),$(KMEANS_SIZE)) --iteration $(ITERATIONS) \
		--input $(DATA_DIR) --data-multiplier $(MULTIPLIER)
	python python/svm_bench.py --core-number $(NUM_THREADS) \
		--fileX data/two/X-$(SVM_VECTORS)x$(SVM_FEATURES).npy \
		--fileY data/two/y-$(SVM_VECTORS)x$(SVM_FEATURES).npy
	python python/svm_bench.py --core-number $(NUM_THREADS) \
		--fileX data/multi/X-$(SVM_VECTORS)x$(SVM_FEATURES).npy \
		--fileY data/multi/y-$(SVM_VECTORS)x$(SVM_FEATURES).npy

data: $(KMEANS_DATA) svm_data

$(KMEANS_DATA): | data/
	python python/kmeans_data.py --size \
		$(shell basename $@ .csv | cut -d _ -f 2) --fname $@ --clusters 10

svm_data: data/two/X-$(SVM_VECTORS)x$(SVM_FEATURES).npy.csv

data/two/X-$(SVM_VECTORS)x$(SVM_FEATURES).npy.csv: | data/
	python python/svm_data.py -v $(SVM_VECTORS) -f $(SVM_FEATURES)
	native/svm_native_data.sh

data/:
	mkdir -p data/
	mkdir -p data/two
	mkdir -p data/multi

clean:
	$(MAKE) -C native clean
	rm -rf data

.PHONY: native python all clean native_data data kmeans_data svm_data
