# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

BENCHMARKS += correlation cosine kmeans linear ridge kmeans_predict two_class_svm multi_class_svm
CXXSRCS = $(addsuffix _bench.cpp,$(BENCHMARKS))

CXX = icc
CXXFLAGS += -m64 -fPIC -fp-model strict -O3 -g -fomit-frame-pointer \
		  -xSSE4.2 -axCORE-AVX2,COMMON-AVX512
CXXFLAGS += -std=c++11
LDFLAGS += -ldaal_core -ldaal_thread
CXXINCLUDE += CLI11/include

ifneq ($(CONDA_PREFIX),)
	LDFLAGS += -L"$(CONDA_PREFIX)/lib"
	CXXINCLUDE += "$(CONDA_PREFIX)/include"
endif

CXXINCLUDE := $(addprefix -I,$(CXXINCLUDE))

all: $(addprefix bin/,$(BENCHMARKS))

bin:
	mkdir -p bin

bin/%: %_bench.cpp | bin
	$(CXX) $< $(CXXINCLUDE) $(CXXFLAGS) $(LDFLAGS) -o $@


clean:
	rm -rf bin/


.PHONY: all clean
