# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

BENCHMARKS += distances kmeans linear ridge pca \
			  two_class_svm multi_class_svm log_reg_lbfgs \
			  decision_forest_regr decision_forest_clsf
FOBJ = $(addprefix lbfgsb/,lbfgsb.o linpack.o timer.o)
CXXSRCS = $(addsuffix _bench.cpp,$(BENCHMARKS))

CXX = icc
CXXFLAGS += -m64 -fPIC -fp-model strict -O3 -fomit-frame-pointer \
		  -xSSE4.2 -axCORE-AVX2,COMMON-AVX512
CXXFLAGS += -std=c++14 -g
LDFLAGS += -ltbb -lstdc++ -lpthread -lm -ldaal_core -ldaal_thread \
		   -Wl,-rpath,$(CONDA_PREFIX)/lib
CXXINCLUDE += include

ifneq ($(CONDA_PREFIX),)
	LDFLAGS += -L"$(CONDA_PREFIX)/lib"
	CXXINCLUDE += "$(CONDA_PREFIX)/include"
endif

CXXINCLUDE := $(addprefix -I,$(CXXINCLUDE))

all: $(addprefix bin/,$(BENCHMARKS))

bin:
	mkdir -p bin

bin/log_reg_lbfgs: log_reg_lbfgs_bench.cpp $(FOBJ) | bin
	$(CXX) $^ $(CXXINCLUDE) $(CXXFLAGS) $(LDFLAGS) -DMKL_ILP64 -mkl=parallel \
		-lmkl_rt -lifcore -limf -o $@


bin/%: %_bench.cpp | bin
	$(CXX) $< $(CXXINCLUDE) $(CXXFLAGS) $(LDFLAGS) -o $@


$(FOBJ):
	make -C lbfgsb


clean:
	rm -rf bin/


.PHONY: all clean
