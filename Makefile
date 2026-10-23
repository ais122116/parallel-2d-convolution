CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall
NVCC     = nvcc
NVFLAGS  = -O3 -std=c++17

all: cpu_reference gen_data compare

cpu_reference: cpu_reference.cpp conv_io.h
	$(CXX) $(CXXFLAGS) -o $@ $<
gen_data: gen_data.cpp conv_io.h
	$(CXX) $(CXXFLAGS) -o $@ $<
compare: compare.cpp conv_io.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# GPU versions (weeks 2-3)
gpu: conv_naive conv_opt
conv_naive: conv_naive.cu conv_io.h
	$(NVCC) $(NVFLAGS) -o $@ $<
conv_opt: conv_opt.cu conv_io.h
	$(NVCC) $(NVFLAGS) -o $@ $<

test: all
	./run_tests.sh

clean:
	rm -f cpu_reference gen_data compare conv_naive conv_opt results.csv
.PHONY: all gpu test clean
