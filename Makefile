# Parallel 2D Convolution with CUDA
#
#   make              EVERYTHING: build, data, verify, test, images, timing
#   make build        only build the programs
#   make data         generate test data into testdata/ (input, filters, CPU reference)
#   make verify       run conv_naive + conv_opt on every case in testdata/ and compare
#   make test         run_tests.sh   (full correctness sweep -> results.csv)
#   make images       run_image_tests.sh (filtered images -> results/)
#   make check        python3 run_tests.py check
#   make bench        python3 run_tests.py bench -> bench.csv
#   make timing       one big case: CPU naive, CPU fast (1 and 22 threads),
#                     conv_naive, conv_opt -> timing.txt (CPU naive takes ~1-2 min)
#   make everything   same as plain make
#   make clean        remove programs and results
#   make distclean    clean + remove testdata/
#
#   make ARCH=sm_86   build for a specific GPU (default: native)

CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -fopenmp -march=native
NVCC     = nvcc
ARCH    ?= native
NVFLAGS  = -O3 -std=c++17 -arch=$(ARCH)

NAIVE_SRC = conv_naive.cu
OPT_SRC   = conv_opt1.cu
DATADIR   = testdata
TIMEDIR   = timingdata
TIME_CASE = 2048 2048 16 32 7      # M N C K W used by "make timing"

CPU_BINS = cpu_reference gen_data compare imgtool
GPU_BINS = conv_naive conv_opt

# test data: "M N C K W" per case
CASES = "1 1 3 8 1"   "5 3 4 4 3"      "33 17 16 32 5"  "64 64 127 3 7" \
        "100 100 8 127 1" "128 96 32 64 3" "127 255 64 16 5" "256 256 3 8 7" \
        "300 201 4 4 1"  "257 129 16 32 3" "512 512 3 4 5"  "1000 1000 3 8 7" \
        "1024 1024 16 32 7" "999 777 32 4 5" "1920 1080 3 8 1"

# plain "make" runs the whole pipeline (stops at the first failure)
all: build data verify test images timing

build: $(CPU_BINS) $(GPU_BINS)

# ---------------------------------------------------------------- build
cpu_reference: cpu_reference.cpp conv_io.h
	$(CXX) $(CXXFLAGS) -o $@ $<
gen_data: gen_data.cpp conv_io.h
	$(CXX) $(CXXFLAGS) -o $@ $<
compare: compare.cpp conv_io.h
	$(CXX) $(CXXFLAGS) -o $@ $<
imgtool: imgtool.cpp conv_io.h
	$(CXX) $(CXXFLAGS) -o $@ $<
conv_naive: $(NAIVE_SRC) conv_io.h
	$(NVCC) $(NVFLAGS) -o $@ $<
conv_opt: $(OPT_SRC) conv_io.h
	$(NVCC) $(NVFLAGS) -o $@ $<

# ---------------------------------------------------------------- test data
data: cpu_reference gen_data
	./cpu_reference --selftest
	rm -rf $(DATADIR) && mkdir -p $(DATADIR)
	@seed=1; for c in $(CASES); do \
	    set -- $$c; d=$(DATADIR)/M$$1_N$$2_C$$3_K$$4_W$$5; mkdir -p $$d; \
	    ./gen_data $$1 $$2 $$3 $$4 $$5 $$seed $$d/input.bin $$d/filters.bin || exit 1; \
	    ./cpu_reference $$d/input.bin $$d/filters.bin $$d/ref.bin > /dev/null || exit 1; \
	    echo "generated $$d"; seed=$$((seed + 1)); \
	done

verify: build
	@[ -d $(DATADIR) ] || $(MAKE) data
	@pass=0; fail=0; for d in $(DATADIR)/*/; do \
	    line=$$(basename $$d); \
	    for p in conv_naive conv_opt; do \
	        t=$$(./$$p $$d/input.bin $$d/filters.bin $$d/out.bin | awk '/time_ms/{print $$2}'); \
	        if ./compare $$d/ref.bin $$d/out.bin > /dev/null; then st=PASS; pass=$$((pass+1)); \
	        else st=FAIL; fail=$$((fail+1)); fi; \
	        line="$$line | $$p $$t ms $$st"; \
	    done; rm -f $$d/out.bin; echo "$$line"; \
	done; echo "Done: $$pass passed, $$fail failed"; [ $$fail -eq 0 ]

# ---------------------------------------------------------------- existing scripts
test: build
	bash run_tests.sh results.csv

images: build
	bash run_image_tests.sh

check: build
	python3 run_tests.py check

bench: build
	python3 run_tests.py bench --out bench.csv
timing_plot:
	@[ -f timing.txt ] || $(MAKE) timing
	python3 plot_timing.py
# ---------------------------------------------------------------- timing
timing: build
	mkdir -p $(TIMEDIR)
	./gen_data $(TIME_CASE) 1 $(TIMEDIR)/input.bin $(TIMEDIR)/filters.bin
	@echo "== CPU naive (1 thread) + CPU fast (all threads)" | tee timing.txt
	./cpu_reference $(TIMEDIR)/input.bin $(TIMEDIR)/filters.bin $(TIMEDIR)/ref.bin both | tee -a timing.txt
	@echo "== CPU fast (1 thread)" | tee -a timing.txt
	OMP_NUM_THREADS=1 ./cpu_reference $(TIMEDIR)/input.bin $(TIMEDIR)/filters.bin $(TIMEDIR)/out.bin | tee -a timing.txt
	@echo "== conv_naive (GPU)" | tee -a timing.txt
	./conv_naive $(TIMEDIR)/input.bin $(TIMEDIR)/filters.bin $(TIMEDIR)/out.bin | tee -a timing.txt
	./compare $(TIMEDIR)/ref.bin $(TIMEDIR)/out.bin | tee -a timing.txt
	@echo "== conv_opt (GPU)" | tee -a timing.txt
	./conv_opt $(TIMEDIR)/input.bin $(TIMEDIR)/filters.bin $(TIMEDIR)/out.bin | tee -a timing.txt
	./compare $(TIMEDIR)/ref.bin $(TIMEDIR)/out.bin | tee -a timing.txt
	rm -f $(TIMEDIR)/out.bin
	@echo "timing results saved in timing.txt"

everything: all

# ---------------------------------------------------------------- cleanup
clean:
	rm -f $(CPU_BINS) $(GPU_BINS) results.csv bench.csv timing.txt
	rm -rf results

distclean: clean
	rm -rf $(DATADIR) $(TIMEDIR)

.PHONY: all build data verify test images check bench timing everything clean distclean
