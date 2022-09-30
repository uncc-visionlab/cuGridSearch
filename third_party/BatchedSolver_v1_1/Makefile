#
# Simple Makefile for the batch solver/inverse routines.
#

# GPU_ARCH should be sm_13, sm_20, sm_30, or sm_35
GPU_ARCH ?= sm_35
ifeq ($(GPU_ARCH),sm_20)
 DEFINES   += -DFERMI
endif
ifeq ($(GPU_ARCH),sm_30)
 DEFINES   += -DKEPLER1
endif
ifeq ($(GPU_ARCH),sm_35)
 DEFINES   += -DKEPLER2
endif

# Set CUDA_VERBOSE to 1 if you want to see registers usage
CUDA_VERBOSE ?=0
ifeq ($(GPU_ARCH),1)
 CUDAOPT=-Xptxas -v 
endif

APPS      = example_batch_solver


example_batch_solver: example.c solve.cu inverse.cu
	nvcc -O3  -arch=$(GPU_ARCH) $(DEFINES) $(CUDAOPT) -o example_batch_solver example.c solve.cu inverse.cu

clean:
	rm -f $(APPS)

