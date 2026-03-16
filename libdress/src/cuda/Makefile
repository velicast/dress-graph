# Makefile for libdress_cuda (shared + static)
#
# Produces: libdress_cuda.so  libdress_cuda.a  dress_cuda.o
#
# When mpicc is available the MPI+CUDA bridge (delta_dress_fit_mpi_cuda)
# is compiled from dress_mpi.c with -DDRESS_CUDA and bundled into the
# library so that downstream examples only need -ldress -ldress_cuda.

NVCC   ?= nvcc
CC     ?= gcc

INC     = -I../../include -I..
CFLAGS  = -O2 -fPIC -DDRESS_CUDA $(INC)
NVFLAGS = -O2 -Xcompiler -fPIC $(INC)

OBJS = dress_cuda.o delta_dress_cuda.o delta_dress_impl_cuda.o

# Include MPI+CUDA bridge when mpicc is available
HAS_MPI := $(shell command -v mpicc >/dev/null 2>&1 && echo 1)
ifeq ($(HAS_MPI),1)
OBJS += dress_mpi_cuda.o
endif

all: libdress_cuda.so libdress_cuda.a

dress_cuda.o: dress_cuda.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<

delta_dress_cuda.o: delta_dress_cuda.c
	$(CC) $(CFLAGS) -c -o $@ $<

delta_dress_impl_cuda.o: ../delta_dress_impl.c
	$(CC) $(CFLAGS) -c -o $@ $<

dress_mpi_cuda.o: ../mpi/dress_mpi.c
	mpicc $(CFLAGS) -c -o $@ $<

LIBDIR  = ../../../build/libdress

libdress_cuda.so: $(OBJS)
	$(NVCC) -shared -o $@ $(OBJS) -lcudart -L$(LIBDIR) -Xlinker --no-as-needed -ldress

libdress_cuda.a: $(OBJS)
	ar rcs $@ $(OBJS)

clean:
	rm -f $(OBJS) libdress_cuda.so libdress_cuda.a

.PHONY: all clean
