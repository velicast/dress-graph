# Makefile for the dress-graph Octave package MEX files.
#
# Octave's pkg system calls `make` in this directory after extracting the
# tarball.  The C sources (dress.c, delta_dress.c, etc.) and headers are
# vendored into the package by build.sh.

MKOCTFILE ?= mkoctfile
NVCC ?= nvcc
MPICC ?= mpicc
CFLAGS_EXTRA = -O2 -fopenmp -I../inst/include
LDFLAGS_EXTRA = -lm -fopenmp
CUDA_CFLAGS_EXTRA = $(CFLAGS_EXTRA) -DDRESS_CUDA
CUDA_LDFLAGS_EXTRA = $(LDFLAGS_EXTRA) -lcudart
MPI_MKOCTFILE = env CC=$(MPICC) CXX=mpicxx $(MKOCTFILE)
MPI_CFLAGS_EXTRA = $(CFLAGS_EXTRA) -DDRESS_MPI
MPI_CUDA_CFLAGS_EXTRA = $(CUDA_CFLAGS_EXTRA) -DDRESS_MPI

DRESS_C       = dress.c
DELTA_DRESS_C = delta_dress.c delta_dress_impl.c dress_histogram.c
NABLA_DRESS_C = nabla_dress.c nabla_dress_impl.c dress_histogram.c
OMP_C         = dress_omp.c delta_dress_omp.c nabla_dress_omp.c
CUDA_KERNEL   = dress_cuda.o
MPI_C         = dress_mpi.c

HAS_CUDA := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)
HAS_MPI := $(shell command -v $(MPICC) >/dev/null 2>&1 && echo 1)

CUDA_TARGETS =
ifeq ($(HAS_CUDA),1)
CUDA_TARGETS += dress_cuda_mex.mex delta_dress_cuda_mex.mex nabla_dress_cuda_mex.mex dress_fit_cuda_obj_mex.mex
endif

MPI_TARGETS =
ifeq ($(HAS_MPI),1)
MPI_TARGETS += delta_dress_mpi_obj_mex.mex
endif

MPI_CUDA_TARGETS =
ifeq ($(HAS_MPI),1)
ifeq ($(HAS_CUDA),1)
MPI_CUDA_TARGETS += delta_dress_mpi_cuda_obj_mex.mex
endif
endif

# ── functional API ──────────────────────────────────────────────────
all: dress_mex.mex delta_dress_mex.mex nabla_dress_mex.mex \
     dress_omp_mex.mex delta_dress_omp_mex.mex nabla_dress_omp_mex.mex \
     dress_init_mex.mex dress_fit_obj_mex.mex dress_get_mex.mex \
	dress_result_mex.mex dress_free_mex.mex \
	$(CUDA_TARGETS) $(MPI_TARGETS) $(MPI_CUDA_TARGETS)

dress_mex.mex: dress_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

delta_dress_mex.mex: delta_dress_mex.c $(DRESS_C) $(DELTA_DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_omp_mex.mex: dress_omp_mex.c $(DRESS_C) $(OMP_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

delta_dress_omp_mex.mex: delta_dress_omp_mex.c $(DRESS_C) $(DELTA_DRESS_C) $(OMP_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

nabla_dress_mex.mex: nabla_dress_mex.c $(DRESS_C) $(NABLA_DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

nabla_dress_omp_mex.mex: nabla_dress_omp_mex.c $(DRESS_C) $(NABLA_DRESS_C) $(OMP_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

$(CUDA_KERNEL): dress_cuda.cu
	$(NVCC) -O2 -Xcompiler -fPIC -I../inst/include -c -o $@ $<

dress_cuda_mex.mex: dress_cuda_mex.c $(DRESS_C) $(CUDA_KERNEL)
	$(MKOCTFILE) --mex $(CUDA_CFLAGS_EXTRA) $(CUDA_LDFLAGS_EXTRA) -o $@ $^

delta_dress_cuda_mex.mex: delta_dress_cuda_mex.c $(DRESS_C) delta_dress_cuda.c delta_dress_impl.c dress_histogram.c $(CUDA_KERNEL)
	$(MKOCTFILE) --mex $(CUDA_CFLAGS_EXTRA) $(CUDA_LDFLAGS_EXTRA) -o $@ $^

nabla_dress_cuda_mex.mex: nabla_dress_cuda_mex.c $(DRESS_C) nabla_dress_cuda.c nabla_dress_impl.c dress_histogram.c $(CUDA_KERNEL)
	$(MKOCTFILE) --mex $(CUDA_CFLAGS_EXTRA) $(CUDA_LDFLAGS_EXTRA) -o $@ $^

delta_dress_mpi_obj_mex.mex: delta_dress_mpi_obj_mex.c $(DRESS_C) $(DELTA_DRESS_C) $(NABLA_DRESS_C) $(OMP_C) $(MPI_C)
	$(MPI_MKOCTFILE) --mex $(MPI_CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

delta_dress_mpi_cuda_obj_mex.mex: delta_dress_mpi_cuda_obj_mex.c $(DRESS_C) delta_dress.c delta_dress_cuda.c delta_dress_impl.c dress_histogram.c $(NABLA_DRESS_C) nabla_dress_cuda.c $(OMP_C) $(MPI_C) $(CUDA_KERNEL)
	$(MPI_MKOCTFILE) --mex $(MPI_CUDA_CFLAGS_EXTRA) $(CUDA_LDFLAGS_EXTRA) -o $@ $^

# ── persistent-object MEX files ─────────────────────────────────────
dress_init_mex.mex: dress_init_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_fit_obj_mex.mex: dress_fit_obj_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_fit_cuda_obj_mex.mex: dress_fit_cuda_obj_mex.c $(DRESS_C) $(CUDA_KERNEL)
	$(MKOCTFILE) --mex $(CUDA_CFLAGS_EXTRA) $(CUDA_LDFLAGS_EXTRA) -o $@ $^

dress_get_mex.mex: dress_get_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_result_mex.mex: dress_result_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_free_mex.mex: dress_free_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

clean:
	rm -f *.mex *.oct *.o
