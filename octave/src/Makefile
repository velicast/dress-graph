# Makefile for the dress-graph Octave package MEX files.
#
# Octave's pkg system calls `make` in this directory after extracting the
# tarball.  The C sources (dress.c, delta_dress.c, etc.) and headers are
# vendored into the package by build.sh.

MKOCTFILE ?= mkoctfile
CFLAGS_EXTRA = -O2 -fopenmp -I../inst/include
LDFLAGS_EXTRA = -lm -fopenmp

DRESS_C       = dress.c
DELTA_DRESS_C = delta_dress.c delta_dress_impl.c

# ── functional API ──────────────────────────────────────────────────
all: dress_mex.oct delta_dress_mex.oct \
     dress_init_mex.oct dress_fit_obj_mex.oct dress_get_mex.oct \
     dress_result_mex.oct dress_free_mex.oct

dress_mex.oct: dress_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

delta_dress_mex.oct: delta_dress_mex.c $(DRESS_C) $(DELTA_DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

# ── persistent-object MEX files ─────────────────────────────────────
dress_init_mex.oct: dress_init_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_fit_obj_mex.oct: dress_fit_obj_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_get_mex.oct: dress_get_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_result_mex.oct: dress_result_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

dress_free_mex.oct: dress_free_mex.c $(DRESS_C)
	$(MKOCTFILE) --mex $(CFLAGS_EXTRA) $(LDFLAGS_EXTRA) -o $@ $^

clean:
	rm -f *.oct *.o
