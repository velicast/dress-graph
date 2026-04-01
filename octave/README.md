# dress-graph (Octave)

**A Continuous Framework for Structural Graph Refinement**

DRESS is a deterministic, parameter-free framework that iteratively refines the structural similarity of edges in a graph to produce a canonical fingerprint: a real-valued edge vector, obtained by converging a non-linear dynamical system to its unique fixed point. The fingerprint is self-contained, isomorphism-invariant by construction, guaranteed bitwise-equal across any vertex labeling, numerically stable (no overflow, no error amplification, no undefined behavior), fast and embarrassingly parallel to compute: DRESS total runtime is O(I * m * d_max) for I iterations to convergence, and convergence is guaranteed by Birkhoff contraction.

This directory contains the Octave package scaffolding. The `publish_octave`
target in `publish.sh` vendors the C sources and `.m` files from `libdress/`
and `matlab/`, then builds `dress-graph-<version>.tar.gz` ready for:

```
pkg install dress-graph-0.4.0.tar.gz
```

The generated package includes the MATLAB-compatible functional and OO APIs:

- CPU: `fit`, `delta_fit`, `DRESS`
- CUDA: `cuda.fit`, `cuda.delta_fit`, `cuda.DRESS`
- MPI: `mpi.DRESS(...).delta_fit(...)`
- MPI+CUDA: `mpi.cuda.DRESS(...).delta_fit(...)`

Examples live in `examples/octave/`:

- `cpu.m`
- `cuda_example.m`
- `rook_vs_shrikhande.m`
- `cpu_oo.m`
- `mpi_oo.m`
- `mpi_cuda_oo.m`

For development, use the MATLAB-compatible files in `matlab/` directly with
GNU Octave; the package build vendors those files and compiles the matching MEX
wrappers for the available CPU, CUDA, MPI, and MPI+CUDA toolchains.
