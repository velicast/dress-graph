# dress-graph (Octave)

**A Continuous Framework for Structural Graph Refinement**

DRESS is a deterministic, parameter-free framework for continuous structural graph refinement. It iterates a nonlinear dynamical system on real-valued edge similarities and produces a graph fingerprint as a sorted edge-value vector once the iteration reaches a prescribed stopping criterion. The resulting fingerprint is self-contained, isomorphism-invariant by construction, reproducible across vertex labelings under the reference implementation, numerically robust in practice, and efficient to compute with straightforward parallelization and distribution.

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
