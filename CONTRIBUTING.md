# Contributing to dress-graph

Thank you for your interest in contributing to DRESS! This document covers how to
get involved, report issues, and submit changes.

## Getting Started

1. Fork the repository and clone your fork
2. Set up the build environment:
   ```bash
   ./setup_env.sh
   ./build.sh --no-test
   ```
3. Run the test suite to make sure everything works:
   ```bash
   ./build.sh
   ```
4. Run the examples to verify the bindings:
   ```bash
   ./run_examples.sh
   ```

## Architecture

DRESS has a layered architecture:

- **`libdress/`** — The core C backend. This is the source of truth for the algorithm.
  All correctness-critical code lives here and is maintained manually with full test coverage.
- **`libdress++/`** — C++ wrapper around the C core.
- **`libdress-igraph/`** — igraph integration layer (C).
- **Language bindings** (`python/`, `rust/`, `go/`, `julia/`, `r/`, `matlab/`, `octave/`, `wasm/`) —
  Thin wrappers that call into the C backend via FFI.

### A Note on Wrappers

The language wrappers are functional but evolving quickly. If you find a bug in a
wrapper, please open an issue with:
- The language and version you are using
- A minimal reproducing example
- The expected vs. actual output

Wrapper contributions (bug fixes, ergonomic improvements, better error messages) are
especially welcome.

## How to Contribute

### Reporting Bugs

Open a [GitHub Issue](https://github.com/velicast/dress-graph/issues) with:
- Your OS, compiler, and language version
- Steps to reproduce
- Expected behavior vs. actual behavior
- If possible, a minimal code snippet

### Suggesting Features

Open an issue tagged `enhancement`. Describe the use case and why it matters.

### Submitting Code

1. Create a branch from `main`
2. Make your changes
3. Add or update tests if applicable
4. Make sure the test suite passes: `./build.sh`
5. Open a Pull Request with a clear description of what changed and why

### Areas Where Help Is Needed

- **Wrapper quality**: Testing and hardening the Python, Rust, Go, Julia, R,
  MATLAB/Octave, and WASM bindings across platforms
- **Documentation**: Tutorials, examples, and guides for specific use cases
  (ML feature engineering, community detection, etc.)
- **Benchmarks**: Performance comparisons on new graph families or real-world datasets
- **Applications**: If you use DRESS for a downstream task (classification, clustering,
  retrieval, etc.), we would love to hear about it

## Code Style

- **C**: Follow the existing style in `libdress/src/`. No external dependencies beyond
  the C standard library (and optionally CUDA, MPI).
- **Python**: Standard PEP 8.
- **Other languages**: Follow the idiomatic style for that language.

## Testing

The C test suite is in `tests/c/`. Python tests are in `tests/python/`.
Run everything with:

```bash
./build.sh           # builds and runs C tests
cd python && pytest  # Python tests
```

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
