# dress-graph (Octave)

Octave Forge package for DRESS edge similarity.

This directory contains the Octave package scaffolding. The `publish_octave`
target in `publish.sh` vendors the C sources and .m files from `libdress/`
and `matlab/`, then builds `dress-graph-<version>.tar.gz` ready for:

```
pkg install dress-graph-0.4.0.tar.gz
```

For development, use the MATLAB-compatible files in `matlab/` directly with
GNU Octave (they already check for `__OCTAVE__`).
