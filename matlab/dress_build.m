% DRESS_BUILD  Compile the dress MEX file.
%
%   Run this script from the matlab/ directory (or adjust paths).
%   Requires: MATLAB with a configured C compiler (mex -setup).
%
%   Usage:
%     >> cd matlab
%     >> dress_build

fprintf('Compiling dress_mex ...\n');

mex('-O', ...
    ['-I' fullfile('..', 'libdress', 'include')], ...
    'CFLAGS=$CFLAGS -fopenmp', ...
    'LDFLAGS=$LDFLAGS -fopenmp', ...
    'dress_mex.c', ...
    fullfile('..', 'libdress', 'src', 'dress.c'), ...
    '-lm');

fprintf('Done. dress_mex.%s is ready.\n', mexext);
