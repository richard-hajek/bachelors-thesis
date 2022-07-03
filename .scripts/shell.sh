#!/usr/bin/env bash

eval "$(conda shell.posix hook)"

export CUDA_HOME="$CONDA_PREFIX"
export TMPDIR="$HOME/.cache"
