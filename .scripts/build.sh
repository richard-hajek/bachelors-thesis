#!/usr/bin/env bash

eval "$(conda shell.posix hook)"

mamba="conda"

$mamba env create -f environment.yaml || true
$mamba env update -f environment.yaml --prune

conda activate csidrl

# do not move up, conda activate throws errros on set -eu
set -eu

export CUDA_HOME="$CONDA_PREFIX"
poetry install

cd lib/SATNet
export CUDA_HOME="$CONDA_PREFIX"
python3 setup.py install
