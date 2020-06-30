#!/bin/bash

# activate environment
conda activate routed-fusion

# install all dependencies
cd deps

# install distance transform
cd distance-transform
pip install -e .
cd ..

# install graphics
cd graphics
pip install -e .
cd ..

# install tsdf
cd tsdf
pip install -e .
cd ..

