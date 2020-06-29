#!/bin/bash

cd deps/mesh-fusion

# install libfusiongpu
cd libfusiongpu
mkdir build
cd build
cmake ..
make
cd ..
python setup.py build_ext --inplace
mv *.so cyfusion.so
cd ..

# install librender
cd librender
python setup.py build_ext --inplace
mv *.so pyrender.so
cd ..

# install libmcubes
cd libmcubes
python setup.py build_ext --inplace
mv *.so mcubes.so
cd ../../..
