#!/bin/bash

# a little script to install TP directly in a new venv
# and run the array tests.
rm -rf .test_venv;
python3 -c 'import sys; print("python version is ", sys.version_info)';
python3 -m venv .test_venv;
. .test_venv/bin/activate;
pip install -v -e .;
pip install scipy;
pip install pytest;
pip install flaky;
TP_COMPILER_CACHE= pytest -k array_test -vs;
