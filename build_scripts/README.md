# Wheel builds for typed_python

Using [Manylinux](https://github.com/pypa/manylinux) we can make typed_python more portable.

## Building wheels

1. Prerequisites: docker
2. Execute the `run_build.sh` script

This process should create a `./wheels` directory under the typed_python repo root with
4 wheels for python 3.7-3.10 built for linux x86_64.

## Release to pypi

1. pip install twine
2. twine upload wheels/*.whl

## Notes

Here we're using the CentOS 7 based manylinux2014 image. 
We cannot move to the newer `manylinux_x_y` image ([PEP 600](https://peps.python.org/pep-0600/)) 
until type_python drops support for python 3.7
