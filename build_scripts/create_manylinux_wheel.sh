#!/bin/bash

# Run this inside a manylinux docker container

# code dir
pushd ${TYPED_PYTHON_DIR:-/opt/apriori/typed_python}

FLAVOR="cp"  # cpython not pypy
ORIG_PATH=$PATH
VERSION=$1

VALID_VERSIONS=("37 38 39 310")
if [[ ! "${VALID_VERSIONS[*]}" =~ "$VERSION" ]]; then
    echo "Valid versions are 37, 38, 39, 310"
    exit 1
fi
    
# Manylinux image has supported python installed like /opt/python/cp38-cp38
PYNAME="${FLAVOR}${VERSION}"

# 37 in maintenance so has special suffix
if [[ ${VERSION} == 37 ]]; then SUFFIX="m"; else SUFFIX=""; fi

# select the right python
PYTHON=/opt/python/${PYNAME}-${PYNAME}${SUFFIX}/bin/python

# `python -m build` respects build deps in pyproject.toml
# `-w` option builds the wheel
${PYTHON} -m build -w

# make wheels that pypi recognizes
for whl in dist/*${PYNAME}*.whl; do
    auditwheel repair "$whl" -w wheels
done
