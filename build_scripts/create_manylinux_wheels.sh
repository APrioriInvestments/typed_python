#!/bin/bash

# Run this inside a manylinux docker container

# Remove build artifacts
rm -rf build dist wheels

# code dir
TP=${TYPED_PYTHON_DIR:-/opt/apriori/typed_python}

FLAVOR="cp"  # not pypy
ORIG_PATH=$PATH
for VERSION in 7 8 9 10; do
  # Manylinux image has supported python installed like /opt/python/cp38-cp38
  PYNAME="${FLAVOR}"3"${VERSION}"
  if [[ ${VERSION} == 7 ]]; then SUFFIX="m"; else SUFFIX=""; fi
  PYHOME=/opt/python/${PYNAME}-${PYNAME}${SUFFIX}
  PATH=${PYHOME}/bin:${ORIG_PATH}
  pip install pipenv
  PIPFILE="Pipfile_3_${VERSION}"
  ln -sf ${PIPFILE} Pipfile
  ln -sf "${PIPFILE}.lock" Pipfile.lock
  pipenv sync --python $(which python) && pipenv run python setup.py bdist_wheel
  rm Pipfile Pipfile.lock
done

for whl in dist/*.whl; do
  auditwheel repair "$whl" -w wheels
done
