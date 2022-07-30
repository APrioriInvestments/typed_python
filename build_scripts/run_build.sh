#!/bin/bash

set -ex

if [[ $# -eq 0 ]]; then
    echo "Usage: ./run_build.sh <python_version>"
    echo "Valid <python_verson>'s are 37 38 39 310"
    exit 1
fi

VERSION=$1

REPO_ROOT=$(realpath $(dirname $(realpath $0))/..)
ARCH=${2:-"x86_64"}
MANYLINUX=${3:-"manylinux2014"}  # PEP 599
BASE_IMAGE="quay.io/pypa/${MANYLINUX}_${ARCH}"

# Unfortunately the code is specific to x86_64 arch
#if [[ $ARCH = "aarch64" ]]; then
#
#  DOCKER_PLATFORM="--platform linux/arm64"
#fi

docker run --rm ${DOCKER_PLATFORM} \
    --volume ${REPO_ROOT}:/opt/apriori/typed_python \
    ${BASE_IMAGE} \
    /opt/apriori/typed_python/build_scripts/create_manylinux_wheel.sh $VERSION
