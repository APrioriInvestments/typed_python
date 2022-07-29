#!/bin/bash

set -ex

REPO_ROOT=$(realpath $(dirname $(realpath $0))/..)
ARCH=${1:-"x86_64"}
MANYLINUX=${2:-"manylinux2014"}  # PEP 599
BASE_IMAGE="quay.io/pypa/${MANYLINUX}_${ARCH}"

if [[ $ARCH = "aarch64" ]]; then
  DOCKER_PLATFORM="--platform linux/arm64"
fi
docker run \
    --rm \
    ${DOCKER_PLATFORM} \
    --volume ${REPO_ROOT}:/opt/apriori/typed_python \
    --workdir /opt/apriori/typed_python \
    ${BASE_IMAGE} \
    /opt/apriori/typed_python/build_scripts/create_manylinux_wheels.sh
