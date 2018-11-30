#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset
set -o xtrace


COMMIT=$(git rev-parse HEAD)

while test $# -gt 0
do
    case "$1" in
        -b|--build )
            rm -rf build
            rm -rf nativepython.egg-info
            docker build . -t nativepython/cloud:"$COMMIT"
            docker tag nativepython/cloud:"$COMMIT"  nativepython/cloud:latest
            ;;
        -w|--webtest )
            #run a dummy webframework
            docker run -it --rm -p 8000:8000 --entrypoint object_database_webtest nativepython/cloud:latest
            ;;
        -t|--test )
            #run unit tests in the debugger
            docker run -it --rm --privileged --entrypoint bash nativepython/cloud:latest -c "gdb -ex run --args /usr/bin/python3 ./test.py -v -s"
            ;;
        -r|--run )
            docker run -v $(pwd):/code -p 8000:8000 --workdir /code -it --rm --entrypoint bash nativepython/cloud:latest
            ;;
        -p|--push )
            docker push nativepython/cloud:"$COMMIT"
            docker push nativepython/cloud:latest
            ;;
        -h)
            echo "usage: ./build.sh [-b|--build] [-t|--test] [-r|--run] [-p|--push]"
            exit 0
            ;;
        *)  echo "unknown argument: $1"
            exit 1
            ;;
    esac

    shift
done
