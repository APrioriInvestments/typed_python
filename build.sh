#!/bin/bash

while test $# -gt 0
do
    case "$1" in
        -b|--build ) 
            docker build . -t nativepython/cloud:latest
            ;;
        -w|--webtest )
            #run a dummy webframework
            docker run -it --rm -p 80:80 --entrypoint object_database_webtest nativepython/cloud:latest
            ;;
        -t|--test )
            #run unit tests
            docker run -it --rm nativepython/cloud:latest ./test.py -v -s
            ;;
        -r|--run )
            docker run -it --rm nativepython/cloud:latest bash
            ;;
        -p|--push )
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
