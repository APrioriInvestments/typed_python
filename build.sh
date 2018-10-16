#!/bin/bash

while test $# -gt 0
do
    case "$1" in
        -b|--build ) 
            docker build . -t nativepython:latest
            ;;
        -w|--webtest )
            #run a dummy webframework
            docker run -it --rm -p 80:80 nativepython:latest object_database_webtest
            ;;
        -t|--test )
            #run unit tests
            docker run -it --rm nativepython:latest ./test.py -v -s
            ;;
        -r|--run )
            docker run -it --rm nativepython:latest bash
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
