FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y install python3 python3-pip redis nano
RUN pip3 install redis nose gevent numpy psutil flask-sockets flask-cors requests websockets

RUN apt-get -y install libtcmalloc-minimal4
ENV LD_PRELOAD=libtcmalloc_minimal.so.4

RUN apt-get -y install gdb python3-dbg
ENV LD_PRELOAD=libtcmalloc_minimal.so.4

COPY . /nativepython/
WORKDIR /nativepython
RUN python3 setup.py install

ENTRYPOINT ["object_database_service_manager", \
    "--source", "/storage/service_source", \
    "--storage", "/storage/service_storage", \
    "--logdir", "/storage/logs"]
