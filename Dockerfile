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

#the gdb-for-python somehow gets confused and thinks that PyUnicodeObject is
#a struct, instead of a typedef. This is probably related to how the shared
#object we're building is exporting symbols in some funny way, but either way
#gdb blows up whenever we do anything after loading typed_python, even if we
#remove all of the content from it. So probably a compiler problem. This
#hack ensures that it will work, instead of giving us
# 
# <class 'RuntimeError'> Type does not have a target.:
#
#everywhere in the gdb stacktraces, which is what happens when you try to 
#ask for .target() on the PyUnicodeObject 'struct' (without importing the module
#it's a typedef)
RUN sed -i 's/global _is_pep393/_is_pep393=True/' /usr/share/gdb/auto-load/usr/bin/python3.6m-gdb.py

ENTRYPOINT ["object_database_service_manager", \
    "--source", "/storage/service_source", \
    "--storage", "/storage/service_storage", \
    "--logdir", "/storage/logs"]
