FROM ubuntu:18.04

RUN apt-get -y update && \
    apt-get -y install \
        python3 python3-pip redis nano \
        libtcmalloc-minimal4 \
        gdb python3-dbg

ENV LD_PRELOAD=libtcmalloc_minimal.so.4

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

ENV APP_PATH /nativepython
WORKDIR $APP_PATH
COPY . .

# install virtualenv
RUN pip3 install --user virtualenv
ENV PATH ${PATH}:/root/.local/bin

# set-up virtualenv
RUN rm -rf .venv  # remove .venv if it already existed
RUN virtualenv --python $(which python3) .venv
ENV PATH $APP_PATH/.venv/bin:$PATH

# pipenv needs these which are normally provided by apt-get locales
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# install pipenv and use it to install our dependencies (and the library itself)
RUN pip install pipenv
RUN pipenv install --dev  --deploy
RUN make testcert.cert

ENTRYPOINT ["object_database_service_manager", \
   "--source", "/storage/service_source", \
   "--storage", "/storage/service_storage", \
   "--logdir", "/storage/logs"]
