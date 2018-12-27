##########################################################################
#  CONFIGURATION

# Path to python binary
PYTHON ?= $(shell which python3)

# Path to virtual environment
VIRTUAL_ENV ?= '.venv'

SRC_PATH ?= typed_python

BUILD_PATH ?= build/temp.linux-x86_64-3.6/typed_python

LIB_PATH ?= build/lib.linux-x86_64-3.6/typed_python

CPP_FLAGS = -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/include/python3.6m -I$(VIRTUAL_ENV)/include/python3.6m -std=c++14 -Wno-sign-compare -Wno-narrowing -Wno-unused-variable

O_FILES = $(BUILD_PATH)/_types.o \
          $(BUILD_PATH)/_runtime.o \
          $(BUILD_PATH)/native_instance_wrapper.o \
          $(BUILD_PATH)/Type.o \
          $(BUILD_PATH)/PythonSerializationContext.o


##########################################################################
#  RULES

$(VIRTUAL_ENV): $(PYTHON)
	virtualenv $(VIRTUAL_ENV) --python=$(PYTHON)

.PHONY: install
install: $(VIRTUAL_ENV)
	. $(VIRTUAL_ENV)/bin/activate; \
		pip install pipenv==2018.11.26; \
		pipenv install --deploy

.PHONY: test
test: testcert.cert testcert.key install
	. $(VIRTUAL_ENV)/bin/activate; \
	./test.py -s

.PHONY: lib
lib: typed_python/_types.cpython-36m-x86_64-linux-gnu.so

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.cc $(BUILD_PATH)
	$(CC) $(CPP_FLAGS) -c $< -o $@

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.cpp $(BUILD_PATH)
	$(CXX) $(CPP_FLAGS) -c $< -o $@

typed_python/_types.cpython-36m-x86_64-linux-gnu.so: $(LIB_PATH)/_types.cpython-36m-x86_64-linux-gnu.so
	cp $(LIB_PATH)/_types.cpython-36m-x86_64-linux-gnu.so  typed_python

$(LIB_PATH)/_types.cpython-36m-x86_64-linux-gnu.so: $(LIB_PATH)  $(O_FILES)
	$(CXX) -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 \
		$(O_FILES) \
		-o $(LIB_PATH)/_types.cpython-36m-x86_64-linux-gnu.so

$(BUILD_PATH):
	mkdir -p $(BUILD_PATH)

$(LIB_PATH):
	mkdir -p $(LIB_PATH)

.PHONY: docker-build
docker-build:
	./build.sh -b

.PHONY: docker-push
docker-push:
	./build.sh -p

.PHONY: docker-test
docker-test:
	./build.sh -t

.PHONY: docker-web
docker-web:
	./build.sh -w

testcert.cert testcert.key:
	openssl req -x509 -newkey rsa:2048 -keyout testcert.key -nodes \
		-out testcert.cert -sha256 -days 1000 \
		-subj '/C=US/ST=New York/L=New York/CN=localhost'

.PHONY: clean
clean:
	rm -rf build/
	rm -rf nativepython.egg-info/
	rm -f nose.*.log
	rm -f typed_python/_types.cpython-*.so
	rm -f testcert.cert testcert.key
	rm -rf .venv
