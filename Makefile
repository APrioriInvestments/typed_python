# Path to python binary
PYTHON ?= $(shell which python3)

# Path to virtual environment
VIRTUAL_ENV ?= '.venv'

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
