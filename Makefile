install: testcert.cert testcert.key
	pipenv install --system --deploy

test: testcert.cert testcert.key
	./test.py -s

docker-build:
	./build.sh -b

docker-push:
	./build.sh -p

docker-test:
	./build.sh -t

docker-web:
	./build.sh -w


testcert.cert testcert.key:
	openssl req -x509 -newkey rsa:2048 -keyout testcert.key -nodes \
		-out testcert.cert -sha256 -days 1000 \
		-subj '/C=US/ST=New York/L=New York/CN=localhost'

clean:
	rm -rf build/
	rm -rf nativepython.egg-info/
	rm -f nose.*.log
	rm -f typed_python/_types.cpython-*.so
	rm -f testcert.cert testcert.key
