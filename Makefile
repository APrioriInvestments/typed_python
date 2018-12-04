install:
	pipenv install --system --deploy

test:
	./test.py -s

docker-build:
	./build.sh -b

docker-push:
	./build.sh -p

docker-test:
	./build.sh -t

docker-web:
	./build.sh -w


clean:
	rm -rf build/
	rm -rf nativepython.egg-info/
	rm -f nose.*.log
	rm -f typed_python/_types.cpython-*.so
