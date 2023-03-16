# Installation

## Development
There are several methods for building and setting up a development environment.

### Manual Method ###
1. Create a new virtualenv with Python 3.6 (`virtualenv --python=<path-to-py3> venv`) and source it.
2. Install requirements via pip. For the moment there are two options:
   * Install with plain pip using `pip install -r requirements.txt`
   * Install using Pipenv (which reads from the Pipfile).
3. Build nativepython libraries using `python setup.py build`
4. Install typed_python in the site-packages directory using `python setup.py install`
5. Move out of the typed_python root directory, as the source files can interfere with the installed package.

### Pipenv Method ###
This method is simple, and can take care of virtual environment creation and installation for you.
1. (Optional) Create a new virtualenv with Python 3.6 (`virtualenv --python=<path-to-py3> venv`) and source it. If you choose to use Pipenv alone, it will create an appropriate virtualenv for you.
2. Run `pipenv install --dev --deploy`

### Makefile Method ###
The included Makefile in this repository contains recipes for building, installing, and testing. For the moment, it is explicitly linked to a specific Python interpreter, so if you are using, say, an interpreter called `python3.6` (as opposed to `python3`), you will need to change the `PYTHON` variable at the top of the Makefile.

You can also customize the name and location of any built virtual environments with the `VIRTUAL_ENV` variable.

## Prerequisites ##

### OSX ###

* Python 3.6 (recommended installed with homebrew)
  * Currently build is tested against `clang`, not `gcc`. For more information about installing `clang` and configuring your environment see [here](https://embeddedartistry.com/blog/2017/2/20/installing-clangllvm-on-osx)
* It is recommended you use Pipenv ([see this link](https://pipenv.readthedocs.io/en/latest/install/#installing-pipenv)) to manage the application.
  * You can also use virtualenv.
* install Redis (`brew install redis`)


### Linux ###
(These instructions are only for Ubuntu for the moment)

Before building the modules in this repository, you will need to make sure that you have the following:
* Python 3.6 with header files (`python3.6-dev python3.6-dbg`)
  Note that for development you will also install the debug interpreter.
  If you are using **Ubuntu 16** or earlier,  you will first need to add the following PPA:
  ```
  sudo add-apt-repository ppa:jonathonf/python-3.6
  ```
  Then run as normal:
  ```
  sudo apt install python3.6-dev python3.6-dbg
  ```
* Pipenv ([see this link](https://pipenv.readthedocs.io/en/latest/install/#installing-pipenv))
* Redis Server (`redis-server`)
