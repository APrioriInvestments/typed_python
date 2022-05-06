# Installation #
## OSX ##
(as of writing tested on both Apple Intel and Silicon Macs, OSX Big Sur+, and Python 3.6 & 3.7 only!)

### Steps ###
1. Install [Anaconda](https://www.anaconda.com/products/distribution). We'll be using their Conda package and environment manager.
2. Clone the repo: `git clone https://github.com/APrioriInvestments/typed_python`
3. Create and activate conda environmet (we'll use python 3.7 here): `conda create --name pyenv37 python=3.7 && conda activate pyenv37` (see this [cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) for other conda environment relate commands).
4. Install typed python: `pip install -e .` (assuming you are in the root of the directory)
5. Test it out: `pytest -v` (note a few performance tests could fail, but the majority should pass) 

## Linux ##

## Development ##
There are several methods for building and setting up a development environment.
  
### Manual Method ###
1. Create a new virtualenv with Python 3.6 (`virtualenv --python=<path-to-py3> venv`) and source it
2. Install requirements via pip. For the moment there are two options:
   * Install with plain pip using the `requirements`
   * Install using Pipenv (which reads from the Pipfile)
3. Build nativepython libraries using `python setup.py build`
4. Append the root of this repository to your `PYTHONPATH`
  
### Pipenv Method ###
This method is simple, and can take care of virtual environment creation and installation for you.
1. (Optional) Create a new virtualenv with Python 3.6 (`virtualenv --python=<path-to-py3> venv`) and source it. If you choose to use Pipenv alone, it will create an appropriate virtualenv for you.
2. Run `pipenv install --dev --deploy`
  
### Makefile Method ###
The included Makefile in this repository contains recipes for building, installing, and testing. For the moment, it is explicitly linked to a specific Python interpreter, so if you are using, say, an interpreter called `python3.6` (as opposed to `python3`), you will need to change the `PYTHON` variable at the top of the Makefile.
  
You can also customize the name and location of any built virtual environments with the `VIRTUAL_ENV` variable.



(These instructions are only for Ubuntu for the moment)

#### Prerequisites ####
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

