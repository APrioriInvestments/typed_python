[![Build Status](https://travis-ci.com/APrioriInvestments/typed_python.svg?branch=dev)](https://travis-ci.com/APrioriInvestments/typed_python.svg?branch=dev)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# `typed_python`

The `typed_python` module provides strong runtime types to Python and a compiler
that can take advantage of them.

It gives you new types you can use to build strongly- and semi-strongly-typed
datastructures, so that your program is easier to understand, and a compiler toolchain
that can take advantage of those datastructures to generate machine code that's
fast, and that doesn't need the GIL.

`typed_python` is a standard modules that run on Python 3.6 and higher. You can use it
incrementally throughout your project -  add a few type constraints here and
there, or compile a couple of small but performance-critical functions. As you
add more type information, more of your program can be compiled. Everything
can still run in interpreter without compilation if you want.

`typed_python` is generously supported by [A Priori Investments](www.aprioriinvestments.com), a quantitative
hedge fund in New York.  If you're interested in working with us, drop us a line at info@aprioriinvestments.com.

## Getting started

You can read the [introductory tutorial](docs/introduction.md) for using `typed_python` or
check out the documentation [typed_python](docs/typed_python.md).

## Where did this come from?

Every time I (Braxton) find myself writing a lot of Python code, I eventually
start to miss C++. As my program gets bigger, I find myself losing track of
what types are supposed to go where. My code gets littered with 'isinstance'
assertions trying to catch mistakes early and provide information about what
kinds of types I expect in certain parts of the code. Compilers solve these
kinds of problems because the type information is out front directly in the code,
and they can find bugs without having to run the program.  And of course, I
miss the performance you get out of C++ - every time I write some overly complicated
numpy code, I think to myself how much easier to understand this code would be
if I could only write a loop.

On the other hand, whenever I write a lot of C++, I find myself missing the
expressiveness of Python, the ability to iterate without a painful compile
cycle, the safety you get from having bad code throw an exception instead of
producing a segfault.  And of course, nobody likes looking through a ten page
error log just to find out that a template parameter is missing.

I developed `typed_python` to try to have the best of both worlds.  `typed_python`
lets me have a single codebase, written entirely in Python, where I can
choose, depending on the context, which style of code I want, from totally
free-form Python with total type chaos, to statically typed, highly performant
code, and anything in between.
