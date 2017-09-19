# nativepython

`nativepython` is an llvm-based code generation framework for the Python programming language.
`nativepython` is not an implementation of the python programming languauge. Instead, it's a library
that lets you build high-performance object code and call it directly from Python. The object model
is similar to C++ (e.g. it's statically typed, objects have a strict lifetime on the stack or heap,
with constructors and destructors, and it allows direct memory manipulation). However, instead of
relying on templates to control how objects get constructed, we allow Python code to control the
type system, object dispatch, etc.

## Why do we need this?

I wanted to build a high-performance real-time event processing and simulation framework, which in
order to perform well, needs a JIT compiler. Rather than building the core object model in C++
around  llvm and then integrating it into Python, I thought it would be nicer to generate all the
code for both the runtime and the user code in the same framework.  The result allows us to  have
both dynamically typed and statically typed code coexisting in a single codebase in a single
language. It also allows us to easily build systems that can specialize themselves for new types of
data as they show up.

## How do I run tests?

run `test.py` from the root. Filter tests with a regex using `test.py -filter=PAT`.
