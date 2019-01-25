# NativePython

The NativePython project is a framework for building high-performance real-time distributed systems in Python.

It consists of three modules:

* `typed_python`, a runtime library for expressing type constraints (not just hints) in Python programs
* `nativepython`, a compiler for Python programs that takes advantage of `typed_python` constraints to generate very efficient code.
* `object_database`, distributed software transactional memory, and a suite of tools to build distributed applications

These are standard modules that run on Python 3.6 and higher. You can use them
incrementally throughout your project -  add a few type constraints here and
there, or compile a couple of small but performance-critical functions. As you
add more type information, more of your program can be compiled. Everything
can still run in interpreter without compilation if you want.

## Where did this come from?

Every time I (Braxton) find myself writing a lot of  Python code, I eventually
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
error log just to find out that you passed a template parameter in the wrong order.

I developed NativePython to try to have the best of both worlds.  Nativepython
lets me have a single codebase, written entirely in Python, where I can
choose, depending on the context, which style of code I want, from totally
free-form Python with total type chaos, to statically typed, highly performant
code, and anything in between.

## How does it work?

We start with `typed_python`, which allows you to express the kinds of
semantics you see in strongly typed languages. `typed_python` provides a set
of datastructures that look and feel like normal Python types such as `list`,
`dict`, `class`, and the rest, but that have explicit type constraints applied
to them.

By using these datastructures, you can catch errors early: for instance, you
can create a `ListOf(str)`, which looks like a `list`, but will allow only
strings. Later, if you accidentally try to insert `None`, you'll get an
exception at insertion time, not later when you walk the list and find an
unexpected value inside of it.

Crucially, we allow you to model the natural variation of types present in
real python programs: for instance, you can say something like

    ListOf(OneOf(None, str, TupleOf(int)))

which is a list whose items must be `None`, strings, or tuples of integers.
This  allows you to continue to write code in a style that's pythonic, where
different values can have different types depending on context, but still add
the kinds of constraints you need to catch errors early.

Separately, the `nativepython` module provides a compiler for functions that
are expressed in terms of `typed_python` types.  For instance, if you write

    @TypedFunction
    def sum(l: ListOf(int)):
        res = 0
        for x in l:
            res += x
        return res

the compiler can generate far more code efficient than a JIT compiler can, because
it can _assume_ that the list contains integers. In fact, the internal memory representation
of the ListOf(int) is the same as a `std::vector<int>` in C++, and the code we generate
is essentially equivalent.  Of course, this function will work just fine in the
interpreter, so if you don't want to recompile your code (say, while debugging a
failing test), you'll get the normal performance you'd expect out of Python. But if you
compile it, you'll get an enormous speedup.

The standard types that `typed_python` provides are one-to-one with python,
and are designed to allow you to annotate existing python programs and have
them 'just work'. However, `typed_python` also provide operations that let you
trade safety for performance. For instance, you can call `unsafeGet` on a
`ListOf` object, indicating that you don't want bounds checking. In the
interpreter, if you make an out-of-bounds call you'll get an
UndefinedNativepythonOperation Exception. In compiled code, we'll drop the
bounds check entirely, which in some cases can be a significant performance
improvement.

## How mature is the project?

As of January 25th, 2019, I use `typed_python` and `object_database` daily in
our production research and trading application. The `nativepython` compiler
is just now getting enough features to be useful.  It produces pretty good code,
but most functions are still missing, and many optimizations remain.  Portions of the
`object_database` api are likely to change substantially as we work through the
best way to use it.

## How do I run tests?

Checkout the project and run `test.py` from the root.  It will automatically
compile the python extension and run tests. You can filter tests with a regex using `test.py --filter=PAT`.
