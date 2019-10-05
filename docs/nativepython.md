## nativepython

`nativepython` is a compiler for Python programs that takes advantage of any
uses of `typed_python` to generate fast code and to sidestep the GIL (Global
Interpreter Lock), giving you real concurrency when you use multithreading.

Currently, it can be used as a JIT compiler. Eventually, we'd like to use it
to compile static binaries with a fixed set of entrypoints
into the compiled binary, and to check programs that use `typed_python` for defects
(say, accessing an attribute that cannot exist based on the types in the program).

`nativepython`'s goal is to maintain exact correspondence with the interpreted
version of any piece of code it compiles, with the following exceptions:

* all integer arithmetic is limited to 64 bits for performance reasons.
You may see overflow in compiled programs you don't see in interpreted ones.
* the messages and tracebacks of exceptions thrown by compiled code may
not exactly match the behavior of the interpreter.
* some multithreaded programs may crash if they access datastructures without locks.

The last point is particularly tricky: some race conditions in normal Python may
succeed without crashing (say, inserting into a dictionary from two threads at the same time)
because Python performs the operation atomically while holding the GIL.
`nativepython` programs release the GIL whenever possible,
and will behave differently in the presense of such races.

`nativepython` interacts with `typed_python` instances directly. This means you can
produce an instance of a `typed_python.ListOf` in interpreter and pass it to a piece
of compiled code with no need for any additional conversion or marshalling. This
lets you pick and choose which portion of your program you compile.

Currently, the only way to invoke `nativepython` is to wrap a function or staticmethod
in the `SpecializedEntrypoint` decorator.

`SpecializedEntrypoint` indicates that you want to cross from the interpreter
into native code. Each time you execute a function marked with
`SpecializedEntrypoint`, `typed_python` examines the types of your arguments
and the annotations on the function itself to decide which version of the
function to branch into.  If your function has type annotations on every
argument, only one version of the function will ever be used. If your function
has no type annotations, `nativepython` will produce a different specialization
for each distinct combination of types you call the function with.

As an example

```
@Entrypoint
def sum(container, zero = 0):
    result = zero

    for element in container:
        result = result + element

    return result
```

will produce different compiled representations if you hand it a `ListOf` or a `Dict` because
the machine code to iterate those two datastructures are completely different.

`nativepython` is still very much a work in progress. Much of Python3 can be compiled,
including much of the core string functionality, most of the typed_python datastructures
including ListOf, Dict, Alternative, etc, and Class instances (with inheritance).

However, large gaps still remain. In particular,

* we don't allow try/catch yet. We do allow you to raise exceptions, but we don't yet give line numbers.
* we can't handle untyped lists, tuples, or dictionaries.
* we can't handle passing functions or function closures as arguments yet
* we have yet to implement calling with keyword arguments for user-defined functions
* lots of little details have deviations from standard python behavior or are simply not implemented yet.

Where `nativepython` cannot compile code, the compilation process currently
throws an exception. Eventually, we intend for the compiler to simply fall
back to the interpreter in those cases.

