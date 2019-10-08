## An introduction to typed_python

`typed_python` provides a library of new datastructures that express different
kinds of type semantics, and a compiler for compiling python
code written using `typed_python`.

The library can be used entirely incrementally - you can use a
`typed_python` datastructure in just one part of your program and affect
nothing else, and you can use `typed_python.Entrypoint` to compile specific functions in
your program that are slow.

`typed_python` provides a collection of datastructures that look and feel like
normal Python types such as `list`, `dict`, `class`, etc, but that
have explicit type constraints applied to them.

For instance, you can create a list that only accepts strings:

```
from typed_python import ListOf

l = ListOf(str)()
l.append("this is a string")
```

`l` looks and feels like a normal python list, except that if you attempt to
append something other than a string you'll get an exception.

Real python programs often have type variation. For instance, I often
write programs where a dictionary value can be a string, or None. We can
model this by writing something like this:

```
from typed_python import Dict, OneOf

d = Dict(str, OneOf(str, None))()

d["hi"] = "bye"
d["deleted"] = None
```

By using strongly typed datastructures like this, you can catch errors in your
program early.  This is far better than littering your program with `isinstance`
checks everywhere, and it also gives the reader of your code more information
about what's in your program.

`typed_python` supports a full set of datastructures, including `ListOf`, `TupleOf`,
`NamedTuple`, `Dict`, `Set`, `ConstDict`, `Alternative`, and most importantly,
`Class`.

If you inherit from `Class` instead of `object` when you define a class, you'll
get a strongly typed class with a packed memory layout. You have to define the
types of the members of your class, but other than that, everything proceeds
the way you would expect:

```
from typed_python import Class, Member

class MyClass(Class):
    x = Member(int)
    y = Member(int)

    def f(self):
        return (self.x, self.y)

m = MyClass(x=10, y=20)

print(m.f())
```

`Class` objects support inheritance the same way normal Python classes do. However, they
also support function overloading for their methods. For instance, you can
write

```
class MyClass(Class):
    def f(self):
        return "No Arguments"

    def f(self, x):
        return "One Argument"

    def f(self, x: int, y: int):
        return "Two Ints"

    def f(self, x, y):
        return "Two Arguments"
```

In this case, if you call `MyClass.f` with zero, one, or two arguments, you'll
get routed to the appropriate function. `typed_python` checks each function
overload in turn to see whether its types match your desired signature.

Using these pieces, you can take pretty much any python program, incrementally add
type annotations to it, and end up with a program that has runtime type safety.

The real power of `typed_python`, however, is that any python code operating
on typed_python Class instances and datastructrures is inherently easier
to reason about, because we can make strong assumptions about the kinds of
values flowing through the program.

This lets us generate efficient, compiled code, on a per-function basis.
Simply stick the `typed_python.Entrypoint` decorator around
any function that uses `typed_python` primitives to get a fast version
of it:

```
@Entrypoint
def sum(someList, zero):
    for x in someList:
        zero += x
    return x
```

will generate specialized code for different data types (`ListOf(int)`, say, or
`ListOf(float)`, or even `Dict(int)`) that's not only many times faster than
the python equivalent, but that can operate using multiple processors.
Compilation occurs each time you call the method with a new combination
of types.
