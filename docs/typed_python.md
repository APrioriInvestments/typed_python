## Typed Python

`typed_python` is a library for describing strongly and semi-strongly typed
data-structures in Python. `typed_python` types look and feel like their
builtin equivalents, but restrict the set of types that can be used with them.
For instance, a `ListOf(OneOf(None, int))` behaves like a `list` that only
contains `int` instances or the value None.

`typed_python` exists for two related reasons. The first is to provide type
annotations that can be enforced at runtime, so that the portions of your
program that use `typed_python` are easier to read, debug, and maintain. The
second is to provide a set of types that can be directly manipulated from
compiled code produced by the `nativepython` module. In particular,  compiled
code can manipulate these objects without interacting with the Python
interpreter, which means they can release the GIL and efficiently use
multithreading.

It's important to note that `typed_python` is a Python module, not a separate
Python implementation. This means you can use it as much or as little as you
want in a project without affecting other code.

### Container Types

Each core python type (`set`, `list`, `dict`, `tuple`) has a corresponding
`typed_python` equivalent: `Set`, `ListOf`, `Dict`, `TupleOf`. These aren't
types, they're type functions: you make concrete types by parameterizing them
with other types. For instance, `ListOf(float)` looks and feels like a regular
`list`, but it insists that all of its members are actually 64-bit floats. If
you attempt to append an object that's not a `float`, `typed_python` will
attempt to convert it to a float, and if it can't, it will throw an exception.
Similarly, a `Dict(int, str)` will only allow integer keys and string values.

### OneOf

In real python programs, a single container or variable often holds several
different types over its lifetime.  You can use use the `OneOf` type constructor to
express this kind of variation.  For instance, you may write
`TupleOf(OneOf(int, None))` to represent a tuple of integers or `None` values.

All type constructors allow you to use python primitive values which get
converted to `Value` types. For instance, you might write `TupleOf(OneOf(1, 2, "hi"))`,
which means a tuple containing only the values 1, 2, and "hi".  This
allows you to represent constraints on the actual values flowing through your
program.

`OneOf` types are never directly instantiated. For instance, the elements of a
`TupleOf(OneOf(int, float))` will always appear in the interpreter as either
an `int` or a `float`, but you'll never have a case where
`type(x) is OneOf(int, float)`.  Instead, you will always use `OneOf` to describe
the kinds of types that may be present in other types.

Also note that you may use python primitives in `OneOf` to indicate that
only certain values are available: `OneOf(1, 2, 3)` will accept only those three
integers.

### Arithmetic types

`typed_python` provides a set of arithmetic types that correspond to the integer and float types
available in hardware. These include `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Int8`, `Int16`, `Int32`,
and `Float32`.  `Int64` is the internal `typed_python` representation of `int` and `Float64` of
`float`. Note that the only place where `typed_python` intentionally deviates from normal
python semantics is its treament of integers. For performance and memory reasons, we chose to map
`int` to `Int64` rather than retaining python's arbitrarily large integers. For most applications
this won't be a problem, but if you attempt to place an integer larger than 64 bits into a
`typed_python` container, you'll see the integer get cast down to 64 bits.

### Object

In some cases, you may have types that need to hold regular python objects. For these cases, you may
use `object`, or any Python type, and `typed_python` will only allow values that are instances of that
type in that context. These values won't be converted in any way (they're held as normal python objects),
and any operations on them will be performed using the interpreter.

### ConstDict

`ConstDict(K, V)`  models an immutable dictionary. It behaves like a `Dict` except that it
can't be modified and its items are iterated in sorted order instead of insertion order, like a `dict`.

New `ConstDict` instances may be created by adding two ConstDicts together, which produces
a new dictionary which is the union of the two dictionaries (with the right hand side of
the operation taking precedence if a key is defined twice). You may remove keys
from a ConstDict by subtracting another iterable from it.

### Alternative

`Alternative` defines strongly typed tagged unions. They are especially useful
for defining syntax trees and mutually exclusive configuration options.

Alternatives are defined by providing a list of subclasses, each of which has
a set of named and typed fields. An instance of a given Alternative type must
be one of the subinstances.  You may access the fields of the subinstance
directly, and find out which subtype of the Alternative you're dealing with
using `matches`.

For instance, you could write

```
ColorDefinition = Alternative(
    "ColorDefinition",
    Rgb=dict(
        red=float,
        blue=float,
        green=float
    ),
    Named=dict(
        name=OneOf('red', 'blue', 'white', 'black', 'green', 'yellow')
    )
)
```

which you can read as "A ColorDefinition is either a ColorDefinition.Rgb with
fields 'red', 'blue', and 'green', which are floats, or it is a
ColorDefinition.Named with a field called 'name' which is one of the strings
'red', 'blue', 'white', 'black', 'green', or 'yellow'"

Each of the specific options in an Alternative is actually a subclass. In
this case, `ColorDefinition.Rgb` and `ColorDefinition.Named` are both
subclasses of `ColorDefinition` with different members, and you
can construct one or the other by calling these subclasses:

```
ColorDefinition.Rgb(red=1.0, blue=.5, green=0.0)
```

If you don't provide an argument for one of the options, typed_python
will attempt to initialize it with its default constructor.

You can determine which subclass you're dealing with using the 'matches'
member:

```
def formatColor(x: ColorDefinition):
    if x.matches.Rgb:
        return f"Rgb({x.red},{x.blue},{x.green})"
    elif x.matches.Named:
        return f"Named({x.name})"
    else:
        raise Exception(f"Unknown subclass of ColorDefinition: {x}"")
```

### NamedTuple

`NamedTuple` allows you to define a tuple with a specific set of typed members. You can
construct a `NamedTuple` type by passing it a set of types by keyword.  You
may call the resulting type with any subset of its members as keyword arguments,
and `typed_python` will construct those members with the given arguments and will
attempt to default-initialize the remaining members.

For instance, you may write

```
T = NamedTuple(x=int, y=str, z=OneOf(None, float))

aTup = T(x=10)
print(f"aTup.y is {aTup.y}")
```

In this case, `T` will be a named tuple with fields `x`, `y`, `z`. You can construct
the named tuple with keyword arguments, and `typed_python` will default construct
any missing arguments. You may also construct a named tuple using a dictionary of
named values.

A `NamedTuple` is still just an ordinary tuple, and can be used with integer
indices. For instance, `T().x == x()[0]`.

### Tuple

You may also describe an unnamed tuple with a fixed set of typed elements using
`Tuple`. For instance, `Tuple(int, float)` is a tuple holding an int followed by
a float. It behaves like `NamedTuple` except that the values have no names.

### Forward types

Some type hierarchies are naturally recursive. For these cases, you can define `Forward` types.
For instance, if we were trying to describe what are acceptable 'json' values,
we would want to write something like

```
Json = OneOf(None, bool, int, float, TupleOf(Json), ConstDict(str, Json))
```

which of course cannot work because `Json` isn't defined yet when we're evaluating
the right-hand side of the assignment.

To solve this problem we introduced `Forward` which allocates an object that
represents a type that hasn't been defined. Once we define the type, any type
using the `Forward` resolves to the definition:

```
Json = Forward("Json")

Json = Json.define(
    OneOf(None, bool, int, float, TupleOf(Json), ConstDict(str, Json))
)
```

This lets you define types that refer to each other or themselves recursively.

### Functions

Typed python also provides facilities for creating typed functions with formal type signatures.

Any python function can be decorated with the `TypedFunction` decorator.  A
`TypedFunction` examines the type annotations placed on the function and uses
them to create an explicit type signature. Any arguments that have no type
annotations will accept any argument. If the arguments to a function call
cannot be coerced to the appropriate types, `typed_python` will throw an
exception. Similarly, if the function has a return type annotation,
`typed_python` will attempt to coerce the result to that type, and will throw
an exception if it is unable to do so.

For example,

```
@TypedFunction
def f(x: int, y: str) -> OneOf(None, int):
...
```

will insist on taking arguments of type `int` and `str` and will always return
`None` or an integer.

### Classes

Typed python classes are created by subclassing `Class`. The class definition
must define each member that the resulting class instance will have by
assigning an instance `Member` in the class namespace.  The resulting members
may be read and assigned by attribute, as with any normal python class
instance. Other attributes will produce an `AttributeError` exception.

`Class` instances may declare an `__init__` method, in which case the members
of the class must be explicitly initialized by the method. If the `__init__`
method is not defined, `typed_python` will generate a default constructor that
allows keyword assignment of the members, and that default initializes any
member not listed in the keyword arguments if the member is of a type that has
a default initialization.  Note that `Class` instances do not ever have a
default initialization, even if they have an `__init__` method that  takes no
arguments, because default initialization must be guaranteed to never raise
exceptions or have side effects.

All `Class` methods are implicitly `TypedFunction` instances, so any
annotations placed on them will actually affect control flow. In addition, you
may define multiple versions of the same function in the class namespace.
`typed_python` will search for the first version of the function that matches
your arguments and invoke that version. This allows you to write different
versions of the same function that take different numbers of arguments or
different types.

As an example,

```
class MyClass(Class):
    anInt = Member(int)
    aString = Member(OneOf(None, str))
    aRandomObject = Member(object)

    def getMyInt(self):
        # this will always be an int
        return self.anInt

    def getMyString(self):
        return self.aString

    def getMyString(self, ifNone):
        # this version gets called if you pass an argument.
        return self.aString if self.aString is not None else ifNone
```

defines a class with three members, `anInt`, `aString`, and `aRandomObject`.
Because it has no `__init__` method and all of its members have default
initializers, it may be constructed as `MyClass()`, `MyClass(anInt=1)`, etc.
It defines `getMyString` twice with different kinds of arguments and will
dispatch to them depending on whether you call it with zero or one argument.

`typed_python` classes support inheritance. In the current implementation,
if you use multiple inheritance, only one of the base classes may have members.
`typed_python` follows the standard method resolution order when attempting
to resolve method calls, checking each method to see whether its type signature
matches.  This means you can override the behavior of a subclass for specific types:

```
class BaseClass(Class):
    def f(self, x):
        return "I'm the base class"

class ChildClass(BaseClass):
    def f(self, x: int):
        return "I'm the child class, and x is an integer"
```

In this example, `ChildClass.f` will only apply to integers, and will use
the `BaseClass.f` for everythign else.

A class may also inherit from `Final`, which indicates that the class
can no longer be subclassed. (This can improve the performance of compiled
code acting on instances of that type, because they can dispatch to the
class' methods without looking in the class vtable).

### Serialization

`typed_python` provides a stable serialization format loosely based on google's
protobuf format.  The functions `typed_python.serialize(T, instance)` and
`typed_python.deserialize(T, someBytes)` can convert between `typed_python` instances
and `bytes` objects.  The format is upgradeable in the sense that new fields can be added
to Classes, Alternatives, and NamedTuples without making old serialized messages
unreadable.

### TypeFunctions

`TypeFunctions` exist to support the description of types that are
parameterized by other types.  They are analagous to templates in C++.  Any
function that accepts types or primitive python values that that returns types
may be decorated with the `TypeFunction` decorator. The decorator memoizes
the result, allowing you to use the call to the function as the definition of the
type.

For instance,

```
@TypeFunction
def LinkedListOf(T):
    class LinkedList(Class):
        head = Member(T)
        tail = OneOf(None, LinkedListOf(T))
    return LinkedList
```

defines a parameterized class 'LinkedList'. You're guaranteed that
if you call the function with the same arguments you'll always get the
same result.  We only call the function once per unique argument set.

Type functions can be used to provide custom implementations of a datastructure
based on its argument types (much like template specialization).

### PointerTo

In order to facilitate creation of fast (but unsafe) code in the compiler, you
may ask for a pointer to the internals of some `typed_python` datastructures.
For instance, you may write something like

```
def sumListFast(l: ListOf(int)):
    p = l.pointerUnsafe(0)
    pMax = l.pointerUnsafe(len(p))

    res = 0
    while p < pMax:
        res += p.get()
        p += 1

    return res
```

The `pointerUnsafe` method gets a pointer to the internals of the `ListOf`. Like
a C++ `std::vector`, a pointer to a held element of a vector is only valid as long as
we don't increase the capacity of the ListOf (say by appending to it), or delete
the relevant value.

In interpreted code, there is no reason to use 'pointerUnsafe'. But in
compiled code, accessing the pointer value using `p.get()` avoids having to
check the bounds of the list, and is therefore faster.

The result of `pointerUsafe` is an instance of `PointerTo`.  You can use
PointerTo in interpreted code or compiled code, and in either case, improper
usage can cause undefined behavior leading to either memory corruption or
a segmentation fault. Use this feature only if you know what you're doing.

### Type Introspection

All `typed_python` types inherit from `Type`. Each of the major type functions
(`ListOf`, `Dict`, `Class`, etc) are subtypes of `Type`, and each of their resulting
types are subtypes of those types (so that `ListOf(int)` is a subclass of `ListOf` and then
`Type`).

Each `typed_python` type describes its type parameters in associated fields present
on the type itself. For instance, the type object `Dict(str, int)` has members `KeyType`
and `ValueType`.

* Each `Alternative` has a list of subclasses defined in `__typed_python_alternatives__`.
* Each `Alternative` subclass knows its alternative as `Alternative`, its index in its parent as `Index`,
its tag name as `Name`, and the `NamedTuple` that makes up its body as `ElementType`.
* Each `Class` has a list of `MemberNames` and `MemberTypes`, along with `BaseClasses`, `MRO`, and `IsFinal`.
* Each `Tuple` has its items in `ElementTypes`
* Each `NamedTuple` has its names in `ElementNames` and its types in `ElementTypes`
* Each `ConstDict` and `Dict` has its key type as `KeyType` and its value type as `ValueType`.
* Each `Function` has a list of the individual function overloads as `overloads`. See `typed_python.internals.FunctionOverload`.
* Each `PointerTo` has the pointed-to type as `ElementType`.
* Each `Set` has the item type as `KeyType`
* Each `ListOf` and `TupleOf` has its element type as `ElementType`.

## The Compiler

The module `typed_python.compiler` contains a compiler for Python programs
that takes advantage of any uses of `typed_python` to generate fast code and
to sidestep the GIL (Global Interpreter Lock), giving you real concurrency
when you use multithreading.

Currently, it can be used as a JIT compiler. Eventually, we'd like to use it
to compile static binaries with a fixed set of entrypoints into the compiled
binary, and to check programs that use `typed_python` for defects (say,
accessing an attribute that cannot exist based on the types in the program).

`typed_python`'s goal is to maintain exact correspondence with the interpreted
version of any piece of code it compiles, with the following exceptions:

* all integer arithmetic is limited to 64 bits for performance reasons.
You may see overflow in compiled programs you don't see in interpreted ones.
* the messages and tracebacks of exceptions thrown by compiled code may
not exactly match the behavior of the interpreter depending on optimization
* some multithreaded programs may crash if they access datastructures without locks.

The last point is particularly tricky: some racey programs written in normal
Python may succeed without crashing (say, inserting into a dictionary from two
threads at the same time) because Python performs the operation atomically
while holding the GIL. Compiled `typed_python` programs release the GIL whenever
possible, and will behave differently in the presense of such races.

The compiler interacts with `typed_python` instances directly. This means you can
produce an instance of a `typed_python.ListOf` in interpreter and pass it to a piece
of compiled code with no need for any additional conversion or marshalling. This
lets you pick and choose which portion of your program you compile. The `typed_python`
datastructures act as a common memory format that both the interpreter and compiler
can use.

Currently, the only way to invoke the compiler is to wrap a function or staticmethod
in the `Entrypoint` decorator.

`Entrypoint` indicates that you want to cross from the interpreter
into native code. Each time you execute a function marked with
`Entrypoint`, the compiler examines the types of your arguments
and the annotations on the function itself to decide which version of the
function to branch into.  If your function has type annotations on every
argument, only one version of the function will ever be used. If your function
has no type annotations, the compiler will produce a different specialization
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

The compiler is still very much a work in progress. Much of Python3 can be compiled,
including much of the core string functionality, most of the typed_python datastructures
including ListOf, Dict, Alternative, etc, and Class instances (with inheritance).

However, large gaps still remain. In particular,

* we don't allow try/catch yet.
* we can't handle untyped lists, tuples, or dictionaries.
* we can't handle passing functions or function closures as arguments yet
* we have yet to implement calling with keyword arguments for user-defined functions
* lots of little details have deviations from standard python behavior or are simply not implemented yet.
* in some cases, where the compiler cannot produce compiled code it doesn't yet know how to defer to the interpreter.

Nevertheless, for the code that it can compile it gives correct results, and
is much (much) faster than the CPython interpreter.
