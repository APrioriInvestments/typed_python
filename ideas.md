# introduction

Our goal is to devise a subset of python that admits strong typing
and therefore native compilation, but that's executable using the normal
python interpreter, and looks 'normal', so that we can incrementally compile
portions of the code using an alternative runtime, but still
think of it as a python program that executes under the normal python
interpreter as well. We'll call this subset 'NativePython'.

NativePython code can seamlessly interoperate with normal python code,
and because it has some typing, admits various optimizations. This means
we can implement different backends that take advantage of different
kinds of optimizations and hints. This is great because

* we can use the strong typing just to check that our code makes sense, even if we don't use a compiler.
* we can build libraries of strongly-typed python code that can be used in a variety of contexts.
* we can explicitly opt in to compilation without modifying our program semantics

As an extension to NativePython, we also provide types that have direct access
to memory (when compiled) and which can therefore create unsafe code. By explicitly
separating which code is 'safe' and 'unsafe', and by allowing users to provide
alternative implementations, we give the library-writer a framework for explicitly
controlling the implementation when they want to.

This is all python3. By convention, user code should never use __ functions,
and we won't guarantee that code that uses them will work properly when
converted using different backends.

Compilers don't have to guarantee the exact semantics of object identity. This
is kind of normal - in python, you're guaranteed that '3+3 is 6' because the interpreter
keeps a table of small integer values. But it's not true that '333333333 is 333333332 + 1'
because that hash table has a limited size.

# Types

Basic types from builtin python:

* str, int, bool, float, long, type(None), bytes
* numpy.float64 etc.

Constants include None, integers, True, False, strings, etc.

We introduce NoneType as the type of 'None', which maps to 'Void' in C land.

Types are considered singleton (unique identity implies unique object),
comparable, hashable, etc.

# Algebraic Types

We can introduce an algebraic type as

    T = Union("T")

and define its subtypes either using this syntax

    T.A = Alternative(x=int, y=T)
    T.B = Alternative(z=float)

or

    T.A = {'x': int, 'y': T}

for some subset of values. Once we 'use' a type (do anything with it
other than define something) that type gets 'frozen' and becomes value-like.

We can then construct instances of 'T' as T.A(x=0, y=...) or T.B(z=1.0) etc.

We may optionally support a syntax like

    T.A = Alternative(x = Field(int, default=20, description='the X value in an A'))

We support 'isinstance' on both T and T.A (T.A is a subclass of T) and issubtype.

We allow the creation of 'multiple' types for field types by writing

    OneOf(int, float)

We allow specifying specific values as an option (if they are concrete):

    OneOf(int, "hello")

or by using the | operator on a member of one of our types:

    T | str | None

Which simply indicates that one of those specific values must be chosen.

We may also use 'object' to mean an arbitrary python object.

Unions have pointer semantics and are naturally 'const' (in the sense that
we can't change their members)

# Lists, Tuples, Dicts, ConstDicts

For any 'T', List(T) is a strongly-typed list of T. Tuple(T)
is a tuple, Dict(K,V) is a dict from K to V. The K type must be hashable and value like.

ConstDict(K, V) is a constant dict from K to V. K must be hashable and value-like.

Kwargs(k1=T1, k2=T2, k3=T3, etc) is a strongly-typed string->value dictionary.

Args(T1, T2, T3) is a strongly-typed tuple with a fixed number of elements.

# Typed classes

The most basic kind of strongly-typed object mimics a normal python object with
a fixed set of settable-fields which are of a particular type. This is intended
to be as close to standard python class semantics as possible, but with strong
types (and strongly-typed dispatch)

We may construct 'classes' (code plus a specific set of named fields) by explicitly
adding code and data to them:

    T = TypedClass("T")

    #add some members
    T.x = int #by default, this is 'uninitialized' and will throw an attribute error
              #until it gets set.

    T.y = TypedClass.Member(int, default=0, description='the y value of an X!')

    #a const member - it may be set exactly once and then never again and _must_ be set
    #by all constructors
    T.z = TypedClass.ConstMember(int)

    #add a constructor
    @TypedClass.Constructor(T)
    def __init__(self, x,y):
        self.x = x
        self.y = y

    #add some methods
    @TypedClass.Method(T)
    def f(self, z):
        return self.x + self.y + z

We can create a template-factory using

    @TypeFunction
    def MyContainer(T):
        ...

which memoizes based on the input types.

We may define the same member function more than once:

    @TypedClass.Method(T)
    def f(self, z):
        ...

    @TypedClass.Method(T)
    def f(self, x, y):
        ...

in which case we'll get dispatch based on the number of arguments.

We may also add type-checking:

    @TypedClass.Method(T)
    def f(self, x: int, y: OneOf(str, None)):
        ...

in which case we'll get dispatch based on types as well.

Type dispatch is linear: we check each overload in sequence to see if it
matches, and we then check each argument sequentially to see if it matches.
The type annotation may be a Type, a lambda function that takes a Kwargs
of matched names (so far) and returns a target type, or an instance of TypeFilter.

If no function matches at call time, you get a TypeError. If you attempt to return a value
of the incorrect type, at return time you'll also get a TypeError.

At runtime you may check that a value has a particular type:

    assert isinstance(x, T)

    if not isinstance(x, T):
        return None

this can be used to restrict the types in play for the compiler layer.

Strongly-typed methods will get Args and Kwargs of appropriate type for
their arguments if they use '*args' or '**kwargs'. They may not type-annotate these.

As shorthand around all of this, you may also write

    @TypedClass
    class MyClass:
        x = Member(int)
        y = Member(int, default=0)

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def f(self, x, y):
            ...

        @overload
        def f(self, x, y: Args.x):
            ...

Note that if you want the overloading behavior, you _have_ to have '@overload' in front
of all definitions of a given function except the first. Otherwise, the python interpreter
will never inform the decorator that this is happening.

Functions may also be marked strongly typed to get the appropriate dispatch:

    @TypedFunction
    def f(x: int) -> int:
        return x+y

    @overload
    def f(x,y: Args.x) -> Args.y:
        return x+y

Individual function terms like this have the type

    @FunctionType(x=int, y=int).Returns(int)

A function type is considered concrete if all the arguments
are types (not functions or TypeFilter objects).

We also have

    @BoundMemberFunctionType(self=T, x=int, ...)

which is the type of a bound member function (with an associated
instance).

# Inheritance and interfaces

We may define an Interface as follows:

    @TypedInterface
    class MyInterface:
        #insists that everyone implementing this interface
        #has a member at least this specific
        x = Member(int)

        @virtual
        def f(self, x: int, y: int) -> int:
            pass

        @virtual
        def f(self, x: float, y: float) -> int:
            pass

        #we allow concrete implementations of some functions
        @overload
        def f(self):
            return self.f(0.0)

A TypedClass may implement a TypedInterface, by specifying it as a base class,
in which case it will always match the typed interface in any type expression,
and we may consider the class to be a subclass of the TypedInterface.
At type-creation time we'll check that the definition is precise by looking
for a function in the implementation that covers the definition.

The only constructor for an object of type TypedInterface is an instance of a
concrete type that implements that interface. E.g. if T implements I, and `t` is
an instance of `T`, then `I(t)` is an instance of 'I'. We may attempt to go
the other direction by calling `Cast(i,T)`. `Cast(I(t), T)` is a no-op.

# static methods

Both classes and interfaces may define @staticmethod methods, which has the
obvious intended effect. These methods may not overload with normal methods or
member names.

# compile-time methods

Some methods may be marked @TypeFunction which means that they are functions
that accept and return Type objects. Calling them with anything else is a
type error. They are assumed to be deterministic and may be cached across
program invocations.

# Function variable typing

By default, functions behave like normal python: variables take references
to objects and can take on multiple types (including the 'Unassigned' state).
This models the behavior of standard python.

You may also specify that all variables in a function have only one single type
through the course of execution. This will invoke the type-checker whenever
a function is first instantiated to insist that this is indeed the case.

    @SinglyTyped
    def f(x, y: Args.x):
        ...

will ensure that every variable in the function has exactly one type and no more.

# Execution model

This model can be executed in a normal python interpreter. However, you can also
wrap any class or function with '@compiled' which will enable JIT compilation. You may
also explicitly 'compile' a class or function to get a compiled version that lives in the cache.

# Caching

Compiling and caching code can be very expensive. Everything we compile or typecheck
gets cached, so that future invocations of it can be fast. Each entry in the cache
contains a set of other entries that must be valid for it to be valid, and a set of files and
hashes that it needs. On compiler boot we invalidate everything in the cache and then
start executing code.

# Interfacing with normal python code

You may specify that the compiler shouldn't try to look into code it can't understand using the
`Python` decorator

    @Python
    def f(x: int) -> int:
        #something that takes an int and produces an int but that's not convertible

This will prevent the type-checker from looking inside (this implies it's a TypedFunction,
but will also prevent the compiler from looking inside.

# Decoration and hints

If the function is pure (and can therefore be optimized away if not used, or will allow
true multithreading), you may decorate it with @Const.

This can be a hint to the compiler to parallelize etc.

# Type ownership

The standard types we're creating so far (TypedClass, Union, etc) are all held as smart-pointers
to objects that are allocated on the heap when we compile them, (except for things like integers
which can be allocated to registers).

Implicitly, we actually have two types: the "held type" and the "reference type". By default, the
held type and the reference type for primitive types are the same. However, for user-defined class types,
we have some additional constructs that allow us to control how the objects are laid out in memory:

    Struct(('k1',T1),('k2',T2),...) - a Type that holds its internal types directly,
        rather than as references.
    PackedStruct(...) - a Type that holds its internal types directly and which packs them
        in unaligned fashion in the order given.
    Owned(T) - a Type that holds its single internal type directly
    InlinePackedArray(T, N) - a fixed number of packed objects

We also provide the following high-level type

    PackedArray(T) - indicates that we hold an array of objects packed in memory - knows the type
        but the count is held in the object itself, and provides safe indexing.

By default, when we hold a 'T' for some user class, it's implemented as a smart-reference.
But we also support the following reference types:

    GuardedReference(T, G) - a reference to a T held within a 'G'. The reference to the 'G'
        is a smart-reference. This is the 'safe' pathway to accessing object internals.
    GuardedArrayReference(T, G) - a reference to a slice of a sized array of type T held inside of a G.
    Pointer(T) - a naked pointer to a 'T'. Member accesses result in StridedPointer objects
    StridedPointer(T, O) - a naked pointer to a 'T' striding by 'O' bytes.

When we access an 'internal' value within a root-level object on the heap, we'll always get a GuardedReference,
which we can then turn into a pointer using the addr() function. We may also directly
update the held value by calling assign(ref_type, other). Classes can define

    def __constructor__(self, args):
        ...

    def __copy_constructor__(self, other):
        ...

    def __assign__(self, other):
        ...

    def __destructor__(self):
        ...

methods if they want more explicit control over how they get constructed and destroyed. These
methods must be valid for a type to be used as an 'internal type', and are constructed by default
if no implementation is provided.  In these cases, instances of 'T' are always naked pointers
(so, oddly, type(self) is Pointer(T), not T, and there's no way to get the 'T' back). In the case of
__constructor__ and __copy_constructor__, 'self' is uninitialized, and we must use the 'T.New(addr, other)'
to initialize the object.

To give us stack semantics, we also introduce

    with Stack.add(T1, ...).add(T2, ...),... as t1, t2, ...:
        #blah

which creates a new stack-frame, valid only for the duration of the context block. t1, t2, etc
are pointers to these values which have been constructed on the stack as values of type T1, T2,
with arguments as given in the 'add' function. Destructors are called in the reverse order. This
gives us the ability to model low-level code in python as needed.

PackedArray gives us an ability to allocate N things on the heap (and hold them as a smart reference).

We also provide low-level types Int8, Int16, Int32, Int64, Int128 (and UInt varieties), Float32, Float64,
which allow us to explicitly model memory and pointers. Pointer objects support casts, offsets, etc.

Our python implementation supports explicit checking of memory validity since everything is modeled as
numpy arrays, and eventually the compiler will support a model for leaving runtime checks
in place on all pointer accesses, or on specific types to try to diagnose failures. The goal here
is to insist on folks using PackedArray[UInt8] and then typecasting with pointer arithmetic so we
can ensure that behavior is correct.

# Native functions

We allow a formal Type ExternalFunction("name", OutputType, [InputTypes]) which has normal C function
linkage.

# Formal equivalences

We can explicitly register handlers for specific singletons and types. For instance, to compile
or typecheck code that uses 'xrange', we need a model for xrange. So we provide a NativePython
implementation of xrange.

Todo:
    * how does this play with the distributed-transactional model?
    * do we have a story around getting the type-object of something at runtime?
    * Threads?
    * What would an auto-parallel framework look like?
    * how about automatic serialization? Would need to work on 'safe' code only
    * const-ness in the PackedArray model











