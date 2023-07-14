import sys

from typed_python import Class, SerializationContext, Held, isForwardDefined, Entrypoint, Forward, Final, Function, ListOf
from typed_python._types import typeWalkRecord, recursiveTypeGroupRepr

def writer():
    class Base(Class):
        def f(self):
            return self

    bytesToWrite = SerializationContext().serialize(Base)

    with open("a.dat", "wb") as f:
        f.write(bytesToWrite)


def reader():
    with open("a.dat", "rb") as f:
        x = SerializationContext().deserialize(f.read())

    print(x.__name__)
    print(x.f.overloads[0].globals)
    print(x().f().f())


if sys.argv[1:] == ['r']:
    reader()
else:
    writer()
