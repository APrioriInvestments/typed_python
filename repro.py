import sys

from typed_python import Class, SerializationContext, Held, isForwardDefined, Entrypoint, Forward, Final
from typed_python._types import typeWalkRecord, recursiveTypeGroupRepr

def writer():
    Base = Forward("Base")

    @Base.define
    class Base(Class):
        def blah(self) -> Base:
            return self

        def f(self, x) -> int:
            return x + 1

    class Child(Base, Final):
        def f(self, x) -> int:
            return -1

    aChild = Child()

    aChildBytes = SerializationContext().serialize(aChild)

    with open("a.dat", "wb") as f:
        f.write(aChildBytes)


def reader():
    with open("a.dat", "rb") as f:
        aChild = SerializationContext().deserialize(f.read())

    # @Entrypoint
    def callF(x):
    	return x.f(10)

    assert callF(aChild) == 11


if sys.argv[1:] == ['r']:
    reader()
else:
    writer()




