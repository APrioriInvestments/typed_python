import sys

from typed_python import Class, SerializationContext, Held, isForwardDefined, Entrypoint, Forward, Final, Function, ListOf, NotCompiled, Member
from typed_python._types import typeWalkRecord, recursiveTypeGroupRepr

import typed_python.compiler.native_compiler.native_ast as n


from typed_python import _types
assert not _types.checkForHashInstability()

print(n.Type.zero)
print(n.Type.zero.overloads)

def writer():
    Cls = Forward("Cls")

    @Cls.define
    class Cls(Class):
        m = Member(str)

        def f(self) -> Cls:
            return Cls(m='HI')

    bytesToWrite = SerializationContext().serialize((Cls, ()))

    with open("a.dat", "wb") as f:
        f.write(bytesToWrite)


def reader():
    with open("a.dat", "rb") as f:
        Cls, args = SerializationContext().deserialize(f.read())

    # print(x.__name__)
    print(Cls(*args).f())


if sys.argv[1:] == ['r']:
    reader()
else:
    writer()
