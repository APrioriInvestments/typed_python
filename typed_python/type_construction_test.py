import pytest

from typed_python import (
    TupleOf, ListOf, OneOf, Forward, isForwardDefined, bytecount, resolveForwardDefinedType,
    Tuple, NamedTuple, PointerTo, RefTo, Dict, Set, Alternative, Function
)


def test_nonforward_definition():
    assert not isForwardDefined(OneOf(int, float))
    assert not isForwardDefined(ListOf(int))
    assert not isForwardDefined(Dict(int, int))
    assert not isForwardDefined(TupleOf(int))
    assert not isForwardDefined(Set(int))
    assert not isForwardDefined(Tuple(int, int))
    assert not isForwardDefined(PointerTo(int))
    assert not isForwardDefined(RefTo(int))
    assert not isForwardDefined(NamedTuple(x=int, y=float))
    assert not isForwardDefined(Alternative("A", x={}, y={}))
    assert not isForwardDefined(Alternative("A", x={}, y={}).x)
    assert not isForwardDefined(Function(lambda x: x))


def test_forward_definition():
    F = Forward()

    assert issubclass(F, Forward)
    assert isForwardDefined(F)

    assert isForwardDefined(Dict(int, F))
    assert isForwardDefined(OneOf(int, F))
    assert isForwardDefined(ListOf(F))
    assert isForwardDefined(TupleOf(F))
    assert isForwardDefined(Set(F))
    assert isForwardDefined(Tuple(int, F))
    assert isForwardDefined(PointerTo(F))
    assert isForwardDefined(RefTo(F))
    assert isForwardDefined(NamedTuple(x=int, y=F))
    assert isForwardDefined(Alternative("A", x={'f': F}))
    assert isForwardDefined(Alternative("A", x={'f': F}).x)

    func = Function(lambda x: x, F)
    assert isForwardDefined(type(func))


def test_forward_one_of():
    F = Forward()

    O = OneOf(int, F)

    assert isForwardDefined(O)

    with pytest.raises(TypeError):
        bytecount(O)

    # F is now defined.
    F.define(float)

    assert resolveForwardDefinedType(F) is float

    O_resolved = resolveForwardDefinedType(O)
    assert not isForwardDefined(O_resolved)

    assert bytecount(O_resolved) == 1 + bytecount(float)


def test_recursive_definition_of_self():
    F = Forward()

    F.define(ListOf(F))

    F = resolveForwardDefinedType(F)

    assert F.__name__ == 'ListOf(^0)'


def test_recursive_list_of_forward():
    F = Forward()
    O = OneOf(None, F)
    F.define(ListOf(O))

    F_resolved = resolveForwardDefinedType(F)
    O_resolved = resolveForwardDefinedType(O)

    assert issubclass(F_resolved, ListOf)
    assert not isForwardDefined(F_resolved)
    assert not isForwardDefined(O_resolved)
    assert F_resolved.ElementType is O_resolved

    assert F_resolved.__name__ == 'ListOf(OneOf(None, ^1))'
    assert O_resolved.__name__ == 'OneOf(None, ListOf(^1))'


def test_recursive_list_of_tuple_and_forward():
    F = Forward()
    NT = NamedTuple(f=F)
    F.define(ListOf(NT))

    F_resolved = resolveForwardDefinedType(F)
    NT_resolved = resolveForwardDefinedType(NT)

    assert issubclass(F_resolved, ListOf)
    assert not isForwardDefined(F_resolved)
    assert not isForwardDefined(NT_resolved)
    assert F_resolved.ElementType is NT_resolved
    assert NT_resolved.ElementTypes == (F_resolved,)
    assert NT_resolved.ElementNames == ('f',)

    assert F_resolved.__name__ == 'ListOf(NamedTuple(f=^1))'
    assert NT_resolved.__name__ == 'NamedTuple(f=ListOf(^1))'


def test_recursive_list_of_forward_memoizes():
    def makeTup():
        F = Forward()
        O = OneOf(None, F)
        F.define(ListOf(O))

        return resolveForwardDefinedType(F)

    assert makeTup() is makeTup()


def test_recursive_pointer_to():
    def makeP():
        F = Forward()
        F.define(PointerTo(F))
        return resolveForwardDefinedType(F)

    assert makeP() is makeP()
    assert makeP().__name__ == 'PointerTo(^0)'


def test_recursive_ref_to():
    def makeP():
        F = Forward()
        F.define(RefTo(F))
        return resolveForwardDefinedType(F)

    assert makeP() is makeP()
    assert makeP().__name__ == 'RefTo(^0)'


def test_recursive_dict():
    def makeP():
        F = Forward()
        F.define(Dict(int, F))
        return resolveForwardDefinedType(F)

    assert makeP() is makeP()
    assert makeP().__name__ == 'Dict(int, ^0)'


def test_recursive_set():
    def makeP():
        F = Forward()
        F.define(Set(F))
        return resolveForwardDefinedType(F)

    assert makeP() is makeP()
    assert makeP().__name__ == 'Set(^0)'


def test_recursive_alternative():
    def makeA():
        F = Forward()
        F.define(
            Alternative(
                "F",
                A={'f': F},
                B={}
            )
        )
        return resolveForwardDefinedType(F)

    assert makeA() is makeA()
    assert makeA().__name__ == 'F'
