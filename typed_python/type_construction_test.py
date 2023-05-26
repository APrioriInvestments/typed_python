import pytest

from typed_python import (
    ListOf, OneOf, Forward, isForwardDefined, bytecount, resolveForwardDefinedType
)


def test_forward_definition():
    F = Forward()

    assert issubclass(F, Forward)
    assert isForwardDefined(F)


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


def test_recursive_tuple_of_forward():
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


def test_recursive_tuple_of_forward_memoizes():
    def makeTup():
        F = Forward()
        O = OneOf(None, F)
        F.define(ListOf(O))

        return resolveForwardDefinedType(F)

    assert makeTup() is makeTup()
