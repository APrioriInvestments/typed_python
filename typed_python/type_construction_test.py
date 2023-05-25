import pytest

from typed_python import (
    ListOf, OneOf, Forward, isForwardDefined, bytecount, resolveForwardDefinedType
)




def test_forward_definition():
    F = Forward("X")

    assert issubclass(F, Forward)
    assert isForwardDefined(F)


def test_forward_one_of():
    F = Forward("X")

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


def test_recursive_tuple_of_forward():
    F = Forward("X")

    O = OneOf(None, F)

    F.define(ListOf(O))

    F_resolved = resolveForwardDefinedType(F)
    O_resolved = resolveForwardDefinedType(O)

    assert issubclass(F_resolved, ListOf)
    assert not isForwardDefined(F_resolved)
    assert not isForwardDefined(O_resolved)
    assert F_resolved.ElementType is O_resolved

    print(O_resolved)
    print(F_resolved)

