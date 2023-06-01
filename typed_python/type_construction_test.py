import pytest

from typed_python import (
    TupleOf, ListOf, OneOf, Forward, isForwardDefined, bytecount, resolveForwardDefinedType,
    Tuple, NamedTuple, PointerTo, RefTo, Dict, Set, Alternative, Function, identityHash, Value,
    Class, Member
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


def test_nonrecursive_list_of_forward_memoizes():
    def makeTup():
        F = Forward()
        O = ListOf(F)
        F.define(int)

        return resolveForwardDefinedType(O)

    assert makeTup() is makeTup()
    assert makeTup() is ListOf(int)


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


def test_cannot_call_forward_function_instances():
    t = Forward("T")

    @Function
    def f(x: t):
        return x

    t.define(int)

    with pytest.raises(TypeError):
        f(10)


def test_function_types_coalesc():
    def makeFNonforward(T):
        @Function
        def f(x: T):
            return x

        return type(f)

    def makeFforward(T):
        t = Forward("T")

        @Function
        def f(x: t):
            return x

        t.define(T)

        return resolveForwardDefinedType(type(f))

    assert makeFNonforward(int) is makeFNonforward(int)
    assert identityHash(makeFforward(int)) == identityHash(makeFforward(int))
    assert makeFforward(int) is makeFforward(int)


def test_create_value_type_with_forward():
    F = Forward("X")
    T = Value(F)
    F.define(int)

    assert isForwardDefined(T)

    T_resolved = resolveForwardDefinedType(T)

    assert not isForwardDefined(T_resolved)
    assert identityHash(T_resolved).hex() == identityHash(Value(int)).hex()
    assert T_resolved is Value(int)


def test_create_class_with_forward():
    def makeClass(T):
        F = Forward("X")

        class C(Class):
            m = Member(F)

        assert isForwardDefined(C)

        F.define(T)

        return resolveForwardDefinedType(C)

    assert makeClass(int) is not makeClass(float)
    assert makeClass(int) is makeClass(int)

    assert makeClass(int)().m == 0
    assert makeClass(str)().m == ""


def test_create_concrete_class():
    def makeClass(T):
        class C(Class):
            m = Member(T)

        assert not isForwardDefined(C)

        return C

    assert makeClass(int) is makeClass(int)
    assert makeClass(int) is not makeClass(str)


def test_create_class_with_recursion():
    def makeClass(T):
        C_fwd = Forward("C")

        class C(Class):
            c = Member(OneOf(None, C_fwd))
            m = Member(T)

        assert isForwardDefined(C_fwd)
        assert isForwardDefined(C)

        C_fwd.define(C)

        return resolveForwardDefinedType(C)

    assert makeClass(int) is not makeClass(float)
    assert makeClass(int) is makeClass(int)

    assert makeClass(int)().m == 0
    assert makeClass(str)().m == ""


def test_create_class_with_nonforward_base():
    def makeClass(T):
        C_child_fwd = Forward("C_child")

        class CBase(Class):
            m = Member(T)

        class CChild(CBase):
            pass

        assert not isForwardDefined(CBase)
        assert isForwardDefined(C_child_fwd)

        C_child_fwd.define(CChild)

        return resolveForwardDefinedType(CChild)

    assert makeClass(int) is not makeClass(float)
    assert makeClass(int) is makeClass(int)

    assert makeClass(int)().m == 0
    assert makeClass(str)().m == ""


def test_create_class_with_forward_base():
    def makeClass(T):
        C_base_fwd = Forward("C_base")
        C_child_fwd = Forward("C_child")

        class CBase(Class):
            c = Member(OneOf(None, C_child_fwd))

        class CChild(CBase):
            m = Member(T)

        assert isForwardDefined(C_base_fwd)
        assert isForwardDefined(C_child_fwd)

        C_child_fwd.define(CChild)
        C_base_fwd.define(CBase)

        return resolveForwardDefinedType(CChild)

    assert makeClass(int) is not makeClass(float)
    assert makeClass(int) is makeClass(int)

    assert makeClass(int)().m == 0
    assert makeClass(str)().m == ""
