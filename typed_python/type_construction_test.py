import pytest

from typed_python import (
    TupleOf, ListOf, OneOf, Forward, isForwardDefined, bytecount, resolveForwardDefinedType,
    Tuple, NamedTuple, PointerTo, RefTo, Dict, Set, Alternative, Function, identityHash, Value,
    Class, Member, ConstDict, TypedCell, Final, typeLooksResolvable
)

from typed_python.test_util import CodeEvaluator


def test_identity_hash_of_alternative_stable():
    e = CodeEvaluator()
    m = {}

    e.evaluateInto("""
        from typed_python._types import identityHash, Alternative

        A = Alternative(
            "A",
            f=lambda x: (A, g)
        )

        h1 = identityHash(A)

        def g():
            pass

        h2 = identityHash(A)
    """, m)

    assert m['h1'] == m['h2']


def test_type_looks_resolvable_alternative():
    # note that these types are not autoresolvable because of their names
    A = Alternative("A_", X={}, f=lambda self: B)

    assert typeLooksResolvable(A, unambiguously=False)
    assert not typeLooksResolvable(A, unambiguously=True)

    B = Alternative("B_", X={}, g=lambda self: A)

    assert typeLooksResolvable(A, unambiguously=False)
    assert typeLooksResolvable(A, unambiguously=True)
    assert typeLooksResolvable(B, unambiguously=False)
    assert typeLooksResolvable(B, unambiguously=True)

    A = resolveForwardDefinedType(A)
    B = resolveForwardDefinedType(B)

    assert A().f() is B
    assert B().g() is A


def test_untyped_functions_are_not_forward():
    # this type is not forward defined because it should be holding 'g' in the closure of
    # 'f' itself
    f = Function(lambda: g())
    assert not isForwardDefined(type(f))

    g = Function(lambda: f())
    assert not isForwardDefined(type(g))


def test_recursive_types():
    # define a forward
    O = Forward("O")

    # give the forward a definition. the thing that comes out is the resolved type
    O = O.define(ListOf(OneOf(int, O)))


def test_dual_recursive_types_direct_instantiation():
    O1 = Forward("O1")
    O2 = Forward("O2")

    O1.define(ListOf(OneOf(int, O2)))
    O2.define(ListOf(OneOf(int, O1)))

    O1 = resolveForwardDefinedType(O1)
    O2 = resolveForwardDefinedType(O2)


def test_dual_recursive_types_direct_instantiation_with_alternatives():
    O1 = Forward("O1")
    O2 = Forward("O2")

    O1.define(Alternative("O1", X={}, Y={'a': O2}, f=lambda self: O2))
    O2.define(Alternative("O2", X={}, Y={'a': O1}, f=lambda self: O1))

    O1 = resolveForwardDefinedType(O1)
    O2 = resolveForwardDefinedType(O2)


def test_dual_recursive_types_implicit_instantiation_with_alternatives():
    O1 = Alternative("O1", X={}, Y={'a': lambda: O2}, f=lambda self: O2)

    # this assignment triggers a resolution
    O2 = Alternative("O2", X={}, Y={'a': lambda: O1}, f=lambda self: O1)


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


def test_oneof():
    assert OneOf(int, float) is OneOf(int, float)
    assert OneOf(int, float) is not OneOf(int, None)


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


def test_recursive_typed_cell():
    def makeP():
        F = Forward()
        F.define(TypedCell(F))
        return resolveForwardDefinedType(F)

    assert makeP() is makeP()
    assert makeP().__name__ == 'TypedCell(^0)'


def test_recursive_dict():
    def makeP():
        F = Forward()
        F.define(Dict(int, F))
        return resolveForwardDefinedType(F)

    assert makeP() is makeP()
    assert makeP().__name__ == 'Dict(int, ^0)'


def test_recursive_const_dict():
    def makeP():
        F = Forward()
        F.define(ConstDict(int, F))
        return resolveForwardDefinedType(F)

    assert makeP() is makeP()
    assert makeP().__name__ == 'ConstDict(int, ^0)'


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
                B={},
                C={'x': int}
            )
        )
        return resolveForwardDefinedType(F)

    assert makeA() is makeA()
    assert makeA().__name__ == 'F'
    assert makeA().A is makeA().A
    makeA().B()
    makeA().C(x=10)


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


def test_instances_of_forward_function_are_forward():
    T = Forward("T")

    @Function
    def f(x: T):
        return x

    assert isForwardDefined(f)
    T.define(int)

    f2 = resolveForwardDefinedType(f)

    assert not isForwardDefined(f2)

    assert f2(1) == 1

    with pytest.raises(TypeError):
        f2("hi")


def test_create_value_type_with_forward():
    F = Forward("X")
    T = Value(F)
    F.define(int)

    assert isForwardDefined(T)

    T_resolved = resolveForwardDefinedType(T)

    assert not isForwardDefined(T_resolved)
    assert identityHash(T_resolved).hex() == identityHash(Value(int)).hex()
    assert T_resolved is Value(int)


def test_create_class_with_methods():
    class C(Class):
        def f(self):
            return "hi"

    assert not isForwardDefined(C)


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


def test_identity_of_leaked_forward():
    def makeClass():
        A = Forward("A")
        A_fwd = A

        A.define(
            Alternative(
                "A",
                X={},
                getA=lambda self: A,
                getAForward=lambda self: A_fwd
            )
        )
        A = resolveForwardDefinedType(A)

        return A, A_fwd

    A1, A1_fwd = makeClass()
    A2, A2_fwd = makeClass()

    # the identity of 'A' should be stable
    assert A1 is A2

    # but the forwards we used to create them could be separate
    assert A1_fwd is not A2_fwd

    # they have to point to the same concrete instance
    assert A1_fwd.get() is A2_fwd.get()
    assert A1_fwd.get() is A1


@pytest.mark.skip(reason='TODO: make this work')
def test_create_class_with_fully_forward_base():
    def makeClass(T):
        C_base_fwd = Forward("C_base")
        C_child_fwd = Forward("C_child")

        class CBase(Class):
            c = Member(OneOf(None, C_child_fwd))

        class CChild(C_base_fwd):
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


# TODO:
# 1. can we get better names and info for forward-defined types? what does ther
#    partially defined state look like
# 2. the CVOV needs to be able to look into TP instances. Value(TupleOf(int)((1, 2, 3))) for
#    instance ought to be visible and compilable
# 3. can regular python classes and functions be 'forward declared'. What do we do if we
#    forward resolve them?
# 4. serialization
# 5. type functions
# 6. are we accidentally creating and installing forward subclasses in the vtable?
#    maybe we need a second pass to wire in 'accepted' forward types
# 7. make sure we thoroughly test the idea that we create a new forward graph with
#    part of the graph being new, part being old
# 8. clean-up in the whole Instance/PyInstance layer - kind of nasty
# 9. we should memoize instances of statless Function objects and instances of regular TP lists/tuple of, etc.
