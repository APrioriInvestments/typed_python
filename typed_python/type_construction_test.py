import pytest

from typed_python import (
    TupleOf, ListOf, OneOf, Forward, isForwardDefined, bytecount, resolveForwardDefinedType,
    Tuple, NamedTuple, PointerTo, RefTo, Dict, Set, Alternative, Function, identityHash, Value,
    Class, Member, ConstDict, TypedCell, typeLooksResolvable, Held, NotCompiled,
    forwardDefinitionsFor, Entrypoint
)

from typed_python.test_util import CodeEvaluator


def test_function_in_anonymous_module_callable():
    c = CodeEvaluator()
    m = {}

    c.evaluateInto("""
        from typed_python import Function

        y = 1

        @Function
        def f(x):
            return x + y
    """, m)
    f = m['f']

    assert f(10) == 11
    assert f.ClosureType.ElementTypes[0].ElementNames[0] == ' _globals'


def test_entrypoint_basic():
    @Entrypoint
    def f(x: int):
        return x + 1

    @Entrypoint
    def g():
        return f(2)

    print(g.ClosureType)
    print(g.overloads[0])

    from typed_python import _types
    g2 = _types.prepareArgumentToBePassedToCompiler(g)
    print("g2: ")
    print(g2.overloads[0].globals)
    print(g2.overloads[0].closureVarLookups)
    print(g2.overloads[0].closureVarLookups['f'][0])
    print(g2.overloads[0].closureVars)
    print()

    assert g() == 3


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


def test_forward_class_exposes_functions():
    class C(Class):
        x = Member(lambda: B)

        def f(self):
            return C

    B = int

    CResolved = resolveForwardDefinedType(C)

    assert C.HeldClass.Class is C
    assert CResolved.HeldClass.Class is CResolved

    assert not C.IsFinal
    assert not CResolved.IsFinal

    assert C.MRO == (C,)
    assert CResolved.MRO == (CResolved,)

    assert CResolved.MemberNames == ('x',)

    assert C.MemberNames == ('x',)

    assert C.f.overloads[0].globals['C'].kind == 'GlobalInCell'


def test_function_global_resolving_to_builtin():
    @NotCompiled
    def f():
        return str(x)

    assert f.overloads[0].globals['str'].getValue() is str


def test_reference_to_builtin_is_resolvable():
    A = Alternative("A", X={}, f=lambda self: str(self) + B)

    # A is forward defined because it depends on B
    assert isForwardDefined(A)

    assert A.f.overloads[0].globals['str'].kind == 'NamedModuleMember'
    assert A.f.overloads[0].globals['str'].moduleName == 'builtins'
    assert A.f.overloads[0].globals['str'].name == 'str'


def test_function_cell_bindings_are_resolved_to_constants():
    def CofX(x):
        class C(Class):
            def f(self):
                return x

        return C

    assert CofX(1).f.overloads[0].globals['x'].kind == 'GlobalInCell'
    assert CofX(1).f.overloads[0].globals['x'].getValue() is 1
    assert CofX(2).f.overloads[0].globals['x'].getValue() is 2


def test_reference_to_builtin_doesnt_prevent_autoresolve():
    A = Alternative("A", X={}, f=lambda self: str(self))

    assert not isForwardDefined(A)

    assert A().f() == str(A())


def test_type_looks_resolvable_alternative():
    A = Alternative("A_", X={}, f=lambda self: B)

    # hide a reference to this forward in a tuple that the autoresolver can't see
    AFwd = (A,)

    assert isForwardDefined(A)

    assert typeLooksResolvable(A, unambiguously=False)
    assert not typeLooksResolvable(A, unambiguously=True)

    B = Alternative("B_", X={}, g=lambda self: A)

    assert typeLooksResolvable(A, unambiguously=False)
    assert typeLooksResolvable(A, unambiguously=True)
    assert typeLooksResolvable(B, unambiguously=False)
    assert typeLooksResolvable(B, unambiguously=True)

    A = resolveForwardDefinedType(A)
    B = resolveForwardDefinedType(B)

    bGlobal = A.f.overloads[0].globals['B']
    assert bGlobal.kind == "GlobalInCell"
    assert bGlobal.getValue() is B

    bFwdGlobal = AFwd[0].f.overloads[0].globals['B']
    assert bFwdGlobal.kind == "GlobalInCell"
    assert bFwdGlobal.getValue() is B

    assert not isForwardDefined(A)
    assert isForwardDefined(AFwd[0])

    assert A().f() is B
    assert B().g() is A


def test_autoresolve_unnamed_type_with_named_forward():
    # this defines 'F' as a forward. When 'F' gets assigned to something other than a forward
    # the type will become resolvable.
    F = Forward(lambda: F)

    # this defines 'G' - there's no reference in the type system to it
    G = OneOf(None, F)

    # this allows 'F' and 'G' to be resolved.
    F = ListOf(G)

    # the system was able to autoresolve both F and G because they were assigned
    assert not isForwardDefined(F)
    assert not isForwardDefined(G)


def test_held_of_forward_class():
    class C(Class):
        def f(self):
            return H

    assert isForwardDefined(C)

    @Held
    class H(Class):
        def f(self):
            return C

    assert not isForwardDefined(H)
    assert not isForwardDefined(C)


def test_autoresolve_on_decorator():
    B = Forward(lambda: B)

    class B(Class):
        def blah(self) -> B:
            return self

    @Function
    def callIt(x: B):
        pass

    assert not isForwardDefined(B)
    assert not isForwardDefined(type(callIt))


def test_class_with_entrypointed_recursive_method():
    class C(Class):
        @NotCompiled
        def f(self):
            return C().g()

    C()


def test_autoresolve_forwards_with_nonforwards():
    class C(Class):
        def f(self):
            return T

    assert isForwardDefined(C)

    with pytest.raises(TypeError):
        C()

    T = int

    assert typeLooksResolvable(C, unambiguously=True)

    C = resolveForwardDefinedType(C)

    C()


def test_autoresolve_unnamed_type_with_anonymous_forward():
    G = OneOf(None, lambda: F)
    F = ListOf(G)

    # the system was able to autoresolve both F and G because they were assigned
    assert not isForwardDefined(F)
    assert not isForwardDefined(G)


def test_untyped_functions_are_not_forward():
    # this type is not forward defined because it should be holding 'g' in the closure of
    # 'f' itself
    f = Function(lambda: g())
    assert not isForwardDefined(type(f))

    g = Function(lambda: f())
    assert not isForwardDefined(type(g))


def test_autoresolve_forwards():
    A = Alternative("A", X={}, Y=dict(inst=lambda: B))
    B = Alternative("B", X={}, Y=dict(inst=lambda: A))

    assert not isForwardDefined(A)
    assert not isForwardDefined(B)


def test_autoresolve_forwards_3cycle():
    A = Alternative("A", X={}, Y=dict(inst=lambda: C))
    B = Alternative("B", X={}, Y=dict(inst=lambda: A))

    assert isForwardDefined(A)
    assert isForwardDefined(B)

    C = Alternative("C", X={}, Y=dict(inst=lambda: B))

    assert not isForwardDefined(A)
    assert not isForwardDefined(B)
    assert not isForwardDefined(C)


def test_invalid_forwards():
    with pytest.raises(TypeError):
        Forward(lambda: 1)

    with pytest.raises(TypeError):
        Forward(lambda: A + B)  # noqa

    X = 10
    with pytest.raises(TypeError):
        Forward(lambda: X + B)  # noqa


def test_implicitly_recursive_types():
    O_fwd = ListOf(OneOf(int, Forward(lambda: O)))

    assert isForwardDefined(O_fwd)
    assert not typeLooksResolvable(O_fwd)

    O = O_fwd

    assert typeLooksResolvable(O_fwd)

    O = resolveForwardDefinedType(O)
    O([O()])


def test_dual_recursive_types_direct_instantiation():
    O1 = Forward("O1")
    O2 = Forward("O2")

    O1.define(ListOf(OneOf(int, O2)))
    O2.define(ListOf(OneOf(int, O1)))

    assert isForwardDefined(O1)
    assert isForwardDefined(O2)

    O1 = resolveForwardDefinedType(O1)
    O2 = resolveForwardDefinedType(O2)


def test_dual_recursive_types_direct_instantiation_with_alternatives():
    O1 = Forward("O1")
    O2 = Forward("O2")

    O1.define(Alternative("O1", X={}, Y={'a': O2}, f=lambda self: O2))
    O2.define(Alternative("O2", X={}, Y={'a': O1}, f=lambda self: O1))

    assert not isForwardDefined(O1)
    assert not isForwardDefined(O2)


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


def test_make_recursive_list_of_autoresolve():
    def makeRecursiveListOf(T):
        L = ListOf(OneOf(T, Forward(lambda: L)))
        return L

    assert not isForwardDefined(makeRecursiveListOf(int))

    assert makeRecursiveListOf(int) is makeRecursiveListOf(int)
    assert makeRecursiveListOf(int) is not makeRecursiveListOf(str)


def test_autoresolve_nonstorage_forwards():
    A = Alternative("A", X={}, fA=lambda: B)
    B = Alternative("B", X={}, fB=lambda: A)

    assert not isForwardDefined(B)
    assert not isForwardDefined(A)


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


def test_define_class_holding_self():
    class C(Class):
        anotherC = Member(OneOf(None, lambda: C))

        def f(self) -> Forward(lambda: C):
            return self

    assert not isForwardDefined(C)

    c = C()
    assert type(c.f()) is C


def test_define_class_holding_self_defined_as_forward():
    C = Forward(lambda: C)

    assert isForwardDefined(C)
    assert not typeLooksResolvable(C, True)

    class C(Class):
        anotherC = Member(OneOf(None, C))

        def f(self) -> C:
            return self

    assert not isForwardDefined(C)

    c = C()
    assert type(c.f()) is C


def test_two_mutually_recursive_classes():
    C1 = Forward(lambda: C1)
    C2 = Forward(lambda: C2)

    class C1(Class):
        holdsC2 = Member(OneOf(None, C2))

        def f(self) -> C1:
            return self

    assert isForwardDefined(C1)

    class C2(Class):
        holdsC1 = Member(OneOf(None, C1))

        def f(self) -> C2:
            return self

    assert not isForwardDefined(C1)
    assert not isForwardDefined(C2)


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


@pytest.mark.skip(reason='what to do with forwards functions?')
def test_cannot_call_forward_function_instances():
    t = Forward("T")

    @Function
    def f(x: t):
        return x

    t.define(int)

    with pytest.raises(TypeError):
        f(10)


@pytest.mark.skip(reason='what to do with forwards functions?')
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


@pytest.mark.skip(reason='what to do with forwards functions?')
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
    assert isForwardDefined(A1_fwd.get())
    assert isForwardDefined(A2_fwd.get())

    assert resolveForwardDefinedType(A1_fwd.get()) is A1
    assert resolveForwardDefinedType(A2_fwd.get()) is A1


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

# CHECK:
# OneOfs containing forwards that resolve to other oneofs changing the number
#     of items - this needs to work
# Forwards resolving to simple types like ints/floats
# MRTG should be completely ignoring Forwards - ideally they go away entirely
#     - if they get leaked, that should be an error!
# how does this play with the ModuleRepresentation stuff?
# how does this play with TypeFunction?
# mutually recursive across modules?
