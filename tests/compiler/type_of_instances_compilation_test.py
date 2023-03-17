from typed_python import Entrypoint, Alternative, Class


def test_type_of_class_is_specific():
    class C(Class):
        pass

    class B(C):
        pass

    @Entrypoint
    def typeOfArg(x: C):
        return type(x)

    assert typeOfArg(B()) is B


def test_type_of_alternative_is_specific():
    for members in [{}, {'a': int}]:
        A = Alternative("A", A=members)

        @Entrypoint
        def typeOfArg(x: A):
            return type(x)

        assert typeOfArg(A.A()) is A.A


def test_type_of_concrete_alternative_is_specific():
    for members in [{}, {'a': int}]:
        A = Alternative("A", A=members)

        @Entrypoint
        def typeOfArg(x: A.A):
            return type(x)

        assert typeOfArg(A.A()) is A.A


def test_type_name():
    A = Alternative("A", X=dict(x=int), Y=dict(y=int))

    class C(Class):
        pass

    class B(C):
        pass

    @Entrypoint
    def typenameOfAInst(a: A):
        return type(a).__name__

    assert typenameOfAInst(A.X()) == 'X'
    assert typenameOfAInst(A.Y()) == 'Y'
    assert typenameOfAInst.resultTypeFor(A.X).typeRepresentation is str

    @Entrypoint
    def typenameOfCInst(c: C):
        return type(c).__name__

    assert typenameOfCInst(C()) == 'C'
    assert typenameOfCInst(B()) == 'B'
    assert typenameOfAInst.resultTypeFor(A.X).typeRepresentation is str
