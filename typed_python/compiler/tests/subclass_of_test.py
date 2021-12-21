from typed_python import Entrypoint, SubclassOf, Class, Final, Function, ListOf


class A(Class):
    pass


class B(A):
    pass


class C(B, Final):
    pass


def test_can_cast_subclass_of_correctly():
    @Function
    def f(c: SubclassOf(C)):
        return "C"

    @f.overload
    def f(c: SubclassOf(B)):
        return "B"

    @f.overload
    def f(c: SubclassOf(A)):
        return "A"

    def checkIt():
        assert f(C) == "C", f(C)
        assert f(B) == "B", f(B)
        assert f(A) == "A", f(A)

    checkIt()

    Entrypoint(checkIt)()

    @Entrypoint
    def checkItList(x):
        res = ListOf(str)()

        for cls in x:
            res.append(f(cls))

        return res

    assert checkItList(ListOf(SubclassOf(A))([A, B, C])) == ["A", "B", "C"]
    assert checkItList(ListOf(SubclassOf(B))([B, C])) == ["B", "C"]
    assert checkItList(ListOf(SubclassOf(C))([C])) == ["C"]
