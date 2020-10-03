from typed_python.macro import Macro
from typed_python import TypeFunction, TupleOf, Entrypoint
from typed_python import Class, NamedTuple, Member, makeNamedTuple  # noqa: F401

from typed_python.compiler.runtime import PrintNewFunctionVisitor

import unittest


class TestMacro(unittest.TestCase):
    def test_main(self):
        @Macro
        def A(T):
            output = ["class R(Class):"]
            output.append("    x = Member(T)")
            if T is int:
                output.append("    def do(self) -> str:")
                output.append("        return 'boom.'")
            else:
                output.append("    pass")
            output.append("return R")
            return {
                "sourceText": output,
                "locals": {"T": T},
            }

        self.assertEqual(A(int)().do(), "boom.")
        with self.assertRaises(AttributeError):
            A(float)().do()

        self.assertIs(A(int), A(int))
        self.assertIs(A(A(str)), A(A(str)))
        self.assertFalse(A(str) is A(A(str)))
        self.assertFalse(A(str) is A(float))

    def test_more(self):
        @TypeFunction
        def F(S, T):
            class A:
                pass
            return A

        @Macro
        def G(S, T):
            output = []
            output.append("R = F(S, T)")
            output.append("return R")
            return {
                "sourceText": output,
                "locals": {"S": S, "T": T, "F": F},
            }

        self.assertEqual(G(int, float), F(int, float))

        @Macro
        def H(S, T):
            output = []
            output.append("R = F(S, T)")
            output.append("return R")
            return {
                "sourceText": output,
                "locals": {"S": S, "T": T},
            }

        with self.assertRaises(NameError):
            H(int, float)

    def test_another(self):
        @Macro
        def A(S, T):
            positions = []
            for name in T.ElementNames:
                assert name in S.ElementNames
                positions.append(S.ElementNames.index(name))

            output = []
            output.append("class X(Class):")
            output.append("    s = Member(S)")
            output.append("    def __init__(self, s):")
            output.append("        self.s = s")
            output.append("    def do(self) -> None:")
            for position in positions:
                output.append(f"        print(self.s[{position}])")
            output.append("return X")

            return {
                "sourceText": output,
                "locals": {"S": S, "T": T},
            }

        a = A(NamedTuple(x=int, y=float), NamedTuple(y=str))
        a = A(NamedTuple(x=int, y=float), NamedTuple(y=str))(makeNamedTuple(x=3, y=4.0))
        print(a)
        print(a.do())

    def test_lazy_named_tuple_row(self):
        @Macro
        def lazyWindow(T):
            output = []
            output.append(f"class X(Class, __name__='LazyWindow({T.__name__})'):")
            output.append("    table = Member(T)")
            output.append("    window = Member(int)")
            output.append("    def __init__(self, table, window=0):")
            output.append("        self.table = table")
            output.append("        self.window = window")
            for i in range(len(T.ElementNames)):
                output.append("    @Entrypoint")
                output.append(f"    def {T.ElementNames[i]}(self) -> T.ElementTypes[{i}].ElementType:")
                output.append(f"        return self.table[{i}][self.window]")
            output.append("return X")

            return {
                "sourceText": output,
                "locals": {"T": T},
            }

        Table = NamedTuple(a=TupleOf(int), b=TupleOf(str))
        Lazy = lazyWindow(Table)

        table = Table(a=(1, 2, 3, 4, 5), b=('a', 'b', 'c', 'd', 'e'))
        lazy = Lazy(table)

        @Entrypoint
        def return_a(lazy):
            return lazy.a()

        def return_b(lazy):
            return lazy.b()

        with PrintNewFunctionVisitor():
            self.assertEqual(return_a(lazy), 1)

        with PrintNewFunctionVisitor():
            self.assertEqual(return_b(lazy), 'a')

        lazy.window += 1

        with PrintNewFunctionVisitor():
            self.assertEqual(return_a(lazy), 2)

        with PrintNewFunctionVisitor():
            self.assertEqual(return_b(lazy), 'b')

    def test_exception(self):
        @Macro
        def f(T):
            output = []
            output.append("R=S")
            output.append("Q=R")
            output.append("return Q")
            return {
                "sourceText": output,
                "locals": {"S": S}
            }

        with self.assertRaises(NameError):
            f(int)

        S = float

        self.assertEqual(f(str), float)

        with self.assertRaises(NameError):
            f(int)

    def test_namespace(self):
        @Macro
        def f(T):
            return {
                "sourceText": ["return A"],
                "locals": {"T": T, "A": A},
            }

        with self.assertRaises(NameError):
            f(int)

        class A(Class):
            pass

        with self.assertRaises(NameError):
            f(int)

        self.assertEqual(f(float).__name__, 'A')

        class A(Class, __name__='B'):  # noqa: F811
            pass

        with self.assertRaises(NameError):
            f(int)

        self.assertEqual(f(float).__name__, 'A')
        self.assertEqual(f(str).__name__, 'B')

        def inside():
            @Macro
            def f(T):
                return {
                    "sourceText": ["return A"],
                    "locals": {"T": T, "A": A},
                }

            with self.assertRaises(NameError):
                f(int)
                # although A existed before, on compilation that A is understood to
                # refer to an A defined in 'inside'

            class A(Class, __name__='C'):
                pass

            with self.assertRaises(NameError):
                f(int)

            self.assertEqual(f(float).__name__, 'C')

            class A(Class, __name__='D'):  # noqa: F811
                pass

            with self.assertRaises(NameError):
                f(int)

            self.assertEqual(f(float).__name__, 'C')
            self.assertEqual(f(str).__name__, 'D')

            return f

        f = inside()

        with self.assertRaises(NameError):
            f(int)
        self.assertEqual(f(float).__name__, 'C')
        self.assertEqual(f(str).__name__, 'D')
        self.assertEqual(f(bool).__name__, 'D')

        class A(Class, __name__='E'):  # noqa: F811
            pass

        self.assertEqual(f(list).__name__, 'D')
