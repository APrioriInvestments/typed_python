#   Copyright 2017-2021 typed_python Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import unittest
from typed_python import Entrypoint, Function, ListOf, isCompiled, OneOf, NotCompiled, Class, Final, TypeOf


class FunctionSignatureTest(unittest.TestCase):
    def test_function_with_signature(self):
        @Function
        def f(x) -> lambda X: ListOf(X):
            return ListOf(type(x))()

        assert f.overloads[0].signatureFunction(int) == ListOf(int)

    def test_function_signature_casts(self):
        @Function
        def f(x) -> lambda X: int:
            return x

        assert isinstance(f(1.0), int)

    def test_function_signature_result_not_convertible(self):
        def notAType():
            pass

        @Function
        def f(x) -> lambda X: notAType:
            return x

        with self.assertRaisesRegex(Exception, "Cannot convert .* to"):
            f(10)

    def test_function_signature_sig_func_throws(self):
        def sigFunc(X):
            raise Exception("Error in signature function")

        @Function
        def f(x) -> sigFunc:
            return x

        with self.assertRaisesRegex(Exception, "Error in signature function"):
            f(10)

    def test_function_signature_sig_func_wrong_number_of_arguments(self):
        def sigFunc(X, Y):
            return X

        @Function
        def f(x) -> sigFunc:
            return x

        with self.assertRaisesRegex(Exception, "missing 1 required positional argument: 'Y'"):
            f(10)

    def test_function_signature_compiles(self):
        def sigFunc(X, Y):
            return X

        @Entrypoint
        def f(x, y) -> lambda X, Y: X:
            assert isCompiled()
            return x + y

        @Entrypoint
        def callF(x, y):
            return f(x, y)

        assert callF.resultTypeFor(int, float).typeRepresentation is int
        assert callF(1, 1.5) == 2

        @Entrypoint
        def callFAsObject(x: object, y):
            return f(x, y)

        assert callFAsObject.resultTypeFor(int, float).typeRepresentation is object
        assert callFAsObject(1, 1.5) == 2

    def test_compile_function_signature_with_value(self):
        @Entrypoint
        def f(x, y) -> lambda X, Y: 1:
            return 1

        @Entrypoint
        def callF(x, y):
            return f(x, y)

        assert callF.resultTypeFor(int, float).typeRepresentation.Value == 1
        assert callF(1, 1.5) == 1

    def test_compile_function_with_oneof(self):
        @Entrypoint
        def f(x) -> lambda X: X:
            return x

        @Entrypoint
        def callF(x):
            return f(x)

        assert callF.resultTypeFor(OneOf(int, float)).typeRepresentation is object

    def test_compile_function_with_overloads(self):
        @Entrypoint
        def f(x: int, y: float) -> lambda X, Y: X:
            return x

        @f.overload
        def f(x: float, y: int) -> lambda X, Y: X:
            return x

        @Entrypoint
        def callF(x, y):
            return f(x, y)

        # the compiler should be able to see that we are only ever going to match one
        # of the two patterns, and that therefore it can successfully determine what to
        # return
        assert callF.resultTypeFor(OneOf(int, float), OneOf(int, float)).typeRepresentation is OneOf(float, int)

        assert callF(1, 1.5) == 1
        assert callF(1.5, 1) == 1.5

    def test_compile_function_with_overloads_indeterminate(self):
        @Entrypoint
        def f(x: int, y) -> lambda X, Y: Y:
            return x

        @f.overload
        def f(x: float, y) -> lambda X, Y: Y:
            return x

        @Entrypoint
        def callF(x, y: object):
            return f(x, y)

        # the compiler will know 'y' as an object, and have to pick, at runtime,
        # between the first and the second overload. Because 'y' is an object,
        # it won't be able to call the signature function
        assert callF.resultTypeFor(OneOf(int, float), float).typeRepresentation is object
        assert callF(1, 1.5) == 1.0
        assert callF(1.5, 1) == 1

    def test_nocompile_function_with_signatures(self):
        @NotCompiled
        def f(x, y) -> lambda X, Y: X:
            return x

        @Entrypoint
        def callF(x, y):
            return f(x, y)

        assert f.resultTypeFor(int, float).typeRepresentation is int
        assert callF.resultTypeFor(int, float).typeRepresentation is int
        assert callF.resultTypeFor(object, float).typeRepresentation is object

        assert callF(1, 1.5) == 1
        assert callF(1.5, 1) == 1.5

        @Entrypoint
        def callFAsObject(x, y: object):
            return f(x, y)

        assert callFAsObject(1, 1.5) == 1
        assert callFAsObject(1.5, 1) == 1.5

    def test_class_function_signatures_work(self):
        class AClass(Class):
            def f(self, x) -> lambda SELF, X: X:
                return x

        assert AClass().f(10) == 10

    def test_compiling_final_class_method_with_signature(self):
        class AClass(Class, Final):
            def f(self, x, y) -> lambda SELF, X, Y: Y:
                return x

        @Entrypoint
        def callIt(x, y):
            return AClass().f(x, y)

        assert callIt.resultTypeFor(int, float).typeRepresentation is float

    def test_compiling_final_class_method_with_signature_and_overloads(self):
        class AClass(Class, Final):
            def f(self, x: float, y) -> lambda SELF, X, Y: Y:
                return x

            def f(self, x: int, y) -> lambda SELF, X, Y: Y:  # noqa
                return x

        @Entrypoint
        def callIt(x, y):
            return AClass().f(x, y)

        assert callIt.resultTypeFor(OneOf(int, float), float).typeRepresentation is float

    def test_compiling_nonfinal_signature_methods(self):
        class BaseClass(Class):
            def f(self, x, y) -> lambda SELF, X, Y: Y:
                return x

        class ChildClass(BaseClass):
            def f(self, x, y) -> lambda SELF, X, Y: Y:
                return x + 1

        @Entrypoint
        def callIt(inst, x, y):
            return inst.f(x, y)

        assert callIt.resultTypeFor(BaseClass, int, float).typeRepresentation is float
        assert callIt.resultTypeFor(ChildClass, int, float).typeRepresentation is float

        @Entrypoint
        def callItAsBase(inst: BaseClass, x, y):
            return inst.f(x, y)

        assert callItAsBase(BaseClass(), 1.5, 1) == 1
        assert callItAsBase(ChildClass(), 1.5, 1) == 2

    def test_compiling_nonfinal_signature_methods_with_type_change(self):
        class BaseClass(Class):
            def f(self, x, y) -> lambda SELF, X, Y: Y:
                return x

        # this shouldn't work because we've promised we're going to return Y, not str.
        # the rule is that return values produced by subclasses must be
        # convertible to the base-class return type at the signature level. We don't
        # do any runtime checking of this invariant, but we'll insist on it at the
        # compiler level.
        class ChildClass(BaseClass):
            def f(self, x, y) -> lambda SELF, X, Y: str:
                return "hi"

        with self.assertRaisesRegex(Exception, "promised a return type"):
            ChildClass().f(1, 2)

        @Entrypoint
        def callIt(inst, x, y):
            return inst.f(x, y)

        assert callIt.resultTypeFor(BaseClass, int, float).typeRepresentation is float

        # we should recognize that we can't convert a str at the signature level to
        # a float, and complain
        assert callIt.resultTypeFor(ChildClass, int, float) is None

        @Entrypoint
        def callItAsBase(inst: BaseClass, x, y):
            return inst.f(x, y)

        assert callItAsBase(BaseClass(), 1.5, 1) == 1

        # if we attempt to actually call the child class masqueraded as a
        # base class, then we should get a runtime error
        with self.assertRaisesRegex(Exception, "promised a return type"):
            assert callItAsBase(ChildClass(), 1.5, 1) == 2

    def test_conflicting_types_dont_affect_refined_filters(self):
        class BaseClass(Class):
            def f(self, x: int) -> int:
                return 0

        class ChildClass(BaseClass):
            def f(self, x) -> str:
                return "hi"

        assert ChildClass().f("0") == "hi"

        with self.assertRaisesRegex(Exception, "promised a return type"):
            ChildClass().f(0)

    def test_compiler_honors_elided_suclass_return_type(self):
        class BaseClass(Class):
            def f(self) -> int:
                return 1

        class ChildClass(BaseClass):
            def f(self):
                return 2

        @Entrypoint
        def callIt(c):
            return c.f()

        @Entrypoint
        def callItAsBase(c: BaseClass):
            return c.f()

        assert callIt(BaseClass()) == 1
        assert callIt(ChildClass()) == 2

        assert callItAsBase(BaseClass()) == 1
        assert callItAsBase(ChildClass()) == 2

    def test_dispatch_to_function_overload(self):
        @Function
        def f(x: str):
            return "str"

        @f.overload
        def f(x: int):
            return "int"

        @f.overload
        def f(x):
            return "object"

        @Entrypoint
        def callF(x: OneOf(int, str)):
            return f(x)

        assert callF("a") == "str"
        assert callF(0) == "int"

    def test_invalid_multi_dispatch(self):
        class BaseClass(Class):
            def f(self, x) -> int:
                return 0

        class ChildClass(BaseClass):
            def f(self, x: str) -> str:
                return x

            def f(self, x: ListOf(int)) -> ListOf(int):  # noqa
                return x

        @Entrypoint
        def callIt(c, x: OneOf(ListOf(int), int, str)):
            return c.f(x)

        # this should puke
        with self.assertRaisesRegex(TypeError, "proposed to return 'str'"):
            callIt(ChildClass(), 'hi')

        with self.assertRaisesRegex(TypeError, "proposed to return 'ListOf"):
            callIt(ChildClass(), [1])

        # this should work
        assert callIt(BaseClass(), 1) == 0

        # redefining the base class should work
        class ChildChildClass(ChildClass, Final):
            def f(self, x: int) -> int:
                return x

        assert callIt(ChildChildClass(), 1) == 1

    def test_type_signature_conflict(self):
        class A(Class):
            pass

        class B(Class):
            pass

        class C(Class):
            pass

        class CFinal(C, Final):
            pass

        class Both(A, B):
            pass

        class BaseClass(Class):
            def f(self, x: A) -> int:
                return 0

        class ChildClass(BaseClass):
            def f(self, x: B) -> str:
                return "hi"

        class FinalChildClass(ChildClass, Final):
            pass

        class FinalChildClassWithDef(ChildClass, Final):
            def f(self, x):
                return ChildClass.f(self, x)

        class FinalChildClassWithBadOverride(ChildClass, Final):
            def f(self, x) -> None:
                return

        assert ChildClass().f(A()) == 0
        assert ChildClass().f(B()) == "hi"

        # this should fail because B is not final, and therefore if we pass
        # Both we end up with a type conflict because it matches in both classes.
        with self.assertRaisesRegex(TypeError, "proposed to return 'str'"):
            ChildClass().f(Both())

        def callItAs(T1, T2):
            @Entrypoint
            def callIt(c: T1, x: T2):
                return c.f(x)

            return callIt

        # check this both ways - with a final child class that's just a passthrough
        # and also a final child class that defers to the base.
        for Subclass in [FinalChildClass, FinalChildClassWithDef]:
            # first, check regular dispatch is correct. In this case, we should be able to determine that
            # the only possible result is "int", because even though an 'A' might match 'B', it would
            # then definitely throw an exception
            assert callItAs(Subclass, A).resultTypeFor(Subclass, A).typeRepresentation is int
            assert callItAs(Subclass, A)(Subclass(), A()) == 0

            # similarly, if we have a 'B', then it could be an A, but that case would throw an
            # exception.
            assert callItAs(Subclass, B).resultTypeFor(Subclass, B).typeRepresentation is str
            assert callItAs(Subclass, B)(Subclass(), B()) == "hi"

            # this should raise, because a 'Both' matches both, which would be an invalid signature
            assert callItAs(Subclass, Both).resultTypeFor(Subclass, Both) is None

            with self.assertRaisesRegex(TypeError, "proposed to return 'str'"):
                callItAs(Subclass, Both)(Subclass(), Both())

            # if we call it as a B, but pass a Both, that should throw an exception
            with self.assertRaisesRegex(TypeError, "proposed to return 'str'"):
                callItAs(Subclass, B)(Subclass(), Both())

        for Subtype in [A, B, Both]:
            assert callItAs(FinalChildClassWithBadOverride, Subtype).resultTypeFor(
                FinalChildClassWithBadOverride, Subtype
            ) is None

            with self.assertRaisesRegex(TypeError, "proposed to return 'NoneType'"):
                callItAs(FinalChildClassWithBadOverride, Subtype)(
                    FinalChildClassWithBadOverride(), Subtype()
                )

        # this should just work
        # assert callItAs(BaseClass, A)(BaseClass(), A()) == 0

        # the compiler should figure out that if we pass A to ChildClass, then we need
        # a check for whether its a subclass of B. If it is, then we should throw an
        # exception
        # assert callItAs(BaseClass, A)(ChildClass(), A()) == 0

        # assert callItAs(BaseClass, A).resultTypeFor(ChildClass, A).typeRepresentation is int

        # compiler should see that return type is int, because if it matches B it'll be
        # with an exception.
        # assert callItAs(FinalChildClass, A).resultTypeFor(FinalChildClass, A).typeRepresentation is int
        # assert callItAs(FinalChildClass, A)(FinalChildClass(), A()) == 0

        return

        # the compiler should figure out that if we pass A to ChildClass, then we need
        # a check for whether its a subclass of B. If it is, then we should throw a
        # runtime error

        assert callItAs(ChildClass, A)(ChildClass(), A()) == 0
        assert callItAs(ChildClass, B)(ChildClass(), B()) == "hi"

        assert callItAs(BaseClass, A)(ChildClass(), A()) == 0
        assert callItAs(BaseClass, B)(ChildClass(), B()) == "hi"

        assert callItAs(ChildClass, object)(ChildClass(), A()) == 0
        assert callItAs(ChildClass, object)(ChildClass(), B()) == "hi"

        assert callItAs(BaseClass, object)(ChildClass(), A()) == 0
        assert callItAs(BaseClass, object)(ChildClass(), B()) == "hi"

        assert ChildClass().f(B()) == "hi"

        with self.assertRaisesRegex(TypeError, "proposed to return 'str'"):
            callItAs(ChildClass, A)(ChildClass(), Both())

    def test_type_inference_classes(self):
        class A(Class):
            pass

        class B(Class, Final):
            pass

        @Entrypoint
        def f(x):
            return ListOf(type(x))()

        assert f.resultTypeFor(int).typeRepresentation == ListOf(int)
        assert f.resultTypeFor(A).typeRepresentation != ListOf(A)
        assert f.resultTypeFor(B).typeRepresentation == ListOf(B)

    def test_typeof(self):
        @Function
        def f(x) -> TypeOf(lambda x: x + x):
            return x

        assert f.resultTypeFor(int).typeRepresentation == int
        assert f.resultTypeFor(float).typeRepresentation == float

    def test_typeof_with_class_arg(self):
        class A(Class):
            def f(self, x) -> float:
                return x

        @Function
        def f(a, x) -> TypeOf(lambda a, x: a.f(x)):
            return a.f(x)

        assert f.resultTypeFor(A, int).typeRepresentation == float
        assert f.resultTypeFor(A, float).typeRepresentation == float

    def test_typeof_with_mutual_recursion(self):
        class A(Class):
            def f(self, x) -> TypeOf(lambda self, x: B().f(x - 1) + 1 if x > 0 else x):
                return x

        class B(Class):
            def f(self, x) -> TypeOf(lambda self, x: A().f(x - 1) + 1 if x > 0 else x):
                return x

        @Function
        def f(x):
            return A().f(x)

        assert f.resultTypeFor(int).typeRepresentation == int
