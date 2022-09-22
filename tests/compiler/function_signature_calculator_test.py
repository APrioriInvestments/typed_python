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


from typed_python.compiler.type_wrappers.function_signature_calculator import (
    FunctionSignatureCalculator,
    CannotBeDetermined,
    SomeInvalidClassReturnType
)

from typed_python import Class, Final, ListOf, Function, OneOf, SubclassOf

from typed_python._types import canConvertToTrivially


def anyIsBad(types):
    for T, ix in types:
        if T is SomeInvalidClassReturnType or T is CannotBeDetermined or T is None:
            return True

    return False


def test_basic():
    class A(Class):
        pass

    class B(Class):
        pass

    class Both(A, B):
        pass

    class BFinal(B, Final):
        pass

    class AFinal(A, Final):
        pass

    class C(Class):
        pass

    class CFinal(C, Final):
        pass

    class BaseClass(Class):
        def f(self, x: A) -> int:
            return 0

    class ChildClass(BaseClass):
        def f(self, x: B) -> str:
            return "hi"

    calc = FunctionSignatureCalculator(ChildClass.f)

    # if we call the base overload with an A, we'll get an int
    assert calc.returnTypeForOverload(1, [ChildClass, A], {}) is int

    # if we were to flow through overload 0 with 'A', then we must have a subclass of 'A'
    # that would then trigger an exception
    assert calc.returnTypeForOverload(0, [ChildClass, A], {}) is SomeInvalidClassReturnType

    # if we match the child class, we _could_ have an exception, but its not guaranteed
    assert calc.returnTypeForOverload(0, [ChildClass, B], {}) is str

    assert calc.returnTypeForLevel(0, [ChildClass, A], {}) is int
    assert calc.returnTypeForLevel(0, [ChildClass, B], {}) is str
    assert calc.returnTypeForLevel(0, [ChildClass, Both], {}) is SomeInvalidClassReturnType

    assert anyIsBad(calc.returnTypesForLevel(0, [ChildClass, A], {})[0])
    assert not anyIsBad(calc.returnTypesForLevel(0, [ChildClass, AFinal], {})[0])

    assert len(calc.overloadInvalidSignatures(0, [ChildClass, B], {})) == 1

    class FinalChildClass(ChildClass, Final):
        pass

    calc = FunctionSignatureCalculator(FinalChildClass.f)

    assert calc.returnTypeFor([FinalChildClass, A], {}) is int
    assert calc.returnTypeFor([FinalChildClass, B], {}) is str
    assert calc.returnTypeFor([FinalChildClass, Both], {}) is SomeInvalidClassReturnType

    class ChildClassWithBadOverride(ChildClass, Final):
        def f(self, x) -> bytes:
            return

    calc = FunctionSignatureCalculator(ChildClassWithBadOverride.f)
    assert calc.returnTypeFor([ChildClassWithBadOverride, B], {}) is SomeInvalidClassReturnType
    assert calc.returnTypeFor([ChildClassWithBadOverride, A], {}) is SomeInvalidClassReturnType
    assert calc.returnTypeFor([ChildClassWithBadOverride, Both], {}) is SomeInvalidClassReturnType


def test_override_diamond():
    class BaseClass(Class):
        def f(self, x) -> int:
            return 0

    class ChildClass(BaseClass):
        def f(self, x: str) -> str:
            return x

        def f(self, x: ListOf(int)) -> ListOf(int):  # noqa
            return x

    class ChildChildClass(ChildClass):
        def f(self, x: int) -> int:
            return x

    calc = FunctionSignatureCalculator(ChildChildClass.f)

    assert calc.returnTypeFor([ChildClass, int], {}) is int
    assert calc.returnTypeFor([ChildChildClass, int], {}) is int

    assert not calc.overloadInvalidSignatures(0, [ChildChildClass, int], {})


def test_understands_more_specific_types():
    @Function
    def f(x: int) -> lambda X: X:
        return x

    calc = FunctionSignatureCalculator(f)

    assert calc.returnTypeFor([OneOf(int, float)], {}) is int


def test_convert_subclass_of_trivially():
    class C(Class):
        pass

    class D(C):
        pass

    assert canConvertToTrivially(SubclassOf(D), SubclassOf(D))
    assert canConvertToTrivially(SubclassOf(D), SubclassOf(C))
    assert not canConvertToTrivially(SubclassOf(C), SubclassOf(D))
