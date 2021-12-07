#   Copyright 2017-2020 typed_python Authors
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

import typed_python
import traceback
import numpy

from typed_python import (
    OneOf, Entrypoint, UInt16, Dict, Set, TupleOf, ListOf, Tuple, Int32,
    NamedTuple
)
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python._types import convertObjectToTypeAtLevel, couldConvertObjectToTypeAtLevel


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class TriggerConvert(CompilableBuiltin):
    """A compilable builtin to exercise 'conversion' semantics.

    If you write

        TriggerConvert(T, conversionLevel)(x)

    then the compiler will generate a convert_to_type(T, x, conversionLevel)
    call.

    We use this to verify that these semantics are correctly implemented.
    """
    def __init__(self, T, conversionLevel):
        super().__init__()

        if isinstance(conversionLevel, int):
            conversionLevel = ConversionLevel.fromIntegerLevel(conversionLevel)

        self.T = T
        self.conversionLevel = conversionLevel

    def __eq__(self, other):
        return isinstance(other, TriggerConvert) and (self.T, self.conversionLevel) == (other.T, other.conversionLevel)

    def __hash__(self):
        return hash(("TriggerConvert", self.T, self.conversionLevel))

    def convert_call(self, context, instance, args, kwargs):
        if len(args) != 1 or len(kwargs) != 0:
            context.pushException(TypeError, "TriggerConvert takes 1 positional argument")

        return args[0].convert_to_type(self.T, self.conversionLevel)


levels = [
    ConversionLevel.Signature,
    ConversionLevel.Upcast,
    ConversionLevel.UpcastContainers,
    ConversionLevel.Implicit,
    ConversionLevel.ImplicitContainers,
    ConversionLevel.New
]


def makeConverterDict():
    @Entrypoint
    def convert0(arg, T):
        return TriggerConvert(T, 0)(arg)

    @Entrypoint
    def convert1(arg, T):
        return TriggerConvert(T, 1)(arg)

    @Entrypoint
    def convert2(arg, T):
        return TriggerConvert(T, 2)(arg)

    @Entrypoint
    def convert3(arg, T):
        return TriggerConvert(T, 3)(arg)

    @Entrypoint
    def convert4(arg, T):
        return TriggerConvert(T, 4)(arg)

    @Entrypoint
    def convert5(arg, T):
        return TriggerConvert(T, 5)(arg)

    return {
        ConversionLevel.Signature: convert0,
        ConversionLevel.Upcast: convert1,
        ConversionLevel.UpcastContainers: convert2,
        ConversionLevel.Implicit: convert3,
        ConversionLevel.ImplicitContainers: convert4,
        ConversionLevel.New: convert5,
    }


compiledConverters = makeConverterDict()


def checkConversionToType(x, TargetType, expectedSuccessLevel):
    for level in levels:
        # check whether the interpreter knows it can convert at this level
        couldBeOfType = couldConvertObjectToTypeAtLevel(x, TargetType, level.LEVEL)

        if not couldBeOfType and expectedSuccessLevel is not None and level >= expectedSuccessLevel:
            raise Exception(
                f"Interpreter assumes that converting {x} of type {type(x)} to {TargetType}"
                f" will fail at level {level} when we expect it to succeed."
            )

        try:
            convertObjectToTypeAtLevel(x, TargetType, level.LEVEL)
            succeeded = True
        except Exception:
            succeeded = False

        if expectedSuccessLevel is None or level < expectedSuccessLevel:
            if succeeded:
                raise Exception(
                    f"converting {x} of type {type(x)} to {TargetType} in the interpreter "
                    f"succeeded at level {level} when we expected it to fail."
                )
        else:
            if not succeeded:
                raise Exception(
                    f"converting {x} of type {type(x)} to {TargetType} in the interpreter "
                    f"failed at level {level} when we expected it to succeed."
                )

        # check whether the compiler knows it can convert at this level
        canConvert = typeWrapper(type(x)).can_convert_to_type(typeWrapper(TargetType), level)

        if canConvert is False and expectedSuccessLevel is not None and level >= expectedSuccessLevel:
            raise Exception(
                f"Compiler assumes that converting {x} of type {type(x)} to {TargetType}"
                f" will fail at level {level} when we expect it to succeed."
            )

        if canConvert is True and (expectedSuccessLevel is None or level < expectedSuccessLevel):
            raise Exception(
                f"Compiler assumes that converting {x} of type {type(x)} to {TargetType}"
                f" will succeed at level {level} when we expect it to fail."
            )

        try:
            compiledConverters[level](x, TargetType)
            succeeded = True
        except Exception:
            succeeded = False
            error = traceback.format_exc()

        if expectedSuccessLevel is None or level < expectedSuccessLevel:
            if succeeded:
                raise Exception(
                    f"converting {x} of type {type(x)} to {TargetType} in the compiler "
                    f"succeeded at level {level} when we expected it to fail."
                )
        else:
            if not succeeded:
                raise Exception(
                    f"converting {x} of type {type(x)} to {TargetType} in the compiler "
                    f"failed at level {level} when we expected it to succeed at {expectedSuccessLevel} or higher:\n\n{error}"
                )


def test_register_conversion_semantics():
    checkConversionToType(False, bool, ConversionLevel.Signature)
    checkConversionToType(False, UInt16, ConversionLevel.Upcast)
    checkConversionToType(False, int, ConversionLevel.Upcast)
    checkConversionToType(False, float, ConversionLevel.Upcast)
    checkConversionToType(False, str, ConversionLevel.New)

    checkConversionToType(UInt16(0), bool, ConversionLevel.Implicit)
    checkConversionToType(UInt16(0), UInt16, ConversionLevel.Signature)
    checkConversionToType(UInt16(0), int, ConversionLevel.Upcast)
    checkConversionToType(UInt16(0), float, ConversionLevel.Upcast)
    checkConversionToType(UInt16(0), str, ConversionLevel.New)

    checkConversionToType(0, bool, ConversionLevel.Implicit)
    checkConversionToType(0, UInt16, ConversionLevel.Implicit)
    checkConversionToType(0, int, ConversionLevel.Signature)
    checkConversionToType(0, float, ConversionLevel.Upcast)
    checkConversionToType(0, str, ConversionLevel.New)

    checkConversionToType(0.0, bool, ConversionLevel.Implicit)
    checkConversionToType(0.0, UInt16, ConversionLevel.Implicit)
    checkConversionToType(0.0, int, ConversionLevel.Implicit)
    checkConversionToType(0.0, float, ConversionLevel.Signature)
    checkConversionToType(0.0, str, ConversionLevel.New)

    checkConversionToType("0", bool, ConversionLevel.New)
    checkConversionToType("0", UInt16, ConversionLevel.New)
    checkConversionToType("0", int, ConversionLevel.New)
    checkConversionToType("0", float, ConversionLevel.New)
    checkConversionToType("0", str, ConversionLevel.Signature)


def test_int_to_list_of_int():
    checkConversionToType(1, ListOf(int), None)


def test_untyped_dict_conversion_semantics():
    aDict = {1: 2}
    checkConversionToType(aDict, Dict(int, int), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, ListOf(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, TupleOf(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, Tuple(int), ConversionLevel.New)
    checkConversionToType(aDict, Tuple(int, int), None)

    checkConversionToType(aDict, Dict(float, float), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, ListOf(float), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, TupleOf(float), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, Tuple(float), ConversionLevel.New)
    checkConversionToType(aDict, Tuple(float, float), None)
    checkConversionToType(aDict, Set(float), ConversionLevel.ImplicitContainers)


def test_untyped_list_of_conversion_semantics():
    aList = [1, 2]

    checkConversionToType(aList, ListOf(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(aList, TupleOf(int), ConversionLevel.UpcastContainers)
    checkConversionToType(aList, Tuple(int, int), ConversionLevel.Upcast)
    checkConversionToType(aList, Tuple(int), None)
    checkConversionToType(aList, Set(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(aList, Set(float), ConversionLevel.ImplicitContainers)


def test_untyped_tuple_of_conversion_semantics():
    aTup = (1, 2)

    checkConversionToType(aTup, ListOf(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(aTup, TupleOf(int), ConversionLevel.UpcastContainers)
    checkConversionToType(aTup, Tuple(int, int), ConversionLevel.Upcast)
    checkConversionToType(aTup, Tuple(int), None)
    checkConversionToType(aTup, Set(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(aTup, Set(float), ConversionLevel.ImplicitContainers)


def test_typed_dict_conversion_semantics():
    aDict = Dict(int, float)({1: 2, 3: 4.5})

    checkConversionToType(aDict, Dict(int, float), ConversionLevel.Signature)
    checkConversionToType(aDict, Dict(float, float), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, ListOf(float), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, ListOf(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(aDict, ListOf(str), None)
    checkConversionToType(aDict, TupleOf(str), None)


def test_numpy_scalar_conversion():
    checkConversionToType(numpy.int64(1), int, ConversionLevel.Signature)
    checkConversionToType(numpy.int64(1), float, ConversionLevel.Upcast)
    checkConversionToType(numpy.int64(1), Int32, ConversionLevel.Implicit)
    checkConversionToType(numpy.int64(1), str, ConversionLevel.New)

    checkConversionToType(Int32(1), int, ConversionLevel.Upcast)
    checkConversionToType(Int32(1), float, ConversionLevel.Upcast)
    checkConversionToType(Int32(1), Int32, ConversionLevel.Signature)
    checkConversionToType(Int32(1), str, ConversionLevel.New)

    checkConversionToType(numpy.int32(1), int, ConversionLevel.Upcast)
    checkConversionToType(numpy.int32(1), float, ConversionLevel.Upcast)
    checkConversionToType(numpy.int32(1), Int32, ConversionLevel.Signature)
    checkConversionToType(numpy.int32(1), str, ConversionLevel.New)


def test_convert_things_to_object():
    checkConversionToType(test_convert_things_to_object, object, ConversionLevel.Signature)
    checkConversionToType(str, object, ConversionLevel.Signature)
    checkConversionToType(1, object, ConversionLevel.Signature)
    checkConversionToType(1.0, object, ConversionLevel.Signature)
    checkConversionToType(False, object, ConversionLevel.Signature)
    checkConversionToType("hi", object, ConversionLevel.Signature)
    checkConversionToType([], object, ConversionLevel.Signature)
    checkConversionToType(ListOf(int)(), object, ConversionLevel.Signature)
    checkConversionToType(TupleOf(int)(), object, ConversionLevel.Signature)
    checkConversionToType(Dict(int, int)(), object, ConversionLevel.Signature)
    checkConversionToType(Set(int)(), object, ConversionLevel.Signature)


def test_list_of_conversion_semantics():
    checkConversionToType(ListOf(int)([1, 2]), Set(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(ListOf(float)([1, 2]), Set(int), ConversionLevel.New)
    checkConversionToType(ListOf(int)([1, 2]), Set(float), ConversionLevel.ImplicitContainers)


def test_typed_set_conversion_semantics():
    checkConversionToType({1}, Set(int), ConversionLevel.ImplicitContainers)


def test_convert_containers_to_oneof():
    checkConversionToType(ListOf(int)(), OneOf(None, ListOf(int)), ConversionLevel.Signature)
    checkConversionToType(ListOf(int)(), OneOf(None, ListOf(float)), ConversionLevel.ImplicitContainers)
    checkConversionToType(ListOf(str)(["hi"]), OneOf(None, ListOf(float)), None)
    checkConversionToType(ListOf(str)(["1.0"]), OneOf(None, ListOf(float)), None)


def test_convert_to_bytes():
    checkConversionToType("hi", bytes, None)
    checkConversionToType(ListOf(int)([1, 2, 3]), bytes, ConversionLevel.New)


def test_convert_numpy_to_listof():
    checkConversionToType(numpy.array([1, 2]), ListOf(int), ConversionLevel.ImplicitContainers)


def test_convert_type_to_type():
    checkConversionToType(str, type, ConversionLevel.Signature)


def test_convert_untyped_classes():
    class C1:
        pass

    class C2(C1):
        pass

    checkConversionToType(C2(), C1, ConversionLevel.Signature)
    checkConversionToType(C1(), C2, None)


def test_convert_to_named_tuple():
    checkConversionToType(NamedTuple(x=int)(x=1), NamedTuple(x=float), ConversionLevel.Upcast)
    checkConversionToType(NamedTuple(x=OneOf(int, float))(x=1), NamedTuple(x=int), ConversionLevel.Upcast)
    checkConversionToType(dict(x=1, y=1), NamedTuple(x=int, y=float), ConversionLevel.UpcastContainers)
    checkConversionToType(dict(x=1.5, y=1), NamedTuple(x=int, y=float), ConversionLevel.Implicit)
    checkConversionToType(ListOf(float)([1.5]), ListOf(int), ConversionLevel.ImplicitContainers)
    checkConversionToType(dict(x=ListOf(float)([1.5]), y=1), NamedTuple(x=ListOf(int), y=float), ConversionLevel.ImplicitContainers)
