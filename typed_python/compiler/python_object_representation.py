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

import threading
import _thread

from typed_python.compiler.typed_expression import TypedExpression
import typed_python.compiler.native_ast as native_ast
from typed_python.type_function import TypeFunction
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python.compiler.type_wrappers.none_wrapper import NoneWrapper
from typed_python.compiler.type_wrappers.method_descriptor_wrapper import MethodDescriptorWrapper
from typed_python.compiler.type_wrappers.python_type_object_wrapper import PythonTypeObjectWrapper
from typed_python.compiler.type_wrappers.module_wrapper import ModuleWrapper
from typed_python.compiler.type_wrappers.typed_cell_wrapper import TypedCellWrapper
from typed_python.compiler.type_wrappers.unresolved_forward_type_wrapper import UnresolvedForwardTypeWrapper
from typed_python.compiler.type_wrappers.python_free_function_wrapper import PythonFreeFunctionWrapper
from typed_python.compiler.type_wrappers.python_free_object_wrapper import PythonFreeObjectWrapper
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import PythonTypedFunctionWrapper
from typed_python.compiler.type_wrappers.value_wrapper import ValueWrapper
from typed_python.compiler.type_wrappers.tuple_of_wrapper import TupleOfWrapper
from typed_python.compiler.type_wrappers.pointer_to_wrapper import PointerToWrapper, PointerToObjectWrapper
from typed_python.compiler.type_wrappers.ref_to_wrapper import RefToWrapper, RefToObjectWrapper
from typed_python.compiler.type_wrappers.list_of_wrapper import ListOfWrapper
from typed_python.compiler.type_wrappers.isinstance_wrapper import IsinstanceWrapper
from typed_python.compiler.type_wrappers.issubclass_wrapper import IssubclassWrapper
from typed_python.compiler.type_wrappers.subclass_of_wrapper import SubclassOfWrapper
from typed_python.compiler.type_wrappers.one_of_wrapper import OneOfWrapper
from typed_python.compiler.type_wrappers.class_wrapper import ClassWrapper
from typed_python.compiler.type_wrappers.held_class_wrapper import HeldClassWrapper
from typed_python.compiler.type_wrappers.const_dict_wrapper import ConstDictWrapper
from typed_python.compiler.type_wrappers.dict_wrapper import DictWrapper
from typed_python.compiler.type_wrappers.set_wrapper import SetWrapper
from typed_python.compiler.type_wrappers.tuple_wrapper import TupleWrapper, NamedTupleWrapper
from typed_python.compiler.type_wrappers.slice_type_object_wrapper import SliceWrapper
from typed_python.compiler.type_wrappers.alternative_wrapper import makeAlternativeWrapper
from typed_python.compiler.type_wrappers.alternative_wrapper import AlternativeMatcherWrapper
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.len_wrapper import LenWrapper
from typed_python.compiler.type_wrappers.hash_wrapper import HashWrapper
from typed_python.compiler.type_wrappers.range_wrapper import range as compilableRange
from typed_python.compiler.type_wrappers.print_wrapper import PrintWrapper
from typed_python.compiler.type_wrappers.super_wrapper import SuperWrapper
from typed_python.compiler.type_wrappers.compiler_introspection_wrappers import (
    IsCompiledWrapper,
    TypeKnownToCompiler,
    LocalVariableTypesKnownToCompiler,
)
from typed_python.compiler.type_wrappers.make_named_tuple_wrapper import MakeNamedTupleWrapper
from typed_python.compiler.type_wrappers.math_wrappers import MathFunctionWrapper
from typed_python.compiler.type_wrappers.builtin_wrappers import BuiltinWrapper
from typed_python.compiler.type_wrappers.bytecount_wrapper import BytecountWrapper
from typed_python.compiler.type_wrappers.arithmetic_wrapper import IntWrapper, FloatWrapper, BoolWrapper
from typed_python.compiler.type_wrappers.string_wrapper import StringWrapper, StringMaketransWrapper
from typed_python.compiler.type_wrappers.bytes_wrapper import BytesWrapper, BytesMaketransWrapper
from typed_python.compiler.type_wrappers.python_object_of_type_wrapper import PythonObjectOfTypeWrapper
from typed_python.compiler.type_wrappers.abs_wrapper import AbsWrapper
from typed_python.compiler.type_wrappers.min_max_wrapper import MinWrapper, MaxWrapper
from typed_python.compiler.type_wrappers.repr_wrapper import ReprWrapper
from types import ModuleType
from typed_python._types import (
    TypeFor, bytecount, prepareArgumentToBePassedToCompiler, allForwardTypesResolved
)
from typed_python import (
    Type, Int32, Int16, Int8, UInt64, UInt32, UInt16,
    UInt8, Float32, makeNamedTuple,
    ListOf, isCompiled,
    typeKnownToCompiler,
    localVariableTypesKnownToCompiler,
    pointerTo, refTo
)

# the type of bound C methods on types.
method_descriptor = type(ListOf(int).append)

_type_to_type_wrapper_cache = {}


def typedPythonTypeToTypeWrapper(t):
    if isinstance(t, Wrapper):
        return t

    if t not in _type_to_type_wrapper_cache:
        _type_to_type_wrapper_cache[t] = _typedPythonTypeToTypeWrapper(t)

    return _type_to_type_wrapper_cache[t]


_concreteWrappers = {
    Int8: IntWrapper(Int8),
    Int16: IntWrapper(Int16),
    Int32: IntWrapper(Int32),
    int: IntWrapper(int),
    UInt8: IntWrapper(UInt8),
    UInt16: IntWrapper(UInt16),
    UInt32: IntWrapper(UInt32),
    UInt64: IntWrapper(UInt64),
    Float32: FloatWrapper(Float32),
    float: FloatWrapper(float),
    bool: BoolWrapper(),
    None: NoneWrapper(),
    type(None): NoneWrapper(),
    str: StringWrapper(),
    bytes: BytesWrapper()
}


def functionIsCompilable(f):
    """Is a python function 'f' compilable?

    Specifically, we try to cordon off the functions we know are not going to compile
    so that we can just represent them as python objects.
    """
    # we know we can't compile standard numpy and scipy functions
    if f.__module__ == "numpy" or f.__module__.startswith("numpy."):
        return False

    if f.__module__ == "scipy" or f.__module__.startswith("scipy."):
        return False

    return True


def _typedPythonTypeToTypeWrapper(t):
    if isinstance(t, Wrapper):
        return t

    if t in _concreteWrappers:
        return _concreteWrappers[t]

    if not hasattr(t, '__typed_python_category__'):
        t = TypeFor(t)

    assert isinstance(t, type), t

    if not allForwardTypesResolved(t):
        return UnresolvedForwardTypeWrapper(t)

    if not hasattr(t, '__typed_python_category__'):
        # this is will be a PythonObjectOfType, but we never actually return such an object
        # to the interpreter
        if t is threading.RLock:
            t = _thread.RLock
        elif t is threading.Lock:
            t = _thread.LockType

        return PythonObjectOfTypeWrapper(t)

    if t.__typed_python_category__ == "Class":
        return ClassWrapper(t)

    if t.__typed_python_category__ == "HeldClass":
        return HeldClassWrapper(t)

    if t.__typed_python_category__ == "Alternative":
        return makeAlternativeWrapper(t)

    if t.__typed_python_category__ == "ConstDict":
        return ConstDictWrapper(t)

    if t.__typed_python_category__ == "Dict":
        return DictWrapper(t)

    if t.__typed_python_category__ == "ConcreteAlternative":
        return makeAlternativeWrapper(t)

    if t.__typed_python_category__ == "NamedTuple":
        return NamedTupleWrapper(t)

    if t.__typed_python_category__ == "Tuple":
        return TupleWrapper(t)

    if t.__typed_python_category__ == "ListOf":
        return ListOfWrapper(t)

    if t.__typed_python_category__ == "PointerTo":
        return PointerToWrapper(t)

    if t.__typed_python_category__ == "RefTo":
        return RefToWrapper(t)

    if t.__typed_python_category__ == "Function":
        return PythonTypedFunctionWrapper(t)

    if t.__typed_python_category__ == "BoundMethod":
        return BoundMethodWrapper(t)

    if t.__typed_python_category__ == "AlternativeMatcher":
        return AlternativeMatcherWrapper(t)

    if t.__typed_python_category__ == "TupleOf":
        return TupleOfWrapper(t)

    if t.__typed_python_category__ == "OneOf":
        return OneOfWrapper(t)

    if t.__typed_python_category__ == "TypedCell":
        return TypedCellWrapper(t)

    if t.__typed_python_category__ == "SubclassOf":
        return SubclassOfWrapper(t)

    if t.__typed_python_category__ == "Value":
        if type(t.Value) in _concreteWrappers or type(t.Value) in (str, int, float, bool):
            return ValueWrapper(t)
        return pythonObjectRepresentation(None, t.Value).expr_type

    if t.__typed_python_category__ == "Set":
        return SetWrapper(t)

    assert False, (t, getattr(t, '__typed_python_category__', None))


def pythonObjectRepresentation(context, f, owningGlobalScopeAndName=None):
    """Create a representation for a python constant.

    Args:
        context - the ExpressionConversionContext
        f - the constant
        owningGlobalScopeAndName - either None, or a pair (dict, name)
            where the object is equivalent to dict[name]
    """
    if isinstance(f, CompilableBuiltin):
        return TypedExpression(context, native_ast.nullExpr, f, False)

    if f is len:
        return TypedExpression(context, native_ast.nullExpr, LenWrapper(), False)

    if f is hash:
        return TypedExpression(context, native_ast.nullExpr, HashWrapper(), False)

    if f is slice:
        return TypedExpression(context, native_ast.nullExpr, SliceWrapper(), False)

    if f is abs:
        return TypedExpression(context, native_ast.nullExpr, AbsWrapper(), False)

    if f is min:
        return TypedExpression(context, native_ast.nullExpr, MinWrapper(), False)

    if f is max:
        return TypedExpression(context, native_ast.nullExpr, MaxWrapper(), False)

    if f is repr:
        return TypedExpression(context, native_ast.nullExpr, ReprWrapper(), False)

    if f is range:
        return pythonObjectRepresentation(context, compilableRange, owningGlobalScopeAndName)

    if f is bytes.maketrans:
        return TypedExpression(context, native_ast.nullExpr, BytesMaketransWrapper(), False)

    if f is str.maketrans:
        return TypedExpression(context, native_ast.nullExpr, StringMaketransWrapper(), False)

    if f is isinstance:
        return TypedExpression(context, native_ast.nullExpr, IsinstanceWrapper(), False)

    if f is issubclass:
        return TypedExpression(context, native_ast.nullExpr, IssubclassWrapper(), False)

    if f is bytecount:
        return TypedExpression(context, native_ast.nullExpr, BytecountWrapper(), False)

    if f is print:
        return TypedExpression(context, native_ast.nullExpr, PrintWrapper(), False)

    if f is super:
        return TypedExpression(context, native_ast.nullExpr, SuperWrapper(), False)

    if f is pointerTo:
        return TypedExpression(context, native_ast.nullExpr, PointerToObjectWrapper(), False)

    if f is refTo:
        return TypedExpression(context, native_ast.nullExpr, RefToObjectWrapper(), False)

    if f is isCompiled:
        return TypedExpression(context, native_ast.nullExpr, IsCompiledWrapper(), False)

    if f is typeKnownToCompiler:
        return TypedExpression(context, native_ast.nullExpr, TypeKnownToCompiler(), False)

    if f is localVariableTypesKnownToCompiler:
        return TypedExpression(context, native_ast.nullExpr, LocalVariableTypesKnownToCompiler(), False)

    if f is makeNamedTuple:
        return TypedExpression(context, native_ast.nullExpr, MakeNamedTupleWrapper(), False)

    if f in MathFunctionWrapper.SUPPORTED_FUNCTIONS:
        return TypedExpression(context, native_ast.nullExpr, MathFunctionWrapper(f), False)

    if f in BuiltinWrapper.SUPPORTED_FUNCTIONS:
        return TypedExpression(context, native_ast.nullExpr, BuiltinWrapper(f), False)

    if f is None:
        return TypedExpression(
            context,
            native_ast.Expression.Constant(
                val=native_ast.Constant.Void()
            ),
            NoneWrapper(),
            False
        )

    if isinstance(f, bool):
        return TypedExpression(
            context,
            native_ast.Expression.Constant(
                val=native_ast.Constant.Int(val=f, bits=1, signed=False)
            ),
            BoolWrapper(),
            False,
            constantValue=f
        )

    if isinstance(f, int):
        return TypedExpression(
            context,
            native_ast.Expression.Constant(
                val=native_ast.Constant.Int(val=f, bits=64, signed=True)
            ),
            IntWrapper(int),
            False,
            constantValue=f
        )

    if isinstance(f, float):
        return TypedExpression(
            context,
            native_ast.Expression.Constant(
                val=native_ast.Constant.Float(val=f, bits=64)
            ),
            FloatWrapper(float),
            False,
            constantValue=f
        )

    if isinstance(f, str):
        return StringWrapper().constant(context, f)

    if isinstance(f, bytes):
        return BytesWrapper().constant(context, f)

    if isinstance(f, type(pythonObjectRepresentation)):
        if functionIsCompilable(f):
            return TypedExpression(
                context,
                native_ast.nullExpr,
                PythonFreeFunctionWrapper(f),
                False
            )
        else:
            return context.constantPyObject(f, owningGlobalScopeAndName)

    if isinstance(f, method_descriptor):
        return TypedExpression(
            context,
            native_ast.nullExpr,
            MethodDescriptorWrapper(f),
            False
        )

    if hasattr(f, '__typed_python_category__') and not isinstance(f, type):
        if f.__typed_python_category__ == "Function":
            f = prepareArgumentToBePassedToCompiler(f)

            if bytecount(f.ClosureType):
                raise Exception(f"Function {f} has nonempty closure {f.ClosureType}")

            return TypedExpression(
                context,
                typedPythonTypeToTypeWrapper(f.ClosureType).getNativeLayoutType().zero(),
                PythonTypedFunctionWrapper(f),
                False
            )

        if f.__typed_python_category__ == "Class":
            # global class instances get held as constants
            return context.constantTypedPythonObject(f, owningGlobalScopeAndName)

    if isinstance(f, type) and issubclass(f, TypeFunction) and len(f.MRO) == 2:
        return TypedExpression(context, native_ast.nullExpr, PythonFreeObjectWrapper(f, False), False)

    if isinstance(f, type):
        return TypedExpression(context, native_ast.nullExpr, PythonTypeObjectWrapper(f), False)

    if isinstance(f, ModuleType):
        return TypedExpression(context, native_ast.nullExpr, ModuleWrapper(f), False)

    if isinstance(f, Type):
        # it's a typed python object at module level scope
        return context.constantTypedPythonObject(f, owningGlobalScopeAndName)

    return context.constantPyObject(f, owningGlobalScopeAndName)


def pythonObjectRepresentationType(f):
    if isinstance(f, str):
        return StringWrapper()

    if isinstance(f, bytes):
        return BytesWrapper()

    return pythonObjectRepresentation(None, f).expr_type
