from nativepython.typed_expression import TypedExpression
import nativepython.native_ast as native_ast
from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.none_wrapper import NoneWrapper
from nativepython.type_wrappers.python_type_object_wrapper import PythonTypeObjectWrapper
from nativepython.type_wrappers.module_wrapper import ModuleWrapper
from nativepython.type_wrappers.python_free_function_wrapper import PythonFreeFunctionWrapper
from nativepython.type_wrappers.python_free_object_wrapper import PythonFreeObjectWrapper
from nativepython.type_wrappers.python_typed_function_wrapper import PythonTypedFunctionWrapper
from nativepython.type_wrappers.tuple_of_wrapper import TupleOfWrapper
from nativepython.type_wrappers.pointer_to_wrapper import PointerToWrapper
from nativepython.type_wrappers.list_of_wrapper import ListOfWrapper
from nativepython.type_wrappers.one_of_wrapper import OneOfWrapper
from nativepython.type_wrappers.class_wrapper import ClassWrapper
from nativepython.type_wrappers.const_dict_wrapper import ConstDictWrapper
from nativepython.type_wrappers.tuple_wrapper import TupleWrapper, NamedTupleWrapper
from nativepython.type_wrappers.alternative_wrapper import makeAlternativeWrapper
from nativepython.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from nativepython.type_wrappers.len_wrapper import LenWrapper
from nativepython.type_wrappers.range_wrapper import _RangeWrapper
from nativepython.type_wrappers.bytecount_wrapper import BytecountWrapper
from nativepython.type_wrappers.arithmetic_wrapper import Int64Wrapper, Float64Wrapper, BoolWrapper
from nativepython.type_wrappers.string_wrapper import StringWrapper
from nativepython.type_wrappers.bytes_wrapper import BytesWrapper
from nativepython.type_wrappers.python_object_of_type_wrapper import PythonObjectOfTypeWrapper
from types import ModuleType
from typed_python._types import TypeFor, bytecount, resolveForwards
from typed_python import *

_type_to_type_wrapper_cache = {}


def typedPythonTypeToTypeWrapper(t):
    if t not in _type_to_type_wrapper_cache:
        _type_to_type_wrapper_cache[t] = _typedPythonTypeToTypeWrapper(t)
    return _type_to_type_wrapper_cache[t]


def _typedPythonTypeToTypeWrapper(t):
    if isinstance(t, Wrapper):
        return t

    if not hasattr(t, '__typed_python_category__'):
        t = TypeFor(t)
        assert hasattr(t, '__typed_python_category__'), t

    resolveForwards(t)

    if t is Int64:
        return Int64Wrapper()

    if t is Float64:
        return Float64Wrapper()

    if t is Bool:
        return BoolWrapper()

    if t is NoneType:
        return NoneWrapper()

    if t is String:
        return StringWrapper()

    if t is Bytes:
        return BytesWrapper()

    if t.__typed_python_category__ == "Class":
        return ClassWrapper(t)

    if t.__typed_python_category__ == "Alternative":
        return makeAlternativeWrapper(t)

    if t.__typed_python_category__ == "ConstDict":
        return ConstDictWrapper(t)

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

    if t.__typed_python_category__ == "Function":
        return PythonTypedFunctionWrapper(t)

    if t.__typed_python_category__ == "BoundMethod":
        return BoundMethodWrapper(t)

    if t.__typed_python_category__ == "TupleOf":
        return TupleOfWrapper(t)

    if t.__typed_python_category__ == "OneOf":
        return OneOfWrapper(t)

    if t.__typed_python_category__ == "PythonObjectOfType":
        return PythonObjectOfTypeWrapper(t)

    assert False, t


def pythonObjectRepresentation(context, f):
    if f is len:
        return TypedExpression(context, native_ast.nullExpr, LenWrapper(), False)

    if f is range:
        return TypedExpression(context, native_ast.nullExpr, _RangeWrapper, False)

    if f is bytecount:
        return TypedExpression(context, native_ast.nullExpr, BytecountWrapper(), False)

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
            False
        )
    if isinstance(f, int):
        return TypedExpression(
            context,
            native_ast.Expression.Constant(
                val=native_ast.Constant.Int(val=f, bits=64, signed=True)
            ),
            Int64Wrapper(),
            False
        )
    if isinstance(f, float):
        return TypedExpression(
            context,
            native_ast.Expression.Constant(
                val=native_ast.Constant.Float(val=f, bits=64)
            ),
            Float64Wrapper(),
            False
        )
    if isinstance(f, str):
        return StringWrapper().constant(context, f)
    if isinstance(f, bytes):
        return BytesWrapper().constant(context, f)

    if isinstance(f, type(pythonObjectRepresentation)):
        return TypedExpression(
            context,
            native_ast.nullExpr,
            PythonFreeFunctionWrapper(f),
            False
        )

    if hasattr(f, '__typed_python_category__'):
        if f.__typed_python_category__ == "Function":
            return TypedExpression(
                context,
                native_ast.nullExpr,
                PythonTypedFunctionWrapper(f),
                False
            )

    if isinstance(f, type):
        return TypedExpression(context, native_ast.nullExpr, PythonTypeObjectWrapper(f), False)

    if isinstance(f, ModuleType):
        return TypedExpression(context, native_ast.nullExpr, ModuleWrapper(f), False)

    return TypedExpression(context, native_ast.nullExpr, PythonFreeObjectWrapper(f), False)
