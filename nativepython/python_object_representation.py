from nativepython.typed_expression import TypedExpression
import nativepython.native_ast as native_ast
from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.none_wrapper import NoneWrapper
from nativepython.type_wrappers.python_type_wrappers import PythonTypeObjectWrapper
from nativepython.type_wrappers.python_free_function_wrapper import PythonFreeFunctionWrapper
from nativepython.type_wrappers.tuple_of_wrapper import TupleOfWrapper
from nativepython.type_wrappers.len_wrapper import LenWrapper
from nativepython.type_wrappers.arithmetic_wrapper import Int64Wrapper, Float64Wrapper, BoolWrapper
from nativepython.type_wrappers.python_object_of_type_wrapper import PythonObjectOfTypeWrapper
from typed_python._types import TypeFor
from typed_python import *

def typedPythonTypeToTypeWrapper(t):
    if isinstance(t, Wrapper):
        return t

    if not hasattr(t, '__typed_python_category__'):
        t = TypeFor(t)
        assert hasattr(t, '__typed_python_category__'), t

    if t is Int64():
        return Int64Wrapper()

    if t is Float64():
        return Float64Wrapper()

    if t is Bool():
        return BoolWrapper()

    if t is NoneType():
        return NoneWrapper()

    if t.__typed_python_category__ == "TupleOf":
        return TupleOfWrapper(t)

    if t.__typed_python_category__ == "PythonObjectOfType":
        return PythonObjectOfTypeWrapper(t)

    assert False, t

def pythonObjectRepresentation(f):
    if f in (int,bool,float,str,type(None)):
        return TypedExpression(native_ast.nullExpr, PythonTypeObjectWrapper(f), False)

    if f is len:
        return TypedExpression(native_ast.nullExpr, LenWrapper(), False)

    if f is None:
        return TypedExpression(
            native_ast.Expression.Constant(
                val=native_ast.Constant.Void()
                ),
            NoneWrapper(),
            False
            )
    if isinstance(f, bool):
        return TypedExpression(
            native_ast.Expression.Constant(
                val=native_ast.Constant.Int(val=f,bits=1,signed=False)
                ),
            BoolWrapper(),
            False
            )
    if isinstance(f, int):
        return TypedExpression(
            native_ast.Expression.Constant(
                val=native_ast.Constant.Int(val=f,bits=64,signed=True)
                ),
            Int64Wrapper(),
            False
            )
    if isinstance(f, float):
        return TypedExpression(
            native_ast.Expression.Constant(
                val=native_ast.Constant.Float(val=f,bits=64)
                ),
            Float64Wrapper(),
            False
            )
    if isinstance(f, type(pythonObjectRepresentation)):
        return TypedExpression(
            native_ast.nullExpr,
            PythonFreeFunctionWrapper(f),
            False
            )

    assert False, f
