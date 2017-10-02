#   Copyright 2017 Braxton Mckee
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

import types
import nativepython
import nativepython.native_ast as native_ast
import nativepython.python_ast as python_ast

from nativepython.type_model.type_base import Type
from nativepython.type_model.typed_expression import TypedExpression
from nativepython.exceptions import ConversionException, UnassignableFieldException


class CompileTimeType(Type):
    @property
    def is_pod(self):
        return True
    
    @property
    def is_compile_time(self):
        return True

    @property
    def null_value(self):
        return native_ast.Constant.Struct(())

    def lower(self):
        return native_ast.Type.Struct(())

    def convert_bin_op(self, op, lref, rref):
        l = lref.dereference()
        r = rref.dereference()

        if op._alternative is python_ast.ComparisonOp:
            if (op.matches.Is or op.matches.IsNot) and isinstance(r.expr_type, CompileTimeType):
                res = l.expr_type.python_object_representation is \
                            r.expr_type.python_object_representation

                if op.matches.IsNot:
                    res = not res

                return TypedExpression(
                    native_ast.trueExpr if res else native_ast.falseExpr,
                    nativepython.type_model.Bool
                    )

            if (op.matches.Eq or op.matches.NotEq) and isinstance(r.expr_type, CompileTimeType):
                res = l.expr_type.python_object_representation == \
                            r.expr_type.python_object_representation

                if op.matches.NotEq:
                    res = not res

                return TypedExpression(
                    native_ast.trueExpr if res else native_ast.falseExpr,
                    nativepython.type_model.Bool
                    )

        return super(Type, self).convert_bin_op(op,lref,rref)
    
    @property
    def python_object_representation(self):
        raise ConversionException("Subclasses must implement")

    def as_typed_expression(self):
        return TypedExpression(
            native_ast.Expression.Constant(
                native_ast.Constant.Struct(())
                ),
            self
            )

    def convert_attribute(self, context, instance, attr, allow_double_refs=False):
        o = self.python_object_representation

        if not hasattr(o, attr):
            raise ConversionException("Can't get attribute %s from %s of type %s" % 
                    (attr,o, type(o)))

        return pythonObjectRepresentation(getattr(o, attr))


def representation_for(obj):
    def decorator(override):
        FreePythonObjectReference.free_python_object_overrides[obj] = override
        return override
    return decorator

class FreePythonObjectReference(CompileTimeType):
    free_python_object_overrides = {}

    def __init__(self, obj):
        object.__init__(self)
        self._original_obj = obj

        if obj in self.free_python_object_overrides:
            obj = self.free_python_object_overrides[obj]
        self._obj = obj

    @property
    def python_object_representation(self):
        return self._original_obj

    def convert_call(self, context, instance, args):
        if isinstance(self._obj, types.FunctionType):
            return context.call_py_function(self._obj, args)

        if self._obj is float:
            assert len(args) == 1
            return args[0].convert_to_type(nativepython.type_model.Float64, False)

        if self._obj is int:
            assert len(args) == 1
            return args[0].convert_to_type(nativepython.type_model.Int64, False)

        if isinstance(self._obj, Type):
            if self._obj.is_ref:
                raise ConversionException("Can't instantiate %s" % self._obj)

            #we are initializing an element of the type
            for a in args:
                assert a.expr is not None

            tmp_ref = context.allocate_temporary(self._obj)

            return TypedExpression(
                self._obj.convert_initialize(context, tmp_ref, args).expr + 
                    context.activates_temporary(tmp_ref) + 
                    tmp_ref.expr,
                tmp_ref.expr_type
                )

        def to_py(x):
            if isinstance(x.expr_type.nonref_type, FreePythonObjectReference):
                return x.expr_type.nonref_type.python_object_representation

            if x.expr is not None and x.expr.matches.Constant:
                if x.expr.val.matches.Int or x.expr.val.matches.Float:
                    return x.expr.val.val
            return x

        call_args = [to_py(x) for x in args]
        try:
            py_call_result = self._obj(*call_args)
        except Exception as e:
            raise ConversionException("Failed to call %s with %s" % (self._obj, call_args))

        return pythonObjectRepresentation(py_call_result)

    def __repr__(self):
        return "FreePythonObject(%s)" % self._obj

def pythonObjectRepresentation(o):
    if isinstance(o,TypedExpression):
        return o

    if isinstance(o, bool):
        return TypedExpression(
            native_ast.Expression.Constant(
                native_ast.Constant.Int(val=1 if o else 0,bits=1,signed=False)
                ), 
            nativepython.type_model.Bool
            )

    if isinstance(o, int):
        return TypedExpression(
            native_ast.Expression.Constant(
                native_ast.Constant.Int(val=o,bits=64,signed=True)
                ), 
            nativepython.type_model.Int64
            )

    if isinstance(o, str):
        return TypedExpression(
            native_ast.Expression.Constant(
                native_ast.Constant.ByteArray(bytes(o,'utf8'))
                ), 
            nativepython.type_model.UInt8.pointer
            )

    if isinstance(o, float):
        return TypedExpression(
            native_ast.Expression.Constant(
                native_ast.Constant.Float(val=o,bits=64)
                ), 
            nativepython.type_model.Float64
            )

    if isinstance(o,CompileTimeType):
        return o.as_typed_expression()

    return FreePythonObjectReference(o).as_typed_expression()

class ExpressionFunction(CompileTimeType):
    def __init__(self, f):
        self.f = f

    def convert_call(self, context, instance, args):
        return self.f(context, args)

    def __repr__(self):
        return self.f.__name__

    @property
    def python_object_representation(self):
        return self

    def __call__(self, *args):
        return self.f(*args)

class TypeFunction(CompileTimeType):
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def convert_call(self, context, instance, args):
        def unwrap(x):
            if not isinstance(x, TypedExpression):
                raise ConversionException("Expected a TypedExpression, not %s" % x)

            t = x.expr_type
            if isinstance(t.nonref_type, CompileTimeType):
                return t.nonref_type.python_object_representation
            
            raise ConversionException(
                "Can't call a TypeFunction with non compile-time argument %s" % t
                )

        def wrap(x):
            return pythonObjectRepresentation(x)

        unwrapped = tuple(unwrap(x) for x in args)

        if unwrapped in self.memo:
            res = self.memo[unwrapped]
        else:
            res = self.memo[unwrapped] = self.f(*unwrapped)
        
        return wrap(res)

    def __call__(self, *args):
        args = tuple(args)

        if args in self.memo:
            return self.memo[args]
        else:
            self.memo[args] = self.f(*args)

        return self.memo[args]

    def __repr__(self):
        return self.f.__name__

    @property
    def python_object_representation(self):
        return self

class ExternalFunction(CompileTimeType):
    def __init__(self, name, output_type, input_types, implicit_type_casting,varargs,intrinsic):
        self.name = name
        self.output_type = output_type
        self.input_types = input_types
        self.implicit_type_casting = implicit_type_casting
        self.varargs = varargs
        self.intrinsic = intrinsic

    def convert_take_address_override(self, instance_ref, context):
        if not self.intrinsic:
            return TypedExpression(
                native_ast.Expression.FunctionPointer(self.get_named_call_target()),
                nativepython.type_model.FunctionPointer(
                    output_type = self.output_type,
                    input_types = self.input_types,
                    varargs = self.varargs,
                    can_throw = False
                    )
                )

    def convert_call(self, context, instance, args):
        if self.varargs:
            assert len(args) >= len(self.input_types)
        else:
            assert len(args) == len(self.input_types)

        args = [a.dereference() for a in args]

        if not self.implicit_type_casting:
            for i in range(len(input_types)):
                assert args[i].expr_type == self.input_types[i]
        else:
            args = list(args)
            for i in range(len(self.input_types)):
                if args[i].expr_type != self.input_types[i]:
                    args[i] = args[i].convert_to_type(self.input_types[i], implicitly=False)
        
        return TypedExpression(
            native_ast.Expression.Call(
                target=native_ast.CallTarget.Named(self.get_named_call_target()),
                args=[a.expr for a in args]
                ),
            self.output_type
            )

    def get_named_call_target(self):
        return native_ast.NamedCallTarget(
            name = self.name,
            arg_types = [i.lower() for i in self.input_types],
            output_type = self.output_type.lower(),
            external=True,
            varargs=self.varargs,
            intrinsic=self.intrinsic,
            can_throw=False
            )

    def __repr__(self):
        return self.name

    @classmethod
    def make(cls, name, output_type, input_types, implicit_type_casting=True,varargs=False,intrinsic=False):
        return ExternalFunction(name, output_type, input_types, implicit_type_casting,varargs,intrinsic)

