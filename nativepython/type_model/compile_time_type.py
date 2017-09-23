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

from nativepython.type_model.type_base import Type
from nativepython.type_model.typed_expression import TypedExpression
from nativepython.exceptions import ConversionException, UnassignableFieldException


class CompileTimeType(Type):
    @property
    def is_pod(self):
        return True
    
    @property
    def null_value(self):
        return native_ast.Constant.Struct(())

    def lower(self):
        return native_ast.Type.Struct(())

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
        
    def convert_attribute(self, context, instance, attr):
        if not hasattr(self._obj, attr):
            raise ConversionException("Can't get attribute %s from %s of type %s" % 
                    (attr,self._obj, type(self._obj)))

        return pythonObjectRepresentation(getattr(self._obj, attr))

    def convert_call(self, context, instance, args):
        if isinstance(self._obj, types.FunctionType):
            return context.call_py_function(self._obj, args)

        if self._obj is float:
            assert len(args) == 1
            return args[0].convert_to_type(nativepython.type_model.Float64)

        if self._obj is int:
            assert len(args) == 1
            return args[0].convert_to_type(nativepython.type_model.Int64)

        ClassType = nativepython.type_model.ClassType

        if ClassType.object_is_class(self._obj):
            return ClassType.convert_class_call(context, self._obj, args)

        if isinstance(self._obj, Type):
            if self._obj.is_ref:
                raise ConversionException("Can't instantiate %s" % self._obj)

            #we are initializing an element of the type
            for a in args:
                assert a.expr is not None

            tmp_ptr = context.allocate_temporary(self._obj)

            return TypedExpression(
                self._obj.convert_initialize(context, tmp_ptr, args).expr + 
                    context.activates_temporary(tmp_ptr) + 
                    tmp_ptr.expr,
                self._obj.reference
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
                native_ast.Constant.ByteArray(bytes(o))
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
