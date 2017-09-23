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

from nativepython.type_model.type_base import Type
from nativepython.type_model.typed_expression import TypedExpression
from nativepython.exceptions import ConversionException

import nativepython.llvm_compiler as llvm_compiler
import nativepython.native_ast as native_ast
import nativepython.python_ast as python_ast

class Pointer(Type):
    def __init__(self, value_type):
        assert isinstance(value_type, Type)
        self.value_type = value_type

    def lower(self):
        return native_ast.Type.Pointer(self.value_type.lower())

    @property
    def is_pointer(self):
        return True

    @property
    def is_pod(self):
        return True

    @property
    def null_value(self):
        return native_ast.Constant.NullPointer(self.value_type.lower())

    def convert_attribute(self, context, instance, attr):
        instance = instance.dereference()
        ref = instance.reference_from_pointer()
        return ref.convert_attribute(context, attr)

    def convert_set_attribute(self, context, instance, attr, val):
        raise ConversionException("no attribute %s in Pointer" % attr)

    def convert_bin_op(self, op, lref, rref):
        l = lref.dereference()
        r = rref.dereference()

        if op._alternative is python_ast.BinaryOp:
            if op.matches.Add or op.matches.Sub:
                if r.expr_type.is_primitive_numeric and r.expr_type.t.matches.Int:
                    if op.matches.Sub:
                        r = r.convert_unary_op(python_ast.PythonASTUnaryOp.USub())

                    return TypedExpression(
                        native_ast.Expression.ElementPtr(
                            left=l.expr,
                            offsets=(r.expr,)
                            ),
                        self
                        )

        return super(Pointer, self).convert_bin_op(op,lref,rref)


    def convert_getitem(self, context, instance, index):
        instance = instance.dereference()
        index = index.dereference()
        
        if not (index.expr_type.is_primitive_numeric
                            and index.expr_type.t.matches.Int):
            raise ConversionException("can only index with integers, not %s" % index.expr_type)

        res = TypedExpression(
            native_ast.Expression.ElementPtr(
                left=instance.expr,
                offsets=(index.expr,)
                ),
            self.value_type.reference
            )

        return res

    def convert_setitem(self, context, instance, index, value):
        return self.convert_getitem(context, instance, index).convert_assign(context, value)

    def convert_to_type(self, instance, to_type):
        instance = instance.dereference()

        if to_type.is_pointer:
            return TypedExpression(
                native_ast.Expression.Cast(left=instance.expr, to_type=to_type.lower()), 
                to_type
                )

        if to_type.is_primitive and to_type.t.matches.Int:
            return TypedExpression(
                native_ast.Expression.Cast(left=instance.expr, to_type=to_type.lower()), 
                to_type
                )

        raise ConversionException("can't convert %s to type %s" % (self, to_type))

    def __repr__(self):
        return "Pointer(%s)" % self.value_type

