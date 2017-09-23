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

import nativepython.python_ast as python_ast
import nativepython.native_ast as native_ast

class PrimitiveType(Type):
    def __init__(self, t):
        Type.__init__(self)
        self.t = t

    @property
    def is_primitive(self):
        return True

    @property
    def null_value(self):
        if self.t.matches.Float:
            return native_ast.Constant.Float(bits=self.t.bits,val=0.0)

        if self.t.matches.Int:
            return native_ast.Constant.Int(bits=self.t.bits,signed=self.t.signed,val=0)

        if self.t.matches.Void:
            return native_ast.Constant.Void()

        raise ConversionException(self.t)

    @property
    def is_pod(self):
        return True

    def lower(self):
        return self.t

    def __repr__(self):
        return str(self.t)

class PrimitiveNumericType(PrimitiveType):
    def __init__(self, t):
        PrimitiveType.__init__(self, t)

        assert t.matches.Float or t.matches.Int

    @property
    def is_primitive_numeric(self):
        return True

    def bigger_type(self, other):
        if self.t.matches.Float and other.t.matches.Int:
            return self
        if self.t.matches.Int and other.t.matches.Float:
            return other
        if self.t.matches.Int:
            return PrimitiveNumericType(
                native_ast.Type.Int(
                    bits = max(self.t.bits, other.t.bits),
                    signed = self.t.signed or other.t.signed
                    )
                )
        else:
            return PrimitiveNumericType(
                native_ast.Type.Float(
                    bits = max(self.t.bits, other.t.bits)
                    )
                )

    def convert_unary_op(self, instance_ref, op):
        instance = instance_ref.dereference()

        if op.matches.UAdd or op.matches.USub:
            return TypedExpression(
                native_ast.Expression.Unaryop(
                    op=native_ast.UnaryOp.Add() if op.matches.UAdd else native_ast.UnaryOp.Negate(),
                    operand=instance.expr
                    ),
                self
                )

        return super(PrimitiveNumericType, self).convert_unary_op(instance, op)

    def convert_bin_op(self, op, l, r):
        l = l.dereference()
        r = r.dereference()

        target_type = self.bigger_type(r.expr_type)

        if r.expr_type != target_type:
            r = r.expr_type.convert_to_type(r, target_type)

        if l.expr_type != target_type:
            l = l.expr_type.convert_to_type(l, target_type)

        if op._alternative is python_ast.BinaryOp:
            for py_op, native_op in [('Add','Add'),('Sub','Sub'),('Mult','Mul'),('Div','Div')]:
                if getattr(op.matches, py_op):
                    return TypedExpression(
                        native_ast.Expression.Binop(
                            op=getattr(native_ast.BinaryOp,native_op)(),
                            l=l.expr,
                            r=r.expr
                            ),
                        target_type
                        )
        if op._alternative is python_ast.ComparisonOp:
            for opname in ['Gt','GtE','Lt','LtE','Eq','NotEq']:
                if getattr(op.matches, opname):
                    return TypedExpression(
                        native_ast.Expression.Binop(
                            op=getattr(native_ast.BinaryOp,opname)(),
                            l=l.expr,
                            r=r.expr
                            ),
                        Bool
                        )

        raise ConversionException(
            "can't handle binary op %s between %s and %s" % (op, l.expr_type, r.expr_type)
            )

    def convert_to_type(self, e, other_type):
        if other_type == self:
            return e

        if other_type.is_primitive:
            return TypedExpression(
                native_ast.Expression.Cast(left=e.expr, to_type=other_type.lower()),
                other_type
                )

        if other_type.is_pointer and self.t.matches.Int:
            return TypedExpression(
                native_ast.Expression.Cast(left=e.expr, to_type=other_type.lower()),
                other_type
                )

        raise ConversionException("can't convert %s to %s" % (self, other_type))

Float64 = PrimitiveNumericType(native_ast.Type.Float(bits=64))
Int64 = PrimitiveNumericType(native_ast.Type.Int(bits=64, signed=True))
Int32 = PrimitiveNumericType(native_ast.Type.Int(bits=32, signed=True))
Int8 = PrimitiveNumericType(native_ast.Type.Int(bits=8, signed=True))
UInt8 = PrimitiveNumericType(native_ast.Type.Int(bits=8, signed=False))
Bool = PrimitiveNumericType(native_ast.Type.Int(bits=1, signed=False))
Void = PrimitiveType(native_ast.Type.Void())

assert Void.sizeof == 0