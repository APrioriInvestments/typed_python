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

import nativepython

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

    @property
    def is_integer(self):
        return self.t.matches.Int

    @property
    def is_bool(self):
        return self.t.matches.Int and self.t.bits == 1

    @property
    def is_float(self):
        return self.t.matches.Float

    @property
    def is_nonbool_integer(self):
        return self.t.matches.Int and self.t.bits > 1

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
            if self.is_bool:
                return self.convert_to_type(Int64, False).convert_unary_op(instance_ref, op)

            return TypedExpression(
                native_ast.Expression.Unaryop(
                    op=native_ast.UnaryOp.Add() if op.matches.UAdd else native_ast.UnaryOp.Negate(),
                    operand=instance.expr
                    ),
                self
                )

        if op.matches.Invert:
            if self.is_bool:
                return self.convert_to_type(Int64, False).convert_unary_op(instance_ref, op)

            return TypedExpression(
                native_ast.Expression.Unaryop(
                    op=native_ast.UnaryOp.BitwiseNot(),
                    operand=instance.expr
                    ),
                self
                )

        if op.matches.Not:
            return TypedExpression(
                native_ast.Expression.Unaryop(
                    op=native_ast.UnaryOp.LogicalNot(),
                    operand=instance.expr
                    ),
                Bool
                )

        return super(PrimitiveNumericType, self).convert_unary_op(instance_ref, op)

    def convert_bin_op(self, op, l, r):
        l = l.dereference()
        r = r.dereference()

        if op._alternative is python_ast.ComparisonOp:
            if op.matches.Is:
                if l.expr_type != r.expr_type:
                    return TypedExpression(native_ast.falseExpr, Bool)

                if l.expr.matches.Constant and r.expr.matches.Constant:
                    return TypedExpression(
                        native_ast.trueExpr if l.expr.val == r.expr.val else native_ast.falseExpr,
                        Bool
                        )
                
                return TypedExpression(
                    native_ast.Expression.Binop(
                        op=native_ast.BinaryOp.Eq(),
                        l=l.expr,
                        r=r.expr
                        ),
                    Bool
                    )

            if op.matches.IsNot:
                if l.expr_type != r.expr_type:
                    return TypedExpression(native_ast.trueExpr, Bool)

                return TypedExpression(
                    native_ast.Expression.Binop(
                        op=native_ast.BinaryOp.NotEq(),
                        l=l.expr,
                        r=r.expr
                        ),
                    Bool
                    )
        
        if op._alternative is python_ast.BinaryOp and (op.matches.LShift or op.matches.RShift):
            if l.expr_type.is_nonbool_integer and r.expr_type.is_nonbool_integer:
                return TypedExpression(
                    native_ast.Expression.Binop(
                        op=getattr(native_ast.BinaryOp,op._which)(),
                        l=l.expr,
                        r=r.expr
                        ),
                    l.expr_type
                    )
        
        if op._alternative is python_ast.BinaryOp and (
                    op.matches.BitOr or op.matches.BitAnd or op.matches.BitXor):
            if l.expr_type.is_integer and r.expr_type.is_integer:
                target_type = self.bigger_type(r.expr_type)

                if r.expr_type != target_type:
                    r = r.expr_type.convert_to_type(r, target_type, False)

                if l.expr_type != target_type:
                    l = l.expr_type.convert_to_type(l, target_type, False)

                return TypedExpression(
                    native_ast.Expression.Binop(
                        op=getattr(native_ast.BinaryOp,op._which)(),
                        l=l.expr,
                        r=r.expr
                        ),
                    target_type
                    )
        
        if op._alternative is python_ast.BinaryOp and op.matches.Pow and l.expr_type.is_float:
            if r.expr_type.is_float:
                target_type = self.bigger_type(r.expr_type)

                if r.expr_type != target_type:
                    r = r.expr_type.convert_to_type(r, target_type, False)

                if l.expr_type != target_type:
                    l = l.expr_type.convert_to_type(l, target_type, False)

                bitcount = l.expr_type.t.bits

                return TypedExpression(
                    native_ast.Expression.Call(
                        target=native_ast.CallTarget.Named(native_ast.NamedCallTarget(
                            name = "llvm.pow.f%s" % bitcount,
                            arg_types = [l.expr_type.lower(), r.expr_type.lower()],
                            output_type = l.expr_type.lower(),
                            external=True,
                            varargs=False,
                            intrinsic=True,
                            can_throw=False
                            )),
                        args=[l.expr, r.expr]
                        ),
                    l.expr_type
                    )
            if r.expr_type.is_integer:
                bitcount = l.expr_type.t.bits

                if r.expr_type != Int32:
                    r = r.expr_type.convert_to_type(r, Int32, False)

                return TypedExpression(
                    native_ast.Expression.Call(
                        target=native_ast.CallTarget.Named(native_ast.NamedCallTarget(
                            name = "llvm.powi.f%s" % bitcount,
                            arg_types = [l.expr_type.lower(), r.expr_type.lower()],
                            output_type = l.expr_type.lower(),
                            external=True,
                            varargs=False,
                            intrinsic=True,
                            can_throw=False
                            )),
                        args=[l.expr, r.expr]
                        ),
                    l.expr_type
                    )

        target_type = self.bigger_type(r.expr_type)

        if r.expr_type != target_type:
            r = r.expr_type.convert_to_type(r, target_type, False)

        if l.expr_type != target_type:
            l = l.expr_type.convert_to_type(l, target_type, False)

        if op._alternative is python_ast.BinaryOp:
            for py_op, native_op in [('Add','Add'),('Sub','Sub'),
                                     ('Mult','Mul'),('Div','Div'),
                                     ('Mod','Mod')
                                     ]:
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

    def convert_assign(self, context, instance_ref, arg):
        if not self.is_pod:
            raise ConversionException("instances of %s need an explicit assignment operator" % self)
        
        self.assert_is_instance_ref(instance_ref)

        arg = arg.dereference()

        if arg.expr_type != self:
            arg = arg.convert_to_type(self, True)

        if arg.expr_type != self:
            raise ConversionException("can't assign %s to %s because type coercion failed" % 
                    (arg.expr_type, self))

        return TypedExpression.Void(
            native_ast.Expression.Store(
                ptr=instance_ref.expr,
                val=arg.expr
                )
            )

    def convert_to_type(self, e, other_type, implicitly):
        if other_type == self:
            return e

        if other_type.is_primitive:
            return TypedExpression(
                native_ast.Expression.Cast(left=e.expr, to_type=other_type.lower()),
                other_type
                )

        if other_type.is_pointer and self.t.matches.Int and not implicitly:
            return TypedExpression(
                native_ast.Expression.Cast(left=e.expr, to_type=other_type.lower()),
                other_type
                )

        raise ConversionException("can't convert %s to %s" % (self, other_type))

Float64 = PrimitiveNumericType(native_ast.Type.Float(bits=64))
Float32 = PrimitiveNumericType(native_ast.Type.Float(bits=32))
Int64 = PrimitiveNumericType(native_ast.Type.Int(bits=64, signed=True))
Int32 = PrimitiveNumericType(native_ast.Type.Int(bits=32, signed=True))
Int8 = PrimitiveNumericType(native_ast.Type.Int(bits=8, signed=True))
UInt8 = PrimitiveNumericType(native_ast.Type.Int(bits=8, signed=False))
Bool = PrimitiveNumericType(native_ast.Type.Int(bits=1, signed=False))
Void = PrimitiveType(native_ast.Type.Void())

assert Void.sizeof == 0