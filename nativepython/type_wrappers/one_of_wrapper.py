#   Copyright 2018 Braxton Mckee
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

from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.typed_expression import TypedExpression
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, Int64, _types, OneOf

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)

class OneOfWrapper(Wrapper):
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        excessBytes = _types.bytecount(t)-1

        self.layoutType = native_ast.Type.Struct(
            element_types=(
                ('which', native_ast.UInt8),
                ('data', native_ast.Type.Array(element_type=native_ast.UInt8, count=excessBytes))
                ),
            name='OneOfLayout'
            )

        self._is_pod = all(typeWrapper(possibility).is_pod for possibility in t.Types)

    @property
    def is_pod(self):
        return self._is_pod

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_bin_op(self, context, left, op, right, isReversed=False):
        types = []
        exprs = []
        typesSeen = set()

        with context.switch(left.expr.ElementPtrIntegers(0,0).load(), range(len(self.typeRepresentation.Types)), False) as indicesAndContexts:
            for i, subcontext in indicesAndContexts:
                with subcontext:
                    if isReversed:
                        exprs.append(right.convert_bin_op(op, self.refAs(context, left, i)))
                    else:
                        exprs.append(self.refAs(context, left, i).convert_bin_op(op, right))

                if exprs[-1] is not None:
                    t = exprs[-1].expr_type
                    if t not in typesSeen:
                        typesSeen.add(t)
                        types.append(t)

            if len(types) == 0:
                #all paths throw exceptions. we're done
                return native_ast.nullExpr

            if len(types) == 1:
                output_type = types[0]
            else:
                output_type = typeWrapper(OneOf(*[t.typeRepresentation for t in types]))

            out_slot = context.allocateUninitializedSlot(output_type)

            for i, subcontext in indicesAndContexts:
                with subcontext:
                    if exprs[i] is not None:
                        converted_res = exprs[i].convert_to_type(output_type)
                        if converted_res is not None:
                            context.pushEffect(
                                out_slot.convert_copy_initialize(converted_res)
                                )

                            context.markUninitializedSlotInitialized(out_slot)

        return out_slot

    def convert_bin_op_reverse(self, context, r, op, l):
        assert r.expr_type == self
        assert r.isReference
        return self.convert_bin_op(context, r, op, l, True)

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        if self.is_pod:
            return expr.expr.store(other.nonref_expr)
        else:
            temp = context.pushMove(expr)
            
            return (
                expr.convert_copy_initialize(other) >> 
                temp.convert_destroy()
                )

    def refAs(self, context, expr, which):
        assert expr.expr_type == self
        assert expr.isReference, expr.expr

        tw = typeWrapper(self.typeRepresentation.Types[which])

        return context.pushReference(
            tw,
            expr.expr.ElementPtrIntegers(0,1).cast(tw.getNativeLayoutType().pointer())
            )

    def convert_copy_initialize(self, context, expr, other):
        assert expr.isReference

        if self.is_pod:
            return expr.expr.store(other.nonref_expr)
        else:
            with context.switch(other.expr.ElementPtrIntegers(0,0).load(), range(len(self.typeRepresentation.Types)), False) as indicesAndContexts:
                for ix, subcontext in indicesAndContexts:
                    with subcontext:
                        context.pushEffect(
                            self.refAs(context, expr, ix).convert_copy_initialize(self.refAs(context, other, ix)) >>
                            expr.expr.ElementPtrIntegers(0,0).store(native_ast.const_uint8_expr(ix))
                            )

            return native_ast.nullExpr

    def convert_destroy(self, context, expr):
        if self.is_pod:
            return native_ast.nullExpr
        else:
            with context.switch(expr.expr.ElementPtrIntegers(0,0).load(), range(len(self.typeRepresentation.Types)), False) as indicesAndContexts:
                for ix, subcontext in indicesAndContexts:
                    with subcontext:
                        context.pushEffect(
                            self.refAs(context, expr, ix).convert_destroy()
                            )

            return native_ast.nullExpr

    def convert_to_type(self, context, expr, otherType):
        if otherType == self:
            return expr

        if otherType.typeRepresentation in self.typeRepresentation.Types:
            #this is wrong - we need to be unpacking each of the alternatives
            #and attempting to convert them. Which should probably be a function...
            assert expr.isReference

            which = tuple(self.typeRepresentation.Types).index(otherType.typeRepresentation)

            return context.push(
                otherType,
                lambda result:
                    native_ast.Expression.Branch(
                        cond=expr.expr.ElementPtrIntegers(0,0).load().eq(native_ast.const_uint8_expr(which)),
                        true=result.convert_copy_initialize(self.refAs(context, expr, which)),
                        false=generateThrowException(context, Exception("Can't convert"))
                        )
                )

        return super().convert_to_type(context, expr, otherType)

    def convert_to_self_native_expr(self, context, result, otherExpr, which):
        assert result.isReference

        return (
            result.expr.ElementPtrIntegers(0,0).store(native_ast.const_uint8_expr(which)) 
                    >> self.refAs(context, result, which).convert_copy_initialize(otherExpr)
            )

    def convert_to_self(self, context, otherExpr):
        if otherExpr.expr_type == self:
            return otherExpr

        return context.push(self, lambda result:
            self.convert_to_self_with_target(context, result, otherExpr)
            )

    def convert_to_self_with_target(self, context, result, otherExpr):
        if otherExpr.expr_type == self:
            return result.convert_copy_initialize(otherExpr)

        if isinstance(otherExpr.expr_type, OneOfWrapper):
            with context.switch(
                    otherExpr.expr.ElementPtrIntegers(0,0).load(), 
                    range(len(otherExpr.expr_type.typeRepresentation.Types)),
                    False
                    ) as indicesAndContexts:
                for i, subcontext in indicesAndContexts:
                    with subcontext:
                        context.pushEffect(
                            self.convert_to_self_with_target(
                                context,
                                result,
                                otherExpr.expr_type.refAs(context, otherExpr, i)
                                )
                            )
            return native_ast.nullExpr

        if otherExpr.expr_type.typeRepresentation in self.typeRepresentation.Types:
            which = tuple(self.typeRepresentation.Types).index(otherExpr.expr_type.typeRepresentation)

            return self.convert_to_self_native_expr(context, result, otherExpr, which)

        return super().convert_to_self(context, otherExpr)
