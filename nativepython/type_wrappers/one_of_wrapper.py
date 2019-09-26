#   Copyright 2017-2019 Nativepython Authors
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

from typed_python import _types, OneOf, PointerTo
from nativepython.typed_expression import TypedExpression

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class OneOfWrapper(Wrapper):
    is_empty = False
    is_pass_by_ref = True
    can_unwrap = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        assert len(t.Types) > 1

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

    def unwrap(self, context, expr, generator):
        types = []
        exprs = []
        typesSeen = set()

        with context.switch(expr.expr.ElementPtrIntegers(0, 0).load(),
                            range(len(self.typeRepresentation.Types)),
                            False) as indicesAndContexts:
            for i, subcontext in indicesAndContexts:
                with subcontext:
                    exprs.append(generator(self.refAs(context, expr, i)))

                if exprs[-1] is not None:
                    t = exprs[-1].expr_type

                    if t not in typesSeen:
                        typesSeen.add(t)
                        types.append(t)

            if len(types) == 0:
                # all paths throw exceptions. we're done
                return None

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

    def convert_attribute(self, context, instance, attribute):
        # just unwrap us
        return self.unwrap(context, instance, lambda realInstance: realInstance.convert_attribute(attribute))

    def convert_call(self, context, left, args, kwargs):
        # just unwrap us
        return self.unwrap(context, left, lambda realInstance: realInstance.convert_call(args, kwargs))

    def convert_bin_op(self, context, left, op, right):
        def generator(leftUnwrapped):
            return leftUnwrapped.convert_bin_op(op, right)

        return self.unwrap(context, left, generator)

    def convert_bin_op_reverse(self, context, r, op, l):
        assert r.expr_type == self
        assert r.isReference

        def generator(rightUnwrapped):
            return l.convert_bin_op(op, rightUnwrapped)

        return self.unwrap(context, r, generator)

    def convert_default_initialize(self, context, target):
        for i, t in enumerate(self.typeRepresentation.Types):
            if _types.is_default_constructible(t):
                self.refAs(context, target, i).convert_default_initialize()
                context.pushEffect(target.expr.ElementPtrIntegers(0, 0).store(native_ast.const_uint8_expr(i)))
                return

        context.pushException(TypeError, "Can't default-initialize any subtypes of %s" % self.typeRepresentation.__qualname__)

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        if self.is_pod:
            context.pushEffect(
                expr.expr.store(other.nonref_expr)
            )
        else:
            temp = context.pushMove(expr)
            expr.convert_copy_initialize(other)
            temp.convert_destroy()

    def refAs(self, context, expr, which):
        assert expr.expr_type == self
        assert expr.isReference, expr.expr

        tw = typeWrapper(self.typeRepresentation.Types[which])

        return context.pushReference(
            tw,
            expr.expr.ElementPtrIntegers(0, 1).cast(tw.getNativeLayoutType().pointer())
        )

    def convert_copy_initialize(self, context, expr, other):
        assert expr.isReference
        assert other.expr_type == self

        if self.is_pod:
            context.pushEffect(
                expr.expr.store(other.nonref_expr)
            )
        else:
            with context.switch(other.expr.ElementPtrIntegers(0, 0).load(),
                                range(len(self.typeRepresentation.Types)),
                                False) as indicesAndContexts:
                for ix, subcontext in indicesAndContexts:
                    with subcontext:
                        self.refAs(context, expr, ix).convert_copy_initialize(self.refAs(context, other, ix))
                        context.pushEffect(
                            expr.expr.ElementPtrIntegers(0, 0).store(native_ast.const_uint8_expr(ix))
                        )

    def convert_destroy(self, context, expr):
        if not self.is_pod:
            with context.switch(expr.expr.ElementPtrIntegers(0, 0).load(),
                                range(len(self.typeRepresentation.Types)),
                                False) as indicesAndContexts:
                for ix, subcontext in indicesAndContexts:
                    with subcontext:
                        self.refAs(context, expr, ix).convert_destroy()

    def convert_to_type_with_target(self, context, expr, targetVal, explicit):
        assert expr.isReference

        isInitialized = context.push(bool, lambda tgt: tgt.expr.store(native_ast.const_bool_expr(False)))

        with context.switch(expr.expr.ElementPtrIntegers(0, 0).load(),
                            range(len(self.typeRepresentation.Types)),
                            False) as indicesAndContexts:
            for ix, subcontext in indicesAndContexts:
                with subcontext:
                    concreteChild = self.refAs(context, expr, ix)

                    converted = concreteChild.expr_type.convert_to_type_with_target(context, concreteChild, targetVal, explicit)
                    context.pushEffect(
                        isInitialized.expr.store(converted.nonref_expr)
                    )

        return isInitialized

    def convert_to_self_with_target(self, context, targetVal, otherExpr, explicit):
        assert targetVal.isReference

        native = context.converter.defineNativeFunction(
            f'convert({otherExpr.expr_type} to {targetVal.expr_type}, explicit={explicit})',
            ('convert', otherExpr.expr_type, targetVal.expr_type, explicit),
            [PointerTo(self.typeRepresentation), otherExpr.expr_type],
            bool,
            lambda *args: self.generateConvertToSelf(*args, explicit=explicit)
        )

        didConvert = context.pushPod(
            bool,
            native.call(
                targetVal.changeType(PointerTo(self.typeRepresentation), False),
                otherExpr
            )
        )

        return didConvert

    def generateConvertToSelf(self, context, _, convertIntoPtr, convertFrom, explicit):
        """Store a conversion of 'convertFrom' into the pointed-to-value at convertIntoPointer."""
        assert not isinstance(convertFrom.expr_type, OneOfWrapper), "This should already have been expanded away"

        # 'convertIntoPtr' is a PointerTo(self.typeRepresentation), and the nonref_expr of that has exactly
        # the same layout as the 'expr' for a reference to 'self'.
        targetVal = TypedExpression(context, convertIntoPtr.nonref_expr, self, True)

        explicitnessPasses = [False, True] if explicit else [False]

        for explicitThisTime in explicitnessPasses:
            # first, try converting without explicit turned on. If that works, that's our preferred
            # conversion.
            for ix, type in enumerate(self.typeRepresentation.Types):
                # get a pointer to the uninitialized target as if it were the 'ix'th type
                typedTarget = self.refAs(context, targetVal, ix)

                if typedTarget.expr_type == convertFrom.expr_type:
                    typedTarget.convert_copy_initialize(convertFrom)

                    context.pushEffect(
                        targetVal.expr.ElementPtrIntegers(0, 0).store(native_ast.const_uint8_expr(ix))
                    )
                    context.pushEffect(
                        native_ast.Expression.Return(arg=native_ast.const_bool_expr(True))
                    )
                    return

                converted = convertFrom.expr_type.convert_to_type_with_target(
                    context,
                    convertFrom,
                    typedTarget,
                    explicitThisTime
                )

                if converted.expr.matches.Constant and converted.expr.val.matches.Int and converted.expr.val.val:
                    # we _definitely_ match
                    context.pushEffect(
                        targetVal.expr.ElementPtrIntegers(0, 0).store(native_ast.const_uint8_expr(ix))
                    )
                    context.pushEffect(
                        native_ast.Expression.Return(arg=native_ast.const_bool_expr(True))
                    )
                    return

                if converted.expr.matches.Constant and converted.expr.val.matches.Int and not converted.expr.val.val:
                    # we definitely didn't match
                    pass
                else:
                    # maybe we matched.
                    with context.ifelse(converted.nonref_expr) as (ifTrue, ifFalse):
                        with ifTrue:
                            context.pushEffect(
                                native_ast.Expression.Return(arg=native_ast.const_bool_expr(True))
                            )

        # at the end, we didn't convert
        context.pushEffect(
            native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
        )
