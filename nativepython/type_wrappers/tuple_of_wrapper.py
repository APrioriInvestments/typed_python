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

from nativepython.type_wrappers.refcounted_wrapper import RefcountedWrapper
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class TupleOrListOfWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        self.is_tuple = t.__typed_python_category__ == "TupleOf"

        self.underlyingWrapperType = typeWrapper(t.ElementType)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('count', native_ast.Int32),
            ('reserved', native_ast.Int32),
            ('data', native_ast.UInt8Ptr)
        ), name='TupleOfLayout' if self.is_tuple else 'ListOfLayout').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        assert instance.isReference

        return (
            context.converter.defineNativeFunction(
                "destructor_" + str(self.typeRepresentation),
                ('destructor', self),
                [self],
                typeWrapper(NoneType),
                self.generateNativeDestructorFunction
            )
            .call(instance)
        )

    def generateNativeDestructorFunction(self, context, out, inst):
        if not self.underlyingWrapperType.is_pod:
            with context.loop(inst.convert_len()) as i:
                inst.convert_getitem_unsafe(i).convert_destroy()

        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 4).load())
        )
        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.cast(native_ast.UInt8Ptr))
        )

    def convert_bin_op(self, context, left, op, right):
        if right.expr_type == left.expr_type:
            if op.matches.Add:
                return context.push(
                    self,
                    lambda new_tuple:
                        context.converter.defineNativeFunction(
                            'concatenate(' + self.typeRepresentation.__name__ + "," + right.expr_type.typeRepresentation.__name__ + ")",
                            ('util', self, 'concatenate', right.expr_type),
                            [self, self],
                            self,
                            self.generateConcatenateTuple
                        ).call(new_tuple, left, right)
                )

        return super().convert_bin_op(context, left, op, right)

    def generateConcatenateTuple(self, context, out, left, right):
        def elt_ref(tupPtrExpr, iExpr):
            return context.pushReference(
                self.underlyingWrapperType,
                tupPtrExpr.ElementPtrIntegers(0, 4).load().cast(
                    self.underlyingWrapperType.getNativeLayoutType().pointer()
                ).elemPtr(iExpr)
            )

        left_size = left.convert_len()
        right_size = right.convert_len()

        context.pushEffect(
            out.expr.store(
                runtime_functions.malloc.call(native_ast.const_int_expr(28))
                    .cast(self.getNativeLayoutType())
            ) >>
            out.expr.load().ElementPtrIntegers(0, 4).store(
                runtime_functions.malloc.call(
                    left_size.nonref_expr
                    .add(right_size.nonref_expr)
                    .mul(native_ast.const_int_expr(self.underlyingWrapperType.getBytecount()))
                ).cast(native_ast.UInt8Ptr)
            ) >>
            out.expr.load().ElementPtrIntegers(0, 0).store(native_ast.const_int_expr(1)) >>
            out.expr.load().ElementPtrIntegers(0, 1).store(native_ast.const_int32_expr(-1)) >>
            out.expr.load().ElementPtrIntegers(0, 2).store(
                left_size.nonref_expr.add(right_size.nonref_expr).cast(native_ast.Int32)
            ) >>
            out.expr.load().ElementPtrIntegers(0, 3).store(
                left_size.nonref_expr.add(right_size.nonref_expr).cast(native_ast.Int32)
            )
        )

        with context.loop(left_size) as i:
            out.convert_getitem_unsafe(i).convert_copy_initialize(left.convert_getitem_unsafe(i))

        with context.loop(right_size) as i:
            out.convert_getitem_unsafe(i+left_size).convert_copy_initialize(right.convert_getitem_unsafe(i))

    def convert_getitem(self, context, expr, item):
        return context.pushReference(
            self.underlyingWrapperType,
            native_ast.Expression.Branch(
                cond=((item >= 0) & (item < self.convert_len(context, expr))).nonref_expr,
                true=expr.nonref_expr.ElementPtrIntegers(0, 4).load().cast(
                    self.underlyingWrapperType.getNativeLayoutType().pointer()
                ).elemPtr(item.toInt64().nonref_expr),
                false=generateThrowException(context, IndexError(("tuple" if self.is_tuple else "list") + " index out of range"))
            )
        )

    def convert_getitem_unsafe(self, context, expr, item):
        return context.pushReference(
            self.underlyingWrapperType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).load().cast(
                self.underlyingWrapperType.getNativeLayoutType().pointer()
            ).elemPtr(item.toInt64().nonref_expr)
        )

    def convert_len_native(self, expr):
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
        )

    def convert_len(self, context, expr):
        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))


class TupleOfWrapper(TupleOrListOfWrapper):
    def convert_default_initialize(self, context, tgt):
        context.pushEffect(
            tgt.expr.store(tgt.expr_type.getNativeLayoutType().zero())
        )
