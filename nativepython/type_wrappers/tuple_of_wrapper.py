#   Coyright 2017-2019 Nativepython Authors
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
from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, Int32
from nativepython.type_wrappers.util import min

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


def tuple_compare_eq(left, right):
    """Compare two 'TupleOf' instances by comparing their individual elements."""
    if len(left) != len(right):
        return False

    for i in range(len(left)):
        if left[i] != right[i]:
            return False

    return True


def tuple_compare_lt(left, right):
    """Compare two 'TupleOf' instances by comparing their individual elements."""
    for i in range(min(len(left), len(right))):
        if left[i] > right[i]:
            return False
        if left[i] < right[i]:
            return True

    return len(left) < len(right)


def tuple_compare_lte(left, right):
    """Compare two 'TupleOf' instances by comparing their individual elements."""
    for i in range(min(len(left), len(right))):
        if left[i] > right[i]:
            return False
        if left[i] < right[i]:
            return True

    return len(left) <= len(right)


def tuple_compare_neq(left, right):
    return not tuple_compare_eq(left, right)


def tuple_compare_gt(left, right):
    return not tuple_compare_lte(left, right)


def tuple_compare_gte(left, right):
    return not tuple_compare_lt(left, right)


def tuple_of_hash(instance):
    val = instance._hash_cache
    if val != -1:
        return val
    val = Int32(0)
    for item in instance:
        val = (val * Int32(1000003)) ^ hash(item)
    if val == Int32(-1):
        val = Int32(-2)

    instance._hash_cache = val

    return val


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

        if right.expr_type == left.expr_type:
            if op.matches.Eq:
                return context.call_py_function(tuple_compare_eq, (left, right), {})
            if op.matches.NotEq:
                return context.call_py_function(tuple_compare_neq, (left, right), {})
            if op.matches.Lt:
                return context.call_py_function(tuple_compare_lt, (left, right), {})
            if op.matches.LtE:
                return context.call_py_function(tuple_compare_lte, (left, right), {})
            if op.matches.Gt:
                return context.call_py_function(tuple_compare_gt, (left, right), {})
            if op.matches.GtE:
                return context.call_py_function(tuple_compare_gte, (left, right), {})

        return super().convert_bin_op(context, left, op, right)

    def convert_attribute(self, context, expr, attr):
        if attr == '_hash_cache':
            return context.pushPod(
                Int32,
                expr.nonref_expr.ElementPtrIntegers(0, 1).load()
            )

        return super().convert_attribute(context, expr, attr)

    def convert_set_attribute(self, context, expr, attr, val):
        if attr == '_hash_cache':
            val = val.convert_to_type(Int32)
            if val is None:
                return None

            return context.pushEffect(
                expr.nonref_expr.ElementPtrIntegers(0, 1).store(val.nonref_expr)
            )

        return super().convert_set_attribute(context, expr, attr, val)

    def convert_hash(self, context, expr):
        return context.call_py_function(tuple_of_hash, (expr,), {})

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
        if item is None or expr is None:
            return None

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

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                TupleOrListOfIteratorWrapper(self.typeRepresentation),
                lambda instance:
                    instance.expr.ElementPtrIntegers(0, 0).store(-1)
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 1)
            ).convert_copy_initialize(instance)

            return res

        return super().convert_method_call(context, instance, methodname, args, kwargs)


class TupleOrListOfIteratorWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, tupType):
        self.tupType = tupType
        super().__init__((tupType, "iterator"))

    def getNativeLayoutType(self):
        return native_ast.Type.Struct(
            element_types=(("pos", native_ast.Int64), ("tup", typeWrapper(self.tupType).getNativeLayoutType())),
            name="tuple_or_list_iterator"
        )

    def convert_next(self, context, expr):
        context.pushEffect(
            expr.expr.ElementPtrIntegers(0, 0).store(
                expr.expr.ElementPtrIntegers(0, 0).load().add(1)
            )
        )
        self_len = self.refAs(context, expr, 1).convert_len()
        canContinue = context.pushPod(
            bool,
            expr.expr.ElementPtrIntegers(0, 0).load().lt(self_len.nonref_expr)
        )

        nextIx = context.pushReference(int, expr.expr.ElementPtrIntegers(0, 0))

        return self.iteratedItemForReference(context, expr, nextIx), canContinue

    def refAs(self, context, expr, which):
        assert expr.expr_type == self

        if which == 0:
            return context.pushReference(int, expr.expr.ElementPtrIntegers(0, 0))

        if which == 1:
            return context.pushReference(
                self.tupType,
                expr.expr
                    .ElementPtrIntegers(0, 1)
                    .cast(typeWrapper(self.tupType).getNativeLayoutType().pointer())
            )

    def iteratedItemForReference(self, context, expr, ixExpr):
        return typeWrapper(self.tupType).convert_getitem_unsafe(
            context,
            self.refAs(context, expr, 1),
            ixExpr
        )

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        for i in range(2):
            self.refAs(context, expr, i).convert_assign(self.refAs(context, other, i))

    def convert_copy_initialize(self, context, expr, other):
        for i in range(2):
            self.refAs(context, expr, i).convert_copy_initialize(self.refAs(context, other, i))

    def convert_destroy(self, context, expr):
        self.refAs(context, expr, 1).convert_destroy()


class TupleOfWrapper(TupleOrListOfWrapper):
    def convert_default_initialize(self, context, tgt):
        context.pushEffect(
            tgt.expr.store(tgt.expr_type.getNativeLayoutType().zero())
        )
