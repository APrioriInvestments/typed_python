#   Copyright 2017-2020 typed_python Authors
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

from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.type_wrappers.const_dict_wrapper import ConstDictWrapper
from typed_python.compiler.type_wrappers.dict_wrapper import DictWrapper
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.tuple_wrapper import TupleWrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.typed_expression import TypedExpression
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin

from typed_python import Int32, TupleOf, ListOf
from typed_python.compiler.type_wrappers.util import min

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def tuple_or_list_contains(tup, elt):
    for x in tup:
        if x == elt:
            return True
    return False


def tuple_or_list_contains_not(tup, elt):
    for x in tup:
        if x == elt:
            return False
    return True


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


class PreReservedTupleOrList(CompilableBuiltin):
    def __init__(self, tupleType):
        super().__init__()

        assert issubclass(tupleType, (ListOf, TupleOf))
        self.tupleType = tupleType
        self.tupleTypeWrapper = typeWrapper(tupleType)
        self.underlyingWrapperType = typeWrapper(tupleType.ElementType)

    def __eq__(self, other):
        return isinstance(other, PreReservedTupleOrList) and other.tupleType == self.tupleType

    def __hash__(self):
        return hash("PreReservedTupleOrList", self.tupleType)

    def convert_call(self, context, instance, args, kwargs):
        if len(args) == 1:
            length = args[0].toInt64()
            if length is None:
                return None

            return context.push(
                self.tupleType,
                lambda out:
                    out.expr.store(
                        runtime_functions.malloc.call(native_ast.const_int_expr(28))
                            .cast(self.tupleTypeWrapper.getNativeLayoutType())
                    ) >>
                    out.expr.load().ElementPtrIntegers(0, 4).store(
                        runtime_functions.malloc.call(
                            length.nonref_expr
                            .mul(native_ast.const_int_expr(self.underlyingWrapperType.getBytecount()))
                        ).cast(native_ast.UInt8Ptr)
                    ) >>
                    out.expr.load().ElementPtrIntegers(0, 0).store(native_ast.const_int_expr(1)) >>
                    out.expr.load().ElementPtrIntegers(0, 1).store(native_ast.const_int32_expr(-1)) >>
                    out.expr.load().ElementPtrIntegers(0, 2).store(native_ast.const_int32_expr(0)) >>
                    out.expr.load().ElementPtrIntegers(0, 3).store(length.nonref_expr.cast(native_ast.Int32))
            )

        return super().convert_call(context, instance, args, kwargs)


class InitializeRef(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, InitializeRef)

    def __hash__(self):
        return hash("InitializeRef")

    def convert_call(self, context, instance, args, kwargs):
        """InitializeRef()(target, sourceVal) -> bool.

        Initializes 'target' with the contents of 'sourceVal', returning True on success
        and False on failure.

        'target' must be a reference expression to an uninitialized value.
        """
        if len(args) == 2:
            return args[0].expr_type.convert_to_type_with_target(
                context,
                args[0],
                args[1],
                True
            )

        return super().convert_call(context, instance, args, kwargs)


def initialize_tuple_or_list_from_other(targetPtr, src):
    ct = len(src)

    target = PreReservedTupleOrList(type(targetPtr).ElementType)(ct)

    ix = 0
    for item in src:
        if not InitializeRef()(item, target._getItemUnsafe(ix)):
            return False
        ix += 1
        target.setSizeUnsafe(ix)

    targetPtr.initialize(target)

    return True


def concatenate_tuple_or_list(l, r):
    result = PreReservedTupleOrList(type(l))(len(l) + len(r))

    ix = 0

    for item in l:
        result._initializeItemUnsafe(ix, item)
        ix += 1
        result.setSizeUnsafe(ix)

    for item in r:
        result._initializeItemUnsafe(ix, item)
        ix += 1
        result.setSizeUnsafe(ix)

    return result


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
                typeWrapper(type(None)),
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

    def convert_bin_op(self, context, left, op, right, inplace):
        if issubclass(right.expr_type.typeRepresentation, (TupleOf, ListOf)):
            if op.matches.Add:
                return context.call_py_function(concatenate_tuple_or_list, (left, right), {})

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

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_bin_op_reverse(self, context, right, op, left, inplace):
        if op.matches.In or op.matches.NotIn:
            left = left.convert_to_type(self.typeRepresentation.ElementType, False)
            if left is None:
                return None
            return context.call_py_function(
                tuple_or_list_contains if op.matches.In else tuple_or_list_contains_not,
                (right, left),
                {}
            )

        return super().convert_bin_op_reverse(context, right, op, left, inplace)

    def convert_attribute(self, context, expr, attr):
        if attr in ("_getItemUnsafe", "_initializeItemUnsafe", "setSizeUnsafe"):
            return expr.changeType(BoundMethodWrapper.Make(self, attr))

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

    def convert_getitem(self, context, expr, item):
        if item is None or expr is None:
            return None

        item = item.toInt64()
        if item is None:
            return None

        actualItem = context.pushPod(
            int,
            native_ast.Expression.Branch(
                cond=item.nonref_expr.lt(0),
                true=item.nonref_expr.add(self.convert_len_native(expr.nonref_expr)),
                false=item.nonref_expr
            )
        )

        with context.ifelse(((actualItem >= 0) & (actualItem < self.convert_len(context, expr))).nonref_expr) as (ifTrue, ifFalse):
            with ifFalse:
                context.pushException(IndexError, ("tuple" if self.is_tuple else "list") + " index out of range")

        return context.pushReference(
            self.underlyingWrapperType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).load().cast(
                self.underlyingWrapperType.getNativeLayoutType().pointer()
            ).elemPtr(actualItem.toInt64().nonref_expr)
        ).heldToRef()

    def convert_getitem_unsafe(self, context, expr, item):
        return context.pushReference(
            self.underlyingWrapperType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).load().cast(
                self.underlyingWrapperType.getNativeLayoutType().pointer()
            ).elemPtr(item.toInt64().nonref_expr)
        )

    def convert_len_native(self, expr):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
        )

    def convert_len(self, context, expr):
        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname == "_getItemUnsafe" and len(args) == 1:
            index = args[0].convert_to_type(int)
            if index is None:
                return None

            return self.convert_getitem_unsafe(context, instance, index)

        if methodname == "setSizeUnsafe":
            if len(args) == 1:
                count = args[0].toInt64()
                if count is None:
                    return

                context.pushEffect(
                    instance.nonref_expr
                    .ElementPtrIntegers(0, 2)
                    .store(count.nonref_expr.cast(native_ast.Int32))
                )

                return context.pushVoid()

        if methodname == "_initializeItemUnsafe" and len(args) == 2:
            index = args[0].convert_to_type(int)
            if index is None:
                return None

            value = args[1].convert_to_type(self.typeRepresentation.ElementType)
            if value is None:
                return None

            self.convert_getitem_unsafe(context, instance, index).convert_copy_initialize(value)

            return context.pushVoid()

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

    def _can_convert_from_type(self, otherType, explicit):
        convertible = (
            TupleOrListOfWrapper,
            typed_python.compiler.type_wrappers.set_wrapper.SetWrapper,
            DictWrapper,
            ConstDictWrapper,
            # TupleWrapper  # doesn't have .ElementType
        )
        if explicit and isinstance(otherType, convertible):
            sourceEltType = typeWrapper(otherType.typeRepresentation.ElementType)
            destEltType = typeWrapper(self.typeRepresentation.ElementType)

            return sourceEltType.can_convert_to_type(destEltType, True)

        return super()._can_convert_from_type(otherType, explicit)

    def convert_to_self_with_target(self, context, targetVal, sourceVal, explicit):
        convertible = (
            TupleOrListOfWrapper,
            typed_python.compiler.type_wrappers.set_wrapper.SetWrapper,
            DictWrapper,
            ConstDictWrapper,
            TupleWrapper
        )
        if explicit and isinstance(sourceVal.expr_type, convertible):
            canConvert = self._can_convert_from_type(sourceVal.expr_type, True)

            if canConvert is False:
                return context.constant(False)

            res = context.call_py_function(
                initialize_tuple_or_list_from_other,
                (targetVal.asPointer(), sourceVal),
                {}
            )

            if canConvert is True:
                return context.constant(True)

            return res

        return super().convert_to_self_with_target(context, targetVal, sourceVal, explicit)

    def convert_type_call_on_container_expression(self, context, typeInst, argExpr):
        if not (argExpr.matches.Tuple or argExpr.matches.List or argExpr.matches.Set):
            return super().convert_type_call_on_container_expression(context, typeInst, argExpr)

        # we're calling TupleOf(T) or ListOf(T) with an expression like [1, 2, 3, ...]

        # first allocate something of the right size
        aTup = PreReservedTupleOrList(self.typeRepresentation).convert_call(
            context,
            None,
            (context.constant(len(argExpr.elts)),),
            {}
        )

        # for each expression, push it onto the front and then update the
        # size. We have to do it this way so that if we throw an exception
        # in the middle of constructing the tuple, we teardown the intermediate
        # tuple the right way.
        for i in range(len(argExpr.elts)):
            val = context.convert_expression_ast(argExpr.elts[i])
            if val is None:
                return None

            aTup.convert_method_call(
                "_initializeItemUnsafe",
                (context.constant(i), val),
                {}
            )

            aTup.convert_method_call(
                "setSizeUnsafe",
                (context.constant(i + 1),),
                {}
            )

        return aTup

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        return super().convert_type_call(context, typeInst, args, kwargs)

    def convert_bool_cast(self, context, expr):
        return context.pushPod(bool, self.convert_len_native(expr.nonref_expr).neq(0))


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
        ).heldToRef()

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

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 1 and args[0].expr_type == self and not kwargs:
            return args[0]

        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self, True)

        return super().convert_type_call(context, typeInst, args, kwargs)
