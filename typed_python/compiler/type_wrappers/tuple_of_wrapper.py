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
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python.compiler.converter_utils import (
    InitializeRefAsImplicit,
    InitializeRefAsImplicitContainers,
    InitializeRefAsUpcastContainers
)
from typed_python.type_function import TypeFunction
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.typed_expression import TypedExpression

from typed_python import Int32, TupleOf, ListOf, Tuple, NamedTuple, Class, Final, Member, pointerTo, PointerTo

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
        self.isTuple = issubclass(tupleType, TupleOf)

    def __eq__(self, other):
        return isinstance(other, PreReservedTupleOrList) and other.tupleType == self.tupleType

    def __hash__(self):
        return hash("PreReservedTupleOrList", self.tupleType)

    def convert_call(self, context, instance, args, kwargs):
        if len(args) == 1:
            length = args[0].toInt64()
            if length is None:
                return None

            if self.isTuple:
                out = context.allocateUninitializedSlot(self.tupleTypeWrapper)

                with context.ifelse(length.nonref_expr) as (ifTrue, ifFalse):
                    with ifFalse:
                        context.pushEffect(
                            out.expr.store(out.expr_type.getNativeLayoutType().zero())
                        )
                    with ifTrue:
                        context.pushEffect(
                            self.initializeEmptyListExpr(out, length)
                        )

                context.markUninitializedSlotInitialized(out)

                return out
            else:
                return context.push(
                    self.tupleType,
                    lambda out: self.initializeEmptyListExpr(out, length)
                )

        return super().convert_call(context, instance, args, kwargs)

    def initializeEmptyListExpr(self, out, length):
        return (
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


def initialize_tuple_or_list_from_other(targetPtr, src, converter_class):
    ct = len(src)

    target = PreReservedTupleOrList(type(targetPtr).ElementType)(ct)

    ix = 0
    for item in src:
        if not converter_class()(item, target._getItemUnsafe(ix)):
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
            left = left.convert_to_type(self.typeRepresentation.ElementType, ConversionLevel.Implicit)

            if left is None:
                return None

            return context.call_py_function(
                tuple_or_list_contains if op.matches.In else tuple_or_list_contains_not,
                (right, left),
                {}
            )

        return super().convert_bin_op_reverse(context, right, op, left, inplace)

    def convert_type_attribute(self, context, typeInst, attr):
        if attr in ('fromBytes',):
            return typeInst.changeType(BoundMethodWrapper.Make(typeInst.expr_type, attr))

        return super().convert_type_attribute(context, typeInst, attr)

    def convert_type_method_call(self, context, typeInst, methodname, args, kwargs):
        if methodname == "fromBytes" and len(args) == 1:
            arg = args[0].convert_to_type(bytes, ConversionLevel.Signature)
            if arg is None:
                return None

            return context.push(
                self,
                lambda newListPtr:
                newListPtr.expr.store(
                    runtime_functions.list_or_tuple_of_from_bytes.call(
                        arg.nonref_expr.cast(native_ast.VoidPtr),
                        context.getTypePointer(self.typeRepresentation).cast(native_ast.VoidPtr)
                    ).cast(newListPtr.expr_type.layoutType)
                )
            )

        return super().convert_type_method_call(context, typeInst, methodname, args, kwargs)

    def has_intiter(self):
        """Does this type support the 'intiter' format?"""
        return True

    def convert_intiter_size(self, context, instance):
        """If this type supports intiter, compute the size of the iterator.

        This function will return a TypedExpression(int) or None if it set an exception."""
        return self.convert_len(context, instance)

    def convert_intiter_value(self, context, instance, valueInstance):
        """If this type supports intiter, compute the value of the iterator.

        This function will return a TypedExpression, or None if it set an exception."""
        return self.convert_getitem(context, instance, valueInstance)

    def convert_attribute(self, context, expr, attr):
        if attr in ("_getItemUnsafe", "_initializeItemUnsafe", "setSizeUnsafe", 'toBytes', '__iter__', 'pointerUnsafe'):
            return expr.changeType(BoundMethodWrapper.Make(self, attr))

        if attr == '_hash_cache':
            return context.pushPod(
                Int32,
                expr.nonref_expr.ElementPtrIntegers(0, 1).load()
            )

        return super().convert_attribute(context, expr, attr)

    def convert_set_attribute(self, context, expr, attr, val):
        if attr == '_hash_cache':
            val = val.convert_to_type(Int32, ConversionLevel.Implicit)
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

        item = item.toIndex()
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
            ).elemPtr(actualItem.nonref_expr)
        )

    def convert_getitem_unsafe(self, context, expr, item):
        return context.pushReference(
            self.underlyingWrapperType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).load().cast(
                self.underlyingWrapperType.getNativeLayoutType().pointer()
            ).elemPtr(item.toIndex().nonref_expr)
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
            index = args[0].toIndex()
            if index is None:
                return None

            return self.convert_getitem_unsafe(context, instance, index)

        if methodname == "toBytes":
            if len(args) == 0:
                return context.push(
                    bytes,
                    lambda bytesRef:
                        bytesRef.expr.store(
                            runtime_functions.list_or_tuple_of_to_bytes.call(
                                instance.nonref_expr.cast(native_ast.VoidPtr),
                                context.getTypePointer(self.typeRepresentation).cast(native_ast.VoidPtr)
                            ).cast(bytesRef.expr_type.layoutType)
                        )
                )

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
            index = args[0].toIndex()
            if index is None:
                return None

            value = args[1].convert_to_type(self.typeRepresentation.ElementType, ConversionLevel.Implicit)
            if value is None:
                return None

            self.convert_getitem_unsafe(context, instance, index).convert_copy_initialize(value)

            return context.pushVoid()

        if methodname == "__iter__" and not args and not kwargs:
            return typeWrapper(TupleOrListOfIterator(self.typeRepresentation)).convert_type_call(
                context,
                None,
                [],
                dict(pos=context.constant(-1), tup=instance)
            )

        if methodname == "pointerUnsafe":
            if len(args) == 1:
                count = args[0].toInt64()
                if count is None:
                    return

                return context.pushPod(
                    PointerTo(self.typeRepresentation.ElementType),
                    instance.nonref_expr.ElementPtrIntegers(0, 4).load().cast(
                        self.underlyingWrapperType.getNativeLayoutType().pointer()
                    ).elemPtr(count.nonref_expr)
                )

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def _can_convert_from_type(self, otherType, conversionLevel):
        if not conversionLevel.isImplicitContainersOrHigher() and issubclass(self.typeRepresentation, ListOf):
            # ListOf only allows 'implicit container' conversions.
            return False

        if conversionLevel < ConversionLevel.UpcastContainers:
            return False

        # note that if the other object is an untyped container, then the
        # pathway in python_object_of_type will take care of it.
        childLevel = (
            ConversionLevel.ImplicitContainers if conversionLevel.isImplicitContainersOrHigher()
            else
            conversionLevel
        )

        if issubclass(otherType.typeRepresentation, (ListOf, TupleOf)):
            # check if we can _definitely_ convert
            if typeWrapper(otherType.typeRepresentation.ElementType).can_convert_to_type(
                self.typeRepresentation.ElementType,
                childLevel
            ) is True:
                return True

            return "Maybe"

        if issubclass(otherType.typeRepresentation, (Tuple, NamedTuple)):
            allConvertible = True

            for t in otherType.typeRepresentation.ElementTypes:
                canDoIt = typeWrapper(t).can_convert_to_type(self.typeRepresentation.ElementType, childLevel)
                if canDoIt is False:
                    return False

                if canDoIt is not True:
                    allConvertible = False

            if allConvertible:
                return True

            return "Maybe"

        if otherType.isIterable() is False:
            return False

        return super()._can_convert_from_type(otherType, conversionLevel)

    def convert_to_self_with_target(self, context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure=False):
        canConvert = self._can_convert_from_type(sourceVal.expr_type, conversionLevel)

        if canConvert is False:
            return super().convert_to_self_with_target(context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure)

        if conversionLevel >= ConversionLevel.ImplicitContainers:
            converter_class = InitializeRefAsImplicitContainers
        elif conversionLevel == ConversionLevel.Implicit:
            converter_class = InitializeRefAsImplicit
        elif conversionLevel == ConversionLevel.UpcastContainers:
            converter_class = InitializeRefAsUpcastContainers

        res = context.call_py_function(
            initialize_tuple_or_list_from_other,
            (targetVal.asPointer(), sourceVal, context.constant(converter_class)),
            {}
        )

        if canConvert is True:
            return context.constant(True)

        return res

    def convert_type_call_on_container_expression(self, context, typeInst, argExpr):
        if argExpr.matches.Tuple or argExpr.matches.List:
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

                val = val.convert_to_type(self.typeRepresentation.ElementType, ConversionLevel.Implicit)
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

        if argExpr.matches.ListComp or argExpr.matches.GeneratorExp:
            # simply build this as a listcomp and remove the masquerade
            res = context.convert_generator_as_list_comprehension(argExpr)
            if res is None:
                return res
            return res.changeType(self)

        return super().convert_type_call_on_container_expression(context, typeInst, argExpr)

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        return super().convert_type_call(context, typeInst, args, kwargs)

    def _can_convert_to_type(self, targetType, conversionLevel):
        if not conversionLevel.isNewOrHigher():
            return False

        if targetType.typeRepresentation is bool:
            return True

        if targetType.typeRepresentation is str:
            return "Maybe"

        return False

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        if targetVal.expr_type.typeRepresentation is bool:
            res = context.pushPod(bool, self.convert_len_native(instance.nonref_expr).neq(0))
            context.pushEffect(
                targetVal.expr.store(res.nonref_expr)
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)


@TypeFunction
def TupleOrListOfIterator(T):
    class TupleOrListOfIterator(Class, Final):
        pos = Member(int)
        tup = Member(T)

        def __fastnext__(self):
            posPtr = pointerTo(self).pos
            tupPtr = pointerTo(self).tup

            posPtr.set(posPtr.get() + 1)

            if posPtr.get() < len(tupPtr.get()):
                return tupPtr.get().pointerUnsafe(posPtr.get())
            else:
                return PointerTo(T.ElementType)()
    return TupleOrListOfIterator


class TupleOfWrapper(TupleOrListOfWrapper):
    def convert_default_initialize(self, context, tgt):
        context.pushEffect(
            tgt.expr.store(tgt.expr_type.getNativeLayoutType().zero())
        )

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 1 and args[0].expr_type == self and not kwargs:
            return args[0]

        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self, ConversionLevel.New)

        return super().convert_type_call(context, typeInst, args, kwargs)
