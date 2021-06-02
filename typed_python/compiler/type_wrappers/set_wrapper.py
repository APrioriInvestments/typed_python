#   Copyright 2020 Braxton Mckee
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

from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.typed_expression import TypedExpression
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.hash_table_implementation import table_next_slot, table_clear, \
    table_contains, set_add, set_add_or_remove, set_remove, set_discard, set_pop
from typed_python import PointerTo, Int32, UInt8, ListOf, TupleOf, Set, Tuple, NamedTuple, Dict, ConstDict

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler
from typed_python.compiler.converter_utils import (
    ConvertUpcastContainers,
    ConvertImplicitContainers
)

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def checkSetSizeAndThrowIfChanged(instance, expectedLen):
    if len(instance) != expectedLen:
        raise RuntimeError("set size changed during iteration")


def initialize_set_from_other_upcast_containers(ptrToOutSet, iterable, mayThrowOnFailure):
    """Initialize a set from an arbitrary iterable object

    This is called internally by the compiler to support initializing
    a Set(T) from any other iterable object.

    Applies 'UpcastContainers' casting to each element in the set.

    Args:
        ptrToOutSet - a pointer to an uninitialized Set instance.
        mappable - something we can iterate over
        mayThrowOnFailure - if True, we throw if we propagate exceptions
            if we can't convert an interior element
    Returns:
        True if we succeeded, and ptrToOutSet is valid. False if
        we failed, and ptrToOutSet will not point to an initialized
        set.
    """
    ptrToOutSet.initialize()
    T = ptrToOutSet.ElementType.ElementType

    try:
        for m in iterable:
            ptrToOutSet.get().add(ConvertUpcastContainers(T)(m))

        return True
    except: # noqa
        ptrToOutSet.destroy()

        if mayThrowOnFailure:
            raise

        return False


def initialize_set_from_other_implicit_containers(ptrToOutSet, iterable, mayThrowOnFailure):
    """Initialize a set from an arbitrary iterable object

    This is called internally by the compiler to support initializing
    a Set(T) from any other iterable object.

    Applies 'ImplicitContainers' casting to each element in the set.

    Args:
        ptrToOutSet - a pointer to an uninitialized Set instance.
        mappable - something we can iterate over
        mayThrowOnFailure - if True, we throw if we propagate exceptions
            if we can't convert an interior element
    Returns:
        True if we succeeded, and ptrToOutSet is valid. False if
        we failed, and ptrToOutSet will not point to an initialized
        set.
    """
    ptrToOutSet.initialize()
    T = ptrToOutSet.ElementType.ElementType

    try:
        for m in iterable:
            ptrToOutSet.get().add(ConvertImplicitContainers(T)(m))

        return True
    except: # noqa
        ptrToOutSet.destroy()

        if mayThrowOnFailure:
            raise

        return False


def set_union(left, right):
    if len(left) >= len(right):
        result = left.copy()
        for i in right:
            result.add(i)
    else:
        result = right.copy()
        for i in left:
            result.add(i)
    return result


def set_intersection(left, right):
    if len(left) > len(right):
        result = right.copy()
        for i in right:
            if i not in left:
                result.discard(i)
    else:
        result = left.copy()
        for i in left:
            if i not in right:
                result.discard(i)
    return result


def set_difference(left, right):
    if len(left) / 4 > len(right):
        result = left.copy()
        for i in right:
            result.discard(i)
    else:
        result = type(left)()
        for i in left:
            if i not in right:
                result.add(i)
    return result


def set_symmetric_difference(left, right):
    if len(left) / 2 > len(right):
        result = left.copy()
        for i in right:
            set_add_or_remove(result, i)
    elif len(right) / 2 > len(left):
        result = right.copy()
        for i in left:
            set_add_or_remove(result, i)
    else:
        result = type(left)()
        for i in left:
            if i not in right:
                result.add(i)
        for i in right:
            if i not in left:
                result.add(i)
    return result


def set_symmetric_difference_update(left, right):
    to_remove = type(left)()
    to_add = type(left)()
    for i in right:
        if i in left:
            to_remove.add(i)
        else:
            to_add.add(i)
            left.add(i)
    left.difference_update(to_remove)
    left.update(to_add)


def set_union_multiple(left, *others):
    result = left
    for o in others:
        result |= o
    return result


def set_update(left, *others):
    for o in others:
        for i in o:
            left.add(i)


def set_intersection_multiple(left, *others):
    result = left
    for o in others:
        result &= o
    return result


# for *others that have __contains__:
# def set_intersection_update0(left, *others):
#    for i in left:
#        for o in others:
#            if i not in o:
#                left.discard(i)

# for generic iterable *others
def set_intersection_update(left, *others):
    for o in others:
        temp = left.copy()
        left.clear()
        for i in o:
            if i in temp:
                left.add(i)


def set_difference_multiple(left, *others):
    result = left
    for o in others:
        result -= o
    return result


def set_difference_update(left, *others):
    for o in others:
        for e in o:
            left.discard(e)


def set_disjoint(left, right):
    for i in right:
        if i in left:
            return False
    return True


# for types that support "not in":
def set_subset(left, right):
    for i in left:
        if i not in right:
            return False
    return True


# for generic iterables:
def set_subset_iterable(left, right):
    if len(left) == 0:
        return True
    shadow = type(left)()
    for i in right:
        if i in left:
            shadow.add(i)
            if len(shadow) == len(left):
                return True
    return False


def set_superset(left, right):
    for i in right:
        if i not in left:
            return False
    return True


def set_proper_subset(left, right):
    for i in left:
        if i not in right:
            return False
    return len(left) < len(right)


def set_equal(left, right):
    return len(left) == len(right) and set_subset(left, right)


def set_not_equal(left, right):
    return len(left) != len(right) or not set_subset(right, left)


def set_duplicate(s):
    res = type(s)()
    for item in s:
        res.add(item)
    return res


class SetWrapperBase(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t, behavior):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t if behavior is None else (t, behavior))

        self.keyType = typeWrapper(t.ElementType)
        self.setType = t

        self.keyBytecount = self.keyType.getBytecount()

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('items', native_ast.UInt8Ptr),
            ('items_populated', native_ast.UInt8Ptr),
            ('items_reserved', native_ast.Int64),
            ('top_item_slot', native_ast.Int64),
            ('hash_table_slots', native_ast.Int32Ptr),
            ('hash_table_hashes', native_ast.Int32Ptr),
            ('hash_table_size', native_ast.Int64),
            ('hash_table_count', native_ast.Int64),
            ('hash_table_empty_slots', native_ast.Int64),
            ('reserved', native_ast.Int64)  # not used
        ), name="SetWrapper").pointer()

    def on_refcount_zero(self, context, instance):
        assert instance.isReference

        return (
            context.converter.defineNativeFunction(
                "destructor_" + str(self.typeRepresentation),
                ('destructor', self),
                [self],
                typeWrapper(type(None)),
                self.generateNativeDestructorFunction
            ).call(instance)
        )

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_len_native(self, expr):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return expr.ElementPtrIntegers(0, 8).load().cast(native_ast.Int64)


class SetWrapper(SetWrapperBase):
    def __init__(self, setType):
        super().__init__(setType, None)

    def convert_default_initialize(self, context, instance):
        context.pushEffect(
            instance.expr.store(
                runtime_functions.table_create.call().cast(self.layoutType)
            )
        )

    def convert_attribute(self, context, expr, attr):
        if attr in (
                "getKeyByIndexUnsafe", "deleteItemByIndexUnsafe",
                "initializeKeyByIndexUnsafe", "_allocateNewSlotUnsafe", "_resizeTableUnsafe",
                "_compressItemTableUnsafe",
                "add", "remove", "discard", "pop", "clear", "copy", "log",
                "union", "intersection", "difference", "symmetric_difference",
                "update", "intersection_update", "difference_update", "symmetric_difference_update",
                "issubset", "issuperset", "isdisjoint"):
            return expr.changeType(BoundMethodWrapper.Make(self, attr))

        if attr == '_items':
            return context.pushPod(
                PointerTo(UInt8),
                expr.nonref_expr.ElementPtrIntegers(0, 1).load()
            )

        if attr == '_items_populated':
            return context.pushPod(
                PointerTo(UInt8),
                expr.nonref_expr.ElementPtrIntegers(0, 2).load()
            )

        if attr == '_items_reserved':
            return context.pushPod(
                int,
                expr.nonref_expr.ElementPtrIntegers(0, 3).load()
            )

        if attr == '_top_item_slot':
            return context.pushPod(
                int,
                expr.nonref_expr.ElementPtrIntegers(0, 4).load()
            )

        if attr == '_hash_table_slots':
            return context.pushPod(
                PointerTo(Int32),
                expr.nonref_expr.ElementPtrIntegers(0, 5).load()
            )

        if attr == '_hash_table_hashes':
            return context.pushPod(
                PointerTo(Int32),
                expr.nonref_expr.ElementPtrIntegers(0, 6).load()
            )

        if attr == '_hash_table_size':
            return context.pushPod(
                int,
                expr.nonref_expr.ElementPtrIntegers(0, 7).load()
            )

        if attr == '_hash_table_count':
            return context.pushPod(
                int,
                expr.nonref_expr.ElementPtrIntegers(0, 8).load()
            )

        if attr == '_hash_table_empty_slots':
            return context.pushPod(
                int,
                expr.nonref_expr.ElementPtrIntegers(0, 9).load()
            )

        return super().convert_attribute(context, expr, attr)

    def convert_set_attribute(self, context, instance, attr, expr):
        if attr == '_items_reserved':
            val = expr.toInt64()
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 3).store(val.nonref_expr)
            )

            return context.pushVoid()

        if attr == '_top_item_slot':
            val = expr.toInt64()
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 4).store(val.nonref_expr)
            )

            return context.pushVoid()

        if attr == '_hash_table_size':
            val = expr.toInt64()
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 7).store(val.nonref_expr)
            )

            return context.pushVoid()

        if attr == '_hash_table_count':
            val = expr.toInt64()
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 8).store(val.nonref_expr)
            )

            return context.pushVoid()

        if attr == '_hash_table_empty_slots':
            val = expr.toInt64()
            if val is None:
                return None
            context.pushEffect(
                instance.nonref_expr.ElementPtrIntegers(0, 9).store(val.nonref_expr)
            )

            return context.pushVoid()

        return super().convert_set_attribute(context, instance, attr, expr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                SetKeysIteratorWrapper(self.setType),
                lambda iteratorInstance:
                iteratorInstance.expr.ElementPtrIntegers(0, 0).store(-1)
                >> iteratorInstance.expr.ElementPtrIntegers(0, 1).store(
                    self.convert_len_native(instance)
                )
                # we initialize the set pointer below, so technically
                # if that were to throw, this would leak a bad value.
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 2)
            ).convert_copy_initialize(instance)

            return res

        if methodname == 'union':
            return context.call_py_function(set_union_multiple, (instance, *args), {})

        if methodname == 'update':
            return context.call_py_function(set_update, (instance, *args), {})

        if methodname == 'intersection':
            return context.call_py_function(set_intersection_multiple, (instance, *args), {})

        if methodname == 'intersection_update':
            return context.call_py_function(set_intersection_update, (instance, *args), {})

        if methodname == 'difference':
            return context.call_py_function(set_difference_multiple, (instance, *args), {})

        if methodname == 'difference_update':
            return context.call_py_function(set_difference_update, (instance, *args), {})

        if methodname == 'symmetric_difference':
            if len(args) != 1:
                return context.pushException(TypeError, f"symmetric_difference() takes exactly one argument ({len(args)} given)")
            return context.call_py_function(set_symmetric_difference, (instance, args[0]), {})

        if methodname == 'symmetric_difference_update':
            if len(args) != 1:
                return context.pushException(TypeError, f"symmetric_difference_update() takes exactly one argument ({len(args)} given)")
            return context.call_py_function(set_symmetric_difference_update, (instance, args[0]), {})

        if len(args) == 0:
            if methodname == "pop":
                return context.call_py_function(set_pop, (instance,), {})
            if methodname == "clear":
                return context.call_py_function(table_clear, (instance,), {})
            if methodname == "_compressItemTableUnsafe":
                context.pushEffect(
                    runtime_functions.table_compress.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr),
                        context.constant(self.keyBytecount)
                    )
                )
                return context.pushVoid()

            if methodname == "_resizeTableUnsafe":
                context.pushEffect(
                    runtime_functions.table_resize.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr)
                    )
                )
                return context.pushVoid()

            if methodname == "copy":
                return context.push(
                    typeWrapper(self.setType),
                    lambda ref: ref.expr.store(
                        runtime_functions.table_copy.call(
                            instance.nonref_expr.cast(native_ast.VoidPtr),
                            context.getTypePointer(self.setType)
                        ).cast(self.layoutType)
                    )
                )

            if methodname == "_allocateNewSlotUnsafe":
                return context.pushPod(
                    Int32,
                    runtime_functions.table_allocate_new_slot.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr),
                        context.constant(self.keyBytecount)
                    )
                )

        if len(args) == 1:
            if methodname == "isdisjoint":
                return context.call_py_function(set_disjoint, (instance, args[0]), {})
            if methodname == "issubset":
                argType = args[0].expr_type.typeRepresentation
                argCat = getattr(argType, "__typed_python_category__", None)
                if argCat in ('Set', 'Dict', 'ConstDict'):  # types that have fast "in" operator
                    return context.call_py_function(set_subset, (instance, args[0]), {})
                else:  # generic iterable type
                    return context.call_py_function(set_subset_iterable, (instance, args[0]), {})
            if methodname == "issuperset":
                return context.call_py_function(set_superset, (instance, args[0]), {})

            if methodname == "add":
                key = args[0].convert_to_type(self.keyType, ConversionLevel.UpcastContainers)
                if key is None:
                    return None

                return context.call_py_function(set_add, (instance, key), {})

            if methodname == "remove":
                key = args[0].convert_to_type(self.keyType, ConversionLevel.UpcastContainers)
                if key is None:
                    return None

                return context.call_py_function(set_remove, (instance, key), {})

            if methodname == "discard":
                key = args[0].convert_to_type(self.keyType, ConversionLevel.UpcastContainers)
                if key is None:
                    return None

                return context.call_py_function(set_discard, (instance, key), {})

            if methodname in ("getKeyByIndexUnsafe", "deleteItemByIndexUnsafe"):
                index = args[0].toInt64()
                if index is None:
                    return None

                key = context.pushReference(
                    self.keyType,
                    instance.nonref_expr.ElementPtrIntegers(0, 1).load().cast(
                        self.keyType.getNativeLayoutType().pointer()
                    ).elemPtr(index.toInt64().nonref_expr)
                )

                if methodname == "getKeyByIndexUnsafe":
                    return key
                elif methodname == "deleteItemByIndexUnsafe":
                    key.convert_destroy()
                    return context.pushVoid()

        if len(args) == 2:
            if methodname == "initializeKeyByIndexUnsafe":
                index = args[0].toInt64()
                if index is None:
                    return None

                key = args[1].convert_to_type(self.keyType, ConversionLevel.UpcastContainers)
                if key is None:
                    return None

                item = context.pushReference(
                    self.keyType,
                    instance.nonref_expr.ElementPtrIntegers(0, 1).load().cast(
                        self.keyType.getNativeLayoutType().pointer()
                    ).elemPtr(index.toInt64().nonref_expr)
                )

                item.convert_copy_initialize(key)

                return context.pushVoid()

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_items_reserved_native(self, expr):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return expr.ElementPtrIntegers(0, 3).load().cast(native_ast.Int64)

    def convert_items_reserved(self, context, expr):
        return context.pushPod(int, self.convert_items_reserved_native(expr))

    def convert_slot_populated_native(self, expr, slotIx):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return expr.ElementPtrIntegers(0, 2).load().elemPtr(slotIx.nonref_expr).load()

    def convert_len(self, context, expr):
        return context.pushPod(int, self.convert_len_native(expr))

    def convert_bin_op(self, context, left, op, right, inplace):
        if right.expr_type == left.expr_type:
            if op.matches.BitOr:
                return context.call_py_function(set_union, (left, right), {})
            if op.matches.BitAnd:
                return context.call_py_function(set_intersection, (left, right), {})
            if op.matches.Sub:
                return context.call_py_function(set_difference, (left, right), {})
            if op.matches.BitXor:
                return context.call_py_function(set_symmetric_difference, (left, right), {})
            if op.matches.Eq:
                return context.call_py_function(set_equal, (left, right), {})
            if op.matches.NotEq:
                return context.call_py_function(set_not_equal, (left, right), {})
            if op.matches.Lt:
                return context.call_py_function(set_proper_subset, (left, right), {})
            if op.matches.LtE:
                return context.call_py_function(set_subset, (left, right), {})
            if op.matches.Gt:
                return context.call_py_function(set_proper_subset, (right, left), {})
            if op.matches.GtE:
                return context.call_py_function(set_subset, (right, left), {})

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_bin_op_reverse(self, context, right, op, left, inplace):
        if op.matches.In:
            left = left.convert_to_type(self.keyType, ConversionLevel.UpcastContainers)
            if left is None:
                return None

            return context.call_py_function(table_contains, (right, left), {})

        return super().convert_bin_op_reverse(context, right, op, left, inplace)

    def convert_getkey_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 1).load()
                .elemPtr(item.nonref_expr.mul(native_ast.const_int_expr(self.keyBytecount)))
                .cast(self.keyType.getNativeLayoutType().pointer())
        )

    def generateNativeDestructorFunction(self, context, out, inst):
        with context.loop(self.convert_items_reserved(context, inst)) as i:
            with context.ifelse(self.convert_slot_populated_native(inst, i)) as (then, otherwise):
                with then:
                    self.convert_getkey_by_index_unsafe(context, inst, i).convert_destroy()

        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 1).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 2).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 5).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.ElementPtrIntegers(0, 6).load().cast(native_ast.UInt8Ptr)) >>
            runtime_functions.free.call(inst.nonref_expr.cast(native_ast.UInt8Ptr))
        )

    def convert_type_call_on_container_expression(self, context, typeInst, argExpr):
        if not (argExpr.matches.Set or argExpr.matches.List or argExpr.matches.Tuple):
            return super().convert_type_call_on_container_expression(context, typeInst, argExpr)

        # we're calling Set(T) with an expression like {1, 2, 3, ...}
        # TODO: construct directly with unsafe calls, like we do for ListOf and TupleOf

        aSet = self.convert_type_call(context, None, [], {})

        for i in range(len(argExpr.elts)):
            val = context.convert_expression_ast(argExpr.elts[i])
            if val is None:
                return None
            aSet.convert_method_call("add", (val,), {})

        return aSet

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1 and not kwargs:
            if args[0].expr_type == self:
                return context.call_py_function(set_duplicate, (args[0],), {})
            return args[0].convert_to_type(self, ConversionLevel.New)

        return super().convert_type_call(context, typeInst, args, kwargs)

    def _can_convert_from_type(self, otherType, conversionLevel):
        if not conversionLevel.isImplicitContainersOrHigher():
            # Set only allows 'implicit containers' conversions or higher
            return False

        # note that if the other object is an untyped container, then the
        # pathway in python_object_of_type will take care of it.
        childLevel = (
            ConversionLevel.ImplicitContainers
            if conversionLevel.isNewOrHigher() else
            ConversionLevel.UpcastContainers
        )

        if issubclass(otherType.typeRepresentation, (ListOf, TupleOf, Set, Dict, ConstDict)):
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

        return super()._can_convert_from_type(otherType, conversionLevel)

    def convert_to_self_with_target(self, context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure=False):
        canConvert = self._can_convert_from_type(sourceVal.expr_type, conversionLevel)

        if canConvert is False:
            return super().convert_to_self_with_target(context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure)

        if conversionLevel.isNewOrHigher():
            converter = initialize_set_from_other_implicit_containers
        else:
            converter = initialize_set_from_other_upcast_containers

        res = context.call_py_function(
            converter,
            (targetVal.asPointer(), sourceVal, context.constant(mayThrowOnFailure)),
            {}
        )

        if canConvert is True:
            return context.constant(True)

        return res

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


class SetMakeIteratorWrapper(SetWrapperBase):
    def convert_method_call(self, context, expr, methodname, args, kwargs):
        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                # self.iteratorType is inherited from our specialized children
                # who pick whether we're an iterator over keys, values, items, etc.
                self.iteratorType,
                lambda iteratorInstance:
                    iteratorInstance.expr.ElementPtrIntegers(0, 0).store(-1)
                    >> iteratorInstance.expr.ElementPtrIntegers(0, 1).store(
                        self.convert_len_native(expr)
                    )
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 2)
            ).convert_copy_initialize(expr)

            return res

        return super().convert_method_call(context, expr, methodname, args, kwargs)


class SetKeysWrapper(SetMakeIteratorWrapper):
    def __init__(self, setType):
        super().__init__(setType, "keys")
        self.iteratorType = SetKeysIteratorWrapper(setType)


class SetIteratorWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, setType, iteratorType):
        self.setType = setType
        self.iteratorType = iteratorType
        super().__init__((setType, "iterator", iteratorType))

    def getNativeLayoutType(self):
        return native_ast.Type.Struct(
            element_types=(
                ("pos", native_ast.Int64),
                ("count", native_ast.Int64),
                ("set", SetWrapper(self.setType).getNativeLayoutType())
            ),
            name="const_set_iterator"
        )

    def convert_fastnext(self, context, expr):
        context.call_py_function(
            checkSetSizeAndThrowIfChanged,
            (
                self.refAs(context, expr, 2),
                self.refAs(context, expr, 1),
            ),
            {}
        )

        nextSlotIx = context.call_py_function(
            table_next_slot,
            (
                self.refAs(context, expr, 2),
                self.refAs(context, expr, 0)
            ),
            {}
        )

        if nextSlotIx is None:
            return None

        context.pushEffect(
            expr.expr.ElementPtrIntegers(0, 0).store(
                nextSlotIx.nonref_expr
            )
        )
        canContinue = context.pushPod(
            bool,
            nextSlotIx.nonref_expr.gte(0)
        )

        nextIx = context.pushReference(int, expr.expr.ElementPtrIntegers(0, 0))

        return self.iteratedItemForReference(context, expr, nextIx).asPointerIf(canContinue)

    def refAs(self, context, expr, which):
        assert expr.expr_type == self

        if which == 0:
            return context.pushReference(int, expr.expr.ElementPtrIntegers(0, 0))

        if which == 1:
            return context.pushReference(int, expr.expr.ElementPtrIntegers(0, 1))

        if which == 2:
            return context.pushReference(
                self.setType,
                expr.expr
                    .ElementPtrIntegers(0, 2)
                    .cast(SetWrapper(self.setType).getNativeLayoutType().pointer())
            )

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        for i in range(3):
            self.refAs(context, expr, i).convert_assign(self.refAs(context, other, i))

    def convert_copy_initialize(self, context, expr, other):
        for i in range(3):
            self.refAs(context, expr, i).convert_copy_initialize(self.refAs(context, other, i))

    def convert_destroy(self, context, expr):
        self.refAs(context, expr, 2).convert_destroy()


class SetKeysIteratorWrapper(SetIteratorWrapper):
    def __init__(self, setType):
        super().__init__(setType, "keys")

    def iteratedItemForReference(self, context, expr, ixExpr):
        return SetWrapper(self.setType).convert_method_call(
            context,
            self.refAs(context, expr, 2),
            "getKeyByIndexUnsafe",
            (ixExpr,),
            {}
        )
