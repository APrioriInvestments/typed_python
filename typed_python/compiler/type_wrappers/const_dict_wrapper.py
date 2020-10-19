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

from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.util import min
from typed_python.compiler.typed_expression import TypedExpression

from typed_python import Tuple

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def const_dict_eq(l, r) -> bool:
    """Given two ConstDicts instances, checks for equality by comparing their individual elements.

    Compiling == or ConstDict.__eq__  compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        l: left operand
        r: right operand

    Returns:
        bool result of equality comparison
    """
    if len(l) != len(r):
        return False

    for i in range(len(l)):
        if l.get_key_by_index_unsafe(i) != r.get_key_by_index_unsafe(i):
            return False

        if l.get_value_by_index_unsafe(i) != r.get_value_by_index_unsafe(i):
            return False

    return True


def const_dict_neq(l, r):
    """Given two ConstDict instances, checks for inequality by comparing their individual elements.

    Compiling != or ConstDict.__ne__  compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        l: left operand
        r: right operand

    Returns:
        bool result of inequality comparison
    """
    return not const_dict_eq(l, r)


def const_dict_lt(left, right):
    """Is a ConstDict instance < another, comparing their individual elements?

    Compiling < or ConstDict.__lt__  compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        l: left operand
        r: right operand

    Returns:
        bool result of < comparison
    """
    for i in range(min(len(left), len(right))):
        if left.get_key_by_index_unsafe(i) > right.get_key_by_index_unsafe(i):
            return False

        if left.get_key_by_index_unsafe(i) < right.get_key_by_index_unsafe(i):
            return True

        if left.get_value_by_index_unsafe(i) > right.get_value_by_index_unsafe(i):
            return False

        if left.get_value_by_index_unsafe(i) < right.get_value_by_index_unsafe(i):
            return True

    return len(left) < len(right)


def const_dict_lte(left, right):
    """Is a ConstDict instance <= another, comparing their individual elements?
    """
    for i in range(min(len(left), len(right))):
        if left.get_key_by_index_unsafe(i) > right.get_key_by_index_unsafe(i):
            return False

        if left.get_key_by_index_unsafe(i) < right.get_key_by_index_unsafe(i):
            return True

        if left.get_value_by_index_unsafe(i) > right.get_value_by_index_unsafe(i):
            return False

        if left.get_value_by_index_unsafe(i) < right.get_value_by_index_unsafe(i):
            return True

    return len(left) <= len(right)


def const_dict_gt(left, right):
    """Is a ConstDict instance > another, comparing their individual elements?
    """
    return not const_dict_lte(left, right)


def const_dict_gte(left, right):
    """Is a ConstDict instance >= another, comparing their individual elements?
    """
    return not const_dict_lt(left, right)


def const_dict_getitem(constDict, key):
    """Get value associated with key from a ConstDict instance.

    Compiling constDict[k] compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        constDict:
        key: key to search for, must be comparable to constDict.elementType

    Returns:
        value of key in constDict

    Raises:
        KeyError if not found
    """
    # perform a binary search
    lowIx = 0
    highIx = len(constDict)

    while lowIx < highIx:
        mid = (lowIx + highIx) >> 1

        keyAtVal = constDict.get_key_by_index_unsafe(mid)

        if keyAtVal < key:
            lowIx = mid + 1
        elif key < keyAtVal:
            highIx = mid
        else:
            return constDict.get_value_by_index_unsafe(mid)

    raise KeyError(key)


def const_dict_get(constDict, key, default):
    """Get value associated with key from a ConstDict instance, using default value if not found.

    Compiling constDict.get(k,default) compiles this function.
    This function is only intended to be executed in this compiled form.

    Args:
        constDict:
        key: key to search for, must be comparable to constDict.elementType

    Returns:
        value of key in constDict, or default if not found
    """
    # perform a binary search
    lowIx = 0
    highIx = len(constDict)

    while lowIx < highIx:
        mid = (lowIx + highIx) >> 1

        keyAtVal = constDict.get_key_by_index_unsafe(mid)

        if keyAtVal < key:
            lowIx = mid + 1
        elif key < keyAtVal:
            highIx = mid
        else:
            return constDict.get_value_by_index_unsafe(mid)

    return default


def const_dict_contains(constDict, key):
    """Does constDict contain key?

    Compiling key in constDict compiles this function.
    This function is only intended to be executed in this compiled form.
    """
    # perform a binary search
    lowIx = 0
    highIx = len(constDict)

    while lowIx < highIx:
        mid = (lowIx + highIx) >> 1

        keyAtVal = constDict.get_key_by_index_unsafe(mid)

        if keyAtVal < key:
            lowIx = mid + 1
        elif key < keyAtVal:
            highIx = mid
        else:
            return True

    return False


class ConstDictWrapperBase(RefcountedWrapper):
    """Common method wrappers for all ConstDicts.

    We subclass this for things like 'keys', 'values', and 'items' since
    they all basically look like a const-dict with different methods
    """
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, constDictType, behavior):
        assert hasattr(constDictType, '__typed_python_category__')
        super().__init__(constDictType if behavior is None else (constDictType, behavior))

        self.constDictType = constDictType
        self.keyType = typeWrapper(constDictType.KeyType)
        self.valueType = typeWrapper(constDictType.ValueType)
        self.itemType = typeWrapper(Tuple(constDictType.KeyType, constDictType.ValueType))

        self.kvBytecount = self.keyType.getBytecount() + self.valueType.getBytecount()
        self.keyBytecount = self.keyType.getBytecount()

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('count', native_ast.Int32),
            ('subpointers', native_ast.Int32),
            ('data', native_ast.UInt8)
        ), name='ConstDictLayout').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        """ Returns code to be executed on ConstDict instance when refcount reaches zero.

        Args:
             context: ExpressionConversionContext
             instance: TypedExpression, reference to ConstDict instance.

        Returns:
            native_ast code
        """
        assert instance.isReference

        if self.keyType.is_pod and self.valueType.is_pod:
            return runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr))
        else:
            return (
                context.converter.defineNativeFunction(
                    "destructor_" + str(self.constDictType),
                    ('destructor', self),
                    [self],
                    typeWrapper(type(None)),
                    self.generateNativeDestructorFunction
                )
                .call(instance)
            )

    def generateNativeDestructorFunction(self, context, out, inst):
        """ Returns code that destroys (tears down) a ConstDict instance.
        """
        with context.loop(inst.convert_len()) as i:
            self.convert_getkey_by_index_unsafe(context, inst, i).convert_destroy()
            self.convert_getvalue_by_index_unsafe(context, inst, i).convert_destroy()

        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.cast(native_ast.UInt8Ptr))
        )


class ConstDictWrapper(ConstDictWrapperBase):
    def __init__(self, constDictType):
        super().__init__(constDictType, None)

    def convert_attribute(self, context, instance, attr):
        """Generates code for ConstDict attribute access.
        """
        if attr in ("get_key_by_index_unsafe", "get_value_by_index_unsafe", "keys", "values", "items", "get"):
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname: str, args, kwargs):
        """Generates code for calling a ConstDict method.

        Generates code raising AttributeError if methodname is invalid.
        Generates code raising TypeError if argument type is invalid.
        Generates code raising ValueError if argument value is invalid.

        Args:
            context: ExpressionConversionContext
            instance: TypedExpression of type ConstDict
            methodname: method name
            args: positional arguments, as tuple of TypedExpressions
            kwargs: keyword arguments, as dict(str, TypedExpression)

        Returns:
            TypedExpression of return value of method call, or None if control does not return
        """
        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                ConstDictKeysIteratorWrapper(self.constDictType),
                lambda instance:
                    instance.expr.ElementPtrIntegers(0, 0).store(-1)
                    # we initialize the dict pointer below, so technically
                    # if that were to throw, this would leak a bad value.
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 1)
            ).convert_copy_initialize(instance)

            return res

        if methodname == "get" and not kwargs:
            if len(args) == 1:
                return self.convert_get(context, instance, args[0], context.constant(None))
            elif len(args) == 2:
                return self.convert_get(context, instance, args[0], args[1])

        if methodname == "keys" and not args and not kwargs:
            return instance.changeType(ConstDictKeysWrapper(self.constDictType))

        if methodname == "values" and not args and not kwargs:
            return instance.changeType(ConstDictValuesWrapper(self.constDictType))

        if methodname == "items" and not args and not kwargs:
            return instance.changeType(ConstDictItemsWrapper(self.constDictType))

        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "get_key_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getkey_by_index_unsafe(context, instance, ix)

        if methodname == "get_value_by_index_unsafe":
            if len(args) == 1:
                ix = args[0].toInt64()
                if ix is None:
                    return

                return self.convert_getvalue_by_index_unsafe(context, instance, ix)

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_getkey_by_index_unsafe(self, context, expr, index):
        """Generates code for compiling unsafe indexed access to a key in a ConstDict instance.
        """
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                index.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
            ).cast(self.keyType.getNativeLayoutType().pointer())
        )

    def convert_getitem_by_index_unsafe(self, context, expr, index):
        """Generates code for compiling unsafe indexed access to an item in a ConstDict instance.
        """
        return context.pushReference(
            self.itemType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                index.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
            ).cast(self.itemType.getNativeLayoutType().pointer())
        )

    def convert_getvalue_by_index_unsafe(self, context, expr, index):
        """Generates code for compiling unsafe indexed access to a value in a ConstDict instance.
        """
        return context.pushReference(
            self.valueType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                index.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
                .add(native_ast.const_int_expr(self.keyBytecount))
            ).cast(self.valueType.getNativeLayoutType().pointer())
        )

    def convert_bin_op(self, context, left, op, right, inplace: bool):
        """Generates code for binary operator op on l and r: l op r

        Args:
            context: ExpressionConversionContext
            l: left operand, a ConstDict TypedExpression
            op: operator, python_ast.BinaryOp
            r: right operand, TypedExpression, possibly of another type
            inplace: Is this operation done in-place? (l = l op r)

        Returns:
            TypedExpression, or None if control does not return
        """
        if right.expr_type == left.expr_type:
            if op.matches.Eq:
                return context.call_py_function(const_dict_eq, (left, right), {})
            if op.matches.NotEq:
                return context.call_py_function(const_dict_neq, (left, right), {})
            if op.matches.Lt:
                return context.call_py_function(const_dict_lt, (left, right), {})
            if op.matches.LtE:
                return context.call_py_function(const_dict_lte, (left, right), {})
            if op.matches.Gt:
                return context.call_py_function(const_dict_gt, (left, right), {})
            if op.matches.GtE:
                return context.call_py_function(const_dict_gte, (left, right), {})

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_bin_op_reverse(self, context, left, op, right, inplace):
        """Generates code for binary operator op on l and r: l op r, where r is of this wrapper's type.

        See Wrapper.convert_bin_op_reverse.
        """
        if op.matches.In:
            right = right.convert_to_type(self.keyType)
            if right is None:
                return None

            return context.call_py_function(const_dict_contains, (left, right), {})

        return super().convert_bin_op_reverse(context, left, op, right, inplace)

    def convert_getitem(self, context, instance, item):
        """Generates code for instance[item].
        """
        item = item.convert_to_type(self.keyType, explicit=False)
        if item is None:
            return None

        return context.call_py_function(const_dict_getitem, (instance, item), {})

    def convert_get(self, context, expr, item, default):
        """Generates code for instance.get(item, default).
        """
        if item is None or expr is None or default is None:
            return None

        item = item.convert_to_type(self.keyType, explicit=False)
        if item is None:
            return None

        return context.call_py_function(const_dict_get, (expr, item, default), {})

    def convert_len_native(self, expr):
        """Returns code for len of a ConstDict expr.
        """
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
        )

    def convert_len(self, context, expr):
        """Generates code for len of a ConstDict expr.
        """
        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))

    def convert_bool_cast(self, context, expr):
        """Generates code to cast this ConstDict to bool.
        """
        return context.pushPod(bool, self.convert_len_native(expr.nonref_expr).neq(0))


class ConstDictMakeIteratorWrapper(ConstDictWrapperBase):
    """Code-generation wrapper for ConstDict iterators.
    """
    def convert_method_call(self, context, expr, methodname, args, kwargs):
        """Generates code to call ConstDict iterator methods.
        """
        if methodname == "__iter__" and not args and not kwargs:
            res = context.push(
                # self.iteratorType is inherited from our specialized children
                # who pick whether we're an interator over keys, values, items, etc.
                self.iteratorType,
                lambda instance:
                    instance.expr.ElementPtrIntegers(0, 0).store(-1)
            )

            context.pushReference(
                self,
                res.expr.ElementPtrIntegers(0, 1)
            ).convert_copy_initialize(expr)

            return res

        return super().convert_method_call(context, expr, methodname, args, kwargs)


class ConstDictKeysWrapper(ConstDictMakeIteratorWrapper):
    """Code-generation wrapper for ConstDict.keys.
    """
    def __init__(self, constDictType):
        super().__init__(constDictType, "keys")
        self.iteratorType = ConstDictKeysIteratorWrapper(constDictType)


class ConstDictValuesWrapper(ConstDictMakeIteratorWrapper):
    """Code-generation wrapper for ConstDict.values.
    """
    def __init__(self, constDictType):
        super().__init__(constDictType, "values")
        self.iteratorType = ConstDictValuesIteratorWrapper(constDictType)


class ConstDictItemsWrapper(ConstDictMakeIteratorWrapper):
    """Code-generation wrapper for ConstDict.items.
    """
    def __init__(self, constDictType):
        super().__init__(constDictType, "items")
        self.iteratorType = ConstDictItemsIteratorWrapper(constDictType)


class ConstDictIteratorWrapper(Wrapper):
    """Code-generation wrapper for ConstDict iterators.
    """
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, constDictType, iteratorType):
        self.constDictType = constDictType
        self.iteratorType = iteratorType
        super().__init__((constDictType, "iterator", iteratorType))

    def getNativeLayoutType(self):
        return native_ast.Type.Struct(
            element_types=(("pos", native_ast.Int64), ("dict", ConstDictWrapper(self.constDictType).getNativeLayoutType())),
            name="const_dict_iterator"
        )

    def convert_next(self, context, expr):
        """Generates code for expr.__next__() for ConstDict iterator.
        """
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

        return self.iteratedElementForReference(context, expr, nextIx), canContinue

    def refAs(self, context, expr, which):
        """Generates code to access an internal field of the ConstDict iterator.
        """
        assert expr.expr_type == self

        if which == 0:
            return context.pushReference(int, expr.expr.ElementPtrIntegers(0, 0))

        if which == 1:
            return context.pushReference(
                self.constDictType,
                expr.expr
                    .ElementPtrIntegers(0, 1)
                    .cast(ConstDictWrapper(self.constDictType).getNativeLayoutType().pointer())
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


class ConstDictKeysIteratorWrapper(ConstDictIteratorWrapper):
    """Code-generation wrapper for ConstDict keys iterator.
    """
    def __init__(self, constDictType):
        super().__init__(constDictType, "keys")

    def iteratedElementForReference(self, context, expr, ixExpr):
        """Generates code to access an iteration key by reference.
        """
        return ConstDictWrapper(self.constDictType).convert_getkey_by_index_unsafe(
            context,
            self.refAs(context, expr, 1),
            ixExpr
        )


class ConstDictItemsIteratorWrapper(ConstDictIteratorWrapper):
    """Code-generation wrapper for ConstDict items iterator.
    """
    def __init__(self, constDictType):
        super().__init__(constDictType, "items")

    def iteratedElementForReference(self, context, expr, ixExpr):
        """Generates code to access an iteration item by reference.
        """
        return ConstDictWrapper(self.constDictType).convert_getitem_by_index_unsafe(
            context,
            self.refAs(context, expr, 1),
            ixExpr
        )


class ConstDictValuesIteratorWrapper(ConstDictIteratorWrapper):
    """Code-generation wrapper for ConstDict values iterator.
    """
    def __init__(self, constDictType):
        super().__init__(constDictType, "values")

    def iteratedElementForReference(self, context, expr, ixExpr):
        """Generates code to access an iteration value by reference.
        """
        return ConstDictWrapper(self.constDictType).convert_getvalue_by_index_unsafe(
            context,
            self.refAs(context, expr, 1),
            ixExpr
        )
