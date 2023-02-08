#   Copyright 2017-2019 typed_python Authors
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
from typed_python.compiler.merge_type_wrappers import mergeTypes
from typed_python import (
    _types, Int32, Tuple, NamedTuple, Function, Dict, Set, ConstDict, ListOf,
    TupleOf, PointerTo, pointerTo, TypeFunction, OneOf, Class, Member, Final
)
import typed_python._types
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
import types
import typed_python.compiler.native_ast as native_ast
import typed_python.python_ast as python_ast
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class _TupleLt(CompilableBuiltin):
    """A builtin for loop-unrolling tuple comparison"""
    IS_LTE = False

    def __eq__(self, other):
        return isinstance(other, _TupleLt)

    def __hash__(self):
        return hash("_TupleLt")

    def convert_call(self, context, expr, args, kwargs):
        assert len(args) == 2
        lhsT = args[0].expr_type.typeRepresentation
        rhsT = args[1].expr_type.typeRepresentation

        assert isinstance(lhsT, type) and issubclass(lhsT, (NamedTuple, Tuple))
        assert isinstance(rhsT, type) and issubclass(rhsT, (NamedTuple, Tuple))

        for i in range(min(len(lhsT.ElementTypes), len(rhsT.ElementTypes))):
            res = args[0].refAs(i) < args[1].refAs(i)
            if res is None:
                return None

            res = res.toBool()
            if res is None:
                return None

            with context.ifelse(res.nonref_expr) as (ifTrue, ifFalse):
                with ifTrue:
                    context.pushReturnValue(context.constant(True))

            resReverse = args[1].refAs(i) < args[0].refAs(i)
            if resReverse is None:
                return None

            resReverse = resReverse.toBool()
            if resReverse is None:
                return None

            with context.ifelse(resReverse.nonref_expr) as (ifTrue, ifFalse):
                with ifTrue:
                    context.pushReturnValue(context.constant(False))

        if self.IS_LTE:
            return context.constant(len(lhsT.ElementTypes) <= len(rhsT.ElementTypes))
        else:
            return context.constant(len(lhsT.ElementTypes) < len(rhsT.ElementTypes))


class _TupleLtE(_TupleLt):
    """A builtin for loop-unrolling tuple comparison"""
    IS_LTE = True

    def __eq__(self, other):
        return isinstance(other, _TupleLtE)

    def __hash__(self):
        return hash("_TupleLtE")


class _TupleEq(CompilableBuiltin):
    """A builtin for loop-unrolling tuple comparison"""
    def __eq__(self, other):
        return isinstance(other, _TupleEq)

    def __hash__(self):
        return hash("_TupleEq")

    def convert_call(self, context, expr, args, kwargs):
        assert len(args) == 2
        lhsT = args[0].expr_type.typeRepresentation
        rhsT = args[1].expr_type.typeRepresentation

        assert isinstance(lhsT, type) and issubclass(lhsT, (NamedTuple, Tuple))
        assert isinstance(rhsT, type) and issubclass(rhsT, (NamedTuple, Tuple))

        if len(lhsT.ElementTypes) != len(rhsT.ElementTypes):
            return context.constant(False)

        for i in range(len(lhsT.ElementTypes)):
            res = args[0].refAs(i) == args[1].refAs(i)
            if res is None:
                return None

            res = res.toBool()
            if res is None:
                return None

            with context.ifelse(res.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushReturnValue(context.constant(False))

        return context.constant(True)


# define these operations directly in terms of the builtins we
# defined up above.
@Function
def _tupleLt(lhs, rhs) -> bool:
    return _TupleLt()(lhs, rhs)


@Function
def _tupleLtE(lhs, rhs) -> bool:
    return _TupleLtE()(lhs, rhs)


@Function
def _tupleEq(lhs, rhs) -> bool:
    return _TupleEq()(lhs, rhs)


# define these operations in terms of the functions we just
# defined, not the CompilableBuiltins, because they push
# return statements directly onto the stack of the caller,
# which means we can't use 'not' on the return value.
@Function
def _tupleGt(lhs, rhs) -> bool:
    return not _tupleLtE(lhs, rhs)


@Function
def _tupleGtE(lhs, rhs) -> bool:
    return not _tupleLt(lhs, rhs)


@Function
def _tupleNe(lhs, rhs) -> bool:
    return not _tupleEq(lhs, rhs)


pyCompOpToTupleFun = {
    python_ast.ComparisonOp.Eq(): _tupleEq,
    python_ast.ComparisonOp.NotEq(): _tupleNe,
    python_ast.ComparisonOp.Lt(): _tupleLt,
    python_ast.ComparisonOp.LtE(): _tupleLtE,
    python_ast.ComparisonOp.Gt(): _tupleGt,
    python_ast.ComparisonOp.GtE(): _tupleGtE
}


class TupleWrapper(Wrapper):
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        self.subTypeWrappers = tuple(typeWrapper(sub_t) for sub_t in t.ElementTypes)
        self._unionType = None

        if not self.subTypeWrappers:
            self.byteOffsets = []
        else:
            self.byteOffsets = [0]

            for i in range(len(self.subTypeWrappers)-1):
                self.byteOffsets.append(self.byteOffsets[-1] + _types.bytecount(t.ElementTypes[i]))

        structArgs = []
        for i in range(len(self.subTypeWrappers)):
            nlt = self.subTypeWrappers[i].getNativeLayoutType()

            assert isinstance(nlt, native_ast.Type)
            assert not nlt.matches.Void, (self.subTypeWrappers[i], type(self.subTypeWrappers[i]))

            structArgs.append(
                (f'elt_{i}', nlt)
            )

        self.layoutType = native_ast.Type.Struct(
            element_types=structArgs,
            packed=True
        )

        self._is_pod = all(typeWrapper(possibility).is_pod for possibility in self.subTypeWrappers)
        self.is_default_constructible = _types.is_default_constructible(t)

    def isIterable(self):
        return True

    @property
    def unionType(self):
        if self._unionType is None and self.typeRepresentation.ElementTypes:
            self._unionType = mergeTypes(self.typeRepresentation.ElementTypes)
        return self._unionType

    def convert_hash(self, context, expr):
        val = context.constant(Int32(0))
        for i in range(len(self.subTypeWrappers)):
            subHash = self.refAs(context, expr, i).convert_hash()
            if subHash is None:
                return None
            val = (val * context.constant(Int32(1000003))) ^ subHash
        return val

    def has_method(self, methodName):
        return False

    @property
    def is_pod(self):
        return self._is_pod

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_attribute(self, context, instance, attribute):
        if attribute in ["__iter__"]:
            return instance.changeType(BoundMethodWrapper.Make(self, attribute))

        return super().convert_attribute(context, instance, attribute)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname == '__iter__' and not args:
            return typeWrapper(TupleIterator(self.typeRepresentation)).convert_type_call(
                context,
                None,
                [],
                dict(pos=context.constant(-1), tup=instance)
            )

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def convert_initialize_from_args(self, context, target, *args):
        assert len(args) <= len(self.byteOffsets)

        for i in range(len(self.byteOffsets)):
            if i < len(args):
                self.refAs(context, target, i).convert_copy_initialize(args[i])
            else:
                self.refAs(context, target, i).convert_default_initialize()

    def convert_default_initialize(self, context, target):
        if not self.is_default_constructible:
            context.pushException(TypeError, "Can't default-initialize any subtypes of %s" % self.typeRepresentation.__qualname__)
            return

        for i, t in enumerate(self.typeRepresentation.ElementTypes):
            if _types.is_default_constructible(t):
                self.refAs(context, target, i).convert_default_initialize()

    def refAs(self, context, expr, which):
        assert isinstance(which, int)

        if not expr.isReference:
            expr = context.pushMove(expr)

        if which < 0:
            which += len(self.subTypeWrappers)

        assert 0 <= which < len(self.subTypeWrappers), (which, self)

        return context.pushReference(
            self.subTypeWrappers[which],
            expr.expr.ElementPtrIntegers(0, which)
        )

    def convert_len(self, context, instance):
        return context.constant(len(self.subTypeWrappers))

    def convert_bin_op(self, context, left, op, right, inplace):
        rhsT = right.expr_type.typeRepresentation
        if isinstance(rhsT, type) and issubclass(rhsT, (NamedTuple, Tuple)) and op in pyCompOpToTupleFun:
            # this is the wrapper for the type, which we can call because it
            # has an empty closure.
            funcTypeWrapper = typeWrapper(type(pyCompOpToTupleFun[op]))

            return funcTypeWrapper.convert_call(context, None, (left, right), {})

        return super().convert_bin_op(context, left, op, right, inplace)

    def convert_bin_op_reverse(self, context, r, op, l, inplace):
        if op.matches.In:
            return self.convert_contains(context, r, l)

        return super().convert_bin_op_reverse(context, r, op, l, inplace)

    def convert_getitem(self, context, expr, index):
        index = index.toIndex()
        if index is None:
            return None

        if not len(self.subTypeWrappers):
            context.pushException(IndexError, "tuple index out of range")
            return None

        # if the argument is a constant, we can be very precise about what type
        # we're going to get out of the indexing operation
        if index.isConstant:
            indexVal = index.constantValue
            assert isinstance(indexVal, int)

            if indexVal >= - len(self.subTypeWrappers) and indexVal < len(self.subTypeWrappers):
                if indexVal < 0:
                    indexVal += len(self.subTypeWrappers)

                return self.refAs(context, expr, indexVal)

        index = index.toIndex()
        if index is None:
            return None

        result = context.allocateUninitializedSlot(self.unionType)
        with context.switch(
            index.nonref_expr,
            range(-len(self.subTypeWrappers), len(self.subTypeWrappers)),
            True
        ) as indicesAndContexts:
            for i, subcontext in indicesAndContexts:
                with subcontext:
                    if i is not None:
                        converted = self.refAs(context, expr, i).convert_to_type(
                            self.unionType, ConversionLevel.Signature
                        )
                        if converted is not None:
                            result.convert_copy_initialize(converted)
                            context.markUninitializedSlotInitialized(result)
                    else:
                        context.pushException(IndexError, "tuple index out of range")

        return result

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        for i in range(len(self.subTypeWrappers)):
            self.refAs(context, expr, i).convert_assign(self.refAs(context, other, i))
        return context.constant(None)

    def convert_copy_initialize(self, context, expr, other):
        for i in range(len(self.subTypeWrappers)):
            self.refAs(context, expr, i).convert_copy_initialize(self.refAs(context, other, i))
        return context.constant(None)

    def convert_destroy(self, context, expr):
        if not self.is_pod:
            for i in range(len(self.subTypeWrappers)):
                self.refAs(context, expr, i).convert_destroy()
        return context.constant(None)

    def get_iteration_expressions(self, context, expr):
        if self.has_intiter() or self.has_method("__iter__"):
            return None

        return [self.refAs(context, expr, i) for i in range(len(self.subTypeWrappers))]

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self, ConversionLevel.New)

        return context.pushException(TypeError, f"Can't initialize {self.typeRepresentation} with this signature")

    def convert_type_call_on_container_expression(self, context, typeInst, argExpr):
        if not (argExpr.matches.Tuple or argExpr.matches.List):
            return super().convert_type_call_on_container_expression(context, typeInst, argExpr)

        if len(self.typeRepresentation.ElementTypes) != len(argExpr.elts):
            context.pushException(TypeError, f"Wrong number of arguments to construct '{self.typeRepresentation}'")
            return

        args = []
        for tupArg in argExpr.elts:
            convertedArg = context.convert_expression_ast(tupArg)
            if convertedArg is None:
                return None
            args.append(convertedArg)

        return self.createFromArgs(context, args)

    def createFromArgs(self, context, args):
        """Initialize a new tuple of this type from a set of arguments.

        This will attempt to convert the tuple.
        """
        if all(a.constantValue is not None for a in args):
            constantVal = self.typeRepresentation([a.constantValue for a in args])
            return context.constantTypedPythonObject(constantVal)

        typeConvertedArgs = []

        for i in range(len(args)):
            typeConvertedArg = args[i].convert_to_type(
                self.typeRepresentation.ElementTypes[i],
                ConversionLevel.ImplicitContainers
            )

            if typeConvertedArg is None:
                return None

            typeConvertedArgs.append(typeConvertedArg)

        uninitializedTuple = context.allocateUninitializedSlot(self)

        for i in range(len(args)):
            uninitializedChildElement = self.refAs(context, uninitializedTuple, i)
            uninitializedChildElement.convert_copy_initialize(typeConvertedArgs[i])

        context.markUninitializedSlotInitialized(uninitializedTuple)

        # the tuple is now initialized
        return uninitializedTuple

    def _indicesInOtherTypeToRead(self, otherType):
        """Determine which field in another type we should read from when initializing 'self'.

        Args:
            otherType - another Tuple or NamedTuple wrapper.

        Returns:
            None if we can't initialize ourself from an instance of otherType, or
                a list of integers giving the index in 'otherType' to initialize
                each of our own slots, or None if we are supposed to default initialze.
        """
        if (
            issubclass(self.typeRepresentation, NamedTuple)
            and issubclass(otherType.typeRepresentation, NamedTuple)
        ):
            slots = [None for _ in self.typeRepresentation.ElementTypes]

            for ix, name in enumerate(otherType.typeRepresentation.ElementNames):
                if name in self.namesToIndices:
                    slots[self.namesToIndices[name]] = ix
                else:
                    return None

            for ix in range(len(slots)):
                if slots[ix] is None and not _types.is_default_constructible(self.typeRepresentation.ElementTypes[ix]):
                    return None

            return slots

        if len(otherType.typeRepresentation.ElementTypes) != len(self.typeRepresentation.ElementTypes):
            return None

        return list(range(len(otherType.typeRepresentation.ElementTypes)))

    def _can_convert_from_type(self, sourceType, conversionLevel):
        if issubclass(sourceType.typeRepresentation, (NamedTuple, Tuple)):
            if conversionLevel < ConversionLevel.Upcast:
                return False

            slots = self._indicesInOtherTypeToRead(sourceType)

            if slots is None:
                return False

            someNotTrue = False

            for destIx, sourceIx in enumerate(slots):
                if sourceIx is not None:
                    canConvert = typeWrapper(sourceType.typeRepresentation.ElementTypes[sourceIx]).can_convert_to_type(
                        typeWrapper(self.typeRepresentation.ElementTypes[destIx]),
                        conversionLevel
                    )

                    if canConvert is False:
                        return False

                    if canConvert is not True:
                        someNotTrue = True

            if someNotTrue:
                return "Maybe"
            return True

        if issubclass(sourceType.typeRepresentation, (ListOf, TupleOf, Dict, Set, ConstDict)):
            return "Maybe"

        return False

    def convert_to_self_with_target(self, context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure=False):
        if issubclass(sourceVal.expr_type.typeRepresentation, (NamedTuple, Tuple)):
            if self._can_convert_from_type(sourceVal.expr_type, conversionLevel) is True:
                slots = self._indicesInOtherTypeToRead(sourceVal.expr_type)

                # this is the simple case
                for destIx, sourceIx in enumerate(slots):
                    if sourceIx is not None:
                        sourceVal.refAs(sourceIx).convert_to_type_with_target(
                            targetVal.refAs(destIx),
                            conversionLevel,
                            mayThrowOnFailure
                        )
                    else:
                        targetVal.refAs(destIx).convert_default_initialize()

                return context.constant(True)
            else:
                native = context.converter.defineNativeFunction(
                    f'type_convert({sourceVal.expr_type} -> {targetVal.expr_type}, conversionLevel={conversionLevel.LEVEL})',
                    ('type_convert', sourceVal.expr_type, targetVal.expr_type, conversionLevel.LEVEL),
                    [self.typeRepresentation, sourceVal.expr_type],
                    bool,
                    lambda *args: self.generateConvertOtherTupToSelf(*args, conversionLevel=conversionLevel)
                )

                return context.pushPod(
                    bool,
                    native.call(
                        targetVal.asPointer(),
                        sourceVal
                    )
                )

        return super().convert_to_self_with_target(context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure)

    def generateConvertOtherTupToSelf(self, context, _, targetVal, sourceVal, conversionLevel):
        convertedValues = []
        destT = targetVal.expr_type.typeRepresentation

        slots = self._indicesInOtherTypeToRead(sourceVal.expr_type)

        if slots is None:
            context.pushEffect(
                native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
            )
            return

        for destIx, sourceIx in enumerate(slots):
            if sourceIx is not None:
                val = context.allocateUninitializedSlot(destT.ElementTypes[destIx])

                res = sourceVal.refAs(sourceIx).convert_to_type_with_target(val, conversionLevel)

                convertedValues.append(val)

                with context.ifelse(res.nonref_expr) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.markUninitializedSlotInitialized(val)

                    with ifFalse:
                        context.pushEffect(
                            native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
                        )
            else:
                convertedValues.append(None)

        # if we're here, we can simply copy the values over to the other tuple
        for i in range(len(convertedValues)):
            if convertedValues[i] is not None:
                targetVal.refAs(i).convert_copy_initialize(convertedValues[i])
            else:
                targetVal.refAs(i).convert_default_initialize()

        context.pushEffect(
            native_ast.Expression.Return(arg=native_ast.const_bool_expr(True))
        )

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
            targetVal.convert_copy_initialize(context.constant(len(self.typeRepresentation.ElementTypes) > 0))
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def convert_contains(self, context, instance, toFind):
        if instance.constantValue is not None and toFind.constantValue is not None:
            return context.constant(toFind.constantValue in instance.constantValue)

        def makeContains(context, unused, instance, toFind):
            for i in range(len(self.typeRepresentation.ElementTypes)):
                isEq = instance.refAs(i) == toFind
                if isEq is None:
                    return

                isEq = isEq.toBool()
                if isEq is None:
                    return

                if isEq.constantValue is True:
                    context.pushReturnValue(context.constant(True))
                    return

                with context.ifelse(isEq.nonref_expr) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushReturnValue(context.constant(True))

            context.pushReturnValue(context.constant(False))

        containsMethod = context.converter.defineNativeFunction(
            f'tuple.contains.{self}.{toFind.expr_type}',
            ('tuple.contains', self, toFind.expr_type),
            (self, toFind.expr_type),
            typeWrapper(bool),
            makeContains
        )

        return context.call_typed_call_target(
            containsMethod,
            [instance, toFind]
        )


class NamedTupleWrapper(TupleWrapper):
    def __init__(self, t):
        super().__init__(t)

        self.isSubclassOfNamedTuple = NamedTuple not in t.__bases__

        self.namesToIndices = {n: i for i, n in enumerate(t.ElementNames)}
        self.namesToTypes = {n: t.ElementTypes[i] for i, n in enumerate(t.ElementNames)}

    def isIterable(self):
        return True

    def has_fastnext_iter(self):
        if self.isSubclassOfNamedTuple:
            return "__iter__" in self.typeRepresentation.__dict__
        return False

    def has_intiter(self):
        """Does this type support the 'intiter' format?"""
        if not self.isSubclassOfNamedTuple:
            return False

        return self.has_method('__typed_python_int_iter_size__')

    def has_method(self, methodName):
        method = getattr(self.typeRepresentation, methodName, None)

        if isinstance(method, (types.FunctionType, typed_python._types.Function)):
            return True

        return False

    def convert_intiter_size(self, context, instance):
        """If this type supports intiter, compute the size of the iterator.

        This function will return a TypedExpression(int) or None if it set an exception."""
        return self.convert_method_call(context, instance, "__typed_python_int_iter_size__", [], {})

    def convert_intiter_value(self, context, instance, valueInstance):
        """If this type supports intiter, compute the value of the iterator.

        This function will return a TypedExpression, or None if it set an exception."""
        return self.convert_method_call(context, instance, "__typed_python_int_iter_value__", [], {})

    def convert_attribute(self, context, instance, attribute):
        if attribute in ["replacing"]:
            return instance.changeType(BoundMethodWrapper.Make(self, attribute))

        ix = self.namesToIndices.get(attribute)
        if ix is not None:
            return self.refAs(context, instance, ix)

        if self.isSubclassOfNamedTuple:
            # check if this method exists in the class object
            # this is a little different than what we do with Class objects, where
            # we have already separated the class definition methods into properties
            # methods, staticmethods, etc.
            methodDef = self.typeRepresentation.__dict__.get(attribute)

            if methodDef is not None:
                if isinstance(methodDef, staticmethod):
                    return typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                        context, methodDef.__func__
                    )

                if isinstance(methodDef, property):
                    return typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                        context, methodDef.fget
                    ).convert_call((instance,), {})

                if isinstance(methodDef, (types.FunctionType, typed_python._types.Function)):
                    return instance.changeType(BoundMethodWrapper.Make(self, attribute))

        return context.pushException(
            AttributeError,
            "'%s' object has no attribute '%s'" % (str(self.typeRepresentation), attribute)
        )

    def convert_fastnext(self, context, instance):
        if self.isSubclassOfNamedTuple:
            return self.convert_method_call(context, instance, "__fastnext__", [], {})

        return super().convert_fastnext()

    def convert_pointerTo(self, context, instance):
        if not instance.isReference:
            instance = context.pushMove(instance)

        return instance.asPointer()

    def convert_attribute_pointerTo(self, context, pointerInstance, attribute):
        if attribute in self.namesToIndices:
            return self.refAs(
                context,
                pointerInstance.asReference(),
                self.namesToIndices[attribute],
            ).asPointer()

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0:
            for name in kwargs:
                if name not in self.namesToTypes:
                    context.pushException(TypeError, f"Couldn't initialize type of {self} with an argument named {name}")
                    return

            needsDefaultInitializer = set()

            for name, argType in self.namesToTypes.items():
                if name not in kwargs:
                    if _types.is_default_constructible(argType):
                        needsDefaultInitializer.add(name)
                    else:
                        context.pushException(TypeError, f"Can't default initialize member '{name}' of {self}")
                        return

            uninitializedNamedTuple = context.allocateUninitializedSlot(self)

            for name, expr in kwargs.items():
                actualExpr = expr.convert_to_type(self.namesToTypes[name], ConversionLevel.ImplicitContainers)
                if actualExpr is None:
                    return None

                uninitializedChildElement = self.refAs(context, uninitializedNamedTuple, self.namesToIndices[name])
                uninitializedChildElement.convert_copy_initialize(actualExpr)

            for name in needsDefaultInitializer:
                self.refAs(context, uninitializedNamedTuple, self.namesToIndices[name]).convert_default_initialize()

            context.markUninitializedSlotInitialized(uninitializedNamedTuple)

            # the tuple is now initialized
            return uninitializedNamedTuple

        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self, ConversionLevel.New)

        return super().convert_type_call(context, typeInst, args, kwargs)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname == 'replacing' and not args:
            return context.push(self, lambda newInstance: self.initializeReplacing(context, newInstance, instance, kwargs))

        method = getattr(self.typeRepresentation, methodname, None)

        if isinstance(method, types.FunctionType):
            return context.call_py_function(method, (instance,) + tuple(args), kwargs)

        if isinstance(method, typed_python._types.Function):
            return typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                context, method
            ).convert_call((instance,) + tuple(args), kwargs)

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def initializeReplacing(self, context, toInitialize, existingInstance, kwargs):
        # check if all the passed arguments are in the list of the names
        additional_arguments = sorted(list(set(kwargs.keys()) - set(self.typeRepresentation.ElementNames)))
        if additional_arguments:
            context.pushException(
                ValueError,
                "The arguments list contain names '{}' which are not in the tuple definition."
                .format(", ".join(additional_arguments))
            )
            return None

        for i in range(len(self.subTypeWrappers)):
            field_name = self.typeRepresentation.ElementNames[i]
            field_type = self.typeRepresentation.ElementTypes[i]

            if field_name not in kwargs:
                self.refAs(context, toInitialize, i).convert_copy_initialize(self.refAs(context, existingInstance, i))
            else:
                converted = kwargs[field_name].convert_to_type(field_type, ConversionLevel.Implicit)
                if converted is None:
                    return None
                self.refAs(context, toInitialize, i).convert_copy_initialize(converted)

    def convert_unary_op(self, context, expr, op):
        magic = "__pos__" if op.matches.UAdd else \
            "__neg__" if op.matches.USub else \
            "__invert__" if op.matches.Invert else \
            "__not__" if op.matches.Not else \
            ""

        if self.has_method(magic):
            return self.convert_method_call(context, expr, magic, (), {})

        return super().convert_unary_op(context, expr, op)

    def convert_bin_op(self, context, l, op, r, inplace):
        magic = (
            "__add__" if op.matches.Add else
            "__sub__" if op.matches.Sub else
            "__mul__" if op.matches.Mult else
            "__truediv__" if op.matches.Div else
            "__floordiv__" if op.matches.FloorDiv else
            "__mod__" if op.matches.Mod else
            "__matmul__" if op.matches.MatMult else
            "__pow__" if op.matches.Pow else
            "__lshift__" if op.matches.LShift else
            "__rshift__" if op.matches.RShift else
            "__or__" if op.matches.BitOr else
            "__xor__" if op.matches.BitXor else
            "__and__" if op.matches.BitAnd else
            "__eq__" if op.matches.Eq else
            "__ne__" if op.matches.NotEq else
            "__lt__" if op.matches.Lt else
            "__gt__" if op.matches.Gt else
            "__le__" if op.matches.LtE else
            "__ge__" if op.matches.GtE else
            ""
        )

        magic_inplace = '__i' + magic[2:] if magic and inplace else None

        if magic_inplace and self.has_method(magic_inplace):
            return self.convert_method_call(context, l, magic_inplace, (r,), {})

        if self.has_method(magic):
            return self.convert_method_call(context, l, magic, (r,), {})

        isComparison = (
            op.matches.Eq or op.matches.NotEq or op.matches.Lt
            or op.matches.Gt or op.matches.LtE or op.matches.GtE
        )

        if isComparison:
            return self.convert_comparison(context, l, op, r)
        else:
            return super().convert_bin_op(context, l, op, r, inplace)

    def convert_comparison(self, context, l, op, r):
        """Return the result of a comparison operator when no magic method exists.

        Subclasses can override this to provide specific implementations.
        """
        return super().convert_bin_op(context, l, op, r, False)

    def convert_bin_op_reverse(self, context, r, op, l, inplace):
        if op.matches.In:
            if self.has_method("__contains__"):
                ret = self.convert_method_call(context, r, "__contains__", (l,), {})
                if ret is not None:
                    ret = ret.toBool()
                return ret

        magic = (
            "__radd__" if op.matches.Add else
            "__rsub__" if op.matches.Sub else
            "__rmul__" if op.matches.Mult else
            "__rtruediv__" if op.matches.Div else
            "__rfloordiv__" if op.matches.FloorDiv else
            "__rmod__" if op.matches.Mod else
            "__rmatmul__" if op.matches.MatMult else
            "__rpow__" if op.matches.Pow else
            "__rlshift__" if op.matches.LShift else
            "__rrshift__" if op.matches.RShift else
            "__ror__" if op.matches.BitOr else
            "__rxor__" if op.matches.BitXor else
            "__rand__" if op.matches.BitAnd else
            "__eq__" if op.matches.Eq else
            "__ne__" if op.matches.NotEq else
            "__gt__" if op.matches.Lt else
            "__lt__" if op.matches.Gt else
            "__ge__" if op.matches.LtE else
            "__le__" if op.matches.GtE else
            ""
        )

        if self.has_method(magic):
            return self.convert_method_call(context, r, magic, (l,), {})

        return super().convert_bin_op_reverse(context, r, op, l, inplace)


@TypeFunction
def TupleIterator(T):
    EltT = OneOf(*T.ElementTypes)

    class TupleIterator(Class, Final, __name__=f"TupleIterator({T.__name__})"):
        pos = Member(int, nonempty=True)
        elt = Member(EltT)
        tup = Member(T, nonempty=True)

        def __fastnext__(self):
            self.pos += 1

            if self.pos < len(self.tup):
                self.elt = self.tup[self.pos]
                return pointerTo(self).elt
            else:
                return PointerTo(EltT)()

    return TupleIterator
