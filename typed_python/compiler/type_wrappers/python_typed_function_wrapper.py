#   Copyright 2017-2021 typed_python Authors
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


from typed_python import PointerTo, bytecount, NamedTuple, Class, OneOf, ListOf, TupleOf, Tuple, Set, ConstDict, Dict, Value
from typed_python.compiler.merge_type_wrappers import mergeTypeWrappers
from typed_python._types import is_default_constructible, allForwardTypesResolved
from typed_python.internals import CellAccess
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import TypedTupleMasqueradingAsTuple
from typed_python.compiler.type_wrappers.named_tuple_masquerading_as_dict_wrapper import NamedTupleMasqueradingAsDict
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.native_ast import VoidPtr
from typed_python.compiler.type_wrappers.type_sets import Either, SubclassOf
from typed_python.compiler.type_wrappers.function_signature_calculator import (
    FunctionSignatureCalculator,
    CannotBeDetermined,
    NoReturnTypeSpecified,
    SomeInvalidClassReturnType
)
import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


def forceSignatureConversion(sigFun, result, *args, **kwargs):
    targetType = sigFun(*[type(a) for a in args], **{name: type(arg) for name, arg in kwargs.items()})
    targetType = OneOf(targetType).Types[0]

    return targetType(result)


class PythonTypedFunctionWrapper(Wrapper):
    is_default_constructible = False

    def __init__(self, f):
        if isinstance(f, typed_python._types.Function):
            f = type(f)

        self.closureWrapper = typeWrapper(f.ClosureType)

        self.signatureCalculator = FunctionSignatureCalculator(f)

        super().__init__(f)

    def getNativeLayoutType(self):
        return self.closureWrapper.getNativeLayoutType()

    @property
    def is_pod(self):
        return self.closureWrapper.is_pod

    @property
    def is_empty(self):
        return self.closureWrapper.is_empty

    @property
    def is_pass_by_ref(self):
        return self.closureWrapper.is_pass_by_ref

    def convert_copy_initialize(self, context, expr, other):
        expr.changeType(self.closureWrapper).convert_copy_initialize(
            other.changeType(self.closureWrapper)
        )

    def convert_assign(self, context, expr, other):
        expr.changeType(self.closureWrapper).convert_assign(
            other.changeType(self.closureWrapper)
        )

    def convert_destroy(self, context, expr):
        expr.changeType(self.closureWrapper).convert_destroy()

    def convert_default_initialize(self, context, target):
        # we are only default initializable if our closure has no data.
        assert is_default_constructible(self.closureWrapper.typeRepresentation)
        return target.changeType(self.closureWrapper).convert_default_initialize()

    @staticmethod
    def computeFunctionOverloadReturnType(overload, argTypes, kwargTypes):
        """Determine what the stated return type a function overload will produce if it
        can be determined.

        If we cannot determine it (because we don't know the types of the arguments
        fully, and we have a function signature) then we return CannotBeDetermined. This means
        that it is possible that the function states that it returns as wide as 'object', but
        that we cannot determine anything about it.

        If we can prove that the overload or method will _definitely_ throw an exception
        because of a mismatch, we return SomeInvalidClassReturnType.

        Otherwise, we return a type that is guaranteed to cover the stated return type we
        would get if we actually knew the values flowing through the overload chain and
        explicitly matched them.

        Or, put another way, if we return a type, then that type covers all values that
        will be returned by this overload because of the pattern matching semantics, but is
        not required to be as tight as possible.

        Args:
            overload - a FunctionOverload object
            argTypes - a list or tuple of TypeWrapper objects
            kwargTypes - a dict from name to TypeWrapper

        Returns:
            NoReturnTypeSpecified, a type, CannotBeDetermined, or SomeInvalidClassReturnType
        """
        if not allForwardTypesResolved(overload.functionTypeObject):
            return NoReturnTypeSpecified

        return typeWrapper(overload.functionTypeObject).signatureCalculator.returnTypeForOverload(
            overload.index, argTypes, kwargTypes
        )

    @staticmethod
    def typeSetFor(T):
        """
        Return what we know when the compiler knows a variable has type "T".

        Specifically, we need to return either T itself, or a TypeSet modeling the set of
        results that might be returned if you call type(x) on x:T.

        There is a distinction between the 'type' the compiler knows for a value and the type
        that you'll get when you write type(x).  For instance, if you know a value as

            OneOf(int, float)

        then type(x) will either be 'int' or 'float', but not OneOf(int, float), since no
        specific value ever has type OneOf(int, float).

        Rather than modeling this as 'OneOf(Value(int), Value(float))' (and, by extension,
        forcing us to pass Value(int) to signature functions), we pass instances

            SubclassOf(T)
            Either(T1, T2)

        which model the various specific subtypes we might receive.
        """
        if issubclass(T, Class):
            if T.IsFinal:
                return T

            return SubclassOf(T)

        if issubclass(T, OneOf):
            return Either([PythonTypedFunctionWrapper.typeSetFor(SubT) for SubT in T.Types])

        if T in (int, float, bool, type(None), str, bytes):
            return T

        if issubclass(T, (Value, ListOf, Set, Dict, ConstDict, TupleOf, Tuple, NamedTuple)):
            return T

        return SubclassOf(object)

    @staticmethod
    def closurePathToCellValue(closurePath, closureExpr):
        """Extract the argument to pass to a function based on the closure path and the closure itself."""
        expr = closureExpr

        for pathEltIx, pathElt in enumerate(closurePath):
            if isinstance(pathElt, int):
                expr = expr.refAs(pathElt)
            elif isinstance(pathElt, str):
                expr = expr.convert_attribute(pathElt, nocheck=True)
            elif pathElt is CellAccess:
                if pathEltIx == len(closurePath) - 1:
                    pass
                else:
                    expr = expr.expr_type.refHeld(expr)
            elif isinstance(pathElt, type):
                expr = expr.changeType(typeWrapper(pathElt))
            else:
                assert False, "invalid cell path"

        return expr

    @staticmethod
    def closurePathToCellType(closurePath, closureType):
        """Calculare a cell type from  a ClosureVariablePath and a closure type.

        The closure determines how a given function stores the data in its closure.
        The 'path' determines how a given variable in the function is stored in the
        closure. This function determines the actual cell type.
        """
        t = closureType

        for pathEltIx, pathElt in enumerate(closurePath):
            if isinstance(pathElt, int):
                t = t.ElementTypes[pathElt]
            elif isinstance(pathElt, str):
                if issubclass(t, NamedTuple):
                    t = t.ElementTypes[t.ElementNames.find(pathElt)]
                elif issubclass(t, Class):
                    if pathElt not in t.MemberNames:
                        # this can happen when be bind a variable to a closure
                        # but it's never been assigned, and so we don't have a type for it.
                        return None

                    t = t.MemberTypes[t.MemberNames.index(pathElt)]
                else:
                    assert False
            elif pathElt is CellAccess:
                if pathEltIx == len(closurePath) - 1:
                    pass
                else:
                    t = t.HeldType
            elif isinstance(pathElt, type):
                t = pathElt
            else:
                assert False, "invalid cell path"

        return t

    def convert_nocompile_call(self, context, instance, args, kwargs):
        if len(self.typeRepresentation.overloads) != 1:
            raise Exception("Can't mark multi-overload functions nocompile yet.")

        overload = self.typeRepresentation.overloads[0]

        actualArgTypes, actualKwargTypes = context.computeOverloadSignature(
            overload,
            [a.expr_type for a in args],
            {name: k.expr_type for name, k in kwargs.items()}
        )

        returnType = self.computeFunctionOverloadReturnType(overload, actualArgTypes, actualKwargTypes)

        if returnType is CannotBeDetermined:
            return instance.convert_to_type(object, ConversionLevel.Signature).convert_call(args, kwargs)

        if returnType is NoReturnTypeSpecified:
            returnType = object

        argsAsObjects = []
        for a in args:
            argsAsObjects.append(a.convert_to_type(object, ConversionLevel.Signature))
            if argsAsObjects[-1] is None:
                return None

        kwargsAsObjects = {}

        for k, a in kwargs.items():
            kwargsAsObjects[k] = a.convert_to_type(object, ConversionLevel.Signature)

            if kwargsAsObjects[k] is None:
                return None

        arguments = []
        kwarguments = []

        for a in argsAsObjects:
            arguments.append(a.nonref_expr.cast(VoidPtr))

        for kwargName, kwargVal in kwargsAsObjects.items():
            kwarguments.append(kwargVal.nonref_expr.cast(VoidPtr))
            kwarguments.append(native_ast.const_utf8_cstr(kwargName))

        if not instance.isReference:
            instance = context.pushMove(instance)

        result = context.push(
            object,
            lambda oPtr:
                oPtr.expr.store(
                    runtime_functions.call_func_as_pyobj.call(
                        context.getTypePointer(self.typeRepresentation).cast(VoidPtr),
                        instance.expr.cast(VoidPtr),
                        native_ast.const_int_expr(len(arguments)),
                        native_ast.const_int_expr(len(kwargsAsObjects)),
                        *arguments,
                        *kwarguments,
                    ).cast(typeWrapper(object).getNativeLayoutType())
                )
        )

        return result.convert_to_type(returnType, ConversionLevel.Implicit)

    def convert_list_comprehension(self, context, instance):
        """We are a generator. Convert a "list comprehension".

        This means we take our code and execute it directly, replacing all 'yield' statements
        with an append to an intermediate list, and then returning the final list comprehension
        at the end.
        """
        from typed_python.compiler.function_conversion_context import ListComprehensionConversionContext

        return self.convert_comprehension(context, instance, ListComprehensionConversionContext)

    def convert_set_comprehension(self, context, instance):
        """We are a generator. Convert a "set comprehension".

        This means we take our code and execute it directly, replacing all 'yield' statements
        with an append to an intermediate set, and then returning the final set comprehension
        at the end.
        """
        from typed_python.compiler.function_conversion_context import SetComprehensionConversionContext

        return self.convert_comprehension(context, instance, SetComprehensionConversionContext)

    def convert_dict_comprehension(self, context, instance):
        """We are a generator. Convert a "dict comprehension".

        This means we take our code and execute it directly, replacing all 'yield' statements
        with an append to an intermediate dict, and then returning the final dict comprehension
        at the end.
        """
        from typed_python.compiler.function_conversion_context import DictComprehensionConversionContext

        return self.convert_comprehension(context, instance, DictComprehensionConversionContext)

    def convert_comprehension(self, context, instance, ConvertionContextType):
        # we should have exactly one overload that takes no arguments and has no return type
        assert len(self.typeRepresentation.overloads) == 1
        overload = self.typeRepresentation.overloads[0]
        assert not overload.args
        assert overload.returnType is None
        assert overload.signatureFunction is None

        # build the actual functions
        argsToPass = context.buildFunctionArguments(
            overload,
            [],
            {}
        )

        closureType = self.typeRepresentation.ClosureType

        # import this wrapper. Note that we have to import it here to break the import cycles.
        # there's definitely a better way to organize this code.

        singleConvertedOverload = context.functionContext.converter.convert(
            overload.name,
            overload.functionCode,
            overload.realizedGlobals,
            overload.functionGlobals,
            list(overload.funcGlobalsInCells),
            list(overload.closureVarLookups),
            [typeWrapper(self.closurePathToCellType(path, closureType)) for path in overload.closureVarLookups.values()],
            None,
            conversionType=ConvertionContextType
        )

        if not singleConvertedOverload:
            # it's still doing type inference
            context.pushException(
                TypeError,
                "Can't build the list comprehension converter."
            )
            return

        closureTuple = instance.changeType(self.closureWrapper)
        closureArgs = [self.closurePathToCellValue(path, closureTuple) for path in overload.closureVarLookups.values()]

        # check if any closureArgs are None, which means that converting them threw an exception.
        for a in closureArgs:
            if a is None:
                return

        return context.call_typed_call_target(
            singleConvertedOverload,
            closureArgs + argsToPass
        )

    def checkInvalidClassReturn(self, context, overloadIx, args, kwargs):
        """Generate code that determines if caling overloadIx with args/kwargs would throw.

        Returns True if it _definitely_ fails, CannotBeDetermined if its unknown.
        """
        argTypes = [a.expr_type for a in args]
        kwargTypes = {k: v.expr_type for k, v in kwargs.items()}

        if not self.signatureCalculator.overloadCouldThrowInvalidReturnType(
            overloadIx, argTypes, kwargTypes
        ):
            return False

        possibleInvalidChecks = self.signatureCalculator.overloadInvalidSignatures(
            overloadIx,
            argTypes,
            kwargTypes
        )

        if possibleInvalidChecks is CannotBeDetermined:
            return CannotBeDetermined

        for check in sorted(
            possibleInvalidChecks,
            key=lambda check: (check.subOverload.index, check.baseOverload.index)
        ):
            if self.checkInvalidClassReturnTypePredicate(context, args, kwargs, check) is True:
                return True

        return False

    def checkInvalidClassReturnTypePredicate(self, context, args, kwargs, predicate):
        """Check if 'args' and 'kwargs' match 'predicate', and if so, throw an exception.

        Predicate takes the form of an InvalidClassReturnTypePredicate, which is a pair of
        function overloads, such that if we match both of them, then we'll

        Args:
            args - a list of actual TypedExpression objects
            kwargs - a dict of TypedExpressin objects
            predicate - an InvalidClassReturnTypePredicate

        Returns:
            True if it definitely throws, False if it definitely doesn't, or "Maybe"

            As a side-effect, it pushes exception checks onto 'context'.
        """
        canConvertToChild = self.checkIfCanCallOverload(context, args, kwargs, predicate.subOverload)

        if canConvertToChild.constantValue is False:
            return False

        canConvertToParent = self.checkIfCanCallOverload(context, args, kwargs, predicate.baseOverload)

        if canConvertToParent.constantValue is False:
            return False

        with context.ifelse(canConvertToChild.nonref_expr) as (ifTrue, ifFalse):
            with ifTrue:
                with context.ifelse(canConvertToParent.nonref_expr) as (ifTrue2, ifFalse2):
                    with ifTrue2:
                        predicate.pushInvalidMethodReturnType(context, self.typeRepresentation.__name__)

        if canConvertToChild.constantValue is True and canConvertToParent.constantValue is True:
            return True

        return "Maybe"

    def convert_call(self, context, left, args, kwargs):
        if left is None:
            assert bytecount(self.typeRepresentation) == 0
            left = context.push(self, lambda expr: None)

        # check if we are marked 'nocompile' in which case we convert to 'object' and dispatch
        # to the interpreter. We do retain any typing information on the return type, however.
        if self.typeRepresentation.isNocompile:
            return self.convert_nocompile_call(context, left, args, kwargs)

        argTypes = [a.expr_type for a in args]
        kwargTypes = {k: v.expr_type for k, v in kwargs.items()}

        # check if there is exactly one overload we could match
        overloadAndConversionLevel = PythonTypedFunctionWrapper.pickSingleOverloadForCall(
            self.typeRepresentation, argTypes, kwargTypes
        )

        if overloadAndConversionLevel is not None:
            overload, conversionLevel = overloadAndConversionLevel

            actualArgTypes, actualKwargTypes = context.computeOverloadSignature(overload, argTypes, kwargTypes)

            actualArgs = [args[i].convert_to_type(actualArgTypes[i], conversionLevel) for i in range(len(actualArgTypes))]
            actualKwargs = {name: kwargs[name].convert_to_type(actualKwargTypes[name], conversionLevel) for name in kwargs}

            # no conversion should ever fail because we are already guaranteed that this one signature
            # is the one we should definitely match.
            assert not any(x is None for x in actualArgs)
            assert not any(x is None for x in actualKwargs.values())

            # build the actual functions
            argsToPass = context.buildFunctionArguments(
                overload,
                actualArgs,
                actualKwargs
            )

            closureType = self.typeRepresentation.ClosureType

            # determine the returnType based on the signature if we have a signatureFunction.
            returnType = self.signatureCalculator.returnTypeForOverload(overload.index, argTypes, kwargTypes)

            if returnType is CannotBeDetermined:
                return left.convert_to_type(object, ConversionLevel.Signature).convert_call(args, kwargs)

            canBail = self.checkInvalidClassReturn(context, overload.index, args, kwargs)

            if canBail is CannotBeDetermined:
                return left.convert_to_type(object, ConversionLevel.Signature).convert_call(args, kwargs)

            if canBail:
                return

            if returnType is SomeInvalidClassReturnType:
                # we should already have forced ourselves to throw an exception
                return

            if returnType is NoReturnTypeSpecified:
                returnType = None

            # just one overload will do. We can just instantiate this particular function
            # with a signature that comes from the method overload signature itself.
            singleConvertedOverload = context.functionContext.converter.convert(
                overload.name,
                overload.functionCode,
                overload.realizedGlobals,
                overload.functionGlobals,
                list(overload.funcGlobalsInCells),
                list(overload.closureVarLookups),
                [typeWrapper(self.closurePathToCellType(path, closureType)) for path in overload.closureVarLookups.values()]
                + [a.expr_type for a in argsToPass],
                returnType
            )

            if not singleConvertedOverload:
                context.pushException(
                    TypeError,
                    f"{self.typeRepresentation} cannot find a valid overload with arguments of type "
                    "(" + ",".join([str(x.expr_type) for x in args] + [k+"="+str(v.expr_type) for k, v in kwargs.items()]) + ")"
                )

                return

            closureTuple = left.changeType(self.closureWrapper)
            closureArgs = [self.closurePathToCellValue(path, closureTuple) for path in overload.closureVarLookups.values()]

            # check if any closureArgs are None, which means that converting them threw an exception.
            for a in closureArgs:
                if a is None:
                    return

            res = context.call_typed_call_target(
                singleConvertedOverload,
                closureArgs + argsToPass
            )

            return res

        # there are multiple possible overloads. We'll need to check each one in turn to see
        # which one to trigger.
        callTarget = self.compileCall(
            context.functionContext.converter,
            None,
            argTypes,
            kwargTypes,
            True
        )

        if not callTarget:
            context.pushException(
                TypeError,
                f"{self.typeRepresentation} cannot find a valid overload with arguments of type "
                "(" + ",".join([str(x.expr_type) for x in args] + [k+"="+str(v.expr_type) for k, v in kwargs.items()]) + ")"
            )
            return

        return context.call_typed_call_target(callTarget, [left] + list(args) + list(kwargs.values()))

    @staticmethod
    def pickSpecializationTypeFor(overloadArg, argType: Wrapper):
        """Given that we are passing 'argType' to 'overloadArg', figure out what Wrapper we
        ought to use.
        """
        if overloadArg.isStarArg:
            assert isinstance(argType, TypedTupleMasqueradingAsTuple)

            if overloadArg.typeFilter is not None:
                return TypedTupleMasqueradingAsTuple(
                    typed_python.Tuple(*[overloadArg.typeFilter for _ in argType.typeRepresentation.ElementTypes])
                )
            else:
                return argType

        elif overloadArg.isKwarg:
            assert isinstance(argType, NamedTupleMasqueradingAsDict)

            if overloadArg.typeFilter is not None:
                return NamedTupleMasqueradingAsDict(
                    typed_python.NamedTuple(
                        **{name: overloadArg.typeFilter for name in argType.typeRepresentation.ElementNames}
                    )
                )
            else:
                return argType
        else:
            if overloadArg.typeFilter is not None:
                return typeWrapper(overloadArg.typeFilter)
            return argType

    def compileCall(
        self,
        converter,
        returnType,
        argTypes,
        kwargTypes,
        provideClosureArgument,
        firstArgConversion=None
    ):
        """Compile this function being called with a particular signature.

        Args:
            converter - the PythonToNativeConverter that needs the concrete definition.
            returnType - the typeWrapper of what we're returning, or None if we don't know
            argTypes - (ListOf(wrapper)) a the actual concrete type wrappers for the arguments
                we're passing.
            kwargTypes - (Dict(str, wrapper)) a the actual concrete type wrappers for the keyword
                arguments we're passing.
            provideClosureArgument - if True, then the first argument is of our closure type.
                If false, then our closure type must be empty, and no argument for it is provided.
            firstArgConversion - if not None, then call this with a TypedExpression containing
                the first argument, and use that instead.
        Returns:
            a TypedCallTarget, or None
        """
        if returnType is None:
            # we have to take the union of the return types we might be dispatching to
            possibleTypes = PythonTypedFunctionWrapper.determinePossibleReturnTypes(
                converter,
                self.typeRepresentation,
                argTypes,
                kwargTypes
            )

            returnType = mergeTypeWrappers(possibleTypes)

            if returnType is None:
                return None
        else:
            returnType = typeWrapper(returnType)

        argNames = [None for _ in argTypes] + list(kwargTypes)

        return converter.defineNativeFunction(
            f'implement_function.{self}{argTypes}.{kwargTypes}->{returnType}',
            ('implement_function.', self, returnType, self, tuple(argTypes), tuple(kwargTypes.items())),
            ([self] if provideClosureArgument else []) + list(argTypes) + list(kwargTypes.values()),
            returnType,
            lambda context, outputVar, *args: (
                self.generateMethodImplementation(
                    context,
                    returnType,
                    args,
                    argNames,
                    provideClosureArgument,
                    firstArgConversion=firstArgConversion
                )
            )
        )

    @staticmethod
    def overloadMatchesSignature(overload, argTypes, kwargTypes, conversionLevel):
        """Is it possible we could dispatch to FunctionOverload 'overload' with 'argTypes/kwargTypes'?

        Returns:
            True if we _definitely_ match
            "Maybe" if we might match
            False if we definitely don't match the arguments.
        """
        assert overload is not None
        ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

        actualArgTypes = ExpressionConversionContext.computeOverloadSignature(overload, argTypes, kwargTypes)

        # check if there's no way to map the arguments
        if actualArgTypes is None:
            return False

        neededArgTypes, neededKwargTypes = actualArgTypes

        if actualArgTypes is None:
            return False

        argTypes = argTypes + list(kwargTypes.values())
        neededArgTypes = neededArgTypes + [neededKwargTypes[name] for name in kwargTypes]

        allTrue = True
        for callType, neededType in zip(argTypes, neededArgTypes):
            canConvert = typeWrapper(callType).can_convert_to_type(
                typeWrapper(neededType),
                conversionLevel
            )

            if canConvert is False:
                return False
            elif canConvert == "Maybe":
                allTrue = False

        if allTrue:
            return allTrue
        else:
            return "Maybe"

    @staticmethod
    def determinePossibleReturnTypes(converter, func, argTypes, kwargTypes):
        returnTypes = []

        for conversionLevel in ConversionLevel.functionConversionSequence():
            for o in func.overloads:
                # check each overload that we might match.
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(o, argTypes, kwargTypes, conversionLevel)

                if mightMatch is False:
                    pass
                else:
                    ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

                    retType = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(
                        o, argTypes, kwargTypes
                    )

                    if retType is CannotBeDetermined:
                        retType = object

                    if retType is SomeInvalidClassReturnType:
                        pass
                    elif retType is not NoReturnTypeSpecified:
                        returnTypes.append(retType)
                    else:
                        # we need to infer the return type
                        actualArgTypes = ExpressionConversionContext.computeFunctionArgumentTypeSignature(
                            o,
                            argTypes,
                            kwargTypes
                        )

                        callTarget = converter.convert(
                            o.name,
                            o.functionCode,
                            o.realizedGlobals,
                            o.functionGlobals,
                            list(o.funcGlobalsInCells),
                            list(o.closureVarLookups),
                            [typeWrapper(PythonTypedFunctionWrapper.closurePathToCellType(path, func.ClosureType))
                             for path in o.closureVarLookups.values()] + actualArgTypes,
                            None
                        )

                        if callTarget is not None and callTarget.output_type is not None:
                            returnTypes.append(callTarget.output_type)

                    if mightMatch is True:
                        return returnTypes

        return returnTypes

    @staticmethod
    def pickSingleOverloadForCall(func, argTypes, kwargTypes):
        """See if there is at most one function overload that could match 'argTypes/kwargTypes'
        in our chain.

        Returns:
            None, or a tuple (FunctionOverload, conversionLevel: ConversionLevel) indicating that one single overload
            is the one version of this function we should try to match.
        """
        possibleMatches = []

        for conversionLevel in ConversionLevel.functionConversionSequence():
            for o in func.overloads:
                # check each overload that we might match.
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(
                    o, argTypes, kwargTypes, conversionLevel
                )

                if mightMatch is False:
                    pass
                elif mightMatch is True:
                    # if we matched something else
                    if possibleMatches:
                        return None

                    return (o, conversionLevel)
                else:
                    possibleMatches.append((o, conversionLevel))

                    if len(possibleMatches) > 1:
                        return None

        # if we had exactly one 'maybe', then that's the one to use.
        if len(possibleMatches) == 1:
            return possibleMatches[0]

        return None

    def checkIfCanCallOverload(self, context, args, kwargs, overload):
        """Return a typed_expression indicating whether we can call overload with args/kwargs

        This function actually builds a subfunction that does the check.
        """
        argNames = [None for _ in args] + list(kwargs)
        argTypes = [a.expr_type for a in args]
        kwargTypes = {k: v.expr_type for k, v in kwargs.items()}

        conversionLevel = ConversionLevel.ImplicitContainers

        possiblyMatches = self.checkIfOverloadArgsDefinitelyMatch(
            context, overload, conversionLevel, args, kwargs
        )

        if possiblyMatches != "Maybe":
            return context.constant(possiblyMatches)

        def makeOverloadImplementor(overload, conversionLevel):
            return lambda context, _, *args: self.generateOverloadArgsMatchCheck(
                context, overload, conversionLevel, args, argNames
            )

        overloadIndex = overload.index

        testSingleOverloadForm = context.converter.defineNativeFunction(
            f'check_can_call_overload.{self}.{overloadIndex}.{conversionLevel.LEVEL}.{argTypes}.{kwargTypes}',
            ('check_can_call_overload', self, overloadIndex, conversionLevel.LEVEL,
                tuple(argTypes), tuple(kwargTypes.items())),
            list(argTypes) + list(kwargTypes.values()),
            typeWrapper(bool),
            makeOverloadImplementor(
                overload,
                conversionLevel
            )
        )

        return context.call_typed_call_target(
            testSingleOverloadForm,
            args + list(kwargs.values())
        )

    def generateMethodImplementation(
        self, context, returnType, args, argNames,
        provideClosureArgument, firstArgConversion=None
    ):
        """Generate native code that calls us with a given return type and set of arguments.

        We try each overload with successively stronger type conversions. The first one that
        succeeds gets to produce the output.

        Args:
            context - an ExpressionConversionContext
            returnType - the output type we are expecting to return. This will be the union
                of the return types of all the overloads that might participate in this dispatch.
            args - the typed_expression for all of our actual arguments, which in this case
                are the closure (if provideClosureArgument), the instance,
                and then the actual arguments we want to convert.
            argNames - a list with one element per args, containing None for each positional
                argument, or the name of the argument if passed as a keyword argument.
            provideClosureArgument - if True, then the first argument is of our closure type.
                If false, then our closure type must be empty, and no argument for it is provided.
            firstArgConversion - if not None, replace the first argument with the result
                of calling this function with the first argument as given.
        """
        func = self.typeRepresentation

        if provideClosureArgument:
            closureArg = args[0]
            args = args[1:]
        else:
            closureArg = None

        if firstArgConversion is not None:
            replacement = firstArgConversion(args[0])
            if replacement is None:
                args = args[1:]
                argNames = argNames[1:]
            else:
                args = (replacement,) + tuple(args[1:])

        argTypes = [a.expr_type for i, a in enumerate(args) if argNames[i] is None]
        kwargTypes = {argNames[i]: a.expr_type for i, a in enumerate(args) if argNames[i] is not None}

        def makeOverloadImplementor(overload, conversionLevel, returnType, forceConvertToSignatureFunc):
            return lambda context, _, outputVar, *args: self.generateOverloadImplement(
                context, overload, returnType, conversionLevel, outputVar, args,
                argNames, provideClosureArgument, forceConvertToSignatureFunc
            )

        for conversionLevel in [
            ConversionLevel.Signature,
            ConversionLevel.Upcast,
            ConversionLevel.UpcastContainers,
            ConversionLevel.Implicit
        ]:
            for overloadIndex, overload in enumerate(func.overloads):
                mightMatch = self.overloadMatchesSignature(overload, argTypes, kwargTypes, conversionLevel)

                if mightMatch is not False:
                    overloadRetType = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(
                        overload,
                        argTypes,
                        kwargTypes
                    )

                    forceConvertToSignatureFunc = False

                    if overloadRetType is CannotBeDetermined:
                        assert returnType.typeRepresentation is object, returnType
                        overloadRetType = object
                        forceConvertToSignatureFunc = True
                    elif overloadRetType is SomeInvalidClassReturnType:
                        # this method produces the wrong type
                        overloadRetType = type(None)
                    elif overloadRetType is NoReturnTypeSpecified:
                        overloadRetType = returnType.typeRepresentation

                    testSingleOverloadForm = context.converter.defineNativeFunction(
                        f'implement_overload.{self}.{overloadIndex}.{conversionLevel.LEVEL}.{argTypes}.{kwargTypes}->{overloadRetType}',
                        ('implement_overload', self, overloadIndex, conversionLevel.LEVEL,
                         overloadRetType, tuple(argTypes), tuple(kwargTypes.items())),
                        [PointerTo(overloadRetType)] + ([self] if provideClosureArgument else [])
                        + list(argTypes) + list(kwargTypes.values()),
                        typeWrapper(bool),
                        makeOverloadImplementor(
                            overload,
                            conversionLevel,
                            overloadRetType,
                            forceConvertToSignatureFunc
                        )
                    )

                    outputSlot = context.allocateUninitializedSlot(overloadRetType)

                    successful = context.call_typed_call_target(
                        testSingleOverloadForm,
                        (outputSlot.changeType(PointerTo(overloadRetType), False),)
                        + tuple([closureArg] if provideClosureArgument else [])
                        + args
                    )

                    with context.ifelse(successful.nonref_expr) as (ifTrue, ifFalse):
                        with ifTrue:
                            context.markUninitializedSlotInitialized(outputSlot)

                            # upcast the result
                            actualResult = outputSlot.convert_to_type(returnType, ConversionLevel.Signature)

                            if actualResult is not None:
                                context.pushReturnValue(actualResult)

                    # if we definitely match, we can return early
                    if mightMatch is True:
                        if successful.constantValue is not True:
                            context.pushException(
                                TypeError,
                                f"Thought we found overload for {self} matching {argTypes} and "
                                f"{kwargTypes}, but it didn't accept the arguments."
                            )
                        return

        # generate a cleanup handler for the cases where we don't match a method signature.
        # this should actually be hitting the interpreter instead.
        context.pushException(TypeError, f"Failed to find an overload for {self} matching {argTypes} and {kwargTypes}")

    def checkIfOverloadArgsDefinitelyMatch(
        self,
        context,
        overload,
        conversionLevel,
        args,
        kwargs
    ):
        """
        Determine whether args/kwargs can match 'overload'

        Returns False, True, or "Maybe"
        """
        argTypes = [a.expr_type for a in args]
        kwargTypes = {k: v.expr_type for k, v in kwargs.items()}

        argAndKwargTypes = context.computeOverloadSignature(overload, argTypes, kwargTypes)

        argConversionMightNotBeSuccessful = False

        argNames = [None for _ in args] + list(kwargs)
        args = list(args) + list(kwargs.values())

        for argIx, argExpr in enumerate(args):
            if argNames[argIx] is None:
                argType = argAndKwargTypes[0][argIx]
            else:
                argType = argAndKwargTypes[1][argNames[argIx]]

            argType = typeWrapper(argType)

            if argExpr.expr_type == argType:
                # nothing to do
                pass
            else:
                successful = argExpr.expr_type.can_convert_to_type(argType, conversionLevel)

                if successful is False:
                    return False

                if successful == "Maybe":
                    argConversionMightNotBeSuccessful = True

        if argConversionMightNotBeSuccessful:
            return "Maybe"

        return True

    def generateOverloadArgsMatchCheck(
        self,
        context,
        overload,
        conversionLevel,
        args,
        argNames
    ):
        """Produce the code that implements this specific overload.

        The generated code returns control flow with a True if it fills out the 'outputVar'
        with data, and False otherwise.

        Args:
            context - an ExpressionConversionContext
            overload - the FunctionOverload we're trying to convert.
            retType - the type we're actually planning on returning here.
            conversionLevel - the level at which we should convert our arguments
            outputVar - a TypedExpression(PointerTo(returnType)) we're supposed to initialize.
            args - the arguments to pass to the method (including the closure if necessary and the instance)
            argNames - a list with one element per args, containing None for each positional
                argument, or the name of the argument if passed as a keyword argument.
        """
        assert len(args) == len(argNames), (args, argNames)

        argTypes = [a.expr_type for i, a in enumerate(args) if argNames[i] is None]
        kwargTypes = {argNames[i]: a.expr_type for i, a in enumerate(args) if argNames[i] is not None}

        argAndKwargTypes = context.computeOverloadSignature(overload, argTypes, kwargTypes)

        convertedArgs = []
        convertedKwargs = {}

        argConversionMightNotBeSuccessful = False

        for argIx, argExpr in enumerate(args):
            if argNames[argIx] is None:
                argType = argAndKwargTypes[0][argIx]
            else:
                argType = argAndKwargTypes[1][argNames[argIx]]

            argType = typeWrapper(argType)

            if argExpr.expr_type == argType:
                # nothing to do
                convertedArg = argExpr
            else:
                convertedArg = context.allocateUninitializedSlot(argType)

                successful = argExpr.convert_to_type_with_target(convertedArg, conversionLevel)

                if successful.isConstant:
                    if successful.constantValue:
                        context.markUninitializedSlotInitialized(convertedArg)
                    else:
                        # we can return early
                        context.pushTerminal(
                            native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
                        )
                        return
                else:
                    argConversionMightNotBeSuccessful = True

                    with context.ifelse(successful.nonref_expr) as (ifTrue, ifFalse):
                        with ifFalse:
                            context.pushTerminal(
                                native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
                            )

                        with ifTrue:
                            context.markUninitializedSlotInitialized(convertedArg)

            if argNames[argIx] is None:
                convertedArgs.append(convertedArg)
            else:
                convertedKwargs[argNames[argIx]] = convertedArg

        context.pushReturnValue(context.constant(True))

        if not argConversionMightNotBeSuccessful:
            context.functionContext.functionMetadata.setConstantReturnValue(
                True
            )

    def generateOverloadImplement(
        self,
        context,
        overload,
        retType,
        conversionLevel,
        outputVar,
        args,
        argNames,
        provideClosureArgument,
        forceConvertToSignatureFunc
    ):
        """Produce the code that implements this specific overload.

        The generated code returns control flow with a True if it fills out the 'outputVar'
        with data, and False otherwise.

        Args:
            context - an ExpressionConversionContext
            overload - the FunctionOverload we're trying to convert.
            retType - the type we're actually planning on returning here.
            conversionLevel - the level at which we should convert our arguments
            outputVar - a TypedExpression(PointerTo(returnType)) we're supposed to initialize.
            args - the arguments to pass to the method (including the closure if necessary and the instance)
            argNames - a list with one element per args, containing None for each positional
                argument, or the name of the argument if passed as a keyword argument.
            provideClosureArgument - if True, then the first argument is of our closure type.
                If false, then our closure type must be empty, and no argument for it is provided.
            forceConvertToSignatureFunc - if True, then force the result to get passed through the
                overload's signature function
        """
        if provideClosureArgument:
            closureArg = args[0]
            args = args[1:]
        else:
            closureArg = None

        assert len(args) == len(argNames)

        argTypes = [a.expr_type for i, a in enumerate(args) if argNames[i] is None]
        kwargTypes = {argNames[i]: a.expr_type for i, a in enumerate(args) if argNames[i] is not None}

        unnamedArgs = [args[i] for i in range(len(argNames)) if argNames[i] is None]
        namedArgs = {argNames[i]: args[i] for i in range(len(argNames)) if argNames[i] is not None}

        canBail = self.checkInvalidClassReturn(context, overload.index, unnamedArgs, namedArgs)

        if canBail is True:
            return

        argAndKwargTypes = context.computeOverloadSignature(overload, argTypes, kwargTypes)

        convertedArgs = []
        convertedKwargs = {}

        argConversionMightNotBeSuccessful = False

        for argIx, argExpr in enumerate(args):
            if argNames[argIx] is None:
                argType = argAndKwargTypes[0][argIx]
            else:
                argType = argAndKwargTypes[1][argNames[argIx]]

            argType = typeWrapper(argType)

            if argExpr.expr_type == argType:
                # nothing to do
                convertedArg = argExpr
            else:
                convertedArg = context.allocateUninitializedSlot(argType)

                successful = argExpr.convert_to_type_with_target(convertedArg, conversionLevel)

                if successful.isConstant:
                    if successful.constantValue:
                        context.markUninitializedSlotInitialized(convertedArg)
                    else:
                        # we can return early
                        context.pushTerminal(
                            native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
                        )
                        return
                else:
                    argConversionMightNotBeSuccessful = True

                    with context.ifelse(successful.nonref_expr) as (ifTrue, ifFalse):
                        with ifFalse:
                            context.pushTerminal(
                                native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
                            )

                        with ifTrue:
                            context.markUninitializedSlotInitialized(convertedArg)

            if argNames[argIx] is None:
                convertedArgs.append(convertedArg)
            else:
                convertedKwargs[argNames[argIx]] = convertedArg

        if outputVar.expr_type.typeRepresentation.ElementType != retType:
            raise Exception(f"Output type mismatch: {outputVar.expr_type.typeRepresentation} vs {retType}")

        if canBail is CannotBeDetermined:
            # we don't know whether we need to throw an exception here because we don't
            # have enough information to call all the parent signature functions.
            # so instead, we just defer to the interpreter for this call.
            if provideClosureArgument and closureArg.expr_type == self:
                instance = closureArg
            elif provideClosureArgument:
                instance = context.constant(self.typeRepresentation).convert_call([closureArg], {})
            else:
                instance = context.constant(self.typeRepresentation())

            if instance is None:
                return None

            res = instance.convert_to_type(object, ConversionLevel.Signature).convert_call(convertedArgs, convertedKwargs)

            if res is not None:
                res = res.convert_to_type(retType, ConversionLevel.Signature)
        else:
            res = context.call_overload(
                overload,
                closureArg,
                convertedArgs,
                convertedKwargs,
                typeWrapper(retType)
            )

        if res is None:
            context.pushException(Exception, "unreachable")
            return

        if forceConvertToSignatureFunc:
            assert retType is object

            res = context.constant(forceSignatureConversion, allowArbitrary=True).convert_call(
                [
                    context.constant(overload.signatureFunction, allowArbitrary=True).convert_to_type(object, ConversionLevel.Signature),
                    res
                ] + list(convertedArgs),
                convertedKwargs
            )

        outputVar.changeType(typeWrapper(retType), True).convert_copy_initialize(res)

        context.pushReturnValue(context.constant(True))

        if not argConversionMightNotBeSuccessful:
            context.functionContext.functionMetadata.setConstantReturnValue(
                True
            )
