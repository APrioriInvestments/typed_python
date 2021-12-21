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

from typed_python import OneOf, Value, ListOf, Set, Dict, ConstDict, TupleOf, Tuple, NamedTuple, Class

from typed_python.compiler.type_wrappers.type_sets import Either, SubclassOf
import typed_python.compiler
from typed_python._types import canConvertToTrivially
from typed_python.compiler.conversion_level import ConversionLevel

from typed_python.compiler.merge_type_wrappers import mergeTypeWrappers


typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


class NoReturnTypeSpecified:
    """Singleton indicating that the function didn't actually specify a return type."""


class SomeInvalidClassReturnType:
    """Singleton used to indicate we definitely throw some kind of match exception, but not sure which."""


class InvalidClassReturnTypePredicate:
    def __init__(
        self,
        baseOverload,
        baseArgTypes,
        baseKwargTypes,
        promisedType,
        subOverload,
        subArgTypes,
        subKwargTypes,
        proposedType,
    ):
        """Indicate that if we match subArg/Kwarg types and also baseArg/Kwarg types,
        then we would want to throw an exception."""
        self.baseOverload = baseOverload
        self.baseArgTypes = baseArgTypes
        self.baseKwargTypes = baseKwargTypes
        self.promisedType = promisedType
        self.subOverload = subOverload
        self.subArgTypes = subArgTypes
        self.subKwargTypes = subKwargTypes
        self.proposedType = proposedType

        assert promisedType is not SomeInvalidClassReturnType
        assert proposedType is not SomeInvalidClassReturnType

    def __eq__(self, other):
        if not isinstance(other, InvalidClassReturnTypePredicate):
            return False

        return self.asTup() == other.asTup()

    def __hash__(self):
        return hash(self.asTup())

    def asTup(self):
        return (
            self.baseOverload,
            tuple(self.baseArgTypes),
            tuple(self.baseKwargTypes.items()),
            self.promisedType,
            self.subOverload,
            tuple(self.subArgTypes),
            tuple(self.subKwargTypes.items()),
            self.proposedType
        )

    def __repr__(self):
        return (
            f"InvalidClassReturnTypePredicate("
            f"[{self.baseArgTypes}/{self.baseKwargTypes}],"
            f"[{self.subArgTypes}/{self.subKwargTypes}],"
            f" -> {self.promisedType} != {self.proposedType}"
        )

    def pushInvalidMethodReturnType(self, context, methodName):
        return context.pushException(
            TypeError,
            f"Method {self.baseOverload.methodOf.Class.__name__}.{methodName} promised a return type "
            f"of '{self.promisedType.__name__}', but subclass {self.subOverload.methodOf.Class.__name__} "
            f"proposed to return '{self.proposedType.__name__}'. "
        )


class CannotBeDetermined:
    """singleton class used as a special return value."""


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

    argTypes = tuple(argTypes) + tuple(kwargTypes.values())
    neededArgTypes = tuple(neededArgTypes) + tuple([neededKwargTypes[name] for name in kwargTypes])

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


class FunctionSignatureCalculator:
    """Utilities for determining the stated signature of a function when called.

    Actually computing this is somewhat complicated: for each overload, we need to determine
    which "higher up" overloads could match it.
    """
    def __init__(self, funcType):
        self.funcType = funcType

        # build a list of lists, each of which is a block of mutually exclusive overloads
        # corresponding to a "level" in the type heirarchy (assuming this is a method)
        self.overloadBlocks = []
        self.indexToBlockIx = {}

        for o in self.funcType.overloads:
            if not self.overloadBlocks or o.methodOf != self.overloadBlocks[-1][-1].methodOf:
                self.overloadBlocks.append([o])
            else:
                self.overloadBlocks[-1].append(o)

            self.indexToBlockIx[o.index] = len(self.overloadBlocks) - 1

    def returnTypeFor(self, argTypes, kwargTypes):
        return self.returnTypeForLevel(0, argTypes, kwargTypes)

    def returnTypeForOverload(self, overloadIx, argTypes, kwargTypes):
        """Determine the tightest return type for a given level.

        If it definitely throws, SomeInvalidClassReturnType.

        If the return type cannot be determined, return CannotBeDetermined. If its not
        specified, return NoReturnTypeSpecified.
        """
        types = self.returnTypesForOverload(overloadIx, argTypes, kwargTypes)

        return self.mergeReturnTypes(types)

    def returnTypeForLevel(self, overloadIx, argTypes, kwargTypes):
        """Determine the tightest return type for a given level.

        If it definitely throws, return SomeInvalidClassReturnType.

        If the return type cannot be determined, return CannotBeDetermined. If its not
        specified, return NoReturnTypeSpecified.
        """
        types, definitelyMatchesAbove = self.returnTypesForLevel(overloadIx, argTypes, kwargTypes)

        return self.mergeReturnTypes(types)

    def mergeReturnTypes(self, types):
        if not types:
            return NoReturnTypeSpecified

        badPathways = []
        goodPathways = []

        for T, ix in types:
            if T is NoReturnTypeSpecified:
                return NoReturnTypeSpecified

            if T is CannotBeDetermined:
                return CannotBeDetermined

            if T is SomeInvalidClassReturnType:
                badPathways.append(T)
            else:
                goodPathways.append(T)

        if goodPathways:
            return mergeTypeWrappers(goodPathways).typeRepresentation

        if len(badPathways) == 1:
            return badPathways[0]

        return SomeInvalidClassReturnType

    def overloadCouldThrowInvalidReturnType(self, overloadIx, argTypes, kwargTypes):
        """Return True if the given overload could be of the wrong type."""
        ourRetType = self.singleReturnTypeFor(overloadIx, argTypes, kwargTypes)

        if ourRetType is CannotBeDetermined:
            return True

        possibleAboveTypes, definitelyMatchesAbove = self.returnTypesForLevel(
            self.indexToBlockIx[overloadIx] + 1,
            argTypes,
            kwargTypes
        )

        for p, ix in possibleAboveTypes:
            if (
                p is SomeInvalidClassReturnType
                or p is CannotBeDetermined
            ):
                return True

        if ourRetType is not None:
            for p, ix in possibleAboveTypes:
                if not canConvertToTrivially(ourRetType, p):
                    return True

        return False

    def overloadInvalidSignatures(self, overloadIx, argTypes, kwargTypes):
        # given that we match overloadIx with argTypes, kwargTypes, what are the
        # possible sets of (argTypes, kwargTypes) that, if we matched them, would produce
        # an invalid typecheck?

        # this function returns a list of InvalidClassReturnTypePredicate objects.
        # if we cannot determine the return type, then we return CannotBeDetermined.
        ourRetType = self.singleReturnTypeFor(overloadIx, argTypes, kwargTypes)

        if ourRetType is CannotBeDetermined:
            return CannotBeDetermined

        possibleAboveTypes, definitelyMatchesAbove = self.returnTypesForLevel(
            self.indexToBlockIx[overloadIx] + 1,
            argTypes,
            kwargTypes
        )

        for p, ix in possibleAboveTypes:
            if p is CannotBeDetermined:
                return CannotBeDetermined

        res = set()

        for p, ix in possibleAboveTypes:
            # we could flow into any of these
            res.update(self.overloadInvalidSignatures(ix, argTypes, kwargTypes))

        if ourRetType is not NoReturnTypeSpecified:
            res = set()

            for p, ix in possibleAboveTypes:
                if p is NoReturnTypeSpecified:
                    canConvert = True
                elif p is SomeInvalidClassReturnType:
                    targetType = self.singleReturnTypeFor(ix, argTypes, kwargTypes)

                    if targetType is CannotBeDetermined:
                        return CannotBeDetermined
                    if targetType is NoReturnTypeSpecified:
                        canConvert = True
                    else:
                        canConvert = False
                        assert isinstance(targetType, type)

                elif not canConvertToTrivially(ourRetType, p):
                    canConvert = False
                    targetType = p
                else:
                    canConvert = True

                if not canConvert:
                    ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

                    baseArgs, baseKwargs = (
                        ExpressionConversionContext.computeOverloadSignature(
                            self.funcType.overloads[ix], argTypes, kwargTypes
                        )
                    )

                    childArgs, childKwargs = (
                        ExpressionConversionContext.computeOverloadSignature(
                            self.funcType.overloads[overloadIx], argTypes, kwargTypes
                        )
                    )

                    res.add(
                        InvalidClassReturnTypePredicate(
                            self.funcType.overloads[ix],
                            baseArgs,
                            baseKwargs,
                            targetType,
                            self.funcType.overloads[overloadIx],
                            childArgs,
                            childKwargs,
                            ourRetType
                        )
                    )

        return res

    def returnTypesForLevel(self, blockLevel, argTypes, kwargTypes):
        """Determine what return types might apply to us if we were to flow into
        a whole 'level' of overload blocks.

        Returns:
            a tuple (possibleTypes, mustMatch)
            where possibleTypes is a list of returnTypesForOverload results for matching overloads,
            and mustMatch is a bool indicating whether we are guaranteed that one of those
            matches definitely applies to us.
        """
        if blockLevel >= len(self.overloadBlocks):
            return [], False

        startOverloadIx = self.overloadBlocks[blockLevel][0].index
        maxOverloadIx = self.overloadBlocks[blockLevel][-1].index + 1

        pathways = set()

        for conversionLevel in ConversionLevel.functionConversionSequence():
            for overloadIx in range(startOverloadIx, maxOverloadIx):
                matches = overloadMatchesSignature(
                    self.funcType.overloads[overloadIx], argTypes, kwargTypes, conversionLevel
                )

                if matches:
                    pathways.update(self.returnTypesForOverload(overloadIx, argTypes, kwargTypes))

                if matches is True:
                    return pathways, True

        # if we can't prove we matched, then its possible that we will flow through
        # to the next level of matches.
        nextLevelPathways, nextLevelApplies = (
            self.returnTypesForLevel(blockLevel + 1, argTypes, kwargTypes)
        )
        pathways.update(nextLevelPathways)

        return pathways, nextLevelApplies

    def returnTypesForOverload(self, overloadIx, argTypes, kwargTypes):
        """Determine what the stated return types for an overload given info about its args.

        If the overload matches, then in the interpreter, there will be some specific
        sequence of overloads within the function that match (exactly one if its a regular
        function, more than one if it happens to be a method on a class). Each of those
        sequence items will be asked for a type, and the interpreter will check that they
        are increasingly broad as we go up the list of overloads.

        This function returns a list of the various possibilities, each one of which represents
        a possible overload (higher in the stack) that might have matched.

        Each term in the resulting list will be a pair of (T, overloadIx) where T is

            CannotBeDetermined - If we cannot determine the type the function will apply to us
                (because we don't know the types of the arguments fully. This means
                that it is possible that the function states that it returns as wide as
                'object', but that we cannot determine anything about it.

            SomeInvalidClassReturnType - this pathway definitely throws

            NoReturnTypeSpecified - this pathway didn't specify a return type at all.

            a type - this pathway will insist that objects have this type.
        """
        argTypes = tuple(typeWrapper(x) for x in argTypes)
        kwargTypes = {name: typeWrapper(val) for name, val in kwargTypes.items()}

        # maybe we match - so at this point, we assume we do match
        ourRetType = self.singleReturnTypeFor(overloadIx, argTypes, kwargTypes)

        # find out the list of possible types we might get if we flow into the parents
        possibleAboveTypes, definitelyMatchesAbove = self.returnTypesForLevel(
            self.indexToBlockIx[overloadIx] + 1,
            argTypes,
            kwargTypes
        )

        if ourRetType is CannotBeDetermined:
            # in theory, we could still do better than this by looking
            # up the stack - all of those signatures will also still apply.
            # is there a representation that's more precise here?
            return [(CannotBeDetermined, overloadIx)]

        if ourRetType is NoReturnTypeSpecified:
            # this overload didn't specify anything. This means it didn't explicitly
            # annotate anything.
            badPathways = []
            goodPathways = []

            # we have constraints that are placed on us by the blocks above us as well
            for p, ix in possibleAboveTypes:
                if p is CannotBeDetermined:
                    return [(CannotBeDetermined, overloadIx)]

                if p is NoReturnTypeSpecified:
                    goodPathways.append((p, ix))

                if p is SomeInvalidClassReturnType:
                    badPathways.append(p)
                else:
                    goodPathways.append((p, ix))

            if badPathways and not goodPathways:
                if len(badPathways) == 1:
                    return [(badPathways[0], overloadIx)]
                else:
                    return [(SomeInvalidClassReturnType, overloadIx)]

            # if all the good pathways are the same, we can return them
            distinctGPs = set([T for T, ix in goodPathways])
            if len(distinctGPs) == 1:
                return [(list(distinctGPs)[0], ix)]

            return goodPathways
        else:
            if not definitelyMatchesAbove:
                # its possible we don't match something above, so we can bail
                return [(ourRetType, overloadIx)]

            # this overload specified we return a specific type. This means we need to
            # check to see if it definitely throws. Otherwise, we can prune it and return it
            goodPathways = []
            badPathways = []

            for p, ix in possibleAboveTypes:
                if p is None:
                    # we might flow into a pathway that doesn't have a flow
                    return [(ourRetType, overloadIx)]

                if p is CannotBeDetermined:
                    # somebody above us cannot be determined, but at least we
                    # know that we don't return less than this
                    return [(ourRetType, overloadIx)]

                if p is SomeInvalidClassReturnType:
                    badPathways.append(p)
                elif not canConvertToTrivially(ourRetType, p):
                    badPathways.append(SomeInvalidClassReturnType)
                else:
                    goodPathways.append(p)

            if not goodPathways and len(badPathways) == 1:
                return [(badPathways[0], overloadIx)]

            if not goodPathways and badPathways:
                # we throw, but we don't know what
                return [(SomeInvalidClassReturnType, overloadIx)]

            return [(ourRetType, overloadIx)]

    def singleReturnTypeFor(self, overloadIx, argTypes, kwargTypes):
        """Determine what return type a specific function overload will produce.

        If we cannot determine it (because we don't know the types of the arguments
        fully, and we have a function signature) then we return CannotBeDetermined.

        If the function doesn't have a stated return type, we return None.

        If we return a type, then we are guaranteed that all values flowing through this
        overload will be covered by that type.

        Args:
            overloadIx - the index of the overload in this function
            argTypes - a list or tuple of TypeWrapper objects
            kwargTypes - a dict from name to TypeWrapper

        Returns:
            NoReturnTypeSpecified, a type, CannotBeDetermined
        """
        overload = self.funcType.overloads[overloadIx]

        ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

        argAndKwargTypes = ExpressionConversionContext.computeOverloadSignature(
            overload, argTypes, kwargTypes
        )

        if argAndKwargTypes is None:
            return NoReturnTypeSpecified

        argTypes, kwargTypes = argAndKwargTypes

        if overload.signatureFunction:
            typeArgs = []
            typeKwargs = {}

            # these are the wrong types - we may not actually know
            # what type is going to go into the function itself!
            for arg in argTypes:
                typeArgs.append(FunctionSignatureCalculator.signatureArgFor(arg))

            for name, kwarg in kwargTypes.items():
                typeKwargs[name] = FunctionSignatureCalculator.signatureArgFor(arg)

            res = overload.signatureFunction(*typeArgs, **typeKwargs)

            if res is NoReturnTypeSpecified:
                return res

            if isinstance(res, (Either, SubclassOf)):
                return CannotBeDetermined

            # signature functions always get their arguments turned into types.
            # this is kind of a round-about way of ensuring that we follow the
            # same rules in the compiler as we do in the interpreter: converting
            # to a 'OneOf' forces the type to get passed as a 'type argument', and
            # simple values get promoted to Value types.
            res = OneOf(res)

            if not res.Types:
                return res

            return res.Types[0]
        elif overload.returnType is None:
            return NoReturnTypeSpecified
        else:
            return overload.returnType

    @staticmethod
    def signatureArgFor(typeWrapper):
        """Given a type wrapper, determine the 'type' we'll pass into a signature function."""
        return FunctionSignatureCalculator.typeSetFor(
            typeWrapper.interpreterTypeRepresentation
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
            return Either([FunctionSignatureCalculator.typeSetFor(SubT) for SubT in T.Types])

        if T in (int, float, bool, type(None), str, bytes):
            return T

        if issubclass(T, (Value, ListOf, Set, Dict, ConstDict, TupleOf, Tuple, NamedTuple)):
            return T

        return SubclassOf(object)
