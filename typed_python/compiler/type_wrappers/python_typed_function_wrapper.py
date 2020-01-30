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


from typed_python import PointerTo, bytecount, NamedTuple, Class
from typed_python._types import is_default_constructible
from typed_python.internals import CellAccess
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.one_of_wrapper import OneOfWrapper
from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import TypedTupleMasqueradingAsTuple
from typed_python.compiler.type_wrappers.named_tuple_masquerading_as_dict_wrapper import NamedTupleMasqueradingAsDict


import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


class PythonTypedFunctionWrapper(Wrapper):
    is_default_constructible = False

    def __init__(self, f):
        if isinstance(f, typed_python._types.Function):
            f = type(f)

        self.closureWrapper = typeWrapper(f.ClosureType)

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
                expr.changeType(typeWrapper(pathElt))
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

    def convert_call(self, context, left, args, kwargs):
        if left is None:
            assert bytecount(self.typeRepresentation) == 0
            left = context.push(self, lambda expr: None)

        # check if we are marked 'nocompile' in which case we convert to 'object' and dispatch
        # to the interpreter. We do retain any typing information on the return type, however.

        # todo: reenable this
        if len(self.typeRepresentation.overloads) == 1 and False:
            overload = self.typeRepresentation.overloads[0]
            functionObj = overload.functionObj

            if hasattr(functionObj, "__typed_python_no_compile__"):
                returnType = overload.returnType or object

                callRes = context.constantPyObject(functionObj).convert_call(
                    args, kwargs
                )

                if callRes is None:
                    return None

                return callRes.convert_to_type(returnType)

        argTypes = [a.expr_type for a in args]
        kwargTypes = {k: v.expr_type for k, v in kwargs.items()}

        # check if there is exactly one overload we could match
        overloadAndIsExplicit = PythonTypedFunctionWrapper.pickSingleOverloadForCall(
            self.typeRepresentation, argTypes, kwargTypes
        )

        if overloadAndIsExplicit is not None:
            overload = overloadAndIsExplicit[0]

            actualArgTypes, actualKwargTypes = context.computeOverloadSignature(overload, argTypes, kwargTypes)

            actualArgs = [args[i].convert_to_type(actualArgTypes[i]) for i in range(len(actualArgTypes))]
            actualKwargs = {name: kwargs[name].convert_to_type(actualKwargTypes[name]) for name in kwargs}

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

            # just one overload will do. We can just instantiate this particular function
            # with a signature that comes from the method overload signature itself.
            singleConvertedOverload = context.functionContext.converter.convert(
                overload.name,
                overload.functionCode,
                overload.functionGlobals,
                [typeWrapper(self.closurePathToCellType(path, closureType)) for path in overload.closureVarLookups.values()]
                + [a.expr_type for a in argsToPass],
                overload.returnType,
                callback=None
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
            None,
            True
        )

        if not callTarget:
            context.pushException(
                TypeError,
                f"{self.typeRepresentation} cannot find a valid overload with arguments of type "
                "(" + ",".join([str(x.expr_type) for x in args] + [k+"="+str(v.expr_type) for k, v in kwargs.items()]) + ")"
            )
            return

        return context.call_typed_call_target(callTarget, [left] + args + list(kwargs.values()))

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

    def compileCall(self, converter, returnType, argTypes, kwargTypes, callback, provideClosureArgument):
        """Compile this function being called with a particular signature.

        Args:
            converter - the PythonToNativeConverter that needs the concrete definition.
            returnType - the typed_python Type of what we're returning, or None if we don't know
            argTypes - (ListOf(wrapper)) a the actual concrete type wrappers for the arguments
                we're passing.
            kwargTypes - (Dict(str, wrapper)) a the actual concrete type wrappers for the keyword
                arguments we're passing.
            callback - the callback to pass to 'convert' so that we can install the compiled
                function pointer in the class vtable at link time.
            provideClosureArgument - if True, then the first argument is of our closure type.
                If false, then our closure type must be empty, and no argument for it is provided.
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

            returnType = OneOfWrapper.mergeTypes(possibleTypes)

            if returnType is None:
                return None

        argNames = [None for _ in argTypes] + list(kwargTypes)

        return converter.defineNativeFunction(
            f'implement_function.{self}{argTypes}.{kwargTypes}->{returnType}',
            ('implement_function.', self, returnType, self, tuple(argTypes), tuple(kwargTypes.items())),
            ([self] if provideClosureArgument else []) + list(argTypes) + list(kwargTypes.values()),
            returnType,
            lambda context, outputVar, *args: (
                self.generateMethodImplementation(context, returnType, args, argNames, provideClosureArgument)
            ),
            callback=callback
        )

    @staticmethod
    def overloadMatchesSignature(overload, argTypes, kwargTypes, isExplicit):
        """Is it possible we could dispatch to FunctionOverload 'overload' with 'argTypes/kwargTypes'?

        Returns:
            True if we _definitely_ match
            "Maybe" if we might match
            False if we definitely don't match the arguments.
        """
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
                isExplicit
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

        for isExplicit in [False, True]:
            for o in func.overloads:
                # check each overload that we might match.
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(o, argTypes, kwargTypes, isExplicit)

                if mightMatch is False:
                    pass
                else:
                    if o.returnType is not None:
                        returnTypes.append(o.returnType)
                    else:
                        ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

                        actualArgTypes = ExpressionConversionContext.computeFunctionArgumentTypeSignature(
                            o,
                            argTypes,
                            kwargTypes
                        )

                        callTarget = converter.convert(
                            o.name,
                            o.functionCode,
                            o.functionGlobals,
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
        """See if there is exactly one function overload that definitely matches 'argTypes/kwargTypes'

        Returns:
            None, or a tuple (FunctionOverload, explicit) indicating that one single overload
            is the one version of this function we definitely match.
        """

        for isExplicit in [False, True]:
            for o in func.overloads:
                # check each overload that we might match.
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(
                    o, argTypes, kwargTypes, isExplicit
                )

                if mightMatch is False:
                    pass
                elif mightMatch is True:
                    return (o, isExplicit)
                else:
                    return None

        return None

    def generateMethodImplementation(self, context, returnType, args, argNames, provideClosureArgument):
        """Generate native code that calls us with a given return type and set of arguments.

        We try each overload, first with 'isExplicit' as False, then with True. The first one that
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
        """
        func = self.typeRepresentation

        if provideClosureArgument:
            closureArg = args[0]
            args = args[1:]
        else:
            closureArg = None

        argTypes = [a.expr_type for i, a in enumerate(args) if argNames[i] is None]
        kwargTypes = {argNames[i]: a.expr_type for i, a in enumerate(args) if argNames[i] is not None}

        def makeOverloadImplementor(overload, isExplicit):
            return lambda context, _, outputVar, *args: self.generateOverloadImplement(
                context, overload, isExplicit, outputVar, args, argNames, provideClosureArgument
            )

        for isExplicit in [False, True]:
            for overloadIndex, overload in enumerate(func.overloads):
                mightMatch = self.overloadMatchesSignature(overload, argTypes, kwargTypes, isExplicit)

                if mightMatch is not False:
                    overloadRetType = overload.returnType or object

                    testSingleOverloadForm = context.converter.defineNativeFunction(
                        f'implement_overload.{self}.{overloadIndex}.{isExplicit}.{argTypes}.{kwargTypes}->{overloadRetType}',
                        ('implement_overload', self, overloadIndex, isExplicit,
                         overloadRetType, tuple(argTypes), tuple(kwargTypes.items())),
                        [PointerTo(overloadRetType)] + ([self] if provideClosureArgument else [])
                        + list(argTypes) + list(kwargTypes.values()),
                        typeWrapper(bool),
                        makeOverloadImplementor(overload, isExplicit)
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
                            actualResult = outputSlot.convert_to_type(returnType)

                            if actualResult is not None:
                                context.pushReturnValue(actualResult)

                    # if we definitely match, we can return early
                    if mightMatch is True:
                        context.pushException(
                            TypeError,
                            f"Thought we found overload for {self} matching {argTypes} and "
                            f"{kwargTypes}, but it didn't accept the arguments."
                        )
                        return

        # generate a cleanup handler for the cases where we don't match a method signature.
        # this should actually be hitting the interpreter instead.
        context.pushException(TypeError, f"Failed to find an overload for {self} matching {argTypes} and {kwargTypes}")

    def generateOverloadImplement(self, context, overload, isExplicit, outputVar, args, argNames, provideClosureArgument):
        """Produce the code that implements this specific overload.

        The generated code returns control flow with a True if it fills out the 'outputVar'
        with data, and False otherwise.

        Args:
            context - an ExpressionConversionContext
            overload - the FunctionOverload we're trying to convert.
            isExplicit - are we using explicit conversion?
            outputVar - a TypedExpression(PointerTo(returnType)) we're supposed to initialize.
            args - the arguments to pass to the method (including the closure if necessary and the instance)
            argNames - a list with one element per args, containing None for each positional
                argument, or the name of the argument if passed as a keyword argument.
            provideClosureArgument - if True, then the first argument is of our closure type.
                If false, then our closure type must be empty, and no argument for it is provided.
        """
        if provideClosureArgument:
            closureArg = args[0]
            args = args[1:]
        else:
            closureArg = None

        assert len(args) == len(argNames)

        argTypes = [a.expr_type for i, a in enumerate(args) if argNames[i] is None]
        kwargTypes = {argNames[i]: a.expr_type for i, a in enumerate(args) if argNames[i] is not None}
        retType = overload.returnType or typeWrapper(object).typeRepresentation

        argAndKwargTypes = context.computeOverloadSignature(overload, argTypes, kwargTypes)

        convertedArgs = []
        convertedKwargs = {}

        for argIx, argExpr in enumerate(args):
            if argNames[argIx] is None:
                argType = argAndKwargTypes[0][argIx]
            else:
                argType = argAndKwargTypes[1][argNames[argIx]]

            argType = typeWrapper(argType)

            convertedArg = context.allocateUninitializedSlot(argType)

            successful = argExpr.convert_to_type_with_target(convertedArg, isExplicit)

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

        outputVar.changeType(typeWrapper(retType), True).convert_copy_initialize(res)

        context.pushReturnValue(context.constant(True))
