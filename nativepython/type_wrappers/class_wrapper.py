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
from nativepython.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, _types, OneOf, PointerTo

import nativepython.native_ast as native_ast
import nativepython


typeWrapper = lambda x: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(x)


native_destructor_function_type = native_ast.Type.Function(
    output=native_ast.Void,
    args=(native_ast.VoidPtr,),
    varargs=False,
    can_throw=False
).pointer()


class_dispatch_table_type = native_ast.Type.Struct(
    element_types=[
        ('implementingClass', native_ast.VoidPtr),
        ('interfaceClass', native_ast.VoidPtr),
        ('funcPtrs', native_ast.VoidPtr.pointer()),
        ('upcastDispatches', native_ast.UInt16.pointer()),
        ('funcPtrsAllocated', native_ast.UInt64),
        ('funcPtrsUsed', native_ast.UInt64),
        ('dispatchIndices', native_ast.VoidPtr),
        ('dispatchDefinitions', native_ast.VoidPtr),
        ('indicesNeedingDefinition', native_ast.VoidPtr),
    ],
    name="ClassDispatchTable"
)


vtable_type = native_ast.Type.Struct(
    element_types=[
        ('heldTypePtr', native_ast.VoidPtr),
        ('destructorFun', native_destructor_function_type),
        ('classDispatchTable', class_dispatch_table_type.pointer())
    ],
    name="VTable"
)


def pickCallSignatureToImplement(overload, name, argTypes):
    """Pick the actual signature to use when calling 'overload' with 'argTypes'

    We can have a function like f(x: int), where we always know the signature.
    But for something like f(x), we may need to generate a different signature
    for each possible 'x' we pass it.

    Args:
        overload - a typed_python.internal.FunctionOverload
        name - the name of the function
        argTypes - a list of typed_python Type objects

    Returns:
        a typed_python.Function object representing the signature we'll implement
        for this overload.
    """
    argTuples = []

    if len(argTypes) != len(overload.args):
        raise Exception(f"Signature mismatch; can't call {overload} with {argTypes}")

    for i, arg in enumerate(overload.args):
        # when choosing the signature we want to generate for a given call signature,
        # we specialize anything with _no signature at all_, but otherwise take the given
        # signature. Otherwise, we'd produce an exorbitant number of signatures (for every
        # possible subtype combination we encounter in code).

        if arg.typeFilter is None:
            argType = argTypes[i].typeRepresentation
        else:
            argType = arg.typeFilter or object

        argTuples.append(
            (arg.name, argType, arg.defaultValue, arg.isStarArg, arg.isKwarg)
        )

    return _types.Function(
        name,
        overload.returnType or object,
        None,
        tuple(argTuples)
    )


def overloadMatchesSignature(overload, argTypes, isExplicit):
    """Is it possible we could dispatch to FunctionOverload 'overload' with 'argTypes'?

    Returns:
        True if we _definitely_ match
        "Maybe" if we might match
        False if we definitely don't match the arguments.
    """
    if not (len(argTypes) == len(overload.args) and not any(x.isStarArg or x.isKwarg for x in overload.args)):
        return False

    allTrue = True
    for i in range(len(argTypes)):
        canConvert = argTypes[i].can_convert_to_type(typeWrapper(overload.args[i].typeFilter or object), isExplicit)

        if canConvert is False:
            return False
        elif canConvert == "Maybe":
            allTrue = False

    if allTrue:
        return allTrue
    else:
        return "Maybe"


class ClassWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    BYTES_BEFORE_INIT_BITS = 16  # the refcount and vtable are both 8 byte integers.

    def __init__(self, t):
        super().__init__(t)

        self.nameToIndex = {}
        self.indexToByteOffset = {}
        self.classType = t

        element_types = [('refcount', native_ast.Int64), ('vtable', vtable_type.pointer()), ('data', native_ast.UInt8)]

        # this follows the general layout of 'held class' which is 1 bit per field for initialization and then
        # each field packed directly according to byte size
        byteOffset = self.BYTES_BEFORE_INIT_BITS + (len(self.classType.MemberNames) // 8 + 1)

        self.bytesOfInitBits = byteOffset - self.BYTES_BEFORE_INIT_BITS

        for i, name in enumerate(self.classType.MemberNames):
            self.nameToIndex[name] = i
            self.indexToByteOffset[i] = byteOffset

            byteOffset += _types.bytecount(self.classType.MemberTypes[i])

        self.layoutType = native_ast.Type.Struct(element_types=element_types, name=t.__qualname__+"Layout").pointer()

        # we need this to actually be a global variable that we fill out, but we don't have the machinery
        # yet in the native_ast. So for now, we just hack it together.
        # because we are writing a pointer value directly into the generated code as a constant, we
        # won't be able to reuse the binary we produced in another program.
        self.vtableExpr = native_ast.const_uint64_expr(
            _types._vtablePointer(self.typeRepresentation)
        ).cast(vtable_type.pointer())

    def _can_convert_to_type(self, otherType, explicit):
        if isinstance(otherType, ClassWrapper):
            if otherType.typeRepresentation in self.typeRepresentation.MRO:
                return True
            elif self.typeRepresentation in otherType.typeRepresentation.MRO:
                return "Maybe"

        return False

    def _can_convert_from_type(self, otherType, explicit):
        return False

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        otherType = targetVal.expr_type

        if isinstance(otherType, ClassWrapper):
            if otherType.typeRepresentation in self.typeRepresentation.MRO:
                # this is an upcast
                index = _types.getDispatchIndexForType(otherType.typeRepresentation, self.typeRepresentation)

                context.pushEffect(
                    targetVal.expr.store(
                        self.withDispatchIndex(e, index)
                    )
                )
                targetVal.convert_incref()

                return context.constant(True)

            elif self.typeRepresentation in otherType.typeRepresentation.MRO:
                raise Exception("Downcast in compiled code not implemented yet")

        return context.constant(False)

    def get_layout_pointer(self, nonref_expr):
        # our layout is 48 bits of pointer and 16 bits of classDispatchTableIndex.
        # so whenever we interact with the pointer we need to chop off the top 16 bits
        return (
            nonref_expr
            .cast(native_ast.UInt64)
            .bitand(native_ast.const_uint64_expr(0xFFFFFFFFFFFF))  # 48 bits of 1s
            .cast(self.layoutType)
        )

    def classDispatchTable(self, instance):
        classDispatchTables = (
            self.get_layout_pointer(instance.nonref_expr)
            .ElementPtrIntegers(0, 1).load().ElementPtrIntegers(0, 2).load()
        )

        # instances of a class can 'masquerade' as any one of their base classes. They have a vtable
        # for each one indicating how to dispatch method calls to the concrete class when they
        # are masquerading as that particular base class. Whenever we represent a child class
        # as a base class, we need to track which of the class' concrete vtable entries we should
        # be using for dispatch. We encode this in the top 16 bits of the pointer because on modern
        # x64 systems, the pointer address space is 48 bits. If somehow we need to compile on
        # itanium, we'll have to rethink this.
        return classDispatchTables.elemPtr(
            self.get_dispatch_index(instance)
        )

    def withDispatchIndex(self, instance, index):
        """Return a native expression representing 'instance' as the 'index'th base class in the MRO.

        To do this we have to look at the 'upcastIndices' in the class dispatch table. This is because
        we might have a child class that's already been cast to an intermediate layer in the type hierarchy,
        and we can't compute at compile time exactly which index the new interface class is. So we have
        to look it up.
        """
        classDispatchTable = self.classDispatchTable(instance)

        actualIndexExpr = classDispatchTable.ElementPtrIntegers(0, 3).load().ElementPtrIntegers(index).load()

        return (
            instance.nonref_expr
            .cast(native_ast.UInt64)
            .bitand(native_ast.const_uint64_expr(0xFFFFFFFFFFFF))
            .add(actualIndexExpr.cast(native_ast.UInt64).lshift(native_ast.const_uint64_expr(48)))
            .cast(self.layoutType)
        )

    def get_refcount_ptr_expr(self, nonref_expr):
        """Return a pointer to the object's refcount. Subclasses can override.

        Args:
            nonref_expr - a native expression equivalent to 'self.nonref_expr'. In most cases
                this will be the pointer to the actual refcounted data structure.
        """
        return self.get_layout_pointer(nonref_expr).ElementPtrIntegers(0, 0)

    def get_dispatch_index(self, instance):
        """Return the integer index of the current class dispatch within this instances' vtable."""
        return (
            instance.nonref_expr
            .cast(native_ast.UInt64)
            .rshift(native_ast.const_uint64_expr(48))
        )

    def convert_default_initialize(self, context, instance):
        return context.pushException(TypeError, f"Can't default initialize instances of {self}")

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        def installDestructorFun(funcPtr):
            _types.installClassDestructor(self.typeRepresentation, funcPtr.fp)

        context.converter.defineNativeFunction(
            "destructor_" + str(self.typeRepresentation),
            ('destructor', self),
            [self],
            typeWrapper(NoneType),
            self.generateNativeDestructorFunction,
            callback=installDestructorFun
        )

        return native_ast.CallTarget.Pointer(
            expr=self.get_layout_pointer(instance.nonref_expr).ElementPtrIntegers(0, 1).load().ElementPtrIntegers(0, 1).load()
        ).call(instance.expr.cast(native_ast.VoidPtr))

    def generateNativeDestructorFunction(self, context, out, instance):
        for i in range(len(self.typeRepresentation.MemberTypes)):
            if not typeWrapper(self.typeRepresentation.MemberTypes[i]).is_pod:
                with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(instance, i))) as (true_block, false_block):
                    with true_block:
                        context.pushEffect(
                            self.convert_attribute(context, instance, i, nocheck=True).convert_destroy()
                        )

        context.pushEffect(runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr)))

    def memberPtr(self, instance, ix):
        return (
            self.get_layout_pointer(instance.nonref_expr)
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(self.indexToByteOffset[ix])
            .cast(
                typeWrapper(self.typeRepresentation.MemberTypes[ix])
                .getNativeLayoutType()
                .pointer()
            )
        )

    def isInitializedNativeExpr(self, instance, ix):
        byte = ix // 8
        bit = ix % 8

        return (
            self.get_layout_pointer(instance.nonref_expr)
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(self.BYTES_BEFORE_INIT_BITS + byte)
            .load()
            .rshift(native_ast.const_uint8_expr(bit))
            .bitand(native_ast.const_uint8_expr(1))
        )

    def setIsInitializedExpr(self, instance, ix):
        byte = ix // 8
        bit = ix % 8

        bytePtr = (
            self.get_layout_pointer(instance.nonref_expr)
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(self.BYTES_BEFORE_INIT_BITS + byte)
        )

        return bytePtr.store(bytePtr.load().bitor(native_ast.const_uint8_expr(1 << bit)))

    def convert_attribute(self, context, instance, attribute, nocheck=False):
        if attribute in self.typeRepresentation.MemberFunctions:
            methodType = BoundMethodWrapper(_types.BoundMethod(self.typeRepresentation, attribute))

            return instance.changeType(methodType)

        if not isinstance(attribute, int):
            ix = self.nameToIndex.get(attribute)
        else:
            ix = attribute

        if ix is None:
            return context.pushTerminal(
                generateThrowException(context, AttributeError("Attribute %s doesn't exist in %s" % (attribute, self.typeRepresentation)))
            )

        if nocheck:
            return context.pushReference(
                self.typeRepresentation.MemberTypes[ix],
                self.memberPtr(instance, ix)
            )

        return context.pushReference(
            self.typeRepresentation.MemberTypes[ix],
            native_ast.Expression.Branch(
                cond=self.isInitializedNativeExpr(instance, ix),
                false=generateThrowException(context, AttributeError("Attribute %s is not initialized" % attribute)),
                true=self.memberPtr(instance, ix)
            )
        )

    def resultTypesForCall(self, func, argTypes):
        resultTypes = set()

        for isExplicit in [False, True]:
            for o in func.overloads:
                # check each overload that we might match.
                mightMatch = overloadMatchesSignature(o, argTypes, isExplicit)

                if mightMatch is False:
                    return resultTypes

                if o.returnType is None:
                    resultTypes.add(object)
                else:
                    resultTypes.add(o.returnType)

                if mightMatch is True:
                    return resultTypes

        return resultTypes

    @staticmethod
    def pickSingleOverloadForCall(func, argTypes):
        """See if there is a single function overload that might match 'argTypes' and nothing else.

        Returns:
            None, or a tuple (FunctionOverload, explicit) indicating that one single overload
            is the one version of this function we might match.
        """

        possibleMaybe = None

        for isExplicit in [False, True]:
            for o in func.overloads:
                # check each overload that we might match.
                mightMatch = overloadMatchesSignature(o, argTypes, isExplicit)

                if mightMatch is False:
                    pass
                elif mightMatch is True:
                    if possibleMaybe is not None:
                        if possibleMaybe == (o, False) and isExplicit:
                            return (o, True)
                        else:
                            return None
                    else:
                        return (o, isExplicit)
                else:
                    if possibleMaybe is None:
                        possibleMaybe = (o, isExplicit)
                    elif possibleMaybe == (o, False) and isExplicit:
                        possibleMaybe = (o, isExplicit)
                    else:
                        return None

        return possibleMaybe

    def convert_method_call(self, context, instance, methodName, args, kwargs):
        # figure out which signature we'd want to use on the given args/kwargs
        if kwargs:
            raise Exception("Can't call methods with keyword args yet.")

        argTypes = [instance.expr_type] + [a.expr_type for a in args]

        func = self.typeRepresentation.MemberFunctions[methodName]

        # each of 'func''s overloads represents one of the functions defined with this name
        # in this class and in its base classes, ordered by the method resolution order.
        # we can think of each one as a pattern that we are sequentially matching against,
        # and we should invoke the first one that matches the specific values that we have
        # in our argTypes. In fact, we may match more than one (for instance if we know all of
        # our values as 'object') in which case we need to generate runtime tests for each value
        # against each type pattern and take the union of the return types.

        # each term that we might match against generates an entrypoint in the class vtable
        # for this class and for all of its children. That entry represents calling the function
        # with name 'methodName' with the given signature.

        # because children can override the behavior of parent signatures, we insist at class
        # definition time that when a child class overrides a parent class its return type
        # signatures become more specific: if a base class defines
        #     def f(self) -> int:
        # then the child class must also return 'int', (or something like OneOf(0, 1)).
        # this is necessary for type inference to work correctly, because if we didn't
        # insist on that we can't make any assumptions about the types that come out
        # of a base class implementation.

        # first, see if there is exacly one possible overload
        overloadAndIsExplicit = ClassWrapper.pickSingleOverloadForCall(func, argTypes)

        if overloadAndIsExplicit is not None:
            return self.dispatchToSingleOverload(
                context,
                overloadAndIsExplicit[0],
                overloadAndIsExplicit[1],
                methodName,
                argTypes,
                instance,
                args
            )

        resultTypes = self.resultTypesForCall(func, argTypes)

        if not resultTypes:
            # we can't call anything
            context.pushException(
                Exception,
                f"No overload could be found for compiled dispatch to "
                f"{self}.{methodName} with args {argTypes}."
            )
            return None

        # compute the return type of dispatching to this function. it will be a one-of if we
        # might dispatch to multiple possible functions.
        if len(resultTypes) == 1:
            output_type = typeWrapper(list(resultTypes)[0])
        else:
            output_type = typeWrapper(OneOf(*resultTypes))

        dispatchToOverloads = context.converter.defineNativeFunction(
            f'call_method.{self}.{methodName}.{argTypes[1:]}',
            ('call_method', self, methodName, tuple(argTypes[1:])),
            list(argTypes),
            output_type,
            lambda context, outputVar, *args: self.generateMethodDispatch(context, methodName, output_type, args)
        )

        return context.call_typed_call_target(dispatchToOverloads, [instance] + args, {})

    def generateMethodDispatch(self, context, methodName, methodReturnType, args):
        """Generate native code that tries to dispatch to each of func's overloads

        We try each overload, first with 'isExplicit' as False, then with True. The first one that
        succeeds gets to produce the output.

        Args:
            context - an ExpressionConversionContext
            methodName - the name of the method we're trying to dispatch to
            methodReturnType - the output type we are expecting to return. This will be the union
                of the return types of all the overloads that might participate in this dispatch.
            args - the typed_expression for all of our actual arguments, which in this case
                are the instance, and then the actual arguments we want to convert.
        """
        func = self.typeRepresentation.MemberFunctions[methodName]

        argTypes = [a.expr_type for a in args]

        def makeOverloadDispatcher(overload, isExplicit):
            return lambda context, _, outputVar, *args: self.generateOverloadDispatch(
                context, methodName, overload, isExplicit, outputVar, args
            )

        for isExplicit in [False, True]:
            for overloadIndex, overload in enumerate(func.overloads):
                mightMatch = overloadMatchesSignature(overload, argTypes, isExplicit)

                if mightMatch is not False:
                    overloadRetType = overload.returnType or object

                    testSingleOverloadForm = context.converter.defineNativeFunction(
                        f'call_overload.{self}.{methodName}.{overloadIndex}.{isExplicit}.{argTypes[1:]}->{overloadRetType}',
                        ('call_overload', self, methodName, overloadIndex, isExplicit, overloadRetType, tuple(argTypes[1:])),
                        [PointerTo(overloadRetType)] + list(argTypes),
                        typeWrapper(bool),
                        makeOverloadDispatcher(overload, isExplicit)
                    )

                    outputSlot = context.allocateUninitializedSlot(overloadRetType)

                    successful = context.call_typed_call_target(
                        testSingleOverloadForm,
                        (outputSlot.changeType(PointerTo(overloadRetType), False),) + args,
                        {}
                    )

                    with context.ifelse(successful.nonref_expr) as (ifTrue, ifFalse):
                        with ifTrue:
                            context.markUninitializedSlotInitialized(outputSlot)

                            # upcast the result
                            actualResult = outputSlot.convert_to_type(methodReturnType)

                            if actualResult is not None:
                                context.pushReturnValue(actualResult)

                    # if we definitely match, we can return early
                    if mightMatch is True:
                        context.pushException(TypeError, f"Failed to find an overload for {self}.{methodName} matching {args}")
                        return

        # generate a cleanup handler for the cases where we don't match a method signature.
        # this should actually be hitting the interpreter instead.
        context.pushException(TypeError, f"Failed to find an overload for {self}.{methodName} matching {args}")

    def generateMethodImplementation(self, context, methodName, methodReturnType, args):
        """Generate native code that implements 'methodName' with a given return type and set of arguments.

        We try each overload, first with 'isExplicit' as False, then with True. The first one that
        succeeds gets to produce the output.

        Args:
            context - an ExpressionConversionContext
            methodName - the name of the method we're trying to dispatch to
            methodReturnType - the output type we are expecting to return. This will be the union
                of the return types of all the overloads that might participate in this dispatch.
            args - the typed_expression for all of our actual arguments, which in this case
                are the instance, and then the actual arguments we want to convert.
        """
        func = self.typeRepresentation.MemberFunctions[methodName]

        argTypes = [a.expr_type for a in args]

        def makeOverloadImplementor(overload, isExplicit):
            return lambda context, _, outputVar, *args: self.generateOverloadImplement(
                context, methodName, overload, isExplicit, outputVar, args
            )

        for isExplicit in [False, True]:
            for overloadIndex, overload in enumerate(func.overloads):
                mightMatch = overloadMatchesSignature(overload, argTypes, isExplicit)

                if mightMatch is not False:
                    overloadRetType = overload.returnType or object

                    testSingleOverloadForm = context.converter.defineNativeFunction(
                        f'implement_overload.{self}.{methodName}.{overloadIndex}.{isExplicit}.{argTypes[1:]}->{overloadRetType}',
                        ('implement_overload', self, methodName, overloadIndex, isExplicit, overloadRetType, tuple(argTypes[1:])),
                        [PointerTo(overloadRetType)] + list(argTypes),
                        typeWrapper(bool),
                        makeOverloadImplementor(overload, isExplicit)
                    )

                    outputSlot = context.allocateUninitializedSlot(overloadRetType)

                    successful = context.call_typed_call_target(
                        testSingleOverloadForm,
                        (outputSlot.changeType(PointerTo(overloadRetType), False),) + args,
                        {}
                    )

                    with context.ifelse(successful.nonref_expr) as (ifTrue, ifFalse):
                        with ifTrue:
                            context.markUninitializedSlotInitialized(outputSlot)

                            # upcast the result
                            actualResult = outputSlot.convert_to_type(methodReturnType)

                            if actualResult is not None:
                                context.pushReturnValue(actualResult)

                    # if we definitely match, we can return early
                    if mightMatch is True:
                        context.pushException(TypeError, f"Failed to find an overload for {self}.{methodName} matching {args}")
                        return

        # generate a cleanup handler for the cases where we don't match a method signature.
        # this should actually be hitting the interpreter instead.
        context.pushException(TypeError, f"Failed to find an overload for {self}.{methodName} matching {args}")

    def generateOverloadImplement(self, context, methodName, overload, isExplicit, outputVar, args):
        """Produce the code that implements this specific overload.

        We return True if successful, False otherwise, and the output is a pointer to the result
        of the function call if we're successful.

        Args:
            context - an ExpressionConversionContext
            methodName - the name of the method we're compiling
            overload - the FunctionOverload we're trying to convert.
            isExplicit - are we using explicit conversion?
            outputVar - a TypedExpression(PointerTo(returnType)) we're supposed to initialize.
            args - the arguments to pass to the method (including the instance)
        """
        signature = pickCallSignatureToImplement(overload, methodName, [a.expr_type for a in args])

        argTypes = [a.typeFilter for a in signature.overloads[0].args]

        retType = overload.returnType or object

        convertedArgs = []

        for argIx, argExpr in enumerate(args):
            argType = argTypes[argIx]

            convertedArg = context.allocateUninitializedSlot(argType)

            successful = argExpr.convert_to_type_with_target(convertedArg, isExplicit)

            with context.ifelse(successful.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushTerminal(
                        native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
                    )

                with ifTrue:
                    context.markUninitializedSlotInitialized(convertedArg)

            convertedArgs.append(convertedArg)

        if outputVar.expr_type.typeRepresentation.ElementType != retType:
            raise Exception(f"Output type mismatch: {outputVar.expr_type.typeRepresentation} vs {retType}")

        res = context.call_py_function(overload.functionObj, convertedArgs, {}, typeWrapper(retType))

        if res is None:
            context.pushException(Exception, "unreachable")
            return

        outputVar.changeType(typeWrapper(retType), True).convert_copy_initialize(res)

        context.pushReturnValue(context.constant(True))

    def generateOverloadDispatch(self, context, methodName, overload, isExplicit, outputVar, args):
        """Produce the code that calls this specific overload.

        We return True if successful, False otherwise, and the output is a pointer to the result
        of the function call if we're successful.

        We attempt to convert each argument in 'args' to the relevant type, returning early if we
        can't.  If we're successful, the value at the output pointer is initialized.

        Args:
            context - an ExpressionConversionContext
            methodName - the name of the method we're compiling
            overload - the FunctionOverload we're trying to convert.
            isExplicit - are we using explicit conversion?
            outputVar - a TypedExpression(PointerTo(returnType)) we're supposed to initialize.
            args - the arguments to pass to the method (including the instance)

        """
        instance = args[0]

        argTypes = [a.expr_type for a in args]

        # get the Function object representing this entrypoint as a signature.
        # we specialize on the types in 'argTypes' because we might be specializing
        # a generic method on a specific subtype.
        signature = pickCallSignatureToImplement(overload, methodName, argTypes)

        assert len(signature.overloads[0].args) == len(argTypes)

        # each entrypoint generates a slot we could call.
        dispatchSlot = _types.allocateClassMethodDispatch(self.typeRepresentation, methodName, signature)

        classDispatchTable = self.classDispatchTable(instance)

        funcPtr = classDispatchTable.ElementPtrIntegers(0, 2).load().elemPtr(dispatchSlot).load()

        retType = overload.returnType or object

        convertedArgs = []

        for argIx, argExpr in enumerate(args):
            argType = typeWrapper(signature.overloads[0].args[argIx].typeFilter or object)

            convertedArg = context.allocateUninitializedSlot(argType)

            successful = argExpr.convert_to_type_with_target(convertedArg, isExplicit)

            with context.ifelse(successful.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushTerminal(
                        native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
                    )

                with ifTrue:
                    context.markUninitializedSlotInitialized(convertedArg)

            convertedArgs.append(convertedArg)

        if outputVar.expr_type.typeRepresentation.ElementType != retType:
            raise Exception(f"Output type mismatch: {outputVar.expr_type.typeRepresentation} vs {retType}")

        outputVar.changeType(typeWrapper(retType), True).convert_copy_initialize(
            context.call_function_pointer(funcPtr, convertedArgs, {}, typeWrapper(retType))
        )

        context.pushReturnValue(context.constant(True))

    def dispatchToSingleOverload(self, context, overload, explicitConversions, methodName, argTypes, instance, args):
        # get the Function object representing this entrypoint as a signature.
        # we specialize on the types in 'argTypes' because we might be specializing
        # a generic method on a specific subtype.
        signature = pickCallSignatureToImplement(overload, methodName, argTypes)

        # each entrypoint generates a slot we could call.
        dispatchSlot = _types.allocateClassMethodDispatch(self.typeRepresentation, methodName, signature)

        classDispatchTable = self.classDispatchTable(instance)

        funcPtr = classDispatchTable.ElementPtrIntegers(0, 2).load().elemPtr(dispatchSlot).load()

        retType = overload.returnType or object

        convertedArgs = []
        actualArgs = (instance,) + tuple(args)

        for argIx, argExpr in enumerate(actualArgs):
            convertedArgs.append(
                argExpr.convert_to_type(
                    signature.overloads[0].args[argIx].typeFilter or object,
                    explicit=explicitConversions
                )
            )

            if convertedArgs[-1] is None:
                return None

        kwargs = {}
        res = context.call_function_pointer(funcPtr, convertedArgs, kwargs, typeWrapper(retType))

        return res

    @staticmethod
    def compileMethodInstantiation(converter, interfaceClass, implementingClass, methodName, signature, callback):
        """Compile a concrete method instantiation.

        In this case, we have a call signature dicatated by a definition in the interface class,
        and our responsibility is to generate code for how the implementing class would implement
        a call of that signature.

        Args:
            converter - the PythonToNativeConverter that needs the concrete definition.
            interfaceClass - the Type for the class that instances will be masquerading as.
            implementingClass - the Type of our 'self' instance in this case
            methodName - (str) the name of the method we're compiling
            signature - (Type) a Function signature representing the signature we're compiling.
                This may be more specific than any of the signatures we're actually working on
                because it will have been specialized by the specific types we called it with.
            callback - the callback to pass to 'convert' so that we can install the compiled
                function pointer in the class vtable at link time.
        """
        assert len(signature.overloads) == 1

        # this is a FunctionOverload object representing the signature we're compiling.
        funcOverload = signature.overloads[0]

        # these are the types that we actually know at this point. They should be concrete (with
        # None replaced with actual signatures).
        argTypes = [implementingClass] + [arg.typeFilter for arg in funcOverload.args[1:]]

        for a in argTypes:
            assert a is not None

        argTypes = [typeWrapper(a) for a in argTypes]

        # this is the function we're implementing. It has its own type signature
        # but we have the signature from the base class as well, which may be more precise
        # in the inputs and less precise in the outputs.
        pyImpl = implementingClass.MemberFunctions[methodName]

        overloadAndIsExplicit = ClassWrapper.pickSingleOverloadForCall(pyImpl, argTypes)

        if overloadAndIsExplicit is not None:
            overload = overloadAndIsExplicit[0]

            # just one overload will do. We can just instantiate this particular function
            # with a signature that comes from the method overload signature itself.
            converter.convert(
                overload.functionObj,
                argTypes,
                funcOverload.returnType if funcOverload.returnType is not None else object,
                callback=callback
            )

            return True

        outputType = funcOverload.returnType or object

        converter.defineNativeFunction(
            f'implement_method.{implementingClass}.{interfaceClass}.{methodName}.{argTypes[1:]}',
            ('implement_method', implementingClass, interfaceClass, methodName, tuple(argTypes[1:])),
            list(argTypes),
            outputType,
            lambda context, outputVar, *args: (
                typeWrapper(implementingClass).generateMethodImplementation(context, methodName, outputType, args)
            ),
            callback=callback
        )

        return True

    def convert_set_attribute(self, context, instance, attribute, value):
        if not isinstance(attribute, int):
            ix = self.nameToIndex.get(attribute)
        else:
            ix = attribute

        if ix is None:
            return context.pushTerminal(
                generateThrowException(context, AttributeError("Attribute %s doesn't exist in %s" % (attribute, self.typeRepresentation)))
            )

        attr_type = typeWrapper(self.typeRepresentation.MemberTypes[ix])

        if attr_type.is_pod:
            return context.pushEffect(
                self.memberPtr(instance, ix).store(value.nonref_expr)
                >> self.setIsInitializedExpr(instance, ix)
            )
        else:
            member = context.pushReference(attr_type, self.memberPtr(instance, ix))

            with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(instance, ix))) as (true_block, false_block):
                with true_block:
                    member.convert_assign(value)
                with false_block:
                    member.convert_copy_initialize(value)
                    context.pushEffect(
                        self.setIsInitializedExpr(instance, ix)
                    )

            return native_ast.nullExpr

    def convert_type_call(self, context, typeInst, args, kwargs):
        if kwargs:
            raise NotImplementedError("can't kwargs-initialize a class yet")
        return context.push(
            self,
            lambda new_class:
                context.converter.defineNativeFunction(
                    'construct(' + self.typeRepresentation.__name__ + ")("
                    + ",".join([a.expr_type.typeRepresentation.__name__ for a in args]) + ")",
                    ('util', self, 'construct', tuple([a.expr_type for a in args])),
                    [a.expr_type for a in args],
                    self,
                    self.generateConstructor
                ).call(new_class, *args)
        )

    def generateConstructor(self, context, out, *args):
        context.pushEffect(
            out.expr.store(
                runtime_functions.malloc.call(
                    native_ast.const_int_expr(
                        _types.bytecount(self.typeRepresentation.HeldClass) + self.BYTES_BEFORE_INIT_BITS
                    )
                ).cast(self.getNativeLayoutType())
            ) >>
            # store a refcount
            out.expr.load().ElementPtrIntegers(0, 0).store(native_ast.const_int_expr(1)) >>
            # store the vtable
            out.expr.load().ElementPtrIntegers(0, 1).store(self.vtableExpr)
        )

        # clear bits of init flags
        for byteOffset in range(self.bytesOfInitBits):
            context.pushEffect(
                out.nonref_expr
                .cast(native_ast.UInt8.pointer())
                .ElementPtrIntegers(self.BYTES_BEFORE_INIT_BITS + byteOffset).store(native_ast.const_uint8_expr(0))
            )

        for i in range(len(self.classType.MemberTypes)):
            if _types.wantsToDefaultConstruct(self.classType.MemberTypes[i]):
                name = self.classType.MemberNames[i]

                if name in self.classType.MemberDefaultValues:
                    defVal = self.classType.MemberDefaultValues.get(name)
                    context.pushReference(self.classType.MemberTypes[i], self.memberPtr(out, i)).convert_copy_initialize(
                        nativepython.python_object_representation.pythonObjectRepresentation(context, defVal)
                    )
                else:
                    context.pushReference(self.classType.MemberTypes[i], self.memberPtr(out, i)).convert_default_initialize()
                context.pushEffect(self.setIsInitializedExpr(out, i))

        if '__init__' in self.typeRepresentation.MemberFunctions:
            initFuncType = typeWrapper(self.typeRepresentation.MemberFunctions['__init__'])
            initFuncType.convert_call(context, context.pushVoid(initFuncType), (out,) + args, {})
        else:
            if len(args):
                context.pushException(
                    TypeError,
                    "Can't construct a " + self.typeRepresentation.__qualname__ +
                    " with positional arguments because it doesn't have an __init__"
                )
