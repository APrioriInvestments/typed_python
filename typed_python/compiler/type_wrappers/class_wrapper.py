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

from typed_python.compiler.typed_expression import TypedExpression
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import PythonTypedFunctionWrapper
from typed_python.compiler.type_wrappers.arithmetic_wrapper import FloatWrapper, IntWrapper
from typed_python.compiler.type_wrappers.one_of_wrapper import OneOfWrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, _types, PointerTo, Bool, Int32

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler
from math import trunc, floor, ceil


typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


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
        ('classDispatchTable', class_dispatch_table_type.pointer()),
        ('initializationBitsBytecount', native_ast.Int64)
    ],
    name="VTable"
)


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
        self.bytesOfInitBits = (len(self.classType.MemberNames) + 7) // 8

        byteOffset = 0

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
        if otherType.typeRepresentation is Bool:
            return True
        if isinstance(otherType, ClassWrapper):
            if otherType.typeRepresentation in self.typeRepresentation.MRO:
                return True
            elif self.typeRepresentation in otherType.typeRepresentation.MRO:
                return "Maybe"
        if isinstance(otherType, IntWrapper):
            return "Maybe"
        if isinstance(otherType, FloatWrapper):
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

        if otherType.typeRepresentation == Bool:
            y = self.generate_method_call(context, "__bool__", (e,))
            if y is not None:
                return y.expr_type.convert_to_type_with_target(context, y, targetVal, False)
            else:
                y = self.generate_method_call(context, "__len__", (e,))
                if y is not None:
                    context.pushEffect(targetVal.expr.store(y.convert_to_type(int).nonref_expr.neq(0)))
                else:
                    context.pushEffect(targetVal.expr.store(context.constant(True)))
                return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)

    def convert_bool_cast(self, context, e):
        y = self.generate_method_call(context, "__bool__", (e,))
        if y is not None:
            return y
        y = self.generate_method_call(context, "__len__", (e,))
        if y is not None:
            return context.pushPod(bool, y.nonref_expr.neq(0))
        return context.constant(True)

    def convert_int_cast(self, context, e, raiseException=True):
        if raiseException:
            return self.generate_method_call(context, "__int__", (e,)) \
                or context.pushException(TypeError, f"__int__ not implemented for {self.typeRepresentation}")
        else:
            return self.generate_method_call(context, "__int__", (e,))

    def convert_float_cast(self, context, e, raiseException=True):
        if raiseException:
            return self.generate_method_call(context, "__float__", (e,)) \
                or context.pushException(TypeError, f"__float__ not implemented for {self.typeRepresentation}")
        else:
            return self.generate_method_call(context, "__float__", (e,))

    def convert_str_cast(self, context, e):
        return self.generate_method_call(context, "__str__", (e,)) \
            or e.convert_repr()

    def convert_bytes_cast(self, context, e):
        return self.generate_method_call(context, "__bytes__", (e,)) \
            or context.pushException(TypeError, f"__bytes__ not implemented for {self.typeRepresentation}")

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

    def bytesOfInitBitsForInstance(self, instance):
        if self.typeRepresentation.IsFinal:
            # if we're final, there are no subclasses, so we can be confident
            # we know the number of bits in the layout
            return native_ast.const_uint64_expr(self.bytesOfInitBits)
        else:
            return (
                self.get_layout_pointer(instance.nonref_expr)
                # get a pointer to the vtable
                .ElementPtrIntegers(0, 1)
                # load it
                .load()
                # get a pointer to the initializer flag area bytecount
                .ElementPtrIntegers(0, 3)
                .load()
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
            .elemPtr(self.bytesOfInitBitsForInstance(instance))
            .ElementPtrIntegers(self.indexToByteOffset[ix])
            .ElementPtrIntegers(self.BYTES_BEFORE_INIT_BITS)
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
            return self.generate_method_call(context, "__getattr__", (instance, context.constant(attribute))) \
                or super().convert_attribute(context, instance, attribute)

        if not nocheck:
            with context.ifelse(self.isInitializedNativeExpr(instance, ix)) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushException(AttributeError, "Attribute %s is not initialized" % attribute)

        return context.pushReference(
            self.typeRepresentation.MemberTypes[ix],
            self.memberPtr(instance, ix)
        )

    def resultTypesForCall(self, func, argTypes):
        resultTypes = set()

        for isExplicit in [False, True]:
            for o in func.overloads:
                # check each overload that we might match.
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(o, argTypes, {}, isExplicit)

                if mightMatch is False:
                    return resultTypes

                if o.returnType is None:
                    resultTypes.add(object)
                else:
                    resultTypes.add(o.returnType)

                if mightMatch is True:
                    return resultTypes

        return resultTypes

    def convert_method_call(self, context, instance, methodName, args, kwargs):
        # figure out which signature we'd want to use on the given args/kwargs
        if kwargs:
            raise Exception("Can't call methods with keyword args yet.")

        func = self.typeRepresentation.MemberFunctions[methodName]

        if self.typeRepresentation.IsFinal:
            # we can sidestep the vtable entirely
            return typeWrapper(func).convert_call(context, None, [instance] + list(args), kwargs)

        argTypes = [instance.expr_type] + [a.expr_type for a in args]

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
        overloadAndIsExplicit = PythonTypedFunctionWrapper.pickSingleOverloadForCall(func, argTypes, {})

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
        output_type = OneOfWrapper.mergeTypes(resultTypes)

        dispatchToOverloads = context.converter.defineNativeFunction(
            f'call_method.{self}.{methodName}.{argTypes[1:]}',
            ('call_method', self, methodName, tuple(argTypes[1:])),
            list(argTypes),
            output_type,
            lambda context, outputVar, *args: self.generateMethodDispatch(context, methodName, output_type, args)
        )

        return context.call_typed_call_target(dispatchToOverloads, [instance] + args)

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
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(overload, argTypes, {}, isExplicit)

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
                        (outputSlot.changeType(PointerTo(overloadRetType), False),) + args
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
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(overload, argTypes, {}, isExplicit)

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
        signature = PythonTypedFunctionWrapper.pickCallSignatureToImplement(overload, [a.expr_type for a in args])

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
        signature = PythonTypedFunctionWrapper.pickCallSignatureToImplement(overload, argTypes)

        assert len(signature.overloads[0].args) == len(argTypes)

        # each entrypoint generates a slot we could call.
        dispatchSlot = _types.allocateClassMethodDispatch(self.typeRepresentation, methodName, signature)

        classDispatchTable = self.classDispatchTable(instance)

        funcPtr = classDispatchTable.ElementPtrIntegers(0, 2).load().elemPtr(dispatchSlot).load()

        retType = overload.returnType or typeWrapper(object).typeRepresentation

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
            context.call_function_pointer(funcPtr, convertedArgs, typeWrapper(retType))
        )

        context.pushReturnValue(context.constant(True))

    def dispatchToSingleOverload(self, context, overload, explicitConversions, methodName, argTypes, instance, args):
        # get the Function object representing this entrypoint as a signature.
        # we specialize on the types in 'argTypes' because we might be specializing
        # a generic method on a specific subtype.
        signature = PythonTypedFunctionWrapper.pickCallSignatureToImplement(overload, argTypes)

        # each entrypoint generates a slot we could call.
        dispatchSlot = _types.allocateClassMethodDispatch(self.typeRepresentation, methodName, signature)

        classDispatchTable = self.classDispatchTable(instance)

        funcPtr = classDispatchTable.ElementPtrIntegers(0, 2).load().elemPtr(dispatchSlot).load()

        retType = overload.returnType or object

        if context.functionContext.converter.generateDebugChecks:
            with context.ifelse(funcPtr.cast(native_ast.Int64)) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushException(TypeError, "EMPTY SLOT")

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

        res = context.call_function_pointer(funcPtr, convertedArgs, typeWrapper(retType))

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

        returnType = funcOverload.returnType if funcOverload.returnType is not None else object

        typeWrapper(pyImpl).compileCall(converter, returnType, argTypes, {}, callback)

        return True

    def convert_getitem(self, context, instance, item):
        return self.generate_method_call(context, "__getitem__", (instance, item)) \
            or super().convert_getitem(context, instance, item)

    def convert_setitem(self, context, instance, item, value):
        return self.generate_method_call(context, "__setitem__", (instance, item, value)) \
            or super().convert_setitem(context, instance, item, value)

    def convert_set_attribute(self, context, instance, attribute, value):
        if not isinstance(attribute, int):
            ix = self.nameToIndex.get(attribute)
        else:
            ix = attribute

        if ix is None:
            if value is None:
                return self.generate_method_call(context, "__delattr__", (instance, context.constant(attribute))) \
                    or super().convert_set_attribute(context, instance, attribute, value)
            return self.generate_method_call(context, "__setattr__", (instance, context.constant(attribute), value)) \
                or super().convert_set_attribute(context, instance, attribute, value)

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
                        typed_python.compiler.python_object_representation.pythonObjectRepresentation(context, defVal)
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

    def convert_call(self, context, expr, args, kwargs):
        return self.generate_method_call(context, "__call__", [expr] + args)

    def convert_len(self, context, expr):
        return self.generate_method_call(context, "__len__", (expr,))

    def convert_abs(self, context, expr):
        return self.generate_method_call(context, "__abs__", (expr,))

    def convert_builtin(self, f, context, expr, a1=None):
        # handle builtins with additional arguments here:
        if f is format:
            if a1 is not None:
                return self.generate_method_call(context, "__format__", (expr, a1))
            else:
                return self.generate_method_call(context, "__format__", (expr, context.constant(''))) \
                    or self.generate_method_call(context, "__str__", (expr,)) \
                    or expr.convert_str_cast()
        if f is round:
            if a1 is not None:
                return self.generate_method_call(context, "__round__", (expr, a1)) \
                    or context.pushPod(
                        float,
                        runtime_functions.round_float64.call(expr.toFloat64().nonref_expr, a1.toInt64().nonref_expr)
                )
            else:
                return self.generate_method_call(context, "__round__", (expr, context.constant(0))) \
                    or context.pushPod(
                        float,
                        runtime_functions.round_float64.call(expr.toFloat64().nonref_expr, context.constant(0))
                )
        if a1 is not None:
            return None
        # handle builtins with no additional arguments here:
        if f is trunc:
            return self.generate_method_call(context, "__trunc__", (expr,)) \
                or context.pushPod(float, runtime_functions.trunc_float64.call(expr.toFloat64().nonref_expr))
        if f is floor:
            expr_float = self.convert_float_cast(context, expr, False)
            return self.generate_method_call(context, "__floor__", (expr,)) \
                or (expr_float and context.pushPod(float, runtime_functions.floor_float64.call(expr_float.nonref_expr)))
        if f is ceil:
            expr_float = self.convert_float_cast(context, expr, False)
            return self.generate_method_call(context, "__ceil__", (expr,)) \
                or (expr_float and context.pushPod(float, runtime_functions.ceil_float64.call(expr_float.nonref_expr)))
        if f is complex:
            return self.generate_method_call(context, "__complex__", (expr,))
        if f is dir:
            return self.generate_method_call(context, "__dir__", (expr,)) \
                or super().convert_builtin(f, context, expr)

        return super().convert_builtin(f, context, expr, a1)

    def convert_unary_op(self, context, expr, op):
        magic = "__pos__" if op.matches.UAdd else \
            "__neg__" if op.matches.USub else \
            "__invert__" if op.matches.Invert else \
            "__not__" if op.matches.Not else \
            ""
        return self.generate_method_call(context, magic, (expr,)) or super().convert_unary_op(context, expr, op)

    def convert_bin_op(self, context, l, op, r, inplace):
        magic = "__add__" if op.matches.Add else \
            "__sub__" if op.matches.Sub else \
            "__mul__" if op.matches.Mult else \
            "__truediv__" if op.matches.Div else \
            "__floordiv__" if op.matches.FloorDiv else \
            "__mod__" if op.matches.Mod else \
            "__matmul__" if op.matches.MatMult else \
            "__pow__" if op.matches.Pow else \
            "__lshift__" if op.matches.LShift else \
            "__rshift__" if op.matches.RShift else \
            "__or__" if op.matches.BitOr else \
            "__xor__" if op.matches.BitXor else \
            "__and__" if op.matches.BitAnd else \
            "__eq__" if op.matches.Eq else \
            "__ne__" if op.matches.NotEq else \
            "__lt__" if op.matches.Lt else \
            "__gt__" if op.matches.Gt else \
            "__le__" if op.matches.LtE else \
            "__ge__" if op.matches.GtE else \
            ""

        magic_inplace = '__i' + magic[2:] if magic and inplace else None

        return (magic_inplace and self.generate_method_call(context, magic_inplace, (l, r))) \
            or self.generate_method_call(context, magic, (l, r)) \
            or self.convert_comparison(context, l, op, r) \
            or super().convert_bin_op(context, l, op, r, inplace)

    def convert_comparison(self, context, left, op, right):
        if op.matches.Eq:
            native_expr = left.nonref_expr.cast(native_ast.UInt64).eq(right.nonref_expr.cast(native_ast.UInt64))
            return TypedExpression(context, native_expr, Bool, False)
        if op.matches.NotEq:
            native_expr = left.nonref_expr.cast(native_ast.UInt64).neq(right.nonref_expr.cast(native_ast.UInt64))
            return TypedExpression(context, native_expr, Bool, False)

        return context.pushException(TypeError, f"Can't compare instances of {left.expr_type.typeRepresentation}"
                                                f" and {right.expr_type.typeRepresentation} with {op}")

    def convert_bin_op_reverse(self, context, r, op, l, inplace):
        if op.matches.In:
            ret = self.generate_method_call(context, "__contains__", (r, l))
            return (ret and ret.toBool()) \
                or super().convert_bin_op_reverse(context, r, op, l, inplace)

        magic = "__radd__" if op.matches.Add else \
            "__rsub__" if op.matches.Sub else \
            "__rmul__" if op.matches.Mult else \
            "__rtruediv__" if op.matches.Div else \
            "__rfloordiv__" if op.matches.FloorDiv else \
            "__rmod__" if op.matches.Mod else \
            "__rmatmul__" if op.matches.MatMult else \
            "__rpow__" if op.matches.Pow else \
            "__rlshift__" if op.matches.LShift else \
            "__rrshift__" if op.matches.RShift else \
            "__ror__" if op.matches.BitOr else \
            "__rxor__" if op.matches.BitXor else \
            "__rand__" if op.matches.BitAnd else \
            ""

        return self.generate_method_call(context, magic, (r, l)) \
            or super().convert_bin_op_reverse(context, r, op, l, inplace)

    def convert_hash(self, context, expr):
        y = self.generate_method_call(context, "__hash__", (expr,))
        if y is not None:
            return y

        # default hash for Class types:
        HELD_CLASS_CAT_NO = 29
        PRIME_NO = 1000003

        vtp = _types._vtablePointer(self.typeRepresentation)
        return context.constant(Int32(((((HELD_CLASS_CAT_NO * PRIME_NO) ^ (vtp >> 32)) * PRIME_NO) ^ vtp) & 0xFFFFFFFF))
