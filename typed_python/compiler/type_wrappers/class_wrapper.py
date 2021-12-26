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

from typed_python.compiler.global_variable_definition import GlobalVariableMetadata
from typed_python.compiler.merge_type_wrappers import mergeTypeWrappers
from typed_python.compiler.typed_expression import TypedExpression
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import (
    PythonTypedFunctionWrapper,
    CannotBeDetermined,
    SomeInvalidClassReturnType,
    NoReturnTypeSpecified
)
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.type_wrappers.class_or_alternative_wrapper_mixin import (
    ClassOrAlternativeWrapperMixin
)
from typed_python import _types, PointerTo, Int32, Tuple, NamedTuple, bytecount, RefTo, SubclassOf, Class, Value

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler


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
        ('initializationBitsBytecount', native_ast.Int64),
        ('classTypePtr', native_ast.VoidPtr)
    ],
    name="VTable"
)


_classCouldBeInstanceOfCache = {}


def classCouldBeInstanceOf(cls, other):
    """Determine whether an instance of cls could be an instance of other.

    Returns True, False, or "Maybe"
    """
    if (cls, other) not in _classCouldBeInstanceOfCache:
        _classCouldBeInstanceOfCache[cls, other] = _classCouldBeInstanceOf(cls, other)

    return _classCouldBeInstanceOfCache[cls, other]


def _classCouldBeInstanceOf(cls, other):
    assert issubclass(cls, Class) or cls is Class, cls
    assert issubclass(other, Class) or other is Class, other

    if other is Class:
        return True

    if cls is Class:
        return "Maybe"

    if other in cls.MRO:
        return True

    elif cls in other.MRO:
        return "Maybe"

    # the only way we could be a subclass of the other is if there is a common subclass
    # of both of us.
    if cls.IsFinal or other.IsFinal:
        # if we're final, and this is not a parent of us, we are not one of it
        return False

    # if both classes have physical members, then there cannot be a common subclass
    if len(cls.MemberNames) and len(other.MemberNames):
        return False

    for name in set(cls.ClassMembers) & set(other.ClassMembers):
        if name not in ['__qualname__', '__name__', '__module__', '__typed_python_template__', '__classcell__']:
            if cls.ClassMembers[name] is not other.ClassMembers[name]:
                return False

    return "Maybe"


class ClassWrapper(ClassOrAlternativeWrapperMixin, RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    # is this wrapping a Class object
    is_class_wrapper = True

    BYTES_BEFORE_INIT_BITS = 16  # the refcount and vtable are both 8 byte integers.
    CAN_BE_NULL = False

    def __init__(self, t):
        super().__init__(t)

        self.nameToIndex = {}
        self.indexToByteOffset = {}
        self.classType = t

        element_types = [('refcount', native_ast.Int64), ('vtable', vtable_type.pointer()), ('data', native_ast.UInt8)]

        # this follows the general layout of 'held class' which is 1 bit per field
        # for initialization and then each field packed directly according to byte size
        if all(
            self.classType.ClassMembers[self.classType.MemberNames[i]].isNonempty
            for i in range(len(self.classType.MemberNames))
        ):
            self.bytesOfInitBits = 0
        else:
            self.bytesOfInitBits = (len(self.classType.MemberNames) + 7) // 8

        byteOffset = 0

        for i, name in enumerate(self.classType.MemberNames):
            self.nameToIndex[name] = i
            self.indexToByteOffset[i] = byteOffset

            byteOffset += _types.bytecount(self.classType.MemberTypes[i])

        self.layoutType = native_ast.Type.Struct(element_types=element_types, name=t.__qualname__+"Layout").pointer()

        self.vtableExpr = native_ast.Expression.GlobalVariable(
            name="cls_vtable_" + str(self.typeRepresentation) + "_" + str(id(self.typeRepresentation)),
            type=vtable_type.pointer(),
            metadata=GlobalVariableMetadata.ClassVtable(
                value=self.typeRepresentation
            )
        ).load()

    def fieldGuaranteedInitialized(self, ix):
        if self.classType.MemberNames[ix] not in self.typeRepresentation.ClassMembers:
            raise Exception(
                f"Can't find "
                f"{self.typeRepresentation.MemberNames[ix]} in "
                f"{list(self.typeRepresentation.ClassMembers)}"
            )
        if self.typeRepresentation.ClassMembers[
            self.classType.MemberNames[ix]
        ].isNonempty:
            return True

        return False

    def convert_refTo(self, context, instance):
        refToType = RefTo(self.typeRepresentation.HeldClass)

        return TypedExpression(
            context,
            (
                self.get_layout_pointer(instance)
                .ElementPtrIntegers(0, 2)
                .cast(typeWrapper(refToType).getNativeLayoutType())
            ),
            refToType,
            False
        )

    def convert_pointerTo(self, context, instance):
        PtrT = PointerTo(self.typeRepresentation.HeldClass)

        return TypedExpression(
            context,
            (
                self.get_layout_pointer(instance)
                .ElementPtrIntegers(0, 2)
                .cast(typeWrapper(PtrT).getNativeLayoutType())
            ),
            PtrT,
            False
        )

    def _can_convert_to_type(self, otherType, conversionLevel):
        if isinstance(otherType, ClassWrapper):
            return classCouldBeInstanceOf(self.typeRepresentation, otherType.typeRepresentation)

        return super()._can_convert_to_type(otherType, conversionLevel)

    def _can_convert_from_type(self, otherType, conversionLevel):
        return False

    def convert_fastnext(self, context, instance):
        return self.convert_method_call(context, instance, "__fastnext__", [], {})

    def convert_to_type_as_expression(
        self,
        context,
        expr,
        target_type,
        level: ConversionLevel,
        mayThrowOnFailure=False,
        assumeSuccessful=False
    ):
        if isinstance(target_type, ClassWrapper):
            if target_type.typeRepresentation in self.typeRepresentation.MRO:
                # this is an upcast
                index = _types.getDispatchIndexForType(target_type.typeRepresentation, self.typeRepresentation)

                return (
                    TypedExpression(
                        context,
                        target_type.withDispatchIndex(expr, index),
                        target_type,
                        False
                    ),
                    context.constant(True)
                )

            if target_type.typeRepresentation.IsFinal:
                if assumeSuccessful:
                    return (
                        target_type.stripClassDispatchIndex(context, expr),
                        context.constant(True)
                    )

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        otherType = targetVal.expr_type

        if isinstance(otherType, ClassWrapper):
            if otherType.typeRepresentation in self.typeRepresentation.MRO:
                # this is an upcast
                index = _types.getDispatchIndexForType(otherType.typeRepresentation, self.typeRepresentation)

                context.pushEffect(
                    targetVal.expr.store(
                        self.withDispatchIndex(instance, index)
                    )
                )
                targetVal.convert_incref()

                return context.constant(True)

            if otherType.typeRepresentation.IsFinal:
                initialized = context.allocateUninitializedSlot(bool)

                with context.ifelse(
                    self.get_class_type_ptr_as_voidptr(instance).cast(native_ast.UInt64).eq(
                        context.getTypePointer(targetVal.expr_type.typeRepresentation)
                        .cast(native_ast.UInt64)
                    )
                ) as (ifTrue, ifFalse):
                    with ifTrue:
                        context.pushEffect(
                            targetVal.expr.store(
                                self.get_layout_pointer(instance)
                            )
                        )
                        context.pushEffect(
                            initialized.expr.store(
                                native_ast.const_bool_expr(True)
                            )
                        )
                        targetVal.convert_incref()

                    with ifFalse:
                        context.pushEffect(
                            initialized.expr.store(
                                native_ast.const_bool_expr(False)
                            )
                        )

                return initialized
            else:
                return context.pushPod(
                    bool,
                    runtime_functions.classObjectDowncast.call(
                        instance.nonref_expr.cast(native_ast.VoidPtr),
                        targetVal.expr.cast(native_ast.VoidPtr),
                        context.getTypePointer(targetVal.expr_type.typeRepresentation)
                    )
                )

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def has_intiter(self):
        """Does this type support the 'intiter' format?"""
        return "__typed_python_int_iter_size__" in self.typeRepresentation.MemberFunctions

    def convert_intiter_size(self, context, instance):
        """If this type supports intiter, compute the size of the iterator.

        This function will return a TypedExpression(int) or None if it set an exception."""
        return self.convert_method_call(context, instance, "__typed_python_int_iter_size__", [], {})

    def convert_intiter_value(self, context, instance, valueInstance):
        """If this type supports intiter, compute the value of the iterator.

        This function will return a TypedExpression, or None if it set an exception."""
        return self.convert_method_call(context, instance, "__typed_python_int_iter_value__", [], {})

    def get_layout_pointer(self, instance):
        # diagnostic you can use to check whether our dispatch indices are getting
        # messed up.
        # if self.typeRepresentation.IsFinal:
        #     from typed_python import UInt64
        #     c = instance.context
        #     with c.ifelse(self.get_dispatch_index(instance)) as (ifTrue, ifFalse):
        #         with ifTrue:
        #             c.logDiagnostic(str(c.functionContext), ": ", str(self), ":",
        #                 c.pushPod(UInt64, self.get_dispatch_index(instance))
        #             )

        return self.get_layout_pointer_native(instance.nonref_expr)

    def get_layout_pointer_native(self, nonref_expr):
        # our layout is 48 bits of pointer and 16 bits of classDispatchTableIndex.
        # so whenever we interact with the pointer we need to chop off the top 16 bits
        if self.typeRepresentation.IsFinal:
            return nonref_expr.cast(self.layoutType)

        return (
            nonref_expr
            .cast(native_ast.UInt64)
            .bitand(native_ast.const_uint64_expr(0xFFFFFFFFFFFF))  # 48 bits of 1s
            .cast(self.layoutType)
        )

    def classDispatchTable(self, instance):
        if isinstance(instance.expr_type, ClassWrapper):
            classDispatchTables = (
                self.get_layout_pointer(instance)
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
        else:
            return runtime_functions.computeTypeClassDispatchTable.call(
                instance.nonref_expr,
                instance.context.getTypePointer(self.typeRepresentation)
            ).cast(class_dispatch_table_type.pointer())

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
                self.get_layout_pointer(instance)
                # get a pointer to the vtable
                .ElementPtrIntegers(0, 1)
                # load it
                .load()
                # get a pointer to the initializer flag area bytecount
                .ElementPtrIntegers(0, 3)
                .load()
            )

    def get_class_type_ptr_as_voidptr(self, instance):
        return (
            self.get_layout_pointer(instance)
            # get a pointer to the vtable
            .ElementPtrIntegers(0, 1)
            # load it
            .load()
            .ElementPtrIntegers(0, 4)
            .load()
        )

    def get_refcount_ptr_expr(self, nonref_expr):
        """Return a pointer to the object's refcount. Subclasses can override.

        Args:
            nonref_expr - a native expression equivalent to 'self.nonref_expr'. In most cases
                this will be the pointer to the actual refcounted data structure.
        """
        return self.get_layout_pointer_native(nonref_expr).ElementPtrIntegers(0, 0)

    def get_dispatch_index(self, instance):
        """Return the integer index of the current class dispatch within this instances' vtable."""
        return (
            instance.nonref_expr
            .cast(native_ast.UInt64)
            .rshift(native_ast.const_uint64_expr(48))
        )

    def convert_default_initialize(self, context, instance, force=False):
        if not _types.is_default_constructible(self.typeRepresentation) and not force:
            return context.pushException(TypeError, f"Can't default initialize instances of {self}")

        # just initialize it. We will not have an __init__ because __init__ makes
        # a class not default constructible.
        newVal = self.convert_type_call(context, None, [], {})

        return instance.convert_copy_initialize(newVal)

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        if self.typeRepresentation.IsFinal:
            self.generateNativeDestructorFunction(context, None, instance)
        else:
            vtablePtr = (
                self.get_layout_pointer(instance)
                .ElementPtrIntegers(0, 1)
                .load()
            )

            destructorPtr = (
                vtablePtr
                .ElementPtrIntegers(0, 1)
                .load()
            )

            with context.ifelse(destructorPtr.cast(native_ast.Int64)) as (ifTrue, ifFalse):
                with ifFalse:
                    # we have an empty slot. We need to compile it
                    context.pushEffect(
                        runtime_functions.compileClassDestructor.call(
                            vtablePtr.cast(native_ast.VoidPtr)
                        )
                    )

            return native_ast.CallTarget.Pointer(destructorPtr).call(
                instance.expr.cast(native_ast.VoidPtr)
            )

    def compileDestructor(self, converter):
        return converter.defineNativeFunction(
            "destructor_" + str(self.typeRepresentation),
            ('destructor', self),
            [self],
            typeWrapper(type(None)),
            self.generateNativeDestructorFunction
        )

    def generateNativeDestructorFunction(self, context, out, instance):
        instance = self.stripClassDispatchIndex(context, instance)

        for i in range(len(self.typeRepresentation.MemberTypes)):
            if not typeWrapper(self.typeRepresentation.MemberTypes[i]).is_pod:
                with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(instance, i))) as (true_block, false_block):
                    with true_block:
                        context.pushEffect(
                            self.convert_attribute(context, instance, i, nocheck=True).convert_destroy()
                        )

        context.pushEffect(runtime_functions.free.call(self.get_layout_pointer(instance).cast(native_ast.UInt8Ptr)))

    def memberPtr(self, instance, ix):
        return (
            self.get_layout_pointer(instance)
            .cast(native_ast.UInt8.pointer())
            .elemPtr(self.bytesOfInitBitsForInstance(instance))
            .ElementPtrIntegers(self.indexToByteOffset[ix] + self.BYTES_BEFORE_INIT_BITS)
            .cast(
                typeWrapper(self.typeRepresentation.MemberTypes[ix])
                .getNativeLayoutType()
                .pointer()
            )
        )

    def isInitializedNativeExpr(self, instance, ix):
        if self.fieldGuaranteedInitialized(ix):
            return native_ast.const_bool_expr(True)

        byte = ix // 8
        bit = ix % 8

        return (
            self.get_layout_pointer(instance)
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(self.BYTES_BEFORE_INIT_BITS + byte)
            .load()
            .rshift(native_ast.const_uint8_expr(bit))
            .bitand(native_ast.const_uint8_expr(1))
        ).cast(native_ast.Bool)

    def setIsInitializedExpr(self, instance, ix):
        if self.fieldGuaranteedInitialized(ix):
            return native_ast.nullExpr

        byte = ix // 8
        bit = ix % 8

        bytePtr = (
            self.get_layout_pointer(instance)
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(self.BYTES_BEFORE_INIT_BITS + byte)
        )

        return bytePtr.store(bytePtr.load().bitor(native_ast.const_uint8_expr(1 << bit)))

    def clearIsInitializedExpr(self, instance, ix):
        if self.fieldGuaranteedInitialized(ix):
            return native_ast.nullExpr

        assert instance.isReference

        byte = ix // 8
        bit = ix % 8

        bytePtr = (
            self.get_layout_pointer(instance)
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(self.BYTES_BEFORE_INIT_BITS + byte)
        )

        return bytePtr.store(bytePtr.load().bitand(native_ast.const_uint8_expr(255 - (1 << bit))))

    def convert_type_attribute(self, context, typeInst, attr):
        if (attr in self.typeRepresentation.StaticMemberFunctions or attr in self.typeRepresentation.ClassMemberFunctions):
            methodType = BoundMethodWrapper(
                _types.BoundMethod(typeInst.expr_type.typeRepresentation, attr)
            )

            return typeInst.changeType(methodType)

        return super().convert_type_attribute(context, typeInst, attr)

    def convert_type_method_call(self, context, typeInst, methodName, args, kwargs):
        if methodName in self.typeRepresentation.ClassMemberFunctions:
            func = self.typeRepresentation.ClassMemberFunctions[methodName]

            return typeWrapper(func).convert_call(context, None, [typeInst] + list(args), kwargs)

        if methodName in self.typeRepresentation.StaticMemberFunctions:
            func = self.typeRepresentation.StaticMemberFunctions[methodName]

            return typeWrapper(func).convert_call(context, None, list(args), kwargs)

        return super().convert_type_method_call(context, typeInst, methodName, args, kwargs)

    def convert_attribute(self, context, instance, attribute, nocheck=False):
        if attribute in self.typeRepresentation.MemberFunctions:
            methodType = BoundMethodWrapper(_types.BoundMethod(self.typeRepresentation, attribute))

            return instance.changeType(methodType)

        if (
            attribute in self.typeRepresentation.ClassMemberFunctions
            or attribute in self.typeRepresentation.StaticMemberFunctions
        ):
            instanceType = instance.convert_typeof()

            methodType = BoundMethodWrapper(
                _types.BoundMethod(instanceType.expr_type.typeRepresentation, attribute)
            )

            return instanceType.changeType(methodType)

        if attribute in self.typeRepresentation.PropertyFunctions:
            return self.convert_method_call(context, instance, attribute, (), {})

        if not isinstance(attribute, int):
            ix = self.nameToIndex.get(attribute)
        else:
            ix = attribute

        if ix is None:
            if self.has_method("__getattr__"):
                return self.convert_method_call(context, instance, "__getattr__", (context.constant(attribute),), {})
            return super().convert_attribute(context, instance, attribute)

        if not nocheck:
            with context.ifelse(self.isInitializedNativeExpr(instance, ix)) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushAttributeError(attribute)

        return context.pushReference(
            self.typeRepresentation.MemberTypes[ix],
            self.memberPtr(instance, ix)
        )

    def resultTypesForCall(self, func, argTypes, kwargTypes):
        resultTypes = set()

        for conversionLevel in ConversionLevel.functionConversionSequence():
            for o in func.overloads:
                # check each overload that we might match.
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(o, argTypes, kwargTypes, conversionLevel)

                if mightMatch is not False:
                    ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

                    actualArgTypes, actualKwargTypes = ExpressionConversionContext.computeOverloadSignature(
                        o,
                        argTypes,
                        kwargTypes
                    )

                    returnType = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(
                        o, actualArgTypes, actualKwargTypes
                    )

                    if returnType is SomeInvalidClassReturnType:
                        pass
                    elif returnType is NoReturnTypeSpecified:
                        resultTypes.add(object)
                    else:
                        assert isinstance(returnType, type)
                        resultTypes.add(returnType)

                    if mightMatch is True:
                        return resultTypes

        return resultTypes

    def getMethodOrPropertyBody(self, name):
        return (
            self.typeRepresentation.MemberFunctions.get(name)
            or self.typeRepresentation.PropertyFunctions.get(name)
        )

    def has_method(self, methodName):
        assert isinstance(methodName, str)
        return self.getMethodOrPropertyBody(methodName) is not None

    def convert_type_method_call_virtual(self, context, typeInst, methodName, args, kwargs):
        """Convert a method call on a Type* which is us, or a subclass."""
        if self.typeRepresentation.IsFinal:
            return self.convert_type_method_call(context, typeInst, methodName, args, kwargs)

        return self.convert_method_call_virtual(context, typeInst, methodName, args, kwargs)

    def convert_method_call(self, context, instance, methodName, args, kwargs):
        # figure out which signature we'd want to use on the given args/kwargs
        func = self.getMethodOrPropertyBody(methodName)

        if func is None:
            context.pushException(AttributeError, methodName)
            return None

        if self.typeRepresentation.IsFinal:
            # we can sidestep the vtable entirely
            return typeWrapper(func).convert_call(context, None, [instance] + list(args), kwargs)

        return self.convert_method_call_virtual(context, instance, methodName, args, kwargs)

    def convert_method_call_virtual(self, context, instance, methodName, args, kwargs):
        """Convert a method call where dispatch is 'virtual'.

        This means we don't know the concrete type of 'self' (or 'cls') - its us or
        a subclass of us, and we have to use a virtual dispatch table to find the
        appropriate implementation.

        This version of the method will work if 'instance' is a ClassWrapper or
        a SubclassOfWrapper, dispatching to MemberFunction if its a classwrapper and
        Static/Classmethods otherwise.

        In both cases, we have to determine which of the visible function signatures
        we match. There may be more than one, in which case we have to generate code
        that checks if we match the signature before dispatching.

        Once we find a match, we'll look in a VTable attached to 'instance' to find
        an appropriate function pointer and use that to actually execute the function.
        """
        argTypes = [instance.expr_type] + [a.expr_type for a in args]
        kwargTypes = {k: v.expr_type for k, v in kwargs.items()}

        if isinstance(instance.expr_type, ClassWrapper):
            # this is a regular method call
            if methodName in self.typeRepresentation.MemberFunctions:
                func = self.typeRepresentation.MemberFunctions[methodName]
            else:
                func = self.typeRepresentation.PropertyFunctions[methodName]
            isStatic = False
        else:
            # this is a call where instance is a Type*.
            if methodName == '__typed_python_virtual_new__':
                @typed_python.Function
                def funcObj(cls, *args, **kwargs):
                    return cls(*args, **kwargs)

                func = type(funcObj)
                isStatic = False
            elif methodName in self.typeRepresentation.ClassMemberFunctions:
                func = self.typeRepresentation.ClassMemberFunctions[methodName]
                isStatic = False
            else:
                func = self.typeRepresentation.StaticMemberFunctions[methodName]
                isStatic = True

        if isStatic:
            matchArgTypes = argTypes[1:]
        else:
            matchArgTypes = argTypes

        # first, see if there is exacly one possible overload we could match. If so,
        # we don't need multiple dispatch code.
        overloadAndConversionLevel = PythonTypedFunctionWrapper.pickSingleOverloadForCall(func, matchArgTypes, kwargTypes)

        if overloadAndConversionLevel is not None:
            return self.dispatchToSingleOverload(
                context,
                overloadAndConversionLevel[0],
                overloadAndConversionLevel[1],
                methodName,
                instance,
                args,
                kwargs,
                isStatic
            )

        resultTypes = self.resultTypesForCall(func, matchArgTypes, kwargTypes)

        if not resultTypes:
            # we can't call anything
            context.pushException(
                Exception,
                f"No overload could be found for compiled dispatch to "
                f"{self}.{methodName} with args {matchArgTypes} and {kwargTypes}."
            )
            return None

        definitelyThrows = False

        if any(r is CannotBeDetermined for r in resultTypes):
            output_type = typeWrapper(object)
        else:
            validTypes = [x for x in resultTypes if x is not SomeInvalidClassReturnType]
            if validTypes:
                output_type = mergeTypeWrappers(validTypes)
            else:
                output_type = type(None)
                definitelyThrows = True

        argSignatureStrings = [str(x) for x in matchArgTypes[1:]]
        argSignatureStrings.extend([f"{k}={v}" for k, v in kwargTypes.items()])

        dispatchToOverloads = context.converter.defineNativeFunction(
            f'call_method.{self}.{methodName}({",".join(argSignatureStrings)})',
            ('call_method', self, methodName, tuple(matchArgTypes), tuple(kwargTypes.items())),
            list(argTypes) + list(kwargTypes.values()),
            output_type,
            lambda context, outputVar, *args: self.generateMethodDispatch(
                context,
                instance,
                methodName,
                output_type,
                args,
                [None for a in argTypes] + list(kwargTypes),
                isStatic
            )
        )

        res = context.call_typed_call_target(
            dispatchToOverloads,
            [instance] + args + list(kwargs.values())
        )

        if definitelyThrows:
            # we defined this as returning None, but because it throws we don't want to
            # actually return None into the typing layer.
            return

        return res

    def generateMethodDispatch(self, context, instance, methodName, methodReturnType, args, argNames, isStatic):
        """Generate native code that tries to dispatch to each of func's overloads

        We try each overload, with conversionLevel going up through the alternatives.
        The first one that succeeds gets to produce the output.

        Args:
            context - an ExpressionConversionContext
            methodName - the name of the method we're trying to dispatch to
            methodReturnType - the output type we are expecting to return. This will be the union
                of the return types of all the overloads that might participate in this dispatch.
            instance - the class or type we are dispatching on.
            args - the typed_expression for our arguments, including 'instance'
            argNames - for each arg, None if it was a positional argument, or the name
                of the argument.
        """
        # these are the arguments we'll pass to the dispatch
        argTypes = [a.expr_type for i, a in enumerate(args) if argNames[i] is None]
        kwargTypes = {argNames[i]: a.expr_type for i, a in enumerate(args) if argNames[i] is not None}

        # these are the arguments we'll use when matching against the function's args
        if isStatic:
            matchArgTypes = argTypes[1:]
        else:
            matchArgTypes = argTypes

        if isinstance(instance.expr_type, ClassWrapper):
            func = self.getMethodOrPropertyBody(methodName)
        elif isStatic:
            func = self.typeRepresentation.StaticMemberFunctions[methodName]
        else:
            func = self.typeRepresentation.ClassMemberFunctions[methodName]

        def makeOverloadDispatcher(overload, outputType, conversionLevel):
            return lambda context, _, outputVar, *args: self.generateOverloadDispatch(
                context, methodName, overload, conversionLevel, outputType, outputVar, args, argNames, isStatic
            )

        for conversionLevel in ConversionLevel.functionConversionSequence():
            for overloadIndex, overload in enumerate(func.overloads):
                mightMatch = PythonTypedFunctionWrapper.overloadMatchesSignature(overload, matchArgTypes, kwargTypes, conversionLevel)

                if mightMatch is not False:
                    overloadRetType = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(overload, matchArgTypes, kwargTypes)

                    if overloadRetType is CannotBeDetermined:
                        assert False, "We should be forcing and checking the type at runtime here"

                    if overloadRetType is SomeInvalidClassReturnType or overloadRetType is NoReturnTypeSpecified:
                        # even if the overloadRetType for the overloads presented here is
                        # invalid, its possible for someone to define an additional override that's
                        # not invalid. So we need to stick with the stated overload type of the
                        # existing overloads
                        overloadRetType = methodReturnType.typeRepresentation

                    assert isinstance(overloadRetType, type), type(overloadRetType)

                    testSingleOverloadForm = context.converter.defineNativeFunction(
                        f'call_overload.{self}.{methodName}.{overloadIndex}.'
                        f'{conversionLevel.LEVEL}.{argTypes[1:]}.{kwargTypes}->{overloadRetType}',
                        ('call_overload', self, methodName, overloadIndex, conversionLevel.LEVEL,
                         overloadRetType, tuple(argTypes[1:]), tuple(kwargTypes.items())),
                        [PointerTo(overloadRetType)] + list(argTypes) + list(kwargTypes.values()),
                        typeWrapper(bool),
                        makeOverloadDispatcher(overload, overloadRetType, conversionLevel)
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
                            actualResult = outputSlot.convert_to_type(methodReturnType, ConversionLevel.Signature)

                            if actualResult is not None:
                                context.pushReturnValue(actualResult)

                    # if we definitely match, we can return early
                    if mightMatch is True:
                        context.pushException(TypeError, f"Failed to find a dispatchable overload for {self}.{methodName} matching {args}")
                        return

        # generate a cleanup handler for the cases where we don't match a method signature.
        # this should actually be hitting the interpreter instead.
        context.pushException(TypeError, f"Failed to find a dispatchable overload for {self}.{methodName} matching {args}")

    def generateOverloadDispatch(self, context, methodName, overload, conversionLevel, retType, outputVar, args, argNames, isStatic):
        """Produce the code that calls this specific overload.

        We return True if successful, False otherwise, and the output is a pointer to the result
        of the function call if we're successful.

        We attempt to convert each argument in 'args' to the relevant type, returning early if we
        can't.  If we're successful, the value at the output pointer is initialized.

        Args:
            context - an ExpressionConversionContext
            methodName - the name of the method we're compiling
            overload - the FunctionOverload we're trying to convert.
            conversionLevel - the degree of conversion we apply to each argument.
            retType - the return type we're expected to match
            outputVar - a TypedExpression(PointerTo(returnType)) we're supposed to initialize.
            args - the arguments to pass to the method (including the instance)
            argNames - a list containing, for each argument, None if the argument is a positional argument,
                or the string name if it was a keyword argument.
            isStatic - is this a staticmethod, so that we don't pass an instance arg
        """
        instance = args[0]

        assert len(args) == len(argNames)

        argTypes = [a.expr_type for i, a in enumerate(args) if argNames[i] is None]
        kwargTypes = {argNames[i]: a.expr_type for i, a in enumerate(args) if argNames[i] is not None}

        if isStatic:
            matchArgTypes = argTypes[1:]
            matchArgs = args[1:]
        else:
            matchArgTypes = argTypes
            matchArgs = args

        ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

        argAndKwargTypes = ExpressionConversionContext.computeOverloadSignature(overload, matchArgTypes, kwargTypes)
        assert argAndKwargTypes is not None, "we should have already guaranteed this signature matches."

        convertedArgs = []

        for argIx, argExpr in enumerate(matchArgs):
            if argIx == 0 and not isStatic:
                argType = args[0].expr_type
            elif argNames[argIx] is None:
                argType = argAndKwargTypes[0][argIx]
            else:
                argType = argAndKwargTypes[1][argNames[argIx]]

            argType = typeWrapper(argType)

            convertedArg = context.allocateUninitializedSlot(argType)

            successful = argExpr.convert_to_type_with_target(convertedArg, conversionLevel)

            with context.ifelse(successful.nonref_expr) as (ifTrue, ifFalse):
                with ifFalse:
                    context.pushTerminal(
                        native_ast.Expression.Return(arg=native_ast.const_bool_expr(False))
                    )

                with ifTrue:
                    context.markUninitializedSlotInitialized(convertedArg)

            convertedArgs.append(convertedArg)

        # determine if we're matching based on the instance (method) or type (classmethod/staticmethod)
        matchesAsInstance = isinstance(args[0].expr_type, ClassWrapper)

        # each entrypoint generates a slot we could call which is indexed by whether we're
        # matching as an instance, and the arguments to the function.

        # note that we explicitly disregard the type of the first argument in the signature, since
        # we don't want to generate a separate entrypoint for each possible overloaded known call type.
        argTupleType = Tuple(
            Value('instance' if matchesAsInstance else 'type'),
            *[x.typeRepresentation for x in argAndKwargTypes[0][0 if isStatic else 1:]]
        )
        kwargTupleType = NamedTuple(**{k: v.typeRepresentation for k, v in argAndKwargTypes[1].items()})

        dispatchSlot = context.allocateClassMethodDispatchSlot(
            self.typeRepresentation,
            methodName,
            retType,
            argTupleType,
            kwargTupleType
        )

        classDispatchTable = self.classDispatchTable(instance)

        funcPtr = classDispatchTable.ElementPtrIntegers(0, 2).load().elemPtr(dispatchSlot).load()

        with context.ifelse(funcPtr.cast(native_ast.Int64)) as (ifTrue, ifFalse):
            with ifFalse:
                # we have an empty slot. We need to compile it
                context.pushEffect(
                    runtime_functions.compileClassDispatch.call(
                        classDispatchTable.cast(native_ast.VoidPtr),
                        dispatchSlot
                    )
                )

        if outputVar.expr_type.typeRepresentation.ElementType != retType:
            raise Exception(f"Output type mismatch: {outputVar.expr_type.typeRepresentation} vs {retType}")

        if isStatic:
            # if we're a staticmethod call we need to stick ourselves on the front.
            # this is because the method signature still expects the argument
            convertedArgs = [args[0]] + convertedArgs

        outputVar.changeType(typeWrapper(retType), True).convert_copy_initialize(
            context.call_function_pointer(funcPtr, convertedArgs, typeWrapper(retType))
        )

        context.pushReturnValue(context.constant(True))

    def dispatchToSingleOverload(
        self,
        context,
        overload,
        conversionLevel,
        methodName,
        instance,
        args,
        kwargs,
        isStatic
    ):
        """Dispatch a single call to a vtable.

        Given that there is a single overload 'overload' that definitely matches our signature,
        we have to generate code that picks the correct function pointer, compiles it if its
        empty, and dispatches to it.
        """

        # pack the args/kwargs into the form that will actually be passed to the function itself.
        argTypes = ([instance.expr_type] if not isStatic else []) + [a.expr_type for a in args]
        kwargTypes = {k: v.expr_type for k, v in kwargs.items()}

        ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

        # figure out the actual call signature we'll use
        argAndKwargTypes = ExpressionConversionContext.computeOverloadSignature(overload, argTypes, kwargTypes)

        assert argAndKwargTypes is not None, "we should have already guaranteed this signature matches."

        for k in argAndKwargTypes[1]:
            assert isinstance(k, str), argAndKwargTypes[1]

        actualArgTypes, actualKwargTypes = context.computeOverloadSignature(overload, argTypes, kwargTypes)

        retType = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(
            overload,
            actualArgTypes,
            actualKwargTypes
        )

        if retType is CannotBeDetermined:
            return instance.convert_to_type(
                object, ConversionLevel.Signature
            ).convert_method_call(methodName, args, kwargs)

        if retType is SomeInvalidClassReturnType:
            funcWrapper = typeWrapper(overload.functionTypeObject)
            canBail = funcWrapper.checkInvalidClassReturn(context, overload.index, [instance] + args, kwargs)
            assert canBail is True
            return

        if retType is NoReturnTypeSpecified:
            # if the user doesn't annotate anything, and we're dispatching
            # then the best we can do is to assume object.
            retType = object

        # determine if we're matching based on the instance (method) or type (classmethod/staticmethod)
        matchesAsInstance = isinstance(instance.expr_type, ClassWrapper)

        # each entrypoint generates a slot we could call which is indexed by whether we're
        # matching as an instance, and the arguments to the function.

        # note that we explicitly disregard the type of the first argument in the signature, since
        # we don't want to generate a separate entrypoint for each possible overloaded known call type.
        argTupleType = Tuple(
            Value('instance' if matchesAsInstance else 'type'),
            *[x.typeRepresentation for x in argAndKwargTypes[0][1 if not isStatic else 0:]]
        )

        kwargTupleType = NamedTuple(**{k: v.typeRepresentation for k, v in argAndKwargTypes[1].items()})

        dispatchSlot = context.allocateClassMethodDispatchSlot(
            self.typeRepresentation,
            methodName,
            retType,
            argTupleType,
            kwargTupleType
        )

        classDispatchTable = self.classDispatchTable(instance)

        funcPtr = classDispatchTable.ElementPtrIntegers(0, 2).load().elemPtr(dispatchSlot).load()

        with context.ifelse(funcPtr.cast(native_ast.Int64)) as (ifTrue, ifFalse):
            with ifFalse:
                # we have an empty slot. We need to compile it
                context.pushEffect(
                    runtime_functions.compileClassDispatch.call(
                        classDispatchTable.cast(native_ast.VoidPtr),
                        dispatchSlot
                    )
                )

        # we always pass the instance, regardless of whether this is a
        # regular method, classmethod or staticmethod call. Receiving code
        # can decide what to do with it. We don't have to worry about the case
        # where the instance we're passing is Void (which would happen if
        # this were a Final class and we knew the type exactly) since then
        # we wouldn't be using virtual dispatch at all
        assert not self.typeRepresentation.IsFinal
        convertedArgs = [instance]

        for argIx in range(len(args)):
            convertedArgs.append(
                args[argIx].convert_to_type(
                    argAndKwargTypes[0][argIx + (1 if not isStatic else 0)],
                    conversionLevel
                )
            )

            if convertedArgs[-1] is None:
                return None

        for argName in kwargs:
            convertedArgs.append(
                kwargs[argName].convert_to_type(
                    argAndKwargTypes[1][argName],
                    conversionLevel
                )
            )

            if convertedArgs[-1] is None:
                return None

        res = context.call_function_pointer(funcPtr, convertedArgs, typeWrapper(retType))

        return res

    def stripClassDispatchIndex(self, context, instance):
        """Return 'instance' with a class-dispatch of 0.

        When we call a method that's overridden in a child class, we need to ensure
        that the pointer argument matches the type that the child is expecting.

        For instance, if class B is the base, C is the child, and they have
        'f(self)' defined, both B and C's version of 'f' expect to receive an instance
        with their own type. However, if we have an instance of C which we know as
        B, then when we call 'f', C's version of 'f' will get 'self' masquerading as
        C with a pointer dispatch of 1.

        This function strips that 1 off.
        """
        return TypedExpression(
            context,
            instance.nonref_expr
            .cast(native_ast.UInt64)
            .bitand(native_ast.const_uint64_expr(0xFFFFFFFFFFFF))  # 48 bits of 1s
            .cast(self.layoutType),
            self,
            False
        )

    @staticmethod
    def compileVirtualMethodInstantiation(
        converter, interfaceClass, implementingClass,
        methodName, retType, argTypeTuple, kwargTypeTuple
    ):
        """Compile a concrete method instantiation.

        In this case, we have a call signature dicatated by a definition in the interface class,
        and our responsibility is to generate code for how the implementing class would implement
        a call of that signature.

        Args:
            converter - the PythonToNativeConverter that needs the concrete definition.
            interfaceClass - the Type for the class that instances will be masquerading as.
            implementingClass - the Type of our 'self' instance in this case
            methodName - (str) the name of the method we're compiling
            retType - (Type) - the return type for this version of the function
            argTypeTuple - (Tuple) - a Tuple type containing the types of the positional arguments
                to the function, including the instance or type object itself.
            kwargTypeTuple - (NamedTuple) - a NamedTuple type containing the types of the
                keyword arguments to this function.
        """
        # the first argument tuple will either be 'instance' or 'type'
        firstArgType = argTypeTuple.ElementTypes[0].Value

        assert firstArgType in ('type', 'instance')

        # these are the types that we actually know from the signature
        argTypesWithoutInstance = [typeWrapper(x) for x in argTypeTuple.ElementTypes[1:]]
        kwargTypes = {
            kwargTypeTuple.ElementNames[i]: typeWrapper(kwargTypeTuple.ElementTypes[i])
            for i in range(len(kwargTypeTuple.ElementNames))
        }

        if firstArgType == 'instance':
            argTypes = [typeWrapper(implementingClass)] + argTypesWithoutInstance
            funcObj = typeWrapper(implementingClass).getMethodOrPropertyBody(methodName)

            def firstArgConversion(arg):
                # Ensure that the function knows 'arg' as implementingClass, not interfaceClass
                return arg.expr_type.stripClassDispatchIndex(arg.context, arg)

        else:
            # either way, we'll get passed a pointer. The interface won't be final
            # since otherwise we wouldn't need to use virtual dispatch
            assert not interfaceClass.IsFinal

            argTypes = [typeWrapper(SubclassOf(interfaceClass))] + argTypesWithoutInstance

            if methodName == '__typed_python_virtual_new__':
                # we're supposed to return a concrete instance
                @typed_python.Function
                def funcInst(cls, *args, **kwargs):
                    return cls(*args, **kwargs)

                funcObj = type(funcInst)

                def firstArgConversion(arg):
                    # Ensure that the function knows 'arg' as implementingClass, not interfaceClass
                    return arg.context.constant(implementingClass)

            elif methodName in implementingClass.ClassMemberFunctions:
                # we'll get passed
                funcObj = implementingClass.ClassMemberFunctions[methodName]

                def firstArgConversion(arg):
                    # Ensure that the function knows 'arg' as implementingClass, not interfaceClass
                    return arg.context.constant(implementingClass)
            else:
                funcObj = implementingClass.StaticMemberFunctions[methodName]

                def firstArgConversion(arg):
                    # Static methods don't take an argument at all
                    return None

        assert bytecount(funcObj.ClosureType) == 0, "Class methods should have empty closures."

        return typeWrapper(funcObj).compileCall(
            converter, retType, argTypes, kwargTypes, False, firstArgConversion=firstArgConversion
        )

    def convert_set_attribute(self, context, instance, attribute, value):
        if not isinstance(attribute, int):
            ix = self.nameToIndex.get(attribute)
        else:
            ix = attribute

        if ix is None:
            if value is None:
                if self.has_method("__delattr__"):
                    return self.convert_method_call(context, instance, "__delattr__", (context.constant(attribute),), {})

                return RefcountedWrapper.convert_set_attribute(self, context, instance, attribute, value)

            if self.has_method("__setattr__"):
                return self.convert_method_call(context, instance, "__setattr__", (context.constant(attribute), value), {})

            return RefcountedWrapper.convert_set_attribute(self, context, instance, attribute, value)

        attr_type = typeWrapper(self.typeRepresentation.MemberTypes[ix])

        if value is None:
            # we're deleting this attribute
            if self.fieldGuaranteedInitialized(ix):
                return context.pushException(
                    AttributeError,
                    f"Attribute '{self.classType.MemberNames[ix]}' cannot be deleted"
                )

            with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(instance, ix))) as (
                true_block, false_block
            ):
                with true_block:
                    if not attr_type.is_pod:
                        member = context.pushReference(attr_type, self.memberPtr(instance, ix))
                        member.convert_destroy()

                    context.pushEffect(
                        self.clearIsInitializedExpr(instance, ix)
                    )

                with false_block:
                    context.pushException(
                        AttributeError,
                        f"Attribute '{self.classType.MemberNames[ix]}' is not initialized"
                    )

            return context.constant(None)

        value = value.convert_to_type(attr_type, ConversionLevel.ImplicitContainers)
        if value is None:
            return None

        if attr_type.is_pod:
            context.pushEffect(
                self.memberPtr(instance, ix).store(value.nonref_expr)
                >> self.setIsInitializedExpr(instance, ix)
            )
            return context.constant(None)
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

            return context.constant(None)

    def convert_type_call(self, context, typeInst, args, kwargs):
        # pack the named arguments onto the back of the call, and pass
        # a tuple of None|string representing the names
        argNames = (None,) * len(args) + tuple(kwargs)
        args = tuple(args) + tuple(kwargs.values())

        return context.push(
            self,
            lambda new_class:
                context.converter.defineNativeFunction(
                    'construct(' + self.typeRepresentation.__name__ + ")("
                    + ",".join([
                        (argNames[i] + '=' if argNames[i] is not None else "") +
                        args[i].expr_type.typeRepresentation.__name__
                        for i in range(len(args))
                    ]) + ")",
                    ('util', self, 'construct', tuple([a.expr_type for a in args]), argNames),
                    [a.expr_type for a in args],
                    self,
                    lambda context, out, *args: self.generateConstructor(context, out, argNames, *args)
                ).call(new_class, *args)
        )

    def generateConstructor(self, context, out, argNames, *args):
        """Generate native code to initialize a Class object from a set of args/kwargs.

        Args:
            context - the NativeFunctionConversionContext governing this conversion
            out - a TypedExpression pointing to an uninitialized Class instance.
            argNames - a tuple of (None|str) with the names of the args as they were passed.
            *args - Typed expressions representing each argument passed to us.
        """
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
            if _types.wantsToDefaultConstruct(self.classType.MemberTypes[i]) or self.fieldGuaranteedInitialized(i):
                name = self.classType.MemberNames[i]

                if name in self.classType.MemberDefaultValues:
                    defVal = self.classType.MemberDefaultValues.get(name)
                    context.pushReference(self.classType.MemberTypes[i], self.memberPtr(out, i)).convert_copy_initialize(
                        typed_python.compiler.python_object_representation.pythonObjectRepresentation(context, defVal)
                    )
                else:
                    context.pushReference(self.classType.MemberTypes[i], self.memberPtr(out, i)).convert_default_initialize()
                context.pushEffect(self.setIsInitializedExpr(out, i))

        # break our args back out to unnamed and named arguments
        unnamedArgs = []
        namedArgs = {}

        for argIx in range(len(args)):
            if argNames[argIx] is not None:
                namedArgs[argNames[argIx]] = args[argIx]
            else:
                unnamedArgs.append(args[argIx])

        if '__init__' in self.typeRepresentation.MemberFunctions:
            initFuncType = typeWrapper(self.typeRepresentation.MemberFunctions['__init__'])
            initFuncType.convert_call(
                context,
                context.push(initFuncType, lambda expr: None),
                (out,) + tuple(unnamedArgs),
                namedArgs
            )
        else:
            if len(unnamedArgs):
                context.pushException(
                    TypeError,
                    "Can't construct a " + self.typeRepresentation.__qualname__ +
                    " with positional arguments because it doesn't have an __init__"
                )

            for name, arg in namedArgs.items():
                self.convert_set_attribute(context, out, name, arg)

    def convert_comparison(self, context, left, op, right):
        if op.matches.Eq:
            native_expr = left.nonref_expr.cast(native_ast.UInt64).eq(right.nonref_expr.cast(native_ast.UInt64))
            return TypedExpression(context, native_expr, bool, False)
        if op.matches.NotEq:
            native_expr = left.nonref_expr.cast(native_ast.UInt64).neq(right.nonref_expr.cast(native_ast.UInt64))
            return TypedExpression(context, native_expr, bool, False)

        return context.pushException(TypeError, f"Can't compare instances of {left.expr_type.typeRepresentation}"
                                                f" and {right.expr_type.typeRepresentation} with {op}")

    def convert_hash(self, context, expr):
        if self.has_method("__hash__"):
            return self.convert_method_call(context, expr, "__hash__", (), {})

        layoutPtr = self.get_layout_pointer(expr).cast(native_ast.UInt64)

        return context.pushPod(
            Int32,
            runtime_functions.hash_class.call(
                layoutPtr.cast(native_ast.VoidPtr),
                context.getTypePointer(expr.expr_type.typeRepresentation)
            )
        )

    def convert_typeof(self, context, instance):
        if self.typeRepresentation.IsFinal:
            return context.constant(self.typeRepresentation)

        return context.pushPod(
            SubclassOf(self.typeRepresentation),
            self.get_class_type_ptr_as_voidptr(instance)
        )
