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

from typed_python import NoneType, _types, Type

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

    def get_layout_pointer(self, nonref_expr):
        # our layout is 48 bits of pointer and 16 bits of classDispatchTableIndex.
        # so whenever we interact with the pointer we need to chop off the top 16 bits
        return (
            nonref_expr
            .cast(native_ast.UInt64)
            .bitand(native_ast.const_uint64_expr(0xFFFFFFFFFFFF))  # 48 bits of 1s
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

    def convert_method_call(self, context, instance, methodName, args, kwargs):
        # figure out which signature we'd want to use on the given args/kwargs
        argTypes = [instance.expr_type.typeRepresentation] + [a.expr_type.typeRepresentation for a in args]

        for a in argTypes:
            assert issubclass(a, Type), a

        func = self.typeRepresentation.MemberFunctions[methodName]

        # each of 'func''s overloads represents one of the functions defined with this name
        # in this class and in its base classes, ordered by the method resolution order.
        # we can think of each one as a pattern that we are sequentially matching against,
        # and we should invoke the first one that matches the specific values that we have
        # in our argTypes. In fact, we may match more than one (for instance if we know all of
        # our values as 'object') in which case we need to generate runtime tests for each value
        # against each type pattern.

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

        for o in func.overloads:
            # for the moment, we fail to generate code if we don't definitely match a single
            # term in the pattern. We should be testing whether our argument types are definitely
            # covered by 'o', or only partially, in which case we should generate code to test
            # the relevant types and then trigger a call if we are successful.
            if o.matchesTypes(argTypes):
                # get the Function object representing this entrypoint as a signature.
                # we specialize on the types in 'argTypes' because we might be specializing
                # a generic method on a specific subtype.
                signature = o.signatureForSubtypes(methodName, argTypes)

                # each entrypoint generates a slot we could call.
                dispatchSlot = _types.allocateClassMethodDispatch(self.typeRepresentation, methodName, signature)

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
                classDispatchTable = classDispatchTables.elemPtr(
                    self.get_dispatch_index(instance)
                )

                funcPtr = classDispatchTable.ElementPtrIntegers(0, 2).load().elemPtr(dispatchSlot).load()

                if o.returnType is None:
                    retType = object
                else:
                    retType = o.returnType

                res = context.call_function_pointer(funcPtr, (instance,) + tuple(args), kwargs, typeWrapper(retType))

                # context.pushEffect(
                #     runtime_functions.print_int64.call(res.nonref_expr.cast(native_ast.Int64))
                # )

                return res

    @staticmethod
    def compileMethodInstantiation(converter, interfaceClass, implementingClass, methodName, signature, callback):
        """Compile a concrete method instantiation.

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

        # this is the function we're actually implementing. It has its own type signature
        # but we have the signature from the base class as well, which may be more precise
        # in the inputs and less precise in the outputs.
        pyImpl = implementingClass.MemberFunctions[methodName]

        # this is a very imprecise proxy for the search we want to do to find the
        # actual concrete method we'd dispatch to in these circumstances. In reality,
        # we might dispatch to multiple methods depending on the signature. For the
        # moment, we just pick one.
        for o2 in pyImpl.overloads:
            if o2.matchesTypes([implementingClass] + [a.typeFilter for a in funcOverload.args[1:]]):
                # really, we should be generating a function that has our signature, and then
                # calling the inner one. For instance, if we are marked to return 'object' and
                # the inner function is marked to return 'int', we need to first check that
                # we return 'int' as our semantics dictate, and then return the object.
                # that won't happen if we are just taking the compiled signature's type.

                argTypes = [implementingClass] + [arg.typeFilter for arg in funcOverload.args[1:]]

                converter.convert(
                    o2.functionObj,
                    argTypes,
                    funcOverload.returnType if funcOverload.returnType is not None else object,
                    callback=callback
                )
                return True

        return False

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
