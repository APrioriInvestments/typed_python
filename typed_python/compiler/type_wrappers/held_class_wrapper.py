#   Copyright 2020 typed_python Authors
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
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.class_or_alternative_wrapper_mixin import (
    ClassOrAlternativeWrapperMixin
)

from typed_python import _types, RefTo

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler


typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


class HeldClassWrapper(Wrapper, ClassOrAlternativeWrapperMixin):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        super().__init__(t)

        self.heldClassType = t
        self.classType = t.Class
        self.refToType = RefTo(t)

        self.classTypeWrapper = typeWrapper(t.Class)
        self.nameToIndex = self.classTypeWrapper.nameToIndex
        self.indexToByteOffset = self.classTypeWrapper.indexToByteOffset
        self.bytesOfInitBits = self.classTypeWrapper.bytesOfInitBits

        element_types = [
            ('data', native_ast.Type.Array(element_type=native_ast.UInt8, count=self.bytesOfInitBits))
        ]

        for i, m in enumerate(self.classType.MemberTypes):
            element_types.append(("member_" + str(i), typeWrapper(m).getNativeLayoutType()))

        self.layoutType = native_ast.Type.Array(element_type=native_ast.UInt8, count=_types.bytecount(self.heldClassType))

    def refTo(self, instance):
        # 'instance' must be a reference to the class, which means it's actually
        # held as a pointer to the HeldClass instance 'C' itself. This has the
        # same native layout as a RefTo(C) does when it's _not_ a reference,
        # so we can simply change the type and rewrite it as not-a-reference
        assert instance.isReference
        return instance.changeType(self.refToType, isReferenceOverride=False)

    def convert_attribute_pointerTo(self, context, pointerInstance, attribute):
        refToSelf = TypedExpression(
            pointerInstance.context,
            pointerInstance.nonref_expr,
            self,
            True
        )

        if attribute in self.nameToIndex:
            return self.memberRef(refToSelf, self.nameToIndex[attribute]).asPointer()

        return super().convert_attribute(context, pointerInstance, attribute)

    def _can_convert_to_type(self, otherType, explicit):
        return False

    def _can_convert_from_type(self, otherType, explicit):
        return False

    def bytesOfInitBitsForInstance(self, instance):
        return native_ast.const_uint64_expr(self.bytesOfInitBits)

    def convert_assign(self, context, expr, other):
        for i in range(len(self.classType.MemberTypes)):
            with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(other, i))) as (true_block, false_block):
                with true_block:
                    with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(expr, i))) as (
                        self_true_block, self_false_block
                    ):
                        with self_true_block:
                            self.memberRef(expr, i).convert_assign(self.memberRef(other, i))
                        with self_false_block:
                            self.memberRef(expr, i).convert_copy_initialize(self.memberRef(other, i))
                            self.setIsInitializedExpr(expr, i)

                with false_block:
                    with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(expr, i))) as (
                        self_true_block, self_false_block
                    ):
                        with self_true_block:
                            self.memberRef(expr, i).convert_destroy()
                            context.pushEffect(self.clearIsInitializedExpr(expr, i))

    def convert_copy_initialize(self, context, expr, other):
        for i in range(len(self.classType.MemberTypes)):
            with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(other, i))) as (true_block, false_block):
                with true_block:
                    self.memberRef(expr, i).convert_copy_initialize(self.memberRef(other, i))
                    self.setIsInitializedExpr(expr, i)
                with false_block:
                    context.pushEffect(self.clearIsInitializedExpr(expr, i))

    def convert_default_initialize(self, context, instance):
        for i in range(len(self.classType.MemberTypes)):
            if _types.wantsToDefaultConstruct(self.classType.MemberTypes[i]):
                name = self.classType.MemberNames[i]

                if name in self.classType.MemberDefaultValues:
                    defVal = self.classType.MemberDefaultValues.get(name)
                    self.memberRef(instance, i).convert_copy_initialize(
                        typed_python.compiler.python_object_representation.pythonObjectRepresentation(context, defVal)
                    )
                else:
                    self.memberRef(instance, i).convert_default_initialize()

                context.pushEffect(self.setIsInitializedExpr(instance, i))
            else:
                context.pushEffect(self.clearIsInitializedExpr(instance, i))

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_destroy(self, context, instance):
        for i in range(len(self.classType.MemberTypes)):
            if not typeWrapper(self.classType.MemberTypes[i]).is_pod:
                with context.ifelse(context.pushPod(bool, self.isInitializedNativeExpr(instance, i))) as (true_block, false_block):
                    with true_block:
                        context.pushEffect(
                            self.memberRef(instance, i).convert_destroy()
                        )

    def memberRef(self, instance, ix):
        return TypedExpression(
            instance.context,
            self.memberPtr(instance, ix),
            self.classType.MemberTypes[ix],
            True
        )

    def memberPtr(self, instance, ix):
        assert instance.isReference

        return (
            instance.expr
            .cast(native_ast.UInt8.pointer())
            .elemPtr(self.bytesOfInitBitsForInstance(instance))
            .ElementPtrIntegers(self.indexToByteOffset[ix])
            .cast(
                typeWrapper(self.classType.MemberTypes[ix])
                .getNativeLayoutType()
                .pointer()
            )
        )

    def isInitializedNativeExpr(self, instance, ix):
        assert instance.isReference

        byte = ix // 8
        bit = ix % 8

        return (
            instance.expr
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(byte)
            .load()
            .rshift(native_ast.const_uint8_expr(bit))
            .bitand(native_ast.const_uint8_expr(1))
        )

    def setIsInitializedExpr(self, instance, ix):
        assert instance.isReference

        byte = ix // 8
        bit = ix % 8

        bytePtr = (
            instance.expr
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(byte)
        )

        return bytePtr.store(bytePtr.load().bitor(native_ast.const_uint8_expr(1 << bit)))

    def clearIsInitializedExpr(self, instance, ix):
        assert instance.isReference

        byte = ix // 8
        bit = ix % 8

        bytePtr = (
            instance.expr
            .cast(native_ast.UInt8.pointer())
            .ElementPtrIntegers(byte)
        )

        return bytePtr.store(bytePtr.load().bitand(native_ast.const_uint8_expr(255 - (1 << bit))))

    def convert_attribute(self, context, instance, attribute, nocheck=False):
        if attribute in self.classType.MemberFunctions:
            instance = instance.heldToRef()

            methodType = BoundMethodWrapper(_types.BoundMethod(self.refToType, attribute))

            return instance.changeType(methodType, isReferenceOverride=False)

        if attribute in self.classType.PropertyFunctions:
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
            self.classType.MemberTypes[ix],
            self.memberPtr(instance, ix)
        )

    def getMethodOrPropertyBody(self, name):
        return (
            self.classType.MemberFunctions.get(name)
            or self.classType.PropertyFunctions.get(name)
        )

    def has_method(self, methodName):
        assert isinstance(methodName, str)
        return self.getMethodOrPropertyBody(methodName) is not None

    def convert_method_call(self, context, instance, methodName, args, kwargs):
        # figure out which signature we'd want to use on the given args/kwargs
        func = self.getMethodOrPropertyBody(methodName)

        return typeWrapper(func).convert_call(context, None, [instance.heldToRef()] + list(args), kwargs)

    def convert_set_attribute(self, context, instance, attribute, value):
        if not isinstance(attribute, int):
            ix = self.nameToIndex.get(attribute)
        else:
            ix = attribute

        if ix is None:
            if value is None:
                if self.has_method("__delattr__"):
                    return self.convert_method_call(context, instance, "__delattr__", (context.constant(attribute),), {})

                return Wrapper.convert_set_attribute(self, context, instance, attribute, value)

            if self.has_method("__setattr__"):
                return self.convert_method_call(context, instance, "__setattr__", (context.constant(attribute), value), {})

            return Wrapper.convert_set_attribute(self, context, instance, attribute, value)

        attr_type = typeWrapper(self.classType.MemberTypes[ix])

        value = value.convert_to_type(attr_type, ConversionLevel.Implicit)
        if value is None:
            return None

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

    def convert_comparison(self, context, left, op, right):
        assert left.isReference and right.isReference

        if op.matches.Eq:
            native_expr = left.expr.cast(native_ast.UInt64).eq(right.expr.cast(native_ast.UInt64))
            return TypedExpression(context, native_expr, bool, False)

        if op.matches.NotEq:
            native_expr = left.expr.cast(native_ast.UInt64).neq(right.expr.cast(native_ast.UInt64))
            return TypedExpression(context, native_expr, bool, False)

        return context.pushException(TypeError, f"Can't compare instances of {left.expr_type.typeRepresentation}"
                                                f" and {right.expr_type.typeRepresentation} with {op}")

    def convert_hash(self, context, expr):
        if self.has_method("__hash__"):
            return self.convert_method_call(context, expr, "__hash__", (), {})

        return context.pushException(TypeError, f"Can't hash instances of {expr.expr_type}")
