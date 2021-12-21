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

import _thread

from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.typed_expression import TypedExpression
from typed_python import OneOf
import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.native_ast import VoidPtr, UInt64
import typed_python
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from math import trunc, floor, ceil


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class PythonObjectOfTypeWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True
    CAN_BE_NULL = False

    def __init__(self, pytype):
        super().__init__(pytype)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('pyObj', VoidPtr)
        ), name='PythonObjectOfTypeWrapper').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        assert instance.isReference

        context.pushEffect(
            runtime_functions.destroy_pyobj_handle.call(
                instance.nonref_expr.cast(VoidPtr)
            )
        )

    def convert_default_initialize(self, context, target, forceToNone=False):
        if isinstance(None, self.typeRepresentation) or forceToNone:
            target.convert_copy_initialize(
                TypedExpression(
                    context,
                    runtime_functions.get_pyobj_None.call().cast(self.getNativeLayoutType()),
                    self,
                    False
                )
            )
            return

        context.pushException(TypeError, "Can't default-initialize %s" % self.typeRepresentation.__qualname__)

    def convert_fastnext(self, context, expr):
        nextRes = context.push(
            object,
            lambda objPtr: objPtr.expr.store(
                runtime_functions.pyobj_iter_next.call(expr.nonref_expr.cast(VoidPtr))
                .cast(self.getNativeLayoutType())
            )
        )

        canContinue = context.pushPod(
            bool,
            nextRes.nonref_expr.cast(native_ast.Int64).gt(0)
        )

        with context.ifelse(nextRes.nonref_expr.cast(native_ast.Int64)) as (ifTrue, ifFalse):
            with ifFalse:
                self.convert_default_initialize(context, nextRes, forceToNone=True)

        return nextRes.asPointerIf(canContinue)

    def convert_attribute(self, context, instance, attr):
        if self.typeRepresentation in (_thread.LockType, _thread.RLock) and attr in ('acquire', 'release', "__enter__", "__exit__"):
            return instance.changeType(BoundMethodWrapper.Make(self, attr))

        assert isinstance(attr, str)

        return context.push(
            object,
            lambda targetSlot: targetSlot.expr.store(
                runtime_functions.getattr_pyobj.call(
                    instance.nonref_expr.cast(VoidPtr),
                    native_ast.const_utf8_cstr(attr)
                ).cast(self.getNativeLayoutType())
            )
        )

    def convert_set_attribute(self, context, instance, attr, val):
        assert isinstance(attr, str)

        valAsObj = val.convert_to_type(object, ConversionLevel.Signature)
        if valAsObj is None:
            return None

        context.pushEffect(
            runtime_functions.setattr_pyobj.call(
                instance.nonref_expr.cast(VoidPtr),
                native_ast.const_utf8_cstr(attr),
                valAsObj.nonref_expr.cast(VoidPtr)
            )
        )

        return context.constant(None)

    def convert_getitem(self, context, instance, item):
        itemAsObj = item.convert_to_type(object, ConversionLevel.Signature)
        if itemAsObj is None:
            return None

        return context.push(
            object,
            lambda targetSlot:
                targetSlot.expr.store(
                    runtime_functions.getitem_pyobj.call(
                        instance.nonref_expr.cast(VoidPtr),
                        itemAsObj.nonref_expr.cast(VoidPtr)
                    ).cast(self.getNativeLayoutType())
                )
        )

    def convert_delitem(self, context, instance, item):
        itemAsObj = item.convert_to_type(object, ConversionLevel.Signature)
        if itemAsObj is None:
            return None

        return context.pushEffect(
            runtime_functions.delitem_pyobj.call(
                instance.nonref_expr.cast(VoidPtr),
                itemAsObj.nonref_expr.cast(VoidPtr)
            )
        )

    def convert_unary_op(self, context, l, op):
        tgt = runtime_functions.pyOpToUnaryCallTarget.get(op)

        if tgt is not None:
            if op.matches.Not:
                return context.pushPod(
                    bool,
                    tgt.call(l.nonref_expr.cast(VoidPtr))
                )

            return context.push(
                object,
                lambda objPtr:
                objPtr.expr.store(
                    tgt.call(l.nonref_expr.cast(VoidPtr))
                    .cast(self.getNativeLayoutType())
                )
            )

        raise Exception(f"Unknown unary operator {op}")

    def convert_bin_op(self, context, l, op, r, inplace):
        rAsObj = r.convert_to_type(object, ConversionLevel.Signature)
        if rAsObj is None:
            return None

        if op.matches.Is:
            return context.pushPod(
                bool,
                l.nonref_expr.ElementPtrIntegers(0, 1).cast(UInt64.pointer()).load().eq(
                    rAsObj.nonref_expr.ElementPtrIntegers(0, 1).cast(UInt64.pointer()).load()
                )
            )

        if op.matches.IsNot:
            return context.pushPod(
                bool,
                l.nonref_expr.ElementPtrIntegers(0, 1).cast(UInt64.pointer()).load().neq(
                    rAsObj.nonref_expr.ElementPtrIntegers(0, 1).cast(UInt64.pointer()).load()
                )
            )

        tgt = runtime_functions.pyOpToBinaryCallTarget.get(op)

        if tgt is not None:
            return context.push(
                object,
                lambda objPtr:
                objPtr.expr.store(
                    tgt.call(
                        l.nonref_expr.cast(VoidPtr),
                        rAsObj.nonref_expr.cast(VoidPtr)
                    ).cast(self.getNativeLayoutType())
                )
            )

        raise Exception(f"Unknown binary operator {op} (inplace={inplace})")

    def convert_bin_op_reverse(self, context, r, op, l, inplace):
        lAsObj = l.convert_to_type(object, ConversionLevel.Signature)
        if lAsObj is None:
            return None

        return lAsObj.convert_bin_op(op, r, inplace)

    def convert_setitem(self, context, instance, index, value):
        indexAsObj = index.convert_to_type(object, ConversionLevel.Signature)
        if indexAsObj is None:
            return None

        valueAsObj = value.convert_to_type(object, ConversionLevel.Signature)
        if valueAsObj is None:
            return None

        context.pushEffect(
            runtime_functions.setitem_pyobj.call(
                instance.nonref_expr.cast(VoidPtr),
                indexAsObj.nonref_expr.cast(VoidPtr),
                valueAsObj.nonref_expr.cast(VoidPtr)
            )
        )

        return context.constant(None)

    def convert_builtin(self, f, context, expr, a1=None):
        if f == ceil:
            return context.pushPod(float, runtime_functions.pyobj_ceil.call(expr.nonref_expr.cast(VoidPtr)))
        elif f == floor:
            return context.pushPod(float, runtime_functions.pyobj_floor.call(expr.nonref_expr.cast(VoidPtr)))
        elif f == trunc:
            return context.pushPod(float, runtime_functions.pyobj_trunc.call(expr.nonref_expr.cast(VoidPtr)))
        return super().convert_builtin(f, context, expr, a1)

    def convert_call(self, context, instance, args, kwargs):
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

        # we converted everything to python objects. We need to pass this
        # ensemble to the interpreter. We use c-style variadic arguments here
        # since everything is a pointer.
        arguments = []
        kwarguments = []

        for a in argsAsObjects:
            arguments.append(a.nonref_expr.cast(VoidPtr))

        for kwargName, kwargVal in kwargsAsObjects.items():
            kwarguments.append(kwargVal.nonref_expr.cast(VoidPtr))
            kwarguments.append(native_ast.const_utf8_cstr(kwargName))

        return context.push(
            object,
            lambda oPtr:
                oPtr.expr.store(
                    runtime_functions.call_pyobj.call(
                        instance.nonref_expr.cast(VoidPtr),
                        native_ast.const_int_expr(len(arguments)),
                        native_ast.const_int_expr(len(kwargsAsObjects)),
                        *arguments,
                        *kwarguments,
                    ).cast(self.getNativeLayoutType())
                )
        )

    def convert_len(self, context, instance):
        return context.push(
            int,
            lambda outLen:
            outLen.expr.store(
                runtime_functions.pyobj_len.call(
                    instance.nonref_expr.cast(VoidPtr)
                )
            )
        )

    def _can_convert_to_type(self, otherType, conversionLevel):
        return super()._can_convert_to_type(otherType, conversionLevel)

    def _can_convert_from_type(self, otherType, conversionLevel):
        if self.typeRepresentation is object:
            return True

        return "Maybe"

    def convert_index_cast(self, context, e):
        return context.pushPod(
            int,
            runtime_functions.pyobj_pynumber_index.call(
                e.nonref_expr.cast(VoidPtr)
            )
        )

    def convert_math_float_cast(self, context, e):
        return context.pushPod(
            float,
            runtime_functions.pyobj_to_float64.call(
                e.nonref_expr.cast(VoidPtr)
            )
        )

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        target_type = targetVal.expr_type

        # check if we are a subclass of targetVal, in which case we are guaranteed
        # that this will succeed.
        if isinstance(targetVal.expr_type, PythonObjectOfTypeWrapper) and (
            issubclass(self.typeRepresentation, targetVal.expr_type.typeRepresentation)
        ):
            targetVal.convert_copy_initialize(instance)
            return context.constant(True)

        t = target_type.typeRepresentation

        if not issubclass(t, OneOf):
            return context.pushPod(
                bool,
                runtime_functions.pyobj_to_typed.call(
                    instance.nonref_expr.cast(VoidPtr),
                    targetVal.expr.cast(VoidPtr),
                    context.getTypePointer(t),
                    context.constant(conversionLevel.LEVEL),
                    native_ast.const_int_expr(1 if mayThrowOnFailure else 0)
                )
            )

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    def convert_to_self_with_target(self, context, targetVal, sourceVal, conversionLevel, mayThrowOnFailure=False):
        if self.typeRepresentation is not object:
            # this isn't a python object yet
            assert not isinstance(sourceVal.expr_type, PythonObjectOfTypeWrapper)

            # convert to an object first, and then convert that down to the subtype
            return sourceVal.convert_to_type(object, ConversionLevel.Signature).convert_to_type_with_target(
                targetVal,
                conversionLevel,
                mayThrowOnFailure
            )

        t = sourceVal.expr_type.typeRepresentation

        if not sourceVal.isReference:
            sourceVal = context.pushMove(sourceVal)

        context.pushEffect(
            targetVal.expr.store(
                runtime_functions.to_pyobj.call(
                    sourceVal.expr.cast(VoidPtr),
                    context.getTypePointer(t)
                )
                .cast(self.getNativeLayoutType())
            )
        )
        return context.constant(True)

    def convert_type_call(self, context, typeInst, args, kwargs):
        # if this is a regular python class, then we need to just convert it to an 'object' and call that.
        return context.constant(self.typeRepresentation).convert_to_type(
            object,
            ConversionLevel.Signature
        ).convert_call(args, kwargs)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if self.typeRepresentation in (_thread.LockType, _thread.RLock) and methodname in ("acquire", "__enter__") and len(args) == 0:
            if self.typeRepresentation is _thread.LockType:
                nativeFun = runtime_functions.pyobj_locktype_lock
            else:
                nativeFun = runtime_functions.pyobj_rlocktype_lock

            return context.pushPod(bool, nativeFun.call(instance.nonref_expr.cast(VoidPtr)))

        if self.typeRepresentation in (_thread.LockType, _thread.RLock) and (
            methodname == "release" and len(args) == 0
            or methodname == "__exit__" and len(args) == 3
        ):
            if self.typeRepresentation is _thread.LockType:
                nativeFun = runtime_functions.pyobj_locktype_unlock
            else:
                nativeFun = runtime_functions.pyobj_rlocktype_unlock

            return context.pushPod(bool, nativeFun.call(instance.nonref_expr.cast(VoidPtr)))

        method = self.convert_attribute(context, instance, methodname)
        if method is None:
            return None

        return method.convert_call(args, kwargs)

    def convert_context_manager_enter(self, context, instance):
        if self.typeRepresentation in (_thread.LockType, _thread.RLock):
            return self.convert_method_call(context, instance, "acquire", (), {})

        return super().convert_context_manager_enter(context, instance)

    def convert_context_manager_exit(self, context, instance, args):
        if self.typeRepresentation in (_thread.LockType, _thread.RLock):
            return self.convert_method_call(context, instance, "release", (), {})

        return super().convert_context_manager_exit(context, instance, args)

    def convert_typeof(self, context, instance):
        pythonObjectRepresentation = (
            typed_python.compiler.python_object_representation.pythonObjectRepresentation
        )

        if self.typeRepresentation is type:
            return pythonObjectRepresentation(context, type)

        return context.push(
            object,
            lambda targetSlot: targetSlot.expr.store(
                runtime_functions.pyobj_typeof.call(
                    instance.nonref_expr.cast(VoidPtr)
                ).cast(self.getNativeLayoutType())
            )
        )

    def convert_issubclass(self, context, instance, ofType, isSubclassCall):
        return context.pushPod(
            bool,
            runtime_functions.pyobj_issubclass.call(
                instance.nonref_expr.cast(VoidPtr),
                ofType.toPyObj().nonref_expr.cast(VoidPtr)
            )
        )
