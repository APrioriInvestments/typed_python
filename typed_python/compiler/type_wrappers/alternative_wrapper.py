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
from typed_python.compiler.type_wrappers.refcounted_wrapper import RefcountedWrapper
from typed_python.compiler.type_wrappers.arithmetic_wrapper import FloatWrapper, IntWrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, _types, OneOf, Bool, Int32, String

import typed_python.compiler.native_ast as native_ast
import typed_python.compiler
from typed_python.compiler.native_ast import VoidPtr
from math import trunc, floor, ceil


typeWrapper = lambda x: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(x)


def makeAlternativeWrapper(t):
    if t.__typed_python_category__ == "ConcreteAlternative":
        if _types.all_alternatives_empty(t):
            return ConcreteSimpleAlternativeWrapper(t)
        else:
            return ConcreteAlternativeWrapper(t)

    if _types.all_alternatives_empty(t):
        return SimpleAlternativeWrapper(t)
    else:
        return AlternativeWrapper(t)


class SimpleAlternativeWrapper(Wrapper):
    """Wrapper around alternatives with all empty arguments."""
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self, t):
        super().__init__(t)

        self.layoutType = native_ast.UInt8

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_hash(self, context, expr):
        y = self.generate_method_call(context, "__hash__", (expr,))
        if y is not None:
            return y
        tp = context.getTypePointer(expr.expr_type.typeRepresentation)
        if tp:
            return context.pushPod(Int32, runtime_functions.hash_alternative.call(expr.nonref_expr.cast(VoidPtr), tp))
        return None

    def convert_copy_initialize(self, context, target, toStore):
        assert target.isReference
        context.pushEffect(
            target.expr.store(toStore.nonref_expr)
        )

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        # there is deliberately no code path for "not explicit" here
        # Alternative conversions must be explicit
        assert targetVal.isReference

        target_type = targetVal.expr_type
        if target_type.typeRepresentation == Bool:
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

    def convert_cast(self, context, e, target_type):
        if isinstance(target_type, IntWrapper):
            y = self.generate_method_call(context, "__int__", (e,))
            if y is not None:
                return y.convert_to_type(target_type)

        if isinstance(target_type, FloatWrapper):
            y = self.generate_method_call(context, "__float__", (e,))
            if y is not None:
                return y.convert_to_type(target_type)

        if target_type.typeRepresentation == Bool:
            return super().convert_cast(context, e, target_type)
        if target_type.typeRepresentation == String:
            return super().convert_cast(context, e, target_type)
        return None

    def convert_call(self, context, expr, args, kwargs):
        return self.generate_method_call(context, "__call__", [expr] + args)

    def convert_len_native(self, context, expr):
        alt = self.typeRepresentation
        if getattr(alt.__len__, "__typed_python_category__", None) == 'Function':
            assert len(alt.__len__.overloads) == 1
            return context.call_py_function(alt.__len__.overloads[0].functionObj, (expr,), {})
        return context.constant(0)

    def convert_len(self, context, expr):
        intermediate = self.convert_len_native(context, expr)
        if intermediate is None:
            return None
        return context.pushPod(int, intermediate.convert_to_type(int).expr)

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
                    or expr.convert_cast(str)
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
        if f is bytes:
            return self.generate_method_call(context, "__bytes__", (expr, ))
        if f is trunc:
            return self.generate_method_call(context, "__trunc__", (expr,)) \
                or context.pushPod(float, runtime_functions.trunc_float64.call(expr.toFloat64().nonref_expr))
        if f is floor:
            expr_float = expr.convert_cast(float)
            return self.generate_method_call(context, "__floor__", (expr,)) \
                or (expr_float and context.pushPod(float, runtime_functions.floor_float64.call(expr_float.nonref_expr)))
        if f is ceil:
            expr_float = expr.convert_cast(float)
            return self.generate_method_call(context, "__ceil__", (expr,)) \
                or (expr_float and context.pushPod(float, runtime_functions.ceil_float64.call(expr_float.nonref_expr)))
        if f is complex:
            return self.generate_method_call(context, "__complex__", (expr, ))
        if f is dir:
            return self.generate_method_call(context, "__dir__", (expr, )) \
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
        # TODO: provide nicer translation from op to Py_ comparison codes
        py_code = 2 if op.matches.Eq else \
            3 if op.matches.NotEq else \
            0 if op.matches.Lt else \
            4 if op.matches.Gt else \
            1 if op.matches.LtE else \
            5 if op.matches.GtE else -1
        if py_code < 0:
            return None
        tp_left = context.getTypePointer(left.expr_type.typeRepresentation)
        tp_right = context.getTypePointer(right.expr_type.typeRepresentation)
        if tp_left and tp_left == tp_right:
            if not left.isReference:
                left = context.push(left.expr_type, lambda x: x.convert_copy_initialize(left))
            if not right.isReference:
                right = context.push(right.expr_type, lambda x: x.convert_copy_initialize(right))
            return context.pushPod(
                Bool,
                runtime_functions.alternative_cmp.call(
                    tp_left,
                    left.expr.cast(VoidPtr),
                    right.expr.cast(VoidPtr),
                    py_code
                )
            )
        return None

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

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        return super().convert_type_call(context, typeInst, args, kwargs)


class ConcreteSimpleAlternativeWrapper(Wrapper):
    """Wrapper around alternatives with all empty arguments, after choosing a specific alternative."""
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self, t):
        super().__init__(t)

        self.layoutType = native_ast.UInt8
        self.alternativeType = t.Alternative

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_default_initialize(self, context, target):
        self.convert_copy_initialize(
            context,
            target,
            typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                context,
                self.typeRepresentation()
            )
        )

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        # there is deliberately no code path for "not explicit" here
        # Alternative conversions must be explicit
        assert targetVal.isReference

        target_type = targetVal.expr_type

        if target_type == typeWrapper(self.alternativeType):
            targetVal.convert_copy_initialize(e.changeType(target_type))
            return context.constant(True)

        if target_type.typeRepresentation == Bool:
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

    def convert_cast(self, context, e, target_type):
        if isinstance(target_type, IntWrapper):
            y = self.generate_method_call(context, "__int__", (e,))
            if y is not None:
                return y.convert_to_type(target_type)

        if isinstance(target_type, FloatWrapper):
            y = self.generate_method_call(context, "__float__", (e,))
            if y is not None:
                return y.convert_to_type(target_type)

        if target_type.typeRepresentation == Bool:
            return super().convert_cast(context, e, target_type)
        if target_type.typeRepresentation == String:
            return super().convert_cast(context, e, target_type)
        return None

    def convert_builtin(self, f, context, expr, a1=None):
        altWrapper = typeWrapper(self.alternativeType)
        return altWrapper.convert_builtin(f, context, expr, a1)

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        return super().convert_type_call(context, typeInst, args, kwargs)


class AlternativeWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        super().__init__(t)

        element_types = [('refcount', native_ast.Int64), ('which', native_ast.Int64), ('data', native_ast.UInt8)]

        self.alternativeType = t
        self.layoutType = native_ast.Type.Struct(element_types=element_types, name=t.__qualname__+"Layout").pointer()
        self.matcherType = AlternativeMatchingWrapper(self.typeRepresentation)
        self._alternatives = None

    @property
    def alternatives(self):
        """Return a list of type wrappers for our alternative bodies.

        This function has to be deferred until after the object is created if we have recursive alternatives.
        """
        if self._alternatives is None:
            self._alternatives = [typeWrapper(x.ElementType) for x in self.typeRepresentation.__typed_python_alternatives__]
        return self._alternatives

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_hash(self, context, expr):
        y = self.generate_method_call(context, "__hash__", (expr,))
        if y is not None:
            return y
        tp = context.getTypePointer(expr.expr_type.typeRepresentation)
        if tp:
            return context.pushPod(Int32, runtime_functions.hash_alternative.call(expr.nonref_expr.cast(VoidPtr), tp))
        return None

    def on_refcount_zero(self, context, instance):
        return (
            context.converter.defineNativeFunction(
                "destructor_" + str(self.typeRepresentation),
                ('destructor', self),
                [self],
                typeWrapper(NoneType),
                self.generateNativeDestructorFunction
            )
            .call(instance)
        )

    def refAs(self, context, instance, whichIx):
        return context.pushReference(
            self.alternatives[whichIx].typeRepresentation,
            instance.nonref_expr.ElementPtrIntegers(0, 2).cast(self.alternatives[whichIx].getNativeLayoutType().pointer())
        )

    def generateNativeDestructorFunction(self, context, out, instance):
        with context.switch(instance.nonref_expr.ElementPtrIntegers(0, 1).load(),
                            range(len(self.alternatives)),
                            False) as indicesAndContexts:
            for ix, subcontext in indicesAndContexts:
                with subcontext:
                    self.refAs(context, instance, ix).convert_destroy()

        context.pushEffect(runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr)))

    def convert_attribute(self, context, instance, attribute, nocheck=False):
        if attribute == 'matches':
            return instance.changeType(self.matcherType)

        possibleTypes = set()
        validIndices = []
        for i, namedTup in enumerate(self.alternatives):
            if attribute in namedTup.namesToTypes:
                possibleTypes.add(namedTup.namesToTypes[attribute])
                validIndices.append(i)

        if not validIndices:
            return self.generate_method_call(context, "__getattr__", (instance, context.constant(attribute))) \
                or super().convert_attribute(context, instance, attribute)
        if len(validIndices) == 1:
            with context.ifelse(instance.nonref_expr.ElementPtrIntegers(0, 1).load().neq(validIndices[0])) as (then, otherwise):
                with then:
                    context.pushException(AttributeError, "Object has no attribute %s" % attribute)
            return self.refAs(context, instance, validIndices[0]).convert_attribute(attribute)
        else:
            outputType = typeWrapper(
                list(possibleTypes)[0] if len(possibleTypes) == 1 else OneOf(*possibleTypes)
            )

            output = context.allocateUninitializedSlot(outputType)

            with context.switch(instance.nonref_expr.ElementPtrIntegers(0, 1).load(), validIndices, False) as indicesAndContexts:
                for ix, subcontext in indicesAndContexts:
                    with subcontext:
                        attr = self.refAs(context, instance, ix).convert_attribute(attribute)
                        attr = attr.convert_to_type(outputType)
                        output.convert_copy_initialize(attr)
                        context.markUninitializedSlotInitialized(output)

            return output

    def convert_getitem(self, context, instance, item):
        return self.generate_method_call(context, "__getitem__", (instance, item)) \
            or super().convert_getitem(context, instance, item)

    def convert_setitem(self, context, instance, item, value):
        return self.generate_method_call(context, "__setitem__", (instance, item, value)) \
            or super().convert_setitem(context, instance, item, value)

    def convert_set_attribute(self, context, instance, attribute, value):
        if value is None:
            return self.generate_method_call(context, "__delattr__", (instance, context.constant(attribute))) \
                or super().convert_set_attribute(context, instance, attribute, value)
        return self.generate_method_call(context, "__setattr__", (instance, context.constant(attribute), value)) \
            or super().convert_set_attribute(context, instance, attribute, value)

    def convert_check_matches(self, context, instance, typename):
        index = -1
        for i in range(len(self.typeRepresentation.__typed_python_alternatives__)):
            if self.typeRepresentation.__typed_python_alternatives__[i].Name == typename:
                index = i

        if index == -1:
            return context.constant(False)
        return context.pushPod(bool, instance.nonref_expr.ElementPtrIntegers(0, 1).load().eq(index))

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        # there is deliberately no code path for "not explicit" here
        # Alternative conversions must be explicit
        assert targetVal.isReference

        target_type = targetVal.expr_type
        if target_type.typeRepresentation == Bool:
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

    def convert_cast(self, context, e, target_type):
        if isinstance(target_type, IntWrapper):
            y = self.generate_method_call(context, "__int__", (e,))
            if y is not None:
                return y.convert_to_type(target_type)

        if isinstance(target_type, FloatWrapper):
            y = self.generate_method_call(context, "__float__", (e,))
            if y is not None:
                return y.convert_to_type(target_type)

        if target_type.typeRepresentation == Bool:
            return super().convert_cast(context, e, target_type)
        if target_type.typeRepresentation == String:
            return super().convert_cast(context, e, target_type)
        return None

    def convert_call(self, context, expr, args, kwargs):
        return self.generate_method_call(context, "__call__", [expr] + args)

    def convert_len_native(self, context, expr):
        return self.generate_method_call(context, "__len__", (expr,)) or context.constant(0)

    def convert_len(self, context, expr):
        intermediate = self.convert_len_native(context, expr)
        if intermediate is None:
            return None
        return context.pushPod(int, intermediate.convert_to_type(int).expr)

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
                    or expr.convert_cast(str)
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
        if f is bytes:
            return self.generate_method_call(context, "__bytes__", (expr, ))
        if f is trunc:
            return self.generate_method_call(context, "__trunc__", (expr,)) \
                or context.pushPod(float, runtime_functions.trunc_float64.call(expr.toFloat64().nonref_expr))
        if f is floor:
            expr_float = expr.convert_cast(float)
            return self.generate_method_call(context, "__floor__", (expr,)) \
                or (expr_float and context.pushPod(float, runtime_functions.floor_float64.call(expr_float.nonref_expr)))
        if f is ceil:
            expr_float = expr.convert_cast(float)
            return self.generate_method_call(context, "__ceil__", (expr,)) \
                or (expr_float and context.pushPod(float, runtime_functions.ceil_float64.call(expr_float.nonref_expr)))
        if f is complex:
            return self.generate_method_call(context, "__complex__", (expr, ))
        if f is dir:
            return self.generate_method_call(context, "__dir__", (expr, )) \
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

    def convert_comparison(self, context, l, op, r):
        # TODO: provide nicer translation from op to Py_ comparison codes
        py_code = 2 if op.matches.Eq else \
            3 if op.matches.NotEq else \
            0 if op.matches.Lt else \
            4 if op.matches.Gt else \
            1 if op.matches.LtE else \
            5 if op.matches.GtE else -1
        if py_code < 0:
            return None
        tp_l = context.getTypePointer(l.expr_type.typeRepresentation)
        tp_r = context.getTypePointer(r.expr_type.typeRepresentation)
        if tp_l and tp_l == tp_r:
            return context.pushPod(
                Bool,
                runtime_functions.alternative_cmp.call(
                    tp_l,
                    l.expr.cast(VoidPtr),
                    r.expr.cast(VoidPtr),
                    py_code
                )
            )
        return None

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

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        return super().convert_type_call(context, typeInst, args, kwargs)


class ConcreteAlternativeWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        super().__init__(t)

        element_types = [('refcount', native_ast.Int64), ('which', native_ast.Int64), ('data', native_ast.UInt8)]

        self.alternativeType = t.Alternative
        self.indexInParent = t.Index
        self.underlyingLayout = typeWrapper(t.ElementType)  # a NamedTuple
        self.layoutType = native_ast.Type.Struct(element_types=element_types, name=t.__qualname__+"Layout").pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        altWrapper = typeWrapper(self.alternativeType)

        return altWrapper.on_refcount_zero(
            context,
            instance.changeType(altWrapper)
        )

    def refToInner(self, context, instance):
        return context.pushReference(
            self.underlyingLayout,
            instance.nonref_expr.ElementPtrIntegers(0, 2).cast(self.underlyingLayout.getNativeLayoutType().pointer())
        )

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        # there is deliberately no code path for "not explicit" here
        # Alternative conversions must be explicit
        assert targetVal.isReference

        target_type = targetVal.expr_type

        if target_type == typeWrapper(self.alternativeType):
            targetVal.convert_copy_initialize(e.changeType(target_type))
            return context.constant(True)

        if target_type.typeRepresentation == Bool:
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

    def convert_cast(self, context, e, target_type):
        if isinstance(target_type, IntWrapper):
            y = self.generate_method_call(context, "__int__", (e,))
            if y is not None:
                return y.convert_to_type(target_type)

        if isinstance(target_type, FloatWrapper):
            y = self.generate_method_call(context, "__float__", (e,))
            if y is not None:
                return y.convert_to_type(target_type)

        if target_type.typeRepresentation == Bool:
            return super().convert_cast(context, e, target_type)
        if target_type.typeRepresentation == String:
            return super().convert_cast(context, e, target_type)
        return None

    def convert_type_call(self, context, typeInst, args, kwargs):
        tupletype = self.typeRepresentation.ElementType

        if len(args) == 1 and not kwargs:
            # check if this is the copy-constructor on ourself
            if args[0].expr_type == self:
                return args[0]

            # check if it's one argument and we have one field exactly
            if len(tupletype.ElementTypes) != 1:
                context.pushException("Can't construct %s with a single positional argument" % self)
                return

            kwargs = {tupletype.ElementNames[0]: args[0]}
            args = ()

        if len(args) > 1:
            context.pushException("Can't construct %s with multiple positional arguments" % self)
            return

        kwargs = dict(kwargs)

        for eltType, eltName in zip(tupletype.ElementTypes, tupletype.ElementNames):
            if eltName not in kwargs and not _types.is_default_constructible(eltType):
                context.pushException(TypeError, "Can't construct %s without an argument for %s of type %s" % (
                    self, eltName, eltType
                ))
                return

        for eltType, eltName in zip(tupletype.ElementTypes, tupletype.ElementNames):
            if eltName not in kwargs:
                kwargs[eltName] = context.push(eltType, lambda out: out.convert_default_initialize())
            else:
                kwargs[eltName] = kwargs[eltName].convert_to_type(typeWrapper(eltType))
                if kwargs[eltName] is None:
                    return

        return context.push(
            self,
            lambda new_alt:
                context.converter.defineNativeFunction(
                    'construct(' + str(self) + ")",
                    ('util', self, 'construct'),
                    tupletype.ElementTypes,
                    self,
                    self.generateConstructor
                ).call(new_alt, *[kwargs[eltName] for eltName in tupletype.ElementNames])
        ).changeType(typeWrapper(self.alternativeType))

    def generateConstructor(self, context, out, *args):
        tupletype = self.typeRepresentation.ElementType

        context.pushEffect(
            out.expr.store(
                runtime_functions.malloc.call(native_ast.const_int_expr(16 + self.underlyingLayout.getBytecount()))
                    .cast(self.getNativeLayoutType())
            ) >>
            out.expr.load().ElementPtrIntegers(0, 0).store(native_ast.const_int_expr(1)) >>  # refcount
            out.expr.load().ElementPtrIntegers(0, 1).store(native_ast.const_int_expr(self.indexInParent))  # which
        )

        assert len(args) == len(tupletype.ElementTypes)

        self.refToInner(context, out).convert_initialize_from_args(*args)

    def convert_attribute(self, context, instance, attribute, nocheck=False):
        return self.refToInner(context, instance).convert_attribute(attribute)

    def convert_check_matches(self, context, instance, typename):
        return context.constant(typename == self.typeRepresentation.Name)

    def convert_builtin(self, f, context, expr, a1=None):
        altWrapper = typeWrapper(self.alternativeType)
        return altWrapper.convert_builtin(f, context, expr, a1)


class AlternativeMatchingWrapper(Wrapper):
    def convert_attribute(self, context, instance, attribute):
        altType = typeWrapper(self.typeRepresentation)

        return altType.convert_check_matches(context, instance.changeType(altType), attribute)
