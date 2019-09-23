#   Copyright 2017-2019 Nativepython Authors
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

from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.refcounted_wrapper import RefcountedWrapper
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, _types, OneOf, Bool, Int32

import nativepython.native_ast as native_ast
import nativepython
from nativepython.native_ast import VoidPtr


typeWrapper = lambda x: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(x)


def makeAlternativeWrapper(t):
    if t.__typed_python_category__ == "ConcreteAlternative":
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

    def convert_default_initialize(self, context, target):
        self.convert_copy_initialize(
            context,
            target,
            nativepython.python_object_representation.pythonObjectRepresentation(context, self.typeRepresentation())
        )

    def convert_destroy(self, context, target):
        pass

    def convert_assign(self, context, target, toStore):
        assert target.isReference
        context.pushEffect(
            target.expr.store(toStore.nonref_expr)
        )

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
            y = self.convert_call_method(context, "__bool__", (e,))
            if y is not None:
                return y.expr_type.convert_to_type_with_target(context, y, targetVal, False)
            else:
                y = self.convert_call_method(context, "__len__", (e,))
                if y is not None:
                    context.pushEffect(targetVal.expr.store(y.convert_to_type(int).nonref_expr.neq(0)))
                else:
                    context.pushEffect(targetVal.expr.store(context.constant(True)))
                return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)

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


class AlternativeWrapper(RefcountedWrapper):
    is_empty = False
    is_pod = False
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
        y = self.convert_call_method(context, "__hash__", (expr,))
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
            return super().convert_attribute(context, instance, attribute)

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
            y = self.convert_call_method(context, "__bool__", (e,))
            if y is not None:
                return y.expr_type.convert_to_type_with_target(context, y, targetVal, False)
            else:
                y = self.convert_call_method(context, "__len__", (e,))
                if y is not None:
                    context.pushEffect(targetVal.expr.store(y.convert_to_type(int).nonref_expr.neq(0)))
                else:
                    context.pushEffect(targetVal.expr.store(context.constant(True)))
                return context.constant(True)
        # TODO: Compile this properly
        # if target_type.typeRepresentation == Bytes:
        #     y = self.convert_call_method(context, "__bytes__", (e,))
        #     if y is not None:
        #         context.pushEffect(targetVal.expr.store(y))
        #         return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)

    def convert_call(self, context, expr, args, kwargs):
        return self.convert_call_method(context, "__call__", [expr] + args)

    def convert_len_native(self, context, expr):
        return self.convert_call_method(context, "__len__", (expr,)) or context.constant(0)

    def convert_len(self, context, expr):
        intermediate = self.convert_len_native(context, expr)
        if intermediate is None:
            return None
        return context.pushPod(int, intermediate.convert_to_type(int).expr)

    def convert_abs(self, context, expr):
        return self.convert_call_method(context, "__abs__", (expr,))

    def convert_unary_op(self, context, expr, op):
        magic = "__pos__" if op.matches.UAdd else \
            "__neg__" if op.matches.USub else \
            "__invert__" if op.matches.Invert else \
            "__not__" if op.matches.Not else \
            ""
        return self.convert_call_method(context, magic, (expr,)) or super().convert_unary_op(context, expr, op)

    def convert_bin_op(self, context, l, op, r):
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

        return self.convert_call_method(context, magic, (l, r)) \
            or self.convert_comparison(context, l, op, r) \
            or super().convert_bin_op(context, l, op, r)

    # Default comparison for Alternative types
    # returns None if no comparison is possible (if op is not a comparison operator to begin with,
    # or if operands are different types)
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

    def convert_bin_op_reverse(self, context, r, op, l):
        if op.matches.In:
            ret = self.convert_call_method(context, "__contains__", (r, l))
            return ret and ret.toBool()
        return super().convert_bin_op_reverse(context, r, op, l)


class ConcreteAlternativeWrapper(RefcountedWrapper):
    is_empty = False
    is_pod = False
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
            y = self.convert_call_method(context, "__bool__", (e,))
            if y is not None:
                return y.expr_type.convert_to_type_with_target(context, y, targetVal, False)
            else:
                y = self.convert_call_method(context, "__len__", (e,))
                if y is not None:
                    context.pushEffect(targetVal.expr.store(y.convert_to_type(int).nonref_expr.neq(0)))
                else:
                    context.pushEffect(targetVal.expr.store(context.constant(True)))
                return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)

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


class AlternativeMatchingWrapper(Wrapper):
    def convert_attribute(self, context, instance, attribute):
        altType = typeWrapper(self.typeRepresentation)

        return altType.convert_check_matches(context, instance.changeType(altType), attribute)
