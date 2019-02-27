#   Copyright 2018 Braxton Mckee
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
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class ConstDictWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        self.keyType = typeWrapper(t.KeyType)
        self.valueType = typeWrapper(t.ValueType)

        self.kvBytecount = self.keyType.getBytecount() + self.valueType.getBytecount()
        self.keyBytecount = self.keyType.getBytecount()

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('count', native_ast.Int32),
            ('subpointers', native_ast.Int32),
            ('data', native_ast.UInt8)
        ), name='TupleOfLayout').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        assert instance.isReference

        if self.keyType.is_pod and self.valueType.is_pod:
            return runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr))
        else:
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

    def generateNativeDestructorFunction(self, context, out, inst):
        with context.loop(inst.convert_len()) as i:
            self.convert_getkey_by_index_unsafe(context, inst, i).convert_destroy()
            self.convert_getvalue_by_index_unsafe(context, inst, i).convert_destroy()

        context.pushEffect(
            runtime_functions.free.call(inst.nonref_expr.cast(native_ast.UInt8Ptr))
        )

    def convert_getkey_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
            ).cast(self.keyType.getNativeLayoutType().pointer())
        )

    def convert_getvalue_by_index_unsafe(self, context, expr, item):
        return context.pushReference(
            self.keyType,
            expr.nonref_expr.ElementPtrIntegers(0, 4).elemPtr(
                item.nonref_expr.mul(native_ast.const_int_expr(self.kvBytecount))
                .add(native_ast.const_int_expr(self.keyBytecount))
            ).cast(self.valueType.getNativeLayoutType().pointer())
        )

    def convert_bin_op_reverse(self, context, left, op, right):
        if op.matches.In or op.matches.NotIn:
            right = right.convert_to_type(self.keyType)
            if right is None:
                return None

            native_contains = context.converter.defineNativeFunction(
                "dict_contains" + str(self.typeRepresentation),
                ('dict_contains', self),
                [self, self.keyType],
                bool,
                self.generateContains()
            )

            if op.matches.In:
                return context.pushPod(bool, native_contains.call(left, right))
            else:
                return context.pushPod(bool, native_contains.call(left, right).logical_not())

        return super().convert_bin_op(context, left, op, right)

    def convert_getitem(self, context, instance, item):
        item = item.convert_to_type(self.keyType)
        if item is None:
            return None

        native_getitem = context.converter.defineNativeFunction(
            "dict_getitem" + str(self.typeRepresentation),
            ('dict_getitem', self),
            [self, self.keyType],
            self.valueType,
            self.generateGetitem()
        )

        if self.valueType.is_pass_by_ref:
            return context.push(
                self.valueType,
                lambda output:
                    native_getitem.call(output, instance, item)
            )
        else:
            return context.push(
                self.valueType,
                lambda output:
                    output.expr.store(native_getitem.call(instance, item))
            )

    def generateGetitem(self):
        return self.generateLookupFun(False)

    def generateContains(self):
        return self.generateLookupFun(True)

    def generateLookupFun(self, containmentOnly):
        def f(context, out, inst, key):
            # a linear scan for now.
            lowIx = context.push(int, lambda x: x.expr.store(native_ast.const_int_expr(0)))
            highIx = context.push(int, lambda x: x.expr.store(self.convert_len_native(inst.nonref_expr)))

            with context.whileLoop(lowIx.nonref_expr.lt(highIx.nonref_expr)):
                mid = context.pushPod(int, lowIx.nonref_expr.add(highIx.nonref_expr).div(2))

                isLt = key < self.convert_getkey_by_index_unsafe(context, inst, mid)
                isEq = key == self.convert_getkey_by_index_unsafe(context, inst, mid)

                if isLt is not None and isEq is not None:
                    with context.ifelse(isEq.nonref_expr) as (true, false):
                        if containmentOnly:
                            with true:
                                context.pushTerminal(native_ast.Expression.Return(arg=native_ast.const_bool_expr(True)))
                        else:
                            with true:
                                result = self.convert_getvalue_by_index_unsafe(context, inst, mid)

                                if out is not None:
                                    context.pushEffect(
                                        out.convert_copy_initialize(result)
                                    )
                                    context.pushTerminal(
                                        native_ast.Expression.Return(arg=None)
                                    )
                                else:
                                    context.pushTerminal(
                                        native_ast.Expression.Return(arg=result.nonref_expr)
                                    )

                    with context.ifelse(isLt.nonref_expr) as (true, false):
                        with true:
                            context.pushEffect(highIx.expr.store(mid.nonref_expr))
                        with false:
                            context.pushEffect(lowIx.expr.store(mid.nonref_expr.add(1)))
            if containmentOnly:
                context.pushTerminal(native_ast.Expression.Return(arg=native_ast.const_bool_expr(False)))
            else:
                context.pushException(KeyError, "Can't find key")
        return f

    def convert_len_native(self, expr):
        return native_ast.Expression.Branch(
            cond=expr,
            false=native_ast.const_int_expr(0),
            true=expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)
        )

    def convert_len(self, context, expr):
        return context.pushPod(int, self.convert_len_native(expr.nonref_expr))
