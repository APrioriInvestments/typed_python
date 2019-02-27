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

from nativepython.typed_expression import TypedExpression
from nativepython.type_wrappers.tuple_of_wrapper import TupleOrListOfWrapper
from nativepython.type_wrappers.bound_compiled_method_wrapper import BoundCompiledMethodWrapper
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import PointerTo

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class ListOfWrapper(TupleOrListOfWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def convert_len_native(self, expr):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64)

    def convert_reserved_native(self, expr):
        if isinstance(expr, TypedExpression):
            expr = expr.nonref_expr
        return expr.ElementPtrIntegers(0, 3).load().cast(native_ast.Int64)

    def convert_reserved(self, context, expr):
        return context.pushPod(int, expr.nonref_expr.ElementPtrIntegers(0, 3).load().cast(native_ast.Int64))

    def convert_attribute(self, context, instance, attr):
        if attr in ("copy", "resize", "reserve", "reserved", "append", "clear", "pop", "setSizeUnsafe", "pointerUnsafe"):
            return instance.changeType(BoundCompiledMethodWrapper(self, attr))

        return super().convert_attribute(context, instance, attr)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if kwargs:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

        if methodname == "pop":
            if len(args) == 0:
                args = (context.constant(-1),)

            if len(args) == 1:
                count = args[0].toInt64()
                if count is None:
                    return

                native = context.converter.defineNativeFunction(
                    'pop(' + self.typeRepresentation.__name__ + ")",
                    ('util', self, 'pop'),
                    [self, int],
                    self.underlyingWrapperType,
                    self.generatePop
                )

                if self.underlyingWrapperType.is_pass_by_ref:
                    return context.push(
                        self.underlyingWrapperType,
                        lambda out: native.call(out, instance, count)
                    )
                else:
                    return context.pushPod(
                        self.underlyingWrapperType,
                        native.call(instance, count)
                    )

        if methodname == "pointerUnsafe":
            if len(args) == 1:
                count = args[0].toInt64()
                if count is None:
                    return

                return context.pushPod(
                    PointerTo(self.typeRepresentation.ElementType),
                    instance.nonref_expr.ElementPtrIntegers(0, 4).load().cast(
                        self.underlyingWrapperType.getNativeLayoutType().pointer()
                    ).elemPtr(count.nonref_expr)
                )

        if methodname == "setSizeUnsafe":
            if len(args) == 1:
                count = args[0].toInt64()
                if count is None:
                    return

                context.pushEffect(instance.nonref_expr.ElementPtrIntegers(0, 2).store(count.nonref_expr.cast(native_ast.Int32)))

                return context.pushVoid()

        if methodname == "resize":
            if len(args) == 1:
                count = args[0].toInt64()
                if count is None:
                    return

                return context.pushPod(
                    None,
                    context.converter.defineNativeFunction(
                        'resize(' + self.typeRepresentation.__name__ + ")",
                        ('util', self, 'resize'),
                        [self, int],
                        None,
                        self.generateResize
                    ).call(instance, count)
                )
            if len(args) == 2:
                count = args[0].toInt64()
                if count is None:
                    return

                val = args[1].convert_to_type(self.underlyingWrapperType)
                if val is None:
                    return

                return context.pushPod(
                    None,
                    context.converter.defineNativeFunction(
                        'resize(' + self.typeRepresentation.__name__ + ")",
                        ('util', self, 'resize'),
                        [self, int, self.underlyingWrapperType],
                        None,
                        self.generateResize
                    ).call(instance, count, val)
                )

        if methodname == "append":
            if len(args) == 1:
                val = args[0].convert_to_type(self.underlyingWrapperType)
                if val is None:
                    return

                return context.pushPod(
                    None,
                    context.converter.defineNativeFunction(
                        'append(' + self.typeRepresentation.__name__ + ")",
                        ('util', self, 'append'),
                        [self, self.underlyingWrapperType],
                        None,
                        self.generateAppend
                    ).call(instance, val)
                )
        if methodname == "copy":
            if len(args) == 0:
                return context.push(
                    self,
                    lambda out:
                        context.converter.defineNativeFunction(
                            'copy(' + self.typeRepresentation.__name__ + ")",
                            ('util', self, 'copy'),
                            [self],
                            self,
                            self.generateCopy
                        ).call(out, instance)
                )
        if methodname == "clear":
            if len(args) == 0:
                return self.convert_method_call(context, instance, "resize", (context.constant(0),), {})

        if methodname == "reserve":
            if len(args) == 1:
                count = args[0].toInt64()
                if count is None:
                    return

                return context.pushPod(
                    None,
                    context.converter.defineNativeFunction(
                        'reserve(' + self.typeRepresentation.__name__ + ")",
                        ('util', self, 'reserve'),
                        [self, int],
                        None,
                        self.generateReserve
                    ).call(instance, count)
                )
        if methodname == "reserved":
            if len(args) == 0:
                return context.pushPod(
                    int,
                    context.converter.defineNativeFunction(
                        'reserved(' + self.typeRepresentation.__name__ + ")",
                        ('util', self, 'reserved'),
                        [self],
                        int,
                        self.generateReserved
                    ).call(instance)
                )

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def generatePop(self, context, out, inst, ix):
        ix = context.push(int, lambda tgt: tgt.expr.store(ix.nonref_expr))

        with context.ifelse(ix < 0) as (then, otherwise):
            with then:
                context.pushEffect(
                    ix.expr.store(ix + inst.convert_len())
                )
        with context.ifelse((ix < 0).nonref_expr.bitor(ix >= inst.convert_len())) as (then, otherwise):
            with then:
                context.pushException(IndexError, "pop index out of range")

        # we are just moving this - we assume no layouts have selfpointers throughout nativepython
        result = context.push(
            self.underlyingWrapperType,
            lambda result: result.expr.store(inst.convert_getitem_unsafe(ix).nonref_expr)
        )

        context.pushEffect(
            inst.nonref_expr.ElementPtrIntegers(0, 2).store(
                inst.nonref_expr.ElementPtrIntegers(0, 2).load().add(native_ast.const_int32_expr(-1))
            )
        )

        data = inst.nonref_expr.ElementPtrIntegers(0, 4).load()

        context.pushEffect(
            runtime_functions.memmove.call(
                data.elemPtr(ix * self.underlyingWrapperType.getBytecount()),
                data.elemPtr((ix+1) * self.underlyingWrapperType.getBytecount()),
                inst.nonref_expr.ElementPtrIntegers(0, 2).load().cast(native_ast.Int64).sub(ix.nonref_expr).mul(
                    self.underlyingWrapperType.getBytecount()
                )
            )
        )

        if not self.underlyingWrapperType.is_pass_by_ref:
            context.pushEffect(
                native_ast.Expression.Return(arg=result.nonref_expr)
            )
        else:
            context.pushEffect(
                out.expr.store(result.nonref_expr)
            )

    def generateResize(self, context, out, listInst, countInst, arg=None):
        with context.ifelse(listInst.convert_len() == countInst) as (if_eq, if_neq):
            with if_eq:
                context.pushEffect(native_ast.Expression.Return(arg=None))

        with context.ifelse(listInst.convert_len() < countInst) as (if_bigger, if_smaller):
            with if_bigger:
                with context.ifelse(listInst.convert_reserved() < countInst) as (if_needs_reserve, _):
                    with if_needs_reserve:
                        self.convert_method_call(context, listInst, "reserve", (countInst,), {})

                with context.loop(countInst - listInst.convert_len()) as i:
                    if arg is None:
                        listInst.convert_getitem_unsafe(i+listInst.convert_len()).convert_default_initialize()
                    else:
                        listInst.convert_getitem_unsafe(i+listInst.convert_len()).convert_copy_initialize(arg)

            with if_smaller:
                with context.loop(listInst.convert_len() - countInst) as i:
                    listInst.convert_getitem_unsafe(i+countInst).convert_destroy()

        context.pushEffect(
            listInst.nonref_expr.ElementPtrIntegers(0, 2).store(countInst.nonref_expr.cast(native_ast.Int32))
        )

    def generateAppend(self, context, out, listInst, arg):
        with context.ifelse(listInst.convert_reserved() < listInst.convert_len()+1) as (if_needs_reserve, _):
            with if_needs_reserve:
                self.convert_method_call(context, listInst, "reserve", ((listInst.convert_len() * 5) / 4 + 1,), {})

        listInst.convert_getitem_unsafe(listInst.convert_len()).convert_copy_initialize(arg)

        context.pushEffect(
            listInst.nonref_expr.ElementPtrIntegers(0, 2).store((listInst.convert_len()+1).nonref_expr.cast(native_ast.Int32))
        )

    def generateCopy(self, context, out, listInst):
        self.convert_default_initialize(context, out)

        self.convert_method_call(context, out, "reserve", (listInst.convert_len(),), {})

        with context.loop(listInst.convert_len()) as i:
            result = listInst.convert_getitem_unsafe(i).convert_to_type(typeWrapper(self.typeRepresentation.ElementType))
            if result is None:
                return None
            out.convert_getitem_unsafe(i).convert_copy_initialize(result)

        self.convert_method_call(context, out, "setSizeUnsafe", (listInst.convert_len(),), {})

    def generateReserve(self, context, out, listInst, countInst):
        countInst = context.push(int, lambda target: target.expr.store(countInst.nonref_expr))

        with context.ifelse(countInst < listInst.convert_len()) as (then, _):
            with then:
                context.pushEffect(countInst.expr.store(listInst.convert_len().nonref_expr))

        context.pushEffect(
            listInst.nonref_expr.ElementPtrIntegers(0, 4).store(
                runtime_functions.realloc.call(
                    listInst.nonref_expr.ElementPtrIntegers(0, 4).load(),
                    countInst.nonref_expr.mul(self.underlyingWrapperType.getBytecount())
                )
            )
        )

        context.pushEffect(
            listInst.nonref_expr.ElementPtrIntegers(0, 3).store(countInst.nonref_expr.cast(native_ast.Int32))
        )

    def generateReserved(self, context, out, listInst):
        context.pushEffect(native_ast.Expression.Return(arg=listInst.convert_reserved().nonref_expr))

    def convert_default_initialize(self, context, tgt):
        context.pushEffect(
            context.converter.defineNativeFunction(
                'empty(' + self.typeRepresentation.__name__ + ")",
                ('util', self, 'empty'),
                [],
                self,
                self.createEmptyList
            ).call(tgt)
        )

    def createEmptyList(self, context, out):
        context.pushEffect(
            out.expr.store(
                runtime_functions.malloc.call(28).cast(self.getNativeLayoutType())
            )
            >> out.nonref_expr.ElementPtrIntegers(0, 0).store(native_ast.const_int_expr(1))  # refcount
            >> out.nonref_expr.ElementPtrIntegers(0, 1).store(native_ast.const_int32_expr(-1))  # hash cache
            >> out.nonref_expr.ElementPtrIntegers(0, 2).store(native_ast.const_int32_expr(0))  # count
            >> out.nonref_expr.ElementPtrIntegers(0, 3).store(native_ast.const_int32_expr(1))  # reserved
            >> out.nonref_expr.ElementPtrIntegers(0, 4).store(
                runtime_functions.malloc.call(self.underlyingWrapperType.getBytecount())
            )  # data
        )

    def convert_setitem(self, context, expr, index, item):
        item = item.convert_to_type(self.underlyingWrapperType)
        if item is None:
            return None

        expr.convert_getitem(index).convert_assign(item)

        return context.pushVoid()

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 1 and args[0].expr_type == self and not kwargs:
            return context.push(
                self,
                lambda out:
                    context.converter.defineNativeFunction(
                        'copy(' + str(self) + "," + str(args[0].expr_type) + ")",
                        ('util', self, 'copy'),
                        [args[0].expr_type],
                        self,
                        self.generateCopy
                    ).call(out, args[0])
            )

        return super().convert_type_call(context, typeInst, args, kwargs)
