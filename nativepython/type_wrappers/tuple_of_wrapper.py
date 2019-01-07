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
from nativepython.typed_expression import TypedExpression
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, Int64

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)

class TupleOfWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)

        self.underlyingWrapperType = typeWrapper(t.ElementType)

        self.layoutType = native_ast.Type.Struct(element_types=(
            ('refcount', native_ast.Int64),
            ('hash_cache', native_ast.Int32),
            ('count', native_ast.Int32),
            ('data', native_ast.UInt8)
            ), name='TupleOfLayout').pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        assert instance.isReference

        if self.underlyingWrapperType.is_pod:
            return runtime_functions.free.call(instance.nonref_expr.cast(native_ast.UInt8Ptr))
        else:
            return (
                context.converter.defineNativeFunction(
                    "destructor_" + str(self.typeRepresentation), 
                    ('destructor', self), 
                    [self],
                    typeWrapper(NoneType()),
                    lambda: self.generateNativeDestructorFunction(context)
                    )
                .call(instance.expr)
                )

    def generateNativeDestructorFunction(self, context):
        inst_expr = native_ast.Expression.Variable(name='input').load()

        body = native_ast.FunctionBody.Internal(body=
            context.loop_expr(
                inst_expr.ElementPtrIntegers(0,2).load().cast(native_ast.Int64),
                lambda ix_expr:
                    context.RefExpr(
                        inst_expr.ElementPtrIntegers(0,3).cast(
                            self.underlyingWrapperType.getNativeLayoutType().pointer()
                            ).elemPtr(ix_expr),
                        self.underlyingWrapperType
                        ).convert_destroy().expr
                ).expr >>
            runtime_functions.free.call(inst_expr.cast(native_ast.UInt8Ptr))
            )

        return native_ast.Function(
            args=[
                ('input', self.getNativeLayoutType().pointer())
                ],
            body=body,
            output_type=native_ast.Type.Void()
            )

    def convert_bin_op(self, context, left, op, right):
        if right.expr_type == left.expr_type:
            if op.matches.Add:
                new_tuple = context.allocate_temporary(self)

                return context.RefExpr(
                    context.converter.defineNativeFunction(
                        'concatenate_tuples(' + self.typeRepresentation.__name__ + "," + right.expr_type.typeRepresentation.__name__ + ")",
                        ('util', self, 'concatenate', right.expr_type),
                        [self, self],
                        self,
                        lambda: self.generateConcatenateTuple(context)
                        ).call(new_tuple.expr, left.expr, right.expr)
                    >> context.activates_temporary(new_tuple)
                    >> new_tuple.expr,
                    self                    
                    )

        return super().convert_bin_op(context, left, op, right)

    def generateConcatenateTuple(self, context):
        out_expr = native_ast.Expression.Variable(name='output')
        left_expr = native_ast.Expression.Variable(name='left').load()
        right_expr = native_ast.Expression.Variable(name='right').load()


        def elt_ref(tupPtrExpr, iExpr):
            return context.RefExpr(
                tupPtrExpr.ElementPtrIntegers(0,3).cast(
                        self.underlyingWrapperType.getNativeLayoutType().pointer()
                        ).elemPtr(iExpr),
                self.underlyingWrapperType
                )

        return native_ast.Function(
            args=[('output', self.getNativePassingType()),
                  ('left', self.getNativePassingType()),
                  ('right', self.getNativePassingType())
                  ],
            output_type=native_ast.Void,
            body=
                native_ast.FunctionBody.Internal(body=
                    context.let(self.convert_len_native(left_expr), lambda left_size_expr:
                    context.let(self.convert_len_native(right_expr), lambda right_size_expr:
                    out_expr.store(
                        runtime_functions.malloc.call(
                            left_size_expr.add(right_size_expr).mul(self.underlyingWrapperType.getBytecount())
                                .add(native_ast.const_int_expr(16))
                            ).cast(self.getNativeLayoutType())
                        ) >>
                    out_expr.load().ElementPtrIntegers(0, 0).store(native_ast.const_int_expr(1)) >>
                    out_expr.load().ElementPtrIntegers(0, 1).store(native_ast.const_int32_expr(-1)) >>
                    out_expr.load().ElementPtrIntegers(0, 2).store(
                        left_size_expr.add(right_size_expr).cast(native_ast.Int32)
                        ) >>
                    context.loop_expr(
                        left_size_expr,
                        lambda i: elt_ref(out_expr.load(), i)
                            .convert_copy_initialize(elt_ref(left_expr, i)).expr
                        ).expr >> 
                    context.loop_expr(
                        right_size_expr,
                        lambda i: elt_ref(out_expr.load(), i.add(left_size_expr))
                            .convert_copy_initialize(elt_ref(right_expr, i)).expr
                        ).expr
                    ))
                )
            )

    def convert_getitem(self, context, expr, item):
        expr = expr.ensureNonReference()

        return context.RefExpr(
            native_ast.Expression.Branch(
                cond=((item >= 0) & (item < self.convert_len(context, expr))).nonref_expr,
                true=expr.expr.ElementPtrIntegers(0,3).cast(
                    self.underlyingWrapperType.getNativeLayoutType().pointer()
                    ).elemPtr(item.toInt64().nonref_expr),
                false=generateThrowException(context, IndexError("tuple index out of range"))
                ),
            self.underlyingWrapperType
            )

    def convert_len_native(self, expr):
        return native_ast.Expression.Branch(
                cond=expr,
                false=native_ast.const_int_expr(0),
                true=expr.ElementPtrIntegers(0,2).load().cast(native_ast.Int64)
                )

    def convert_len(self, context, expr):
        return context.ValueExpr(
            self.convert_len_native(expr.nonref_expr),
            Int64()
            )



