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
from typed_python import Int32
import typed_python.compiler.native_ast as native_ast


class NoneWrapper(Wrapper):
    is_pod = True
    is_empty = True
    is_pass_by_ref = False
    is_compile_time_constant = True

    def __init__(self):
        super().__init__(type(None))

    def getCompileTimeConstant(self):
        return None

    def convert_default_initialize(self, context, target):
        pass

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_assign(self, context, target, toStore):
        pass

    def convert_copy_initialize(self, context, target, toStore):
        pass

    def convert_destroy(self, context, instance):
        pass

    def convert_hash(self, context, expr):
        return context.constant(Int32(0))

    def convert_bin_op(self, context, left, op, right, inplace):
        if right.expr_type == self:
            if op.matches.Eq:
                return context.constant(True)
            if op.matches.NotEq or op.matches.Lt or op.matches.LtE or op.matches.Gt or op.matches.GtE:
                return context.constant(False)
            if op.matches.Is:
                return context.constant(True)
            if op.matches.IsNot:
                return context.constant(False)

        return super().convert_bin_op(context, left, op, right, inplace)

    def _can_convert_to_type(self, targetType, conversionLevel):
        if not conversionLevel.isNewOrHigher():
            return False

        return targetType.typeRepresentation in (bool, str)

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        if targetVal.expr_type.typeRepresentation is bool:
            context.pushEffect(targetVal.expr.store(context.constant(False).nonref_expr))
            return context.constant(True)

        if targetVal.expr_type.typeRepresentation is str:
            targetVal.convert_copy_initialize(context.constant("None"))
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)
