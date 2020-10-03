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

from typed_python import OneOf
from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast


class ValueWrapper(Wrapper):
    """A type wrapper for all 'Value' typed_python types.

    We use Value types to model constants: if you write `OneOf(1, 2, 3)`, you
    have three Value types modeling the three integers. When you actually interact
    with such values at the interpreter level, you always deal with them as
    real python objects because typed_python unboxes them.

    This type wrapper is responsible for implementing these same semantics
    at the compiler level.
    """
    is_pod = True
    is_empty = True
    is_pass_by_ref = False
    is_compile_time_constant = True
    can_unwrap = True

    def getCompileTimeConstant(self):
        return self.typeRepresentation

    def __init__(self, valueType):
        super().__init__(valueType)
        assert getattr(valueType, '__typed_python_category__', '') == 'Value'

    def __str__(self):
        return f"Value({self.typeRepresentation})"

    def __repr__(self):
        return f"Value({self.typeRepresentation})"

    def unwrap(self, context, expr, generator):
        return generator(context.constant(self.typeRepresentation.Value))

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

    def convert_unary_op(self, context, left, op):
        return context.constant(self.typeRepresentation.Value).convert_unary_op(op)

    def convert_bin_op(self, context, left, op, right, inplace):
        """Apply a binary operator to a Value and something else."""
        return context.constant(self.typeRepresentation.Value).convert_bin_op(op, right)

    def convert_bin_op_reverse(self, context, left, op, right, inplace):
        return right.convert_bin_op(op, context.constant(self.typeRepresentation.Value))

    def convert_getitem(self, context, expr, item):
        return context.constant(self.typeRepresentation.Value).convert_getitem(item)

    def convert_attribute(self, context, instance, attribute):
        return context.constant(self.typeRepresentation.Value).convert_attribute(attribute)

    def convert_abs(self, context, expr):
        return context.constant(self.typeRepresentation.Value).convert_abs()

    def _can_convert_to_type(self, otherType, conversionLevel) -> OneOf(False, True, "Maybe"):  # noqa
        return "Maybe"

    def convert_to_type_with_target(self, context, expr, targetVal, conversionLevel, mayThrowOnFailure=False):
        return context.constant(self.typeRepresentation.Value).convert_to_type_with_target(targetVal, conversionLevel, mayThrowOnFailure)

    def convert_to_self_with_target(self, context, targetVal, otherExpr, conversionLevel, mayThrowOnFailure=False):
        return context.constant(self.typeRepresentation.Value) == otherExpr
