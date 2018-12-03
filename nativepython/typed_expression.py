#   Copyright 2017 Braxton Mckee
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

import nativepython.native_ast as native_ast

from nativepython.type_wrappers.wrapper import Wrapper
from nativepython.type_wrappers.none_wrapper import NoneWrapper

class TypedExpression(object):
    def __init__(self, expr, t, isReference):
        """Initialize a TypedExpression

        expr - a native_ast containing an expression
        t - a subclass of Wrapper
        isReference - is this a pointer to a memory location, or the actual value?
        """
        super().__init__()
        assert isinstance(t, Wrapper) or t is None, t

        self.expr = expr
        self.expr_type = t
        self.isReference = isReference

    def convert_assign(self, context, toStore):
        return self.expr_type.convert_assign(context, self, toStore)

    def convert_copy_initialize(self, context, target, toStore):
        return self.expr_type.convert_copy_initialize(context, self, toStore)

    def convert_bin_op(self, context, op, rhs):
        return self.expr_type.convert_bin_op(context, self, op, rhs)

    def unwrap(self):
        return self.expr_type.unwrap(self)

    def asNonref(self):
        return self.expr_type.asNonref(self)

    def toFloat64(self):
        return self.expr_type.toFloat64(self)

    def toInt64(self):
        return self.expr_type.toInt64(self)

    def __str__(self):
        return "Expression(%s%s)" % (self.expr_type, ",[ref]" if self.isReference else "")

    @staticmethod
    def Void(expr=None):
        return TypedExpression(expr if expr is not None else native_ast.nullExpr, NoneWrapper(), False)

    def __add__(self, other):
        return TypedExpression(self.expr + other.expr, other.expr_type, other.isReference)