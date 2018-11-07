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

class TypedExpression(object):
    def __init__(self, expr, t, isSlotRef):
        super().__init__()
        self.expr = expr
        self.expr_type = t
        self.isSlotRef = isSlotRef

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