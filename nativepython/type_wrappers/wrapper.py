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
import nativepython

from nativepython.type_wrappers.exceptions import generateThrowException

class Wrapper(object):
    #properties of all objects:

    #is this 'plain old data' with no constructor/destructor semantics
    is_pod = False

    #does this boil down to a void type?
    is_empty = False

    #do we pass this as a reference to a stackslot or as registers?
    is_pass_by_ref = True

    def __repr__(self):
        return "Wrapper(%s)" % self.typeRepresentation

    def __init__(self, typeRepresentation):
        super().__init__()

        self.typeRepresentation = typeRepresentation

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.typeRepresentation == other.typeRepresentation

    def __hash__(self):
        return hash(self.typeRepresentation)

    def ensureNonReference(self, e):
        if e.isReference:
            if self.is_empty:
                return e.context.NoneExpr(e.expr + native_ast.nullExpr)
            return e.context.ValueExpr(e.expr.load(), e.expr_type)
        return e

    def getNativePassingType(self):
        if self.is_pass_by_ref:
            return self.getNativeLayoutType().pointer()
        else:
            return self.getNativeLayoutType()

    def convert_incref(self, context, expr):
        if self.is_pod:
            return expr
        raise NotImplementedError(self)

    def convert_attribute(self, context, instance, attribute):
        return nativepython.typed_expression.TypedExpression(
            generateThrowException(context, AttributeError("object has no attribute " + attribute)),
            None,
            False
            )

    def convert_set_attribute(self, context, instance, attribute, value):
        return nativepython.typed_expression.TypedExpression(
            generateThrowException(context, AttributeError("object has no attribute " + attribute)),
            None,
            False
            )

    def convert_assign(self, context, target, toStore):
        raise NotImplementedError()

    def convert_copy_initialize(self, context, target, toStore):
        raise NotImplementedError()

    def convert_destroy(self, context, instance):
        raise NotImplementedError()

    def convert_call(self, context, left, args):
        raise NotImplementedError()

    def convert_len(self, context, expr):
        raise NotImplementedError()

    def convert_to_type(self, context, expr, target_type):
        return context.TerminalExpr(
            generateThrowException(context, TypeError("Can't convert from type %s to type %s" % (self, target_type)))
            )

