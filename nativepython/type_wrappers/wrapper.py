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
from typed_python import _types

from nativepython.type_wrappers.exceptions import generateThrowException

class Wrapper(object):
    #properties of all objects:

    #is this 'plain old data' with no constructor/destructor semantics?
    #if so, we can dispense with destructors entirely.
    is_pod = False

    #does this boil down to a void type? if so, it will always be excluded
    #from function argument lists (both in the defeinitions and in calls)
    is_empty = False

    #do we pass this as a reference to a stackslot or as registers?
    #if true, then when this is a return value, we also have to pass a pointer
    #to the output location as the first argument (and return void) rather
    #than returning registers.
    is_pass_by_ref = True

    def __repr__(self):
        return "Wrapper(%s)" % self.typeRepresentation.__qualname__

    def __str__(self):
        return self.typeRepresentation.__qualname__

    def __init__(self, typeRepresentation):
        super().__init__()

        self.typeRepresentation = typeRepresentation

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.typeRepresentation == other.typeRepresentation

    def __hash__(self):
        return hash(self.typeRepresentation)

    def getNativePassingType(self):
        if self.is_pass_by_ref:
            return self.getNativeLayoutType().pointer()
        else:
            return self.getNativeLayoutType()

    def getBytecount(self):
        if self.is_empty:
            return 0

        return _types.bytecount(self.typeRepresentation)

    def convert_incref(self, context, expr):
        if self.is_pod:
            return

        raise NotImplementedError(self)

    def convert_attribute(self, context, instance, attribute):
        return context.pushTerminal(
            generateThrowException(context, AttributeError("object has no attribute " + attribute))
            )

    def convert_set_attribute(self, context, instance, attribute, value):
        return context.pushTerminal(
            generateThrowException(context, AttributeError("object has no attribute " + attribute))
            )

    def convert_getitem(self, context, instance, item):
        return context.pushTerminal(
            generateThrowException(context, AttributeError("%s is not subscriptable" % str(self)))
            )

    def convert_setitem(self, context, instance, index, value):
        return context.pushTerminal(
            generateThrowException(context, AttributeError("%s does not support item assignment" % str(self)))
            )

    def convert_assign(self, context, target, toStore):
        raise NotImplementedError(self)

    def convert_default_initialize(self, context, target):
        raise NotImplementedError(self)

    def convert_copy_initialize(self, context, target, toStore):
        raise NotImplementedError(self)

    def convert_destroy(self, context, instance):
        raise NotImplementedError(self)

    def convert_call(self, context, left, args):
        return context.pushException(TypeError, "Can't call %s with args of type (%s)" % (
            self.typeRepresentation.__qualname__,
            ",".join([str(a.expr_type) for a in args])
            ))

    def convert_len(self, context, expr):
        return context.pushTerminal(
            generateThrowException(context, TypeError("Can't take 'len' of instance of type '%s'" % (str(self),)))
            )

    def convert_unary_op(self, context, expr, op):
        return context.pushTerminal(
            generateThrowException(context, TypeError("Can't apply unary op %s to type '%s'" % (op, expr.expr_type)))
            )

    def convert_to_type(self, context, expr, target_type):
        return target_type.convert_to_self(context, expr)

    def convert_to_self(self, context, expr):
        if expr.expr_type == self:
            return expr
        return context.pushTerminal(
            generateThrowException(context,
                TypeError("Can't convert from type %s to type %s" % (
                    expr.expr_type.typeRepresentation.__name__,
                    self.typeRepresentation.__name__)
                    )
                )
            )

    def convert_bin_op(self, context, l, op, r):
        return r.expr_type.convert_bin_op_reverse(context, r, op, l)

    def convert_bin_op_reverse(self, context, r, op, l):
        return context.pushTerminal(
            generateThrowException(context, TypeError("Can't apply op %s to expressions of type %s and %s" %
                (op, str(l.expr_type), str(r.expr_type))))
            )

    def convert_type_call(self, context, typeInst, args):
        if len(args) == 0:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1:
            return args[0].convert_to_type(self)

        return context.pushException(
            TypeError,
            "%s() takes at most 1 argument" % str(self)
            )

    def convert_method_call(self, context, instance, methodname, args):
        return context.pushException(TypeError, "Can't call %s.%s with args of type (%s)" % (
            self.typeRepresentation.__qualname__, methodname,
            ",".join([a.expr_type.typeRepresentation.__qualname__ for a in args])
            ))
