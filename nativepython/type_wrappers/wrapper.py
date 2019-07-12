#   Coyright 2017-2019 Nativepython Authors
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

import nativepython

from typed_python import _types

from nativepython.type_wrappers.exceptions import generateThrowException


def replaceTupElt(tup, index, newValue):
    return tuple(tup[:index]) + (newValue,) + tuple(tup[index+1:])


def replaceDictElt(dct, key, newValue):
    res = dict(dct)
    res[key] = newValue
    return res


class Wrapper(object):
    """Represents a code-generation wrapper for objects of a particular type.

    For each type we can represent in typed python, and for types that have
    an injection into compiled code but don't take on runtime values
    (types whose values are known, or singleton functions like 'len' or 'range'),
    we have a corresponding 'wrapper' type.

    The wrapper type is responsible for controlling how we generate code
    to perform the relevant operations on objects of the wrapped type.
    """

    # is this 'plain old data' with no constructor/destructor semantics?
    # if so, we can dispense with destructors entirely.
    is_pod = False

    # does this boil down to a void type? if so, it will always be excluded
    # from function argument lists (both in the defeinitions and in calls)
    is_empty = False

    # do we pass this as a reference to a stackslot or as registers?
    # if true, then when this is a return value, we also have to pass a pointer
    # to the output location as the first argument (and return void) rather
    # than returning registers.
    is_pass_by_ref = True

    # are we a Value or OneOf, where we need to be lowered to a different representation
    # for most operations to succeed?
    can_unwrap = False

    # are we a simple arithmetic type
    is_arithmetic = False

    def __repr__(self):
        return "Wrapper(%s)" % str(self)

    def __str__(self):
        rep = self.typeRepresentation

        if isinstance(rep, type):
            return rep.__qualname__
        if isinstance(rep, tuple) and len(rep) == 2 and isinstance(rep[0], type):
            return "(" + rep[0].__qualname__ + "," + str(rep[1]) + ")"
        return str(rep)

    def __init__(self, typeRepresentation):
        super().__init__()

        self.typeRepresentation = typeRepresentation
        self._conversionCache = {}

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

    def convert_next(self, context, expr):
        """Return a pair of typed_expressions (next_value, continue_iteration) for the result of __next__.

        If continue_iteration is False, then next_value will be ignored. It should be a reference.
        """
        return context.pushTerminal(
            generateThrowException(context, AttributeError("%s object cannot be iterated" % self))
        )

    def convert_attribute(self, context, instance, attribute):
        """Produce code to access 'attribute' on an object represented by TypedExpression 'instance'."""
        return context.pushTerminal(
            generateThrowException(context, AttributeError("%s object has no attribute %s" % (self, attribute)))
        )

    def convert_set_attribute(self, context, instance, attribute, value):
        return context.pushTerminal(
            generateThrowException(context, AttributeError("%s object has no attribute %s" % (self, attribute)))
        )

    def convert_delitem(self, context, instance, item):
        return context.pushTerminal(
            generateThrowException(context, AttributeError("%s is not subscriptable" % str(self)))
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
        if self.is_pod:
            assert target.isReference
            context.pushEffect(
                target.expr.store(toStore.nonref_expr)
            )
        else:
            raise NotImplementedError()

    def convert_copy_initialize(self, context, target, toStore):
        assert target.isReference
        if self.is_pod:
            context.pushEffect(
                target.expr.store(toStore.nonref_expr)
            )
        else:
            raise NotImplementedError()

    def convert_destroy(self, context, instance):
        if self.is_pod:
            pass
        else:
            raise NotImplementedError()

    def convert_default_initialize(self, context, target):
        raise NotImplementedError(self)

    def convert_call(self, context, left, args, kwargs):
        return context.pushException(TypeError, "Can't call %s with args of type (%s)" % (
            self,
            ",".join([str(a.expr_type) for a in args] + ["%s=%s" % (k, str(v.expr_type)) for k, v in kwargs.items()])
        ))

    def convert_len(self, context, expr):
        return context.pushTerminal(
            generateThrowException(context, TypeError("Can't take 'len' of instance of type '%s'" % (str(self),)))
        )

    def convert_hash(self, context, expr):
        return context.pushTerminal(
            generateThrowException(context, TypeError("Can't hash instance of type '%s'" % (str(self),)))
        )

    def convert_unary_op(self, context, expr, op):
        return context.pushTerminal(
            generateThrowException(context, TypeError("Can't apply unary op %s to type '%s'" % (op, expr.expr_type)))
        )

    def convert_to_type(self, context, expr, target_type, explicit=True):
        """Convert to 'target_type' and return a handle on the resulting expression.

        If 'explicit', then we're requesting an agressive conversion that may lose information.

        If non-explicit, then we only allow conversion that's an obvious upcast (say, from float
        to OneOf(None, float))

        We return a TypedExpression, or None if the operation always throws an exception.

        Subclasses are not generally expected to override this function.

        Args:
            context - an ExpressionConversionContext
            expr - a TypedExpression for the instance we're converting
            target_type - a Wrapper for the target type we're converting to
            explicit (bool) - should we allow conversion or not?
        """

        # check if there's nothing to do
        if target_type == self:
            return expr

        # put conversion into its own function
        targetVal = context.allocateUninitializedSlot(target_type)
        succeeded = expr.expr_type.convert_to_type_with_target(context, expr, targetVal, explicit)
        if succeeded is None:
            return
        succeeded = succeeded.convert_to_type(bool)
        if succeeded is None:
            return

        with context.ifelse(succeeded.nonref_expr) as (ifTrue, ifFalse):
            with ifTrue:
                context.markUninitializedSlotInitialized(targetVal)

            with ifFalse:
                context.pushException(TypeError, "Can't convert from type %s to type %s" % (self, target_type))

        return targetVal

    def convert_to_type_with_target(self, context, expr, targetVal, explicit):
        """Convert 'expr' into the slot contained by 'targetVal', returning True if initialized.

        This is the method child classes are expected to override in order to control how they convert.
        If no conversion to the target type is available, we're expected to call the super implementation
        which defers to 'convert_to_self_with_target'
        """
        return targetVal.expr_type.convert_to_self_with_target(context, targetVal, expr, explicit)

    def convert_to_self_with_target(self, context, targetVal, sourceVal, explicit):
        return context.constant(False)

    def convert_bin_op(self, context, l, op, r):
        return r.expr_type.convert_bin_op_reverse(context, r, op, l)

    def convert_bin_op_reverse(self, context, r, op, l):
        if op.matches.Is:
            return context.constant(False)

        if op.matches.IsNot:
            return context.constant(True)

        if op.matches.Eq and l.expr_type != r.expr_type:
            return context.constant(False)

        if op.matches.NotEq and l.expr_type != r.expr_type:
            return context.constant(True)

        return context.pushTerminal(
            generateThrowException(
                context,
                TypeError("Can't apply op %s to expressions of type %s and %s" %
                          (op, str(l.expr_type), str(r.expr_type)))
            )
        )

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0 and not kwargs:
            return context.push(self, lambda x: x.convert_default_initialize())

        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self)

        return context.pushException(
            TypeError,
            "%s() takes at most 1 positional argument" % str(self)
        )

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        return context.pushException(
            TypeError,
            "Can't call %s.%s with args of type (%s)" % (
                self,
                methodname,
                ",".join(
                    [str(a.expr_type) for a in args] +
                    ["%s=%s" % (k, str(v.expr_type)) for k, v in kwargs.items()]
                )
            )
        )

    def get_iteration_expressions(self, context, expr):
        """Return a fixed list of TypedExpressions iterating the object.

        In cases where iteration produces a fixed set of values of possibly
        different types, this lets us 'iterate' the object without having to jam
        all the values down into a single OneOf

        Args:
            context - an ExpressionConversionContext
            expr - a TypedExpression representing the current instance.

        Returns:
            None if we can't do this, or a list of TypedExpressions representing
            the values of the expression.
        """
        return None

    @staticmethod
    def unwrapOneOfAndValue(f):
        """Decorator for 'f' to unwrap value and oneof arguments to their composite forms.

        We loop over each argument to 'f' and check if its a TypedExpression. if so, and if its
        'unwrappable', we call 'f' with the unwrapped form. For OneOf and Value, this means we
        never see actual instances of those objects, just their lowered forms.

        We also loop over lists, tuples, and dicts (at the first level of the argument tree)
        and break those apart.
        """
        def inner(self, context, *args):
            TypedExpression = nativepython.typed_expression.TypedExpression

            for i in range(len(args)):
                if isinstance(args[i], TypedExpression) and args[i].canUnwrap():
                    return args[i].unwrap(
                        lambda newArgI: inner(self, context, *replaceTupElt(args, i, newArgI))
                    )

                if isinstance(args[i], (tuple, list)):
                    for j in range(len(args[i])):
                        if isinstance(args[i][j], TypedExpression) and args[i][j].canUnwrap():
                            return args[i][j].unwrap(
                                lambda newArgIJ: inner(self, context, *replaceTupElt(args, i, replaceTupElt(args[i], j, newArgIJ)))
                            )

                if isinstance(args[i], dict):
                    for key, val in args[i].items():
                        if isinstance(val, TypedExpression) and val.canUnwrap():
                            return val.unwrap(
                                lambda newVal: inner(self, context, *replaceTupElt(args, i, replaceDictElt(args[i], key, newVal)))
                            )

            return f(self, context, *args)

        inner.__name__ = f.__name__
        return inner
