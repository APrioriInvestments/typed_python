#   Copyright 2017-2020 typed_python Authors
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

import typed_python.compiler
from typed_python.python_ast import ComparisonOp, UnaryOp
from typed_python import _types, OneOf, ListOf
from typed_python.hash import Hash
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.native_ast import VoidPtr


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


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

    # can we be converted to a pure python representation?
    # if this is true, then we must also have a 'getCompileTimeConstant' method
    is_compile_time_constant = False

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

        # this is the representation of this type _in the compiler_
        self.typeRepresentation = typeRepresentation
        self._conversionCache = {}

    def identityHash(self):
        return (
            Hash(_types.identityHash(self.typeRepresentation))
            + Hash(_types.identityHash(self.interpreterTypeRepresentation))
        )

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.typeRepresentation == other.typeRepresentation

    def __hash__(self):
        return hash((type(self), self.typeRepresentation))

    @property
    def interpreterTypeRepresentation(self):
        """Return the typeRepresentation we should use _at the interpreter_ level.

        This can be different than self.typeRepresentation if we are masquerading
        as another type. This should be the type we would expect if we called

            type(x)

        where x is an instance of the type covered by the wrapper.
        """
        return self.typeRepresentation

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
        context.pushException(
            AttributeError,
            "%s object cannot be iterated" % self
        )

        return None, None

    def convert_attribute(self, context, instance, attribute):
        """Produce code to access 'attribute' on an object represented by TypedExpression 'instance'."""
        if isinstance(self.typeRepresentation, type) and hasattr(self.typeRepresentation, attribute):
            return typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                context,
                self.typeRepresentation
            ).convert_attribute(attribute)

        return context.pushException(
            AttributeError,
            "%s object has no attribute '%s'" % (self, attribute)
        )

    def convert_set_attribute(self, context, instance, attribute, value):
        return context.pushException(
            AttributeError,
            "%s object has no attribute '%s'" % (self, attribute)
        )

    def convert_delitem(self, context, instance, item):
        return context.pushException(
            AttributeError,
            "%s is not subscriptable" % str(self)
        )

    def convert_getitem(self, context, instance, item):
        return context.pushException(
            AttributeError,
            "%s is not subscriptable" % str(self)
        )

    def convert_getslice(self, context, instance, lower, upper, step):
        return context.pushException(
            AttributeError,
            "%s is not sliceable" % str(self)
        )

    def convert_setitem(self, context, instance, index, value):
        return context.pushException(
            AttributeError,
            "%s does not support item assignment" % str(self)
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
        raise NotImplementedError(type(self))

    def convert_masquerade_to_untyped(self, context, instance):
        """If we are masquerading as an untyped type, convert us to that type."""
        return instance

    def can_cast_to_primitive(self, context, e, primitiveType):
        """Returns true if we can call one of the 'convert_X_cast' functions.

        Args:
            primitiveType - one of bool, int, float, str, bytes
        """
        if primitiveType is str:
            return True
        if primitiveType in (int, float, bool, bytes):
            return False
        assert False, "Invalid primitive type argument " + str(primitiveType)

    def convert_call(self, context, left, args, kwargs):
        return context.pushException(TypeError, "Can't call %s with args of type (%s)" % (
            str(self) + " (of type " + str(self.typeRepresentation) + ")",
            ",".join([str(a.expr_type) for a in args] + ["%s=%s" % (k, str(v.expr_type)) for k, v in kwargs.items()])
        ))

    def convert_len(self, context, expr):
        return context.pushException(
            TypeError,
            "Can't take 'len' of instance of type '%s'" % (str(self),)
        )

    def convert_hash(self, context, expr):
        return context.pushException(
            TypeError,
            "Can't hash instance of type '%s'" % (str(self),)
        )

    def convert_abs(self, context, expr):
        return context.pushException(
            TypeError,
            "Can't take 'abs' of instance of type '%s'" % (str(self),)
        )

    def convert_bool_cast(self, context, expr):
        return context.pushException(
            TypeError,
            "Can't take 'bool' of instance of type '%s'" % (str(self),)
        )

    def convert_int_cast(self, context, expr):
        return context.pushException(
            TypeError,
            "Can't take 'int' of instance of type '%s'" % (str(self),)
        )

    def convert_index_cast(self, context, expr):
        return context.pushException(
            TypeError,
            "Can't take instance of type '%s' to an integer index" % (str(self),)
        )

    def convert_float_cast(self, context, expr):
        return context.pushException(
            TypeError,
            "Can't take 'float' of instance of type '%s'" % (str(self),)
        )

    def convert_str_cast(self, context, instance):
        t = instance.expr_type.typeRepresentation

        if not instance.isReference:
            instance = context.pushMove(instance)

        return context.push(
            str,
            lambda newStr:
                newStr.expr.store(
                    runtime_functions.np_str.call(
                        instance.expr.cast(VoidPtr),
                        context.getTypePointer(t)
                    ).cast(typeWrapper(str).getNativeLayoutType())
                )
        )

    def convert_bytes_cast(self, context, expr):
        return context.pushException(
            TypeError,
            "Can't take 'bytes' of instance of type '%s'" % (str(self),)
        )

    def convert_builtin(self, f, context, expr, a1=None):
        if f is dir and a1 is None:
            if not expr.isReference:
                expr = context.pushMove(expr)

            retT = ListOf(str)

            return context.push(
                typeWrapper(retT),
                lambda Ref: Ref.expr.store(
                    runtime_functions.np_dir.call(
                        expr.expr.cast(VoidPtr),
                        context.getTypePointer(expr.expr_type.typeRepresentation)
                    ).cast(typeWrapper(retT).layoutType)
                )
            )

        if f is format and a1 is None:
            return expr.convert_str_cast()

        return context.pushException(
            TypeError,
            "Can't compile '%s' on instance of type '%s'%s"
            % (str(f), str(self), " with additional parameter" if a1 else "")
        )

    def convert_repr(self, context, expr):
        if not expr.isReference:
            expr = context.pushMove(expr)

        return context.push(
            str,
            lambda r: r.expr.store(
                runtime_functions.np_repr.call(
                    expr.expr.cast(VoidPtr),
                    context.getTypePointer(expr.expr_type.typeRepresentation)
                ).cast(typeWrapper(str).layoutType)
            )
        )

    def convert_unary_op(self, context, expr, op):
        if op.matches.Not:
            res = self.convert_bool_cast(context, expr)
            if res is None:
                return res
            return res.convert_unary_op(op)

        return context.pushException(
            TypeError,
            "Can't apply unary op %s to type '%s'" % (op, expr.expr_type)
        )

    def can_convert_to_type(self, otherType, explicit) -> OneOf(False, True, "Maybe"):  # noqa
        """Can we convert to another type? This should match what typed_python does.

        Subclasses may not override this! If either of (self, otherType) knows what to do here,
        we assume that that works. If either has 'Maybe', then we're 'Maybe'

        Args:
            otherType - another Wrapper instance.
            explicit - are we allowing explicit conversion?

        Returns:
            True if we can always convert to this other type
            False if we can never convert
            "Maybe" if it depends on the types involved.
        """
        otherType = typeWrapper(otherType)

        if otherType == self:
            return True

        toType = self._can_convert_to_type(otherType, explicit)
        fromType = otherType._can_convert_from_type(self, explicit)

        if toType is True or fromType is True:
            return True

        if fromType is False and toType is False:
            return False

        return "Maybe"

    def _can_convert_to_type(self, otherType, explicit) -> OneOf(False, True, "Maybe"):  # noqa
        """Does this wrapper know how to convert to 'otherType'?

        Return True if we can convert to this type in all cases. Return False if we
        definitely don't know how. Return "Maybe" if we sometimes can.
        """
        if otherType == self:
            return True

        return "Maybe"

    def _can_convert_from_type(self, otherType, explicit) -> OneOf(False, True, "Maybe"):  # noqa
        """Analagous to _can_convert_to_type.
        """
        if otherType == self:
            return True

        return "Maybe"

    def convert_to_type(self, context, expr, target_type, explicit=True):
        """Convert to 'target_type' and return a handle on the resulting expression.

        If 'explicit', then we're requesting an agressive conversion that may lose information.

        If non-explicit, then we only allow conversion that's an obvious upcast (say, from float
        to OneOf(None, float))

        We return a TypedExpression, or None if the operation always throws an exception.

        Subclasses are not generally expected to override this function. Instead they
        should override _can_convert_to_type, _can_convert_from_type, convert_to_type_with_target,
        and convert_to_self_with_target.

        Note that this is not the pathway used by 'float(x)', 'bool(x)', 'int(x)', 'str(x)' etc,
        which are converted by the convert_(float|int|bool|str|bytes)_cast functions etc. This is
        to ensure that objects (particularly, class instances) that define __float__ don't end
        up being _implicitly_ convertible to float.

        Args:
            context - an ExpressionConversionContext
            expr - a TypedExpression for the instance we're converting
            target_type - a Wrapper for the target type we're converting to
            explicit (bool) - should we allow conversion or not?
        """
        # check if there's nothing to do
        if target_type == self.typeRepresentation or target_type == self:
            return expr

        canConvert = self.can_convert_to_type(target_type, explicit)

        if canConvert is False:
            context.pushException(TypeError, "Couldn't initialize type %s from %s" % (target_type, self))
            return None

        # put conversion into its own function
        targetVal = context.allocateUninitializedSlot(target_type)

        succeeded = expr.expr_type.convert_to_type_with_target(context, expr, targetVal, explicit)

        if succeeded is None:
            return

        # if we know with certainty that we can convert, then don't produce the exception
        # code.
        if canConvert is True:
            if not (succeeded.expr.matches.Constant and succeeded.expr.val.truth_value()):
                raise Exception(
                    f"Trying to convert {self} to {target_type}, we "
                    f"were promised conversion would succeed, but it didn't. Expr was {succeeded.expr}"
                )

            context.markUninitializedSlotInitialized(targetVal)
            return targetVal

        succeeded = succeeded.convert_to_type(bool)
        if succeeded is None:
            return

        with context.ifelse(succeeded.nonref_expr) as (ifTrue, ifFalse):
            with ifTrue:
                context.markUninitializedSlotInitialized(targetVal)

            with ifFalse:
                context.pushException(TypeError, f"Can't convert from type {self} to type {target_type}")

        return targetVal

    def convert_to_type_with_target(self, context, expr, targetVal, explicit):
        """Convert 'expr' into the slot contained by 'targetVal', returning True if initialized.

        This is the method child classes are expected to override in order to control how they convert.
        If no conversion to the target type is available, we're expected to call the super implementation
        which defers to 'convert_to_self_with_target'
        """
        return targetVal.expr_type.convert_to_self_with_target(context, targetVal, expr, explicit)

    def convert_to_self_with_target(self, context, targetVal, sourceVal, explicit):
        if sourceVal.expr_type == self:
            targetVal.convert_copy_initialize(sourceVal)
            return context.constant(True)

        return context.constant(False)

    def convert_bin_op(self, context, l, op, r, inplace):
        return r.expr_type.convert_bin_op_reverse(context, r, op, l, inplace)

    def convert_bin_op_reverse(self, context, r, op, l, inplace):
        if op.matches.Is:
            return context.constant(False)

        if op.matches.IsNot:
            return context.constant(True)

        if op.matches.Eq and l.expr_type != r.expr_type:
            return context.constant(False)

        if op.matches.NotEq and l.expr_type != r.expr_type:
            return context.constant(True)

        if op.matches.NotIn:
            res = l.convert_bin_op(ComparisonOp.In(), r, False)
            if not res:
                return
            res = res.convert_bool_cast()
            if not res:
                return
            return res.convert_unary_op(UnaryOp.Not())

        return context.pushException(
            TypeError,
            "Can't apply op %s to expressions of type %s and %s" %
            (op, str(l.expr_type), str(r.expr_type))
        )

    def convert_format(self, context, instance, formatSpecOrNone=None):
        if formatSpecOrNone is None:
            return instance.convert_str_cast()
        else:
            raise context.pushException(TypeError, "We don't support conversion in the base wrapper.")

    def convert_type_call(self, context, typeInst, args, kwargs):
        context.pushException(
            TypeError,
            f"We can't call type {self.typeRepresentation} with args {args} and kwargs {kwargs}"
        )

    def convert_call_on_container_expression(self, context, inst, argExpr):
        """Convert a case where we are calling x([...]) or similar.

        We can't just convert expressions like (1, 2, 3) to containers directly because
        we'd lose the information about what kind of object they are (they have to be
        represented as 'list' or 'tuple' which loses the typing information). This is
        especially true in the case of list and dict because they're mutable, so whatever
        type information we thought we could infer isn't stable. So instead, we look
        for cases where we can see that the resulting container is directly passed to a
        type function and give it the chance to do something efficient if it knows
        how to.

        The default implementation simply renders the expression and calls 'self' with
        it.

        Args:
            context - an ExpressionConversionContext
            inst - a TypedExpression giving the instance being called.
            argExpr - a python_ast.Expr object representing the expression we're converting
        """
        argVal = context.convert_expression_ast(argExpr)

        if argVal is None:
            return argVal

        return inst.convert_call((argVal,), {})

    def convert_type_call_on_container_expression(self, context, typeInst, argExpr):
        """Like convert_call_on_container_expression, but on us as a TYPE.

        Args:
            context - an ExpressionConversionContext
            typeInst - a TypedExpression giving the instance of the type object itself,
                which we probably don't care about since most type objects are represented
                as 'void' and have no actual instance information.
            argExpr - a python_ast.Expr object representing the expression we're converting
        """
        argVal = context.convert_expression_ast(argExpr)

        if argVal is None:
            return argVal

        return typeInst.convert_call((argVal,), {})

    def has_method(self, context, instance, methodName):
        assert isinstance(methodName, str)
        return False

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

    def convert_context_manager_enter(self, context, instance):
        return instance.convert_method_call("__enter__", (), {})

    def convert_context_manager_exit(self, context, instance, args):
        return instance.convert_method_call("__exit__", args, {})

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
            TypedExpression = typed_python.compiler.typed_expression.TypedExpression

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
