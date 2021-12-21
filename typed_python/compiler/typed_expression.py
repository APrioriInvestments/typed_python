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

from typed_python import UInt64, PointerTo
from typed_python.python_ast import BinaryOp, ComparisonOp, BooleanOp
import typed_python.compiler.native_ast as native_ast
import typed_python.compiler
from typed_python.compiler.conversion_level import ConversionLevel


from typed_python.compiler.type_wrappers.wrapper import Wrapper

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class TypedExpression:
    def __init__(self, context, expr, t, isReference, constantValue=None):
        """Initialize a TypedExpression

        context - an ExpressionConversionContext
        expr - a native_ast containing an expression
        t - a subclass of Wrapper, or a type that we'll convert to a wrapper, or None (meaning
            control doesn't return)
        isReference - is this a pointer to a memory location holding the value, or the actual value?
            if it's a reference, the reference is guaranteed to be valid for the lifetime of the
            expression. if its the value, then the value contains an implicit incref which must
            either be transferred or decreffed.
        constantValue - if this is a simple inline constant, what is it? otherwise, None.
        """
        super().__init__()
        if isinstance(t, type) or hasattr(t, "__typed_python_category__"):
            t = typeWrapper(t)

        assert isinstance(t, Wrapper) or t is None, t
        assert isinstance(expr, native_ast.Expression), expr

        self.context = context
        self.expr = expr
        self.expr_type = t
        self.isReference = isReference
        self.constantValue = constantValue

        if self.constantValue is None and self.expr_type.is_compile_time_constant:
            self.constantValue = self.expr_type.getCompileTimeConstant()

    @property
    def isConstant(self):
        return self.constantValue is not None

    def changeContext(self, newContext):
        return TypedExpression(newContext, self.expr, self.expr_type, self.isReference)

    def withConstantValue(self, newVal):
        return TypedExpression(
            self.context,
            self.expr,
            self.expr_type,
            self.isReference,
            constantValue=newVal
        )

    def asPointer(self):
        """Change from being a reference to 'T' to being a _value_ of type 'T*'"""
        assert self.isReference

        return self.changeType(
            typeWrapper(PointerTo(self.expr_type.typeRepresentation)),
            False
        )

    def asPointerIf(self, ifExpr):
        """Return 'self' as a PointerTo(T) which is null if ifExpr is False.

        Args:
            self - a reference to a T
            ifExpr - a TypedExpression to something which can be compared to zero.

        Returns:
            a TypedExpression(PointerTo(T)) which is _not_ a reference.
        """
        assert self.isReference

        return TypedExpression(
            self.context,
            native_ast.Expression.Branch(
                cond=ifExpr.nonref_expr,
                true=self.expr,
                false=self.expr_type.getNativeLayoutType().pointer().zero()
            ),
            typeWrapper(PointerTo(self.expr_type.typeRepresentation)),
            False,
            None
        )

    def asReference(self):
        """Change from being a value 'PointerTo(T)' to being a reference of type 'T'"""
        if self.isReference:
            self = TypedExpression(
                self.context,
                self.nonref_expr,
                self.expr_type,
                False
            )

        assert (
            isinstance(self.expr_type.typeRepresentation, type)
            and issubclass(self.expr_type.typeRepresentation, PointerTo)
        )

        return self.changeType(
            typeWrapper(self.expr_type.typeRepresentation.ElementType),
            True
        )

    def changeType(self, newType, isReferenceOverride=None):
        """Return a TypedExpression with the same native_ast content but a different type-wrapper."""
        return TypedExpression(
            self.context,
            self.expr,
            typeWrapper(newType),
            self.isReference if isReferenceOverride is None else isReferenceOverride
        )

    def as_native_call_arg(self):
        """Convert this expression to a call-argument form."""
        if self.expr_type.is_pass_by_ref:
            assert self.isReference
            return self.expr
        else:
            return self.nonref_expr

    def canUnwrap(self):
        return self.expr_type.can_unwrap

    def unwrap(self, generator):
        """If we 'canUnwrap', call generator back with 'self' in the lowered form.

        In the case of a OneOf, this may produce a compound expression that merges the operation
        over the possible subtypes.
        """
        return self.expr_type.unwrap(self.context, self, generator)

    @property
    def nonref_expr(self):
        """Get our expression (dereferenced if necessary) so that it definitely represents the real object, not its location"""
        if self.expr_type is None:
            return self.expr

        if self.isReference:
            if self.expr_type.is_empty:
                return self.expr >> native_ast.nullExpr
            return self.expr.load()
        else:
            return self.expr

    def ensureNonReference(self):
        return self.expr_type.ensureNonReference(self)

    def ensureIsReference(self):
        if self.isReference:
            return self

        return self.context.pushReference(self.expr_type, self.expr)

    def convert_typeof(self):
        return self.expr_type.convert_typeof(self.context, self)

    def convert_incref(self):
        return self.expr_type.convert_incref(self.context, self)

    def convert_set_attribute(self, attribute, expr):
        return self.expr_type.convert_set_attribute(self.context, self, attribute, expr)

    def convert_assign(self, toStore):
        return self.expr_type.convert_assign(self.context, self, toStore)

    def convert_initialize_from_args(self, *args):
        return self.expr_type.convert_initialize_from_args(self.context, self, *args)

    def convert_default_initialize(self, **kwargs):
        return self.expr_type.convert_default_initialize(self.context, self, **kwargs)

    def convert_destroy(self):
        return self.expr_type.convert_destroy(self.context, self)

    def convert_copy_initialize(self, toStore):
        return self.expr_type.convert_copy_initialize(self.context, self, toStore)

    def convert_attribute(self, attribute, **kwargs):
        # we have 'kwargs' because 'class' convert_attribute accepts some keyword args
        return self.expr_type.convert_attribute(self.context, self, attribute, **kwargs)

    def convert_setitem(self, index, value):
        return self.expr_type.convert_setitem(self.context, self, index, value)

    def convert_format(self, format_spec):
        return self.expr_type.convert_format(self.context, self, format_spec)

    def convert_delitem(self, item):
        return self.expr_type.convert_delitem(self.context, self, item)

    def convert_getslice(self, lower, upper, step):
        return self.expr_type.convert_getslice(self.context, self, lower, upper, step)

    def convert_getitem(self, item):
        return self.expr_type.convert_getitem(self.context, self, item)

    def convert_getitem_unsafe(self, item):
        return self.expr_type.convert_getitem_unsafe(self.context, self, item)

    def convert_len(self):
        return self.expr_type.convert_len(self.context, self)

    def convert_hash(self):
        return self.expr_type.convert_hash(self.context, self)

    def convert_abs(self):
        return self.expr_type.convert_abs(self.context, self)

    def has_method(self, methodName):
        return self.expr_type.has_method(methodName)

    def convert_bool_cast(self):
        return self.convert_to_type(bool, ConversionLevel.New)

    def convert_int_cast(self):
        return self.convert_to_type(int, ConversionLevel.New)

    def convert_float_cast(self):
        return self.convert_to_type(float, ConversionLevel.New)

    def convert_str_cast(self):
        return self.convert_to_type(str, ConversionLevel.New)

    def convert_bytes_cast(self):
        return self.convert_to_type(bytes, ConversionLevel.New)

    def convert_builtin(self, f, a1=None):
        return self.expr_type.convert_builtin(f, self.context, self, a1)

    def convert_repr(self):
        return self.expr_type.convert_repr(self.context, self)

    def convert_reserved(self):
        return self.expr_type.convert_reserved(self.context, self)

    def convert_unary_op(self, op):
        return self.expr_type.convert_unary_op(self.context, self, op)

    def convert_bin_op(self, op, rhs, inplace=False):
        return self.expr_type.convert_bin_op(self.context, self, op, rhs, inplace)

    def convert_bin_op_reverse(self, op, rhs, inplace=False):
        return self.expr_type.convert_bin_op_reverse(self.context, self, op, rhs, inplace)

    def convert_call(self, args, kwargs):
        return self.expr_type.convert_call(self.context, self, args, kwargs)

    def convert_method_call(self, methodname, args, kwargs):
        return self.expr_type.convert_method_call(self.context, self, methodname, args, kwargs)

    def convert_to_type(
        self,
        target_type,
        conversionLevel: ConversionLevel,
        assumeSuccessful=False
    ):
        """Convert to a target type as a function argument.

        If 'explicit', then allow conversions that may change type (e.g. int->float). Otherwise
        insist on strict conversion.
        """
        target_type = typeWrapper(target_type)

        return self.expr_type.convert_to_type(
            self.context,
            self,
            target_type,
            conversionLevel,
            assumeSuccessful=assumeSuccessful
        )

    def convert_context_manager_enter(self):
        return self.expr_type.convert_context_manager_enter(self.context, self)

    def convert_context_manager_exit(self, args):
        return self.expr_type.convert_context_manager_exit(self.context, self, args)

    def convert_to_type_with_target(self, targetVal, conversionLevel, mayThrowOnFailure=False):
        """Convert this value to another type whose storage is already allocated.

        Args:
            targetVal - a TypedExpression containing the value to initialize. This must be
                a reference to uninitialized storage for the value.
            conversionLevel - a ConversionLevel indicating how strictly to convert values
                that don't match in type.
            mayThrowOnFailure - if True, the function may choose to push an exception onto the
                stack and return None.

        Returns:
            None, or a TypedExpression(bool) indicating whether conversion succeeded or failed. If True,
            then targetVal must be initialized.  If returning None, then the value must not be
            initialized and control flow may not return. This can only happen if 'mayThrowOnFailure'
            is True
        """
        return self.expr_type.convert_to_type_with_target(
            self.context,
            self,
            targetVal,
            conversionLevel,
            mayThrowOnFailure
        )

    def get_iteration_expressions(self):
        return self.expr_type.get_iteration_expressions(self.context, self)

    def convert_issubclass(self, ofType, isSubclassCall):
        return self.expr_type.convert_issubclass(self.context, self, ofType, isSubclassCall)

    def convert_masquerade_to_untyped(self):
        return self.expr_type.convert_masquerade_to_untyped(self.context, self)

    def convert_masquerade_to_typed(self):
        return self.expr_type.convert_masquerade_to_typed(self.context, self)

    def convert_fastnext(self):
        """Call '__fastnext__' on the object.

        Returns:
            a PointerTo the result, which will be None if iteration is
            stopped.
        """
        return self.expr_type.convert_fastnext(self.context, self)

    def toPyObj(self):
        return self.convert_to_type(object, ConversionLevel.Signature)

    def toFloat64(self):
        return self.convert_to_type(float, ConversionLevel.New)

    def toInt64(self):
        return self.convert_to_type(int, ConversionLevel.New)

    def toUInt64(self):
        return self.convert_to_type(UInt64, ConversionLevel.New)

    def toBool(self):
        return self.convert_to_type(bool, ConversionLevel.New)

    def toIndex(self):
        """Equivalent to __index__"""
        res = self.expr_type.convert_index_cast(self.context, self)

        if not res:
            return None

        if res.expr_type.typeRepresentation is not int:
            raise Exception(f"{self.expr_type}.toIndex() returned {res.expr_type}, not int.")

        return res

    def toFloatMath(self):
        """Equivalent to __float__"""
        res = self.expr_type.convert_math_float_cast(self.context, self)

        if not res:
            return None

        if res.expr_type.typeRepresentation is not float:
            raise Exception(f"{self.expr_type}.toFloatMath() returned {res.expr_type}, not float.")

        return res

    def refAs(self, i):
        return self.expr_type.refAs(self.context, self, i)

    @staticmethod
    def asBool(typedExpressionOrNone):
        if typedExpressionOrNone is not None:
            return typedExpressionOrNone.toBool()
        else:
            return None

    def __str__(self):
        return "TypedExpression(%s%s%s)" % (
            self.expr_type,
            ",[ref]" if self.isReference else "",
            f",constant={self.constantValue}" if self.isConstant else ""
        )

    def __repr__(self):
        return str(self)

    def __rshift__(self, other):
        return TypedExpression(self.context, self.expr >> other.expr, other.expr_type, other.isReference)

    @staticmethod
    def sugar_operator(left, right, opname):
        if isinstance(right, (int, float, bool)):
            right = left.context.constant(right)

        if hasattr(BinaryOp, opname):
            op = getattr(BinaryOp, opname)()
        elif hasattr(ComparisonOp, opname):
            op = getattr(ComparisonOp, opname)()
        elif hasattr(BooleanOp, opname):
            op = getattr(BooleanOp, opname)()
        else:
            assert False, opname

        return left.convert_bin_op(op, right)

    def __add__(self, other):
        return TypedExpression.sugar_operator(self, other, "Add")

    def __sub__(self, other):
        return TypedExpression.sugar_operator(self, other, "Sub")

    def __mul__(self, other):
        return TypedExpression.sugar_operator(self, other, "Mult")

    def __truediv__(self, other):
        return TypedExpression.sugar_operator(self, other, "Div")

    def __and__(self, other):
        return TypedExpression.sugar_operator(self, other, "BitAnd")

    def __or__(self, other):
        return TypedExpression.sugar_operator(self, other, "BitOr")

    def __xor__(self, other):
        return TypedExpression.sugar_operator(self, other, "BitXor")

    def __lt__(self, other):
        return TypedExpression.sugar_operator(self, other, "Lt")

    def __le__(self, other):
        return TypedExpression.sugar_operator(self, other, "LtE")

    def __gt__(self, other):
        return TypedExpression.sugar_operator(self, other, "Gt")

    def __ge__(self, other):
        return TypedExpression.sugar_operator(self, other, "GtE")

    def __eq__(self, other):
        return TypedExpression.sugar_operator(self, other, "Eq")

    def __ne__(self, other):
        return TypedExpression.sugar_operator(self, other, "NotEq")
