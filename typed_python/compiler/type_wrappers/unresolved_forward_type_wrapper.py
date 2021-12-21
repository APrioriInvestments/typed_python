from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.conversion_level import ConversionLevel


class UnresolvedForwardTypeWrapper(Wrapper):
    def __init__(self, t):
        super().__init__(t)

    def getNativeLayoutType(self):
        return native_ast.Void

    def throwUnresolvedForwardException(self, context):
        return context.pushException(
            TypeError,
            f"Type {self.typeRepresentation.__name__} has unresolved forwards"
        )

    def convert_incref(self, context, expr):
        return self.throwUnresolvedForwardException(context)

    def convert_fastnext(self, context, expr):
        return self.throwUnresolvedForwardException(context)

    def convert_attribute_pointerTo(self, context, pointerInstance, attribute):
        return self.throwUnresolvedForwardException(context)

    def convert_attribute(self, context, instance, attribute):
        return self.throwUnresolvedForwardException(context)

    def convert_set_attribute(self, context, instance, attribute, value):
        return self.throwUnresolvedForwardException(context)

    def convert_delitem(self, context, instance, item):
        return self.throwUnresolvedForwardException(context)

    def convert_getitem(self, context, instance, item):
        return self.throwUnresolvedForwardException(context)

    def convert_getslice(self, context, instance, start, stop, step):
        return self.throwUnresolvedForwardException(context)

    def convert_setitem(self, context, instance, index, value):
        return self.throwUnresolvedForwardException(context)

    def convert_assign(self, context, target, toStore):
        return self.throwUnresolvedForwardException(context)

    def convert_copy_initialize(self, context, target, toStore):
        return self.throwUnresolvedForwardException(context)

    def convert_destroy(self, context, instance):
        return self.throwUnresolvedForwardException(context)

    def convert_default_initialize(self, context, target):
        return self.throwUnresolvedForwardException(context)

    def convert_typeof(self, context, instance):
        return self.throwUnresolvedForwardException(context)

    def convert_issubclass(self, context, instance, ofType, isSubclassCall):
        return self.throwUnresolvedForwardException(context)

    def convert_masquerade_to_untyped(self, context, instance):
        return self.throwUnresolvedForwardException(context)

    def convert_call(self, context, left, args, kwargs):
        return self.throwUnresolvedForwardException(context)

    def convert_len(self, context, expr):
        return self.throwUnresolvedForwardException(context)

    def convert_hash(self, context, expr):
        return self.throwUnresolvedForwardException(context)

    def convert_abs(self, context, expr):
        return self.throwUnresolvedForwardException(context)

    def convert_index_cast(self, context, expr):
        return self.throwUnresolvedForwardException(context)

    def convert_math_float_cast(self, context, expr):
        return self.throwUnresolvedForwardException(context)

    def convert_builtin(self, f, context, expr, a1=None):
        return self.throwUnresolvedForwardException(context)

    def convert_repr(self, context, expr):
        return self.throwUnresolvedForwardException(context)

    def convert_unary_op(self, context, expr, op):
        return self.throwUnresolvedForwardException(context)

    def convert_to_type(self, context, expr, target_type, level: ConversionLevel, assumeSuccessful=False):
        return self.throwUnresolvedForwardException(context)

    def convert_to_type_with_target(self, context, expr, targetVal, level: ConversionLevel, mayThrowOnFailure=False):
        return self.throwUnresolvedForwardException(context)

    def convert_to_self_with_target(self, context, targetVal, sourceVal, level: ConversionLevel, mayThrowOnFailure=False):
        return self.throwUnresolvedForwardException(context)

    def convert_bin_op(self, context, l, op, r, inplace):
        return self.throwUnresolvedForwardException(context)

    def convert_bin_op_reverse(self, context, r, op, l, inplace):
        return self.throwUnresolvedForwardException(context)

    def convert_format(self, context, instance, formatSpecOrNone=None):
        return self.throwUnresolvedForwardException(context)

    def convert_type_attribute(self, context, typeInst, attribute):
        return self.throwUnresolvedForwardException(context)

    def convert_type_call(self, context, typeInst, args, kwargs):
        return self.throwUnresolvedForwardException(context)

    def convert_call_on_container_expression(self, context, inst, argExpr):
        return self.throwUnresolvedForwardException(context)

    def convert_type_call_on_container_expression(self, context, typeInst, argExpr):
        return self.throwUnresolvedForwardException(context)

    def convert_type_method_call(self, context, typeInst, methodname, args, kwargs):
        return self.throwUnresolvedForwardException(context)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        return self.throwUnresolvedForwardException(context)

    def get_iteration_expressions(self, context, expr):
        return self.throwUnresolvedForwardException(context)
