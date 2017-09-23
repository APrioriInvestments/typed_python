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

import types
import nativepython
import nativepython.native_ast as native_ast
import nativepython.llvm_compiler as llvm_compiler

from nativepython.type_model.typed_expression import TypedExpression
from nativepython.exceptions import ConversionException

class Type(object):
    def __init__(self):
        object.__init__(self)

    def unwrap_reference(self, completely=False):
        return self

    @property
    def nonref_type(self):
        return self.unwrap_reference(completely=True)

    @property
    def is_ref(self):
        return False

    @property
    def is_create_reference(self):
        return False

    @property
    def is_pointer(self):
        return False

    @property
    def is_primitive(self):
        return False

    @property
    def is_primitive_numeric(self):
        return False

    @property
    def is_pod(self):
        raise ConversionException("can't directly references instances of %s" % self)

    def assert_is_instance_ref(self, instance_ref):
        if not instance_ref.expr_type.is_ref or instance_ref.expr_type.value_type != self:
            raise ConversionException(
                "Expected argument to be a reference to %s, not %s"
                 % (self, instance_ref.expr_type)
                )
        
    @property
    def null_value(self):
        raise ConversionException("can't construct a null value of type %s" % self)

    def lower_as_function_arg(self):
        if self.is_pod:
            return self.lower()
        return native_ast.Type.Pointer(self.lower())

    def lower(self):
        raise ConversionException("Can't directly reference instances of %s" % self)

    def convert_initialize_copy(self, context, instance_ref, other_instance):
        if not self.is_pod:
            raise ConversionException("can't initialize %s - need a real implementation" % self)

        other_instance = other_instance.dereference()

        if other_instance.expr_type != self:
            other_instance = other_instance.expr_type.convert_to_type(other_instance, self)

        return TypedExpression.Void(
            native_ast.Expression.Store(
                ptr=instance_ref.expr,
                val=other_instance.expr
                )
            )

    def convert_destroy(self, context, instance_ref):
        if not self.is_pod:
            raise ConversionException("can't destroy %s - need a real implementation" % self)

        return TypedExpression.Void(
            native_ast.nullExpr
            )

    def convert_initialize(self, context, instance_ref, args):
        self.assert_is_instance_ref(instance_ref)

        assert len(args) <= 1

        if len(args) == 1:
            return self.convert_initialize_copy(context, instance_ref, args[0])
        else:
            if not self.is_pod:
                raise ConversionException("can't initialize %s - need a real implementation" % self)
            return TypedExpression.Void(
                native_ast.Expression.Store(
                    ptr=instance_ref.expr,
                    val=native_ast.Expression.Constant(self.null_value)
                    )
                )

    def convert_assign(self, context, instance_ref, arg):
        if not self.is_pod:
            raise ConversionException("instances of %s need an explicit assignment operator" % self)
        
        self.assert_is_instance_ref(instance_ref)

        arg = arg.dereference()

        if arg.expr_type != self:
            raise ConversionException("can't assign %s to %s" % (arg.expr_type, self))

        return TypedExpression.Void(
            native_ast.Expression.Store(
                ptr=instance_ref.expr,
                val=arg.expr
                )
            )

    def convert_unary_op(self, instance, op):
        raise ConversionException("can't handle unary op %s on %s" % (op, self))

    def convert_bin_op(self, op, l, r):
        raise ConversionException("can't handle binary op %s between %s and %s" % (op, l.expr_type, r.expr_type))

    def convert_to_type(self, instance, to_type):
        raise ConversionException("can't convert %s to type %s" % (self, to_type))

    def convert_attribute(self, context, instance, attr):
        raise ConversionException("%s has no attribute %s" % (self, attr))

    def convert_set_attribute(self, context, instance, attr, value):
        raise ConversionException("%s has no attribute %s" % (self, attr))

    def convert_getitem(self, context, instance, index):
        raise ConversionException("%s doesn't support getting items" % self)

    def convert_setitem(self, context, instance, index, value):
        raise ConversionException("%s doesn't support setting items" % self)

    @property
    def sizeof(self):
        return llvm_compiler.sizeof_native_type(self.lower())

    @property
    def pointer(self):
        return nativepython.type_model.Pointer(self)

    @property
    def as_call_arg(self):
        return self

    def is_valid_as_variable(self):
        return True

    @property
    def reference(self):
        return nativepython.type_model.Reference(self)

    @property
    def create_reference(self):
        return nativepython.type_model.CreateReference(self)

    def __cmp__(self, other):
        if not isinstance(other, type(self)):
            return cmp(type(self), type(other))
        for k in sorted(self.__dict__):
            c = cmp(getattr(self,k), getattr(other,k))
            if c:
                return c
        return 0

    def __hash__(self):
        try:
            return hash(tuple(sorted(self.__dict__.iteritems())))
        except:
            print "failed on ", self, " with ", self.__dict__
            raise
