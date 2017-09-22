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

import nativepython
import nativepython.native_ast as native_ast
import nativepython.llvm_compiler as llvm_compiler

from nativepython.type_model.type_base import Type
from nativepython.type_model.typed_expression import TypedExpression
from nativepython.exceptions import ConversionException

class Reference(Type):
    def __init__(self, value_type):
        assert isinstance(value_type, Type)
        self.value_type = value_type

    @property
    def is_ref(self):
        return True

    def unwrap_reference(self, completely=False):
        if completely:
            return self.value_type.unwrap_reference(True)
        else:
            return self.value_type

    def lower(self):
        return native_ast.Type.Pointer(self.value_type.lower())

    @property
    def as_call_arg(self):
        return self.value_type

    @property
    def create_reference(self):
        return CreateReference(self.value_type)

    @property
    def is_pod(self):
        return True

    @property
    def null_value(self):
        return native_ast.Constant.NullPointer(self.value_type.lower())

    @property
    def sizeof(self):
        return llvm_compiler.pointer_size

    @property
    def pointer(self):
        return nativepython.type_model.Pointer(self)

    def convert_attribute(self, context, instance, attr):
        raise ConversionException("References cannot be used directly")

    def convert_set_attribute(self, context, instance, attr, val):
        raise ConversionException("References cannot be used directly")

    def convert_bin_op(self, op, l, r):
        raise ConversionException("References cannot be used directly")

    def convert_getitem(self, context, instance, index):
        raise ConversionException("References cannot be used directly")

    def convert_setitem(self, context, instance, index, value):
        raise ConversionException("References cannot be used directly")

    def convert_to_type(self, instance, to_type):
        raise ConversionException("References cannot be used directly")

    def convert_assign(self, context, instance_ref, other):
        raise ConversionException("References cannot be used directly")

    def convert_initialize_copy(self, context, instance_ref, other_instance):
        other_instance = other_instance.drop_create_reference()

        if other_instance.expr_type != self:
            other_instance = other_instance.expr_type.convert_to_type(other_instance, self)

        return TypedExpression.Void(
            native_ast.Expression.Store(
                ptr=instance_ref.expr,
                val=other_instance.expr
                )
            )

    def __repr__(self):
        return "Reference(%s)" % self.value_type

    @property
    def is_valid_as_variable(self):
        return not self.value_type.is_ref

class CreateReference(Reference):
    @property
    def is_create_reference(self):
        return True

    @property
    def is_valid_as_variable(self):
        return False

    @property
    def reference(self):
        raise ConversionException("Can't make a reference out of %s" % self)

    @property
    def as_call_arg(self):
        return self.value_type.reference

    @property
    def create_reference(self):
        raise ConversionException("can't make a CreateReference out of %s" % self)

    def __repr__(self):
        return "CreateReference(%s)" % self.value_type
