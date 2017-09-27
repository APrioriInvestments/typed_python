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

from nativepython.type_model.type_base import Type
from nativepython.type_model.typed_expression import TypedExpression
from nativepython.exceptions import ConversionException, UnassignableFieldException

import nativepython
import nativepython.python.string_util as string_util
import nativepython.native_ast as native_ast

class ElementTypesUnresolved:
    pass
class ElementTypesBeingResolved:
    pass

class FunctionType(Type):
    def __init__(self, f, output_type, input_types, named_call_target):
        Type.__init__(self)

        self.f = f
        self.output_type = output_type
        self.input_types = tuple(input_types)
        self.named_call_target = named_call_target

    @property
    def is_pod(self):
        return True

    @property
    def is_function(self):
        return True

    @property
    def null_value(self):
        return native_ast.Constant.Struct([])

    def lower(self):
        return native_ast.Type.Struct([])

    def __str__(self):
        return "Function(f=%s,(%s)->%s)" % (
            self.f.__name__,
            ",".join([str(x) for x in self.input_types]), 
            str(self.output_type)
            )

    def convert_take_address_override(self, instance_ref, context):
        return TypedExpression(
            native_ast.Expression.FunctionPointer(self.named_call_target),
            nativepython.type_model.FunctionPointer(
                output_type = self.output_type,
                input_types = self.input_types,
                varargs = False,
                can_throw = True
                )
            )

    def convert_call(self, context, instance, args):
        if len(args) != len(self.input_types):
            raise ConversionException("Can't call %s with %s arguments" % (self, len(args)))

        return context.call_typed_function(
            native_ast.CallTarget.Named(self.named_call_target),
            self.output_type,
            self.input_types,
            self.varargs,
            args
            )
