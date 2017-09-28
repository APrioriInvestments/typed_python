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
from nativepython.exceptions import ConversionException

import nativepython
import nativepython.python.string_util as string_util
import nativepython.native_ast as native_ast

class FunctionPointer(Type):
    def __init__(self, output_type, input_types, varargs=False, can_throw=True):
        self.output_type = output_type
        self.input_types = tuple(input_types)
        self.varargs = varargs
        self.can_throw = can_throw

        if self.varargs:
            for a in input_types:
                if not a.is_pod:
                    raise ConversionException("Varargs functions can only accept POD types")

            if not output_type.is_pod:
                raise ConversionException("Varargs functions can only return POD types")

        if self.output_type.is_pod:
            self.native_type = native_ast.Type.Pointer(
                native_ast.Type.Function(
                    output=self.output_type.lower(),
                    args=[i.lower() for i in self.input_types],
                    varargs=self.varargs,
                    can_throw=self.can_throw
                    )
                )
        else:
            self.native_type = native_ast.Type.Pointer(
                native_ast.Type.Function(
                    output=native_ast.Void,
                    args=[self.output_type.pointer.lower()] + [i.lower() for i in self.input_types],
                    varargs=self.varargs,
                    can_throw=self.can_throw
                    )
                )

    @property
    def is_pod(self):
        return True

    @property
    def is_function_pointer(self):
        return True

    @property
    def null_value(self):
        return native_ast.Constant.NullPointer(self.lower())

    def lower(self):
        return self.native_type

    def __str__(self):
        return "FuncPtr((%s)->%s%s)" % (
            ",".join([str(x) for x in (c.input__types + ["..."] if self.varargs else [])]), 
            str(self.output_type),
            ",nothrow" if not self.can_throw else ""
            )

    def convert_call(self, context, instance, args):
        instance = instance.dereference()

        if (not self.varargs and len(args) != len(self.input_types) 
                or self.varargs and len(args) < len(self.input_types)):
            raise ConversionException("Can't call %s with %s arguments" % (self, len(args)))

        return context.call_typed_function(
            native_ast.CallTarget.Pointer(instance.expr),
            self.output_type,
            self.input_types,
            self.varargs,
            args
            )
