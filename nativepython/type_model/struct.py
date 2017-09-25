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

from nativepython.type_model.type_base import Type
from nativepython.type_model.class_type import ClassType
from nativepython.type_model.typed_expression import TypedExpression
from nativepython.exceptions import ConversionException

import nativepython.native_ast as native_ast

class Struct_:
    pass

class Struct(ClassType):
    def __init__(self, element_types=()):
        ClassType.__init__(self, Struct_, element_types)

    def __str__(self):
        return "Struct(%s)" % (",".join(["%s=%s" % t for t in self.element_types]))

    @property
    def is_class(self):
        return False

    @property
    def is_struct(self):
        return True

    def with_field(self, name, type):
        if isinstance(name, TypedExpression):
            assert name.expr.matches.Constant and name.expr.val.matches.ByteArray, name.expr
            name = name.expr.val.val

        return Struct(element_types=self.element_types + ((name,type),))

    def convert_getitem(self, context, instance, index):
        if not (index.expr.matches.Constant and index.expr.val.matches.Int):
            raise ConversionException("can't index %s with %s" % (self, index.expr))

        i = index.expr.val.val
        
        if not (i >= 0 and i < len(self.element_types)):
            raise ConversionException("can't index %s with %s" % (self, i))

        return self.convert_attribute(context, instance, self.element_types[i][0])
