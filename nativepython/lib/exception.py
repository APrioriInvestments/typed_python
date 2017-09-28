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
import nativepython.util as util
import nativepython.type_model as type_model

from nativepython.exceptions import ConversionException

Int8 = type_model.Int8
Int64 = type_model.Int64
Void = type_model.Void

VoidPtrFunc = type_model.FunctionPointer(
    input_types=[Int8.pointer],
    output_type=Void
    )

@type_model.cls
class InFlightException:
    def __types__(cls):
        cls.data_ptr = Int8.pointer
        cls.destructor = VoidPtrFunc
        cls.typeid = Int64

    def __init__(self, data_ptr, destructor, typeid):
        self.data_ptr.__init__(data_ptr)
        self.destructor.__init__(destructor)
        self.typeid.__init__(typeid)

    def teardown(self):
        #note - not handling exceptions here!
        self.destructor(self.data_ptr)
        util.free(self.data_ptr)
        util.free(util.addr(self))

def createException(e):
    T = util.typeof(e).nonref_type

    data_ptr = Int8.pointer(util.malloc(T.sizeof))

    util.in_place_new(T.pointer(data_ptr), e)

    exception_ptr = InFlightException.pointer(util.malloc(InFlightException.sizeof))

    util.in_place_new(exception_ptr, 
        InFlightException(
            data_ptr, 
            VoidPtrFunc(util.addr(util.destructor(T))),
            util.typeid(T)
            )
        )

    return exception_ptr

def exception_matches(T, exception_ptr):
    return exception_ptr[0].typeid == util.typeid(T)

def bind_exception_into(exception_ptr, target):
    T = util.typeof(target).nonref_type

    source_ptr = T.pointer(exception_ptr[0].data_ptr)

    util.in_place_new(util.addr(target), source_ptr[0])

    exception_ptr[0].teardown()

def exception_teardown(exception_ptr):
    exception_ptr[0].teardown()

@util.exprfun
def throw_pointer(context, args):
    if len(args) != 1:
        raise ConversionException("throw takes one argument")
    arg = args[0].dereference()

    if not arg.expr_type.is_pointer:
        raise ConversionException("throw can only take a pointer")

    return type_model.TypedExpression(native_ast.Expression.Throw(arg.expr), None)

def throw(value):
    throw_pointer(createException(value))
