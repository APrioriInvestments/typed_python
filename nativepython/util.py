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

import nativepython.native_ast as native_ast
import nativepython.type_model as type_model

from nativepython.exceptions import ConversionException

TypedExpression = type_model.TypedExpression
Float64 = type_model.Float64
Int64 = type_model.Int64
Int32 = type_model.Int32
Bool = type_model.Bool
Void = type_model.Void
UInt8 = type_model.UInt8
Int8 = type_model.Int8
Struct = type_model.Struct

def typefun(f):
    return type_model.TypeFunction(f)

def exprfun(f):
    return type_model.ExpressionFunction(f)

@typefun
def assert_types_same(t1, t2):
    if t1 != t2:
        raise ConversionException("Types are not the same: %s != %s" % (t1, t2))

@exprfun
def addr(context, args):
    if len(args) != 1:
        raise ConversionException("addr takes 1 argument")

    return args[0].convert_take_address(context)

@exprfun
def ref(context, args):
    if len(args) != 1:
        raise ConversionException("ref takes 1 argument")
    return args[0].as_creates_reference

@exprfun
def ref_if_ref(context, args):
    if len(args) != 1:
        raise ConversionException("ref_if_ref takes 1 argument")
    if args[0].expr_type.is_ref and not args[0].expr_type.is_ref_to_temp:
        return args[0].as_creates_reference
    return args[0]


@exprfun
def nonref(context, args):
    if len(args) != 1:
        raise ConversionException("ref takes 1 argument")

    if args[0].expr_type.is_create_ref:
        return TypedExpression(
            args[0].expr,
            args[0].expr_type.value_type.reference
            )
    return args[0]

@typefun
def deref(t):
    return t.nonref_type

@exprfun
def typeof(context, args):
    if len(args) != 1:
        raise ConversionException("typeof takes 1 argument")
    return type_model.pythonObjectRepresentation(args[0].expr_type)

@exprfun
def typestring(context, args):
    if len(args) != 1:
        raise ConversionException("typestring takes 1 argument")

    return type_model.pythonObjectRepresentation(str(args[0].expr_type))

@exprfun
def in_place_new(context, args):
    if len(args) != 2:
        raise ConversionException("in_place_new takes 2 arguments")

    ptr = args[0].dereference()

    if not ptr.expr_type.is_pointer:
        raise ConversionException("in_place_new needs a pointer for its first argument")

    object_type = ptr.expr_type.value_type

    return object_type.convert_initialize_copy(context, ptr.reference_from_pointer(), args[1])

@exprfun
def in_place_destroy(context, args):
    if len(args) != 1:
        raise ConversionException("in_place_destroy takes 1 arguments")

    ptr = args[0].dereference()

    if not ptr.expr_type.is_pointer:
        raise ConversionException("in_place_destroy needs a pointer for its first argument")

    object_type = ptr.expr_type.value_type

    return object_type.convert_destroy(context, ptr.reference_from_pointer())

def attribute_getter(attr):
    @exprfun
    def getter(context, args):
        assert len(args) == 1
        return args[0].convert_attribute(context, attr)

    return getter

malloc = type_model.ExternalFunction.make("malloc", UInt8.pointer, [Int64])
realloc = type_model.ExternalFunction.make("realloc", UInt8.pointer, [Int64])
free = type_model.ExternalFunction.make("free", Void, [UInt8.pointer])
printf = type_model.ExternalFunction.make("printf", Int64, [UInt8.pointer], varargs=True)

@exprfun
def map_struct(context, args):
    if len(args) != 2:
        raise ConversionException("map_struct takes two arguments")
    if not args[0].expr_type.nonref_type.is_struct:
        raise ConversionException("first argument of map_struct must be of type Struct")

    struct_type = args[0].expr_type.nonref_type

    exprs = []

    for fieldname,_ in struct_type.element_types:
        exprs.append(
            args[1].convert_call(context, [args[0].convert_attribute(context, fieldname)])
                .as_call_arg(context)
            )

    new_struct_type = Struct(
        [(struct_type.element_types[i][0], exprs[i].expr_type)
            for i in xrange(len(exprs))]
        )

    tmp_ref = context.allocate_temporary(new_struct_type)

    return type_model.TypedExpression(
        new_struct_type.convert_initialize(context, tmp_ref, exprs).expr + 
            context.activates_temporary(tmp_ref) + 
            tmp_ref.expr,
        tmp_ref.expr_type
        )

@typefun
def struct_size(t):
    if t.nonref_type.is_struct:
        return len(t.nonref_type.element_types)
    else:
        return None

@type_model.representation_for(len)
def len_override(x):
    if typeof(x).is_struct:
        return struct_size(typeof(x))
    else:
        return x.__len__()

@type_model.representation_for(xrange)
@type_model.cls
class xrange_override:
    def __types__(cls):
        cls.start = int
        cls.stop = int
        cls.step = int

    def __init__(self, *args):
        if struct_size(typeof(args)) is 0:
            self.start = 0
            self.stop = 0
            self.step = 1
        if struct_size(typeof(args)) is 1:
            self.start = 0
            self.stop = args[0]
            self.step = 1
        else:
            self.start = args[0]
            self.stop = args[1]

            if struct_size(typeof(args)) is 3:
                self.step = args[2]
            else:
                self.step = 1

    def __iter__(self):
        return xrange_iterator(self.start, self.stop, self.step)

@type_model.cls
class xrange_iterator:
    def __types__(cls):
        cls.cur_value = int
        cls.stop = int
        cls.step = int

    def __init__(self, cur_value, stop, step):
        self.cur_value = cur_value
        self.stop = stop
        self.step = step

    def has_next(self):
        if self.step > 0:
            return self.cur_value < self.stop
        else:
            return self.cur_value > self.stop

    def next(self):
        val = self.cur_value
        self.cur_value += self.step
        return val

@exprfun
def throw_pointer(context, args):
    if len(args) != 1:
        raise ConversionException("throw takes one argument")
    arg = args[0].dereference()

    if not arg.expr_type.is_pointer:
        raise ConversionException("throw can only take a pointer")

    return TypedExpression(native_ast.Expression.Throw(arg.expr), None)

def throw(value):
    throw_pointer(value)

@exprfun
def typed_function(context, args):
    actual_types = []

    if len(args) < 1:
        raise ConversionException("Expected an argument")

    for a in args:
        t = a.expr_type
        if not t.nonref_type.is_compile_time:
            raise ConversionException("'typed_function' only takes types as arguments")
        actual_types.append(t.nonref_type.python_object_representation)

    if not isinstance(actual_types[0], types.FunctionType):
        raise ConversionException("First argument should be a function")

    def map_internal(t):
        if t is int:
            return type_model.Int64
        if t is float:
            return type_model.Float64
        if t is bool:
            return type_model.Bool
        if t is None:
            return type_model.Void

        return t

    actual_types = [map_internal(a) for a in actual_types]

    target = context.converter.convert(actual_types[0], actual_types[1:])

    func_type = type_model.FunctionType(
            actual_types[0],
            target.output_type,
            actual_types[1:],
            target.named_call_target
            )

    return TypedExpression(
        native_ast.Expression.Constant(func_type.null_value),
        func_type
        )


        
