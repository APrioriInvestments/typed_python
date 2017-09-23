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

import nativepython.python_to_native_ast as python_to_native_ast
import nativepython.native_ast as native_ast
import nativepython.type_model as type_model

from nativepython.exceptions import ConversionException

Float64 = type_model.Float64
Int64 = type_model.Int64
Int32 = type_model.Int32
Bool = type_model.Bool
Void = type_model.Void
UInt8 = type_model.UInt8
Struct = type_model.Struct

class ExpressionFunction(type_model.CompileTimeType):
    def __init__(self, f):
        self.f = f

    def convert_call(self, context, instance, args):
        return self.f(context, args)

    def __repr__(self):
        return self.f.func_name

    @property
    def python_object_representation(self):
        return self

class TypeFun(type_model.CompileTimeType):
    def __init__(self, f):
        self.f = f

    def convert_call(self, context, instance, args):
        def unwrap(x):
            if not isinstance(x, type_model.TypedExpression):
                raise ConversionException("Expected a TypedExpression, not %s" % x)
            t = x.expr_type
            if isinstance(t.nonref_type, type_model.FreePythonObjectReference):
                return t.nonref_type.python_object_representation
            else:
                return t

        def wrap(x):
            return type_model.pythonObjectRepresentation(x)

        unwrapped = [unwrap(x) for x in args]

        res = self.f(*unwrapped)
        
        return wrap(res)

    def __call__(self, *args):
        return self.f(*args)

    def __repr__(self):
        return self.f.func_name

    @property
    def python_object_representation(self):
        return self

def typefun(f):
    return TypeFun(f)

def exprfun(f):
    return ExpressionFunction(f)

@typefun
def assert_types_same(t1, t2):
    if t1 != t2:
        raise ConversionException("Types are not the same: %s != %s" % (t1, t2))

@exprfun
def addr(context, args):
    if len(args) != 1:
        raise ConversionException("addr takes 1 argument")

    return args[0].address_of

@exprfun
def ref(context, args):
    if len(args) != 1:
        raise ConversionException("ref takes 1 argument")
    return args[0].as_creates_reference

@exprfun
def nonref(context, args):
    if len(args) != 1:
        raise ConversionException("ref takes 1 argument")

    if args[0].expr_type.is_create_reference:
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

class ExternalFunction(type_model.CompileTimeType):
    def __init__(self, name, output_type, input_types, implicit_type_casting,varargs):
        self.name = name
        self.output_type = output_type
        self.input_types = input_types
        self.implicit_type_casting = implicit_type_casting
        self.varargs = varargs

    def convert_call(self, context, instance, args):
        if self.varargs:
            assert len(args) >= len(self.input_types)
        else:
            assert len(args) == len(self.input_types)

        args = [a.dereference() for a in args]

        if not self.implicit_type_casting:
            for i in xrange(len(input_types)):
                assert args[i].expr_type == self.input_types[i]
        else:
            args = list(args)
            for i in xrange(len(self.input_types)):
                if args[i].expr_type != self.input_types[i]:
                    args[i] = args[i].convert_to_type(self.input_types[i])

        return type_model.TypedExpression(
            native_ast.Expression.Call(
                target=native_ast.CallTarget(
                    name = self.name,
                    arg_types = [i.lower() for i in self.input_types],
                    output_type = self.output_type.lower(),
                    external=True,
                    varargs=self.varargs
                    ),
                args=[a.expr for a in args]
                ),
            self.output_type
            )

    def __repr__(self):
        return self.name

    @classmethod
    def make(cls, name, output_type, input_types, implicit_type_casting=True,varargs=False):
        return ExternalFunction(name, output_type, input_types, implicit_type_casting,varargs)

malloc = ExternalFunction.make("malloc", UInt8.pointer, [Int64])
realloc = ExternalFunction.make("realloc", UInt8.pointer, [Int64])
free = ExternalFunction.make("free", Void, [UInt8.pointer])
printf = ExternalFunction.make("printf", Int64, [UInt8.pointer], varargs=True)

@typefun
def is_struct(t):
    return isinstance(t, type_model.Struct)

@typefun
def struct_size(t):
    if isinstance(t, type_model.Struct):
        return len(t.element_types)
    else:
        return None

@type_model.representation_for(len)
def len_override(x):
    if is_struct(typeof(x)):
        return struct_size(typeof(x))
    else:
        return x.__len__()

@type_model.representation_for(xrange)
class xrange_override:
    def __init__(self, top):
        self.top = top

    def __iter__(self):
        return xrange_iterator(0, self.top)

class xrange_iterator:
    def __init__(self, cur_value, top):
        self.cur_value = cur_value
        self.top = top

    def has_next(self):
        return self.cur_value < self.top

    def next(self):
        val = self.cur_value
        self.cur_value += 1
        return val
