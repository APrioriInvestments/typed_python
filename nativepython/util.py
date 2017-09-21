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


Float64 = python_to_native_ast.Float64
Int64 = python_to_native_ast.Int64
Int32 = python_to_native_ast.Int32
Bool = python_to_native_ast.Bool
Void = python_to_native_ast.Void
UInt8 = python_to_native_ast.UInt8
Struct = python_to_native_ast.Struct

class ExpressionFunction(python_to_native_ast.RepresentationlessType):
    def __init__(self, f):
        self.f = f

    def convert_call(self, context, instance, args):
        return self.f(context, args)

    def __repr__(self):
        return self.f.func_name

    @property
    def python_object_representation(self):
        return self

class TypeFun(python_to_native_ast.RepresentationlessType):
    def __init__(self, f):
        self.f = f

    def convert_call(self, context, instance, args):
        def unwrap(x):
            assert isinstance(x, python_to_native_ast.TypedExpression)
            t = x.expr_type
            if isinstance(t, python_to_native_ast.FreePythonObjectReference):
                return t._obj
            else:
                return t

        def wrap(x):
            return python_to_native_ast.pythonObjectRepresentation(x)

        return wrap(self.f(*[unwrap(x) for x in args]))

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

@exprfun
def addr(context, args):
    assert len(args) == 1

    return args[0].address

@exprfun
def ref(context, args):
    assert len(args) == 1
    return args[0].reference

@exprfun
def deref(context, args):
    assert len(args) == 1
    return args[0].dereference

@exprfun
def typeof(context, args):
    assert len(args) == 1

    return python_to_native_ast.pythonObjectRepresentation(args[0].expr_type)

@exprfun
def typestring(context, args):
    assert len(args) == 1

    return python_to_native_ast.pythonObjectRepresentation(str(args[0].expr_type))

@exprfun
def in_place_new(context, args):
    assert len(args) == 2

    assert isinstance(args[0].expr_type, python_to_native_ast.Pointer)

    object_type = args[0].expr_type.value_type

    return object_type.convert_initialize_copy(context, args[0], args[1])

@exprfun
def in_place_destroy(context, args):
    assert len(args) == 1

    assert isinstance(args[0].expr_type, python_to_native_ast.Pointer)

    object_type = args[0].expr_type.value_type

    return object_type.convert_destroy(context, args[0])

def attribute_getter(attr):
    @exprfun
    def getter(context, args):
        assert len(args) == 1
        return args[0].expr_type.convert_attribute(args[0], attr)

    return getter

class ExternalFunction(python_to_native_ast.RepresentationlessType):
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

        if not self.implicit_type_casting:
            for i in xrange(len(input_types)):
                assert args[i].expr_type == self.input_types[i]
        else:
            args = list(args)
            for i in xrange(len(self.input_types)):
                if args[i].expr_type != self.input_types[i]:
                    args[i] = args[i].convert_to_type(self.input_types[i])

        return python_to_native_ast.TypedExpression(
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
    return isinstance(t, python_to_native_ast.Struct)

@typefun
def struct_size(t):
    if isinstance(t, python_to_native_ast.Struct):
        return len(t.element_types)
    else:
        return None

@python_to_native_ast.representation_for(len)
def len_override(x):
    if is_struct(typeof(x)):
        return struct_size(typeof(x))
    else:
        return x.__len__()

@python_to_native_ast.representation_for(xrange)
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
