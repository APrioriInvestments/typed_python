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
import nativepython.type_model as type_model
import nativepython.native_ast as native_ast
import nativepython.util as util
import nativepython.llvm_compiler as llvm_compiler

@util.typefun
def is_simple_type(t):
    t = t.nonref_type

    return (
        t == type_model.Float64 or
        t == type_model.Int64 or 
        t == type_model.Bool or 
        t == type_model.Void or
        isinstance(t, type_model.CompileTimeType)
        )

class ReferenceHeldAsPointer:
    def __init__(self, p):
        self.p = p

@util.typefun
def isRefHeldAsPointer(t):
    return t.nonref_type.is_class and t.nonref_type.cls is ReferenceHeldAsPointer

def dereferenceRefHeldAsPointer(x):
    if isRefHeldAsPointer(x):
        return util.ref(x.p[0])
    return util.ref(x)

def wrapping_function_call(f):
    def new_f(*in_args):
        args = util.map_struct(in_args, dereferenceRefHeldAsPointer)

        output_type = util.typeof(f(*args)).nonref_type
        output_type_raw = util.typeof(f(*args))

        if is_simple_type(output_type):
            return f(*args)

        if output_type_raw.is_ref:
            if output_type_raw.is_ref_to_temp:
                #this function creates a new object and returns it to us
                raw_ptr = output_type.pointer(util.malloc(output_type.sizeof))

                util.in_place_new(raw_ptr, f(*args))

                return raw_ptr
            else:
                #this function is returning an internal reference
                ref_as_ptr = ReferenceHeldAsPointer(util.addr(f(*args)))

                reftype = util.typeof(ref_as_ptr).nonref_type

                raw_ptr = reftype.pointer(util.malloc(reftype.sizeof))

                util.in_place_new(raw_ptr, ref_as_ptr)

                return raw_ptr
        else:
            #this function creates a new object and returns it to us by value
            raw_ptr = output_type.pointer(util.malloc(output_type.sizeof))

            util.in_place_new(raw_ptr, f(*args))

            return raw_ptr

    new_f.func_name = "<wrapped f=%s>" % str(f)
    return new_f


class WrappedFunction(object):
    def __init__(self, runtime, f, wants_wrapping=True):
        self.runtime = runtime
        self.f = f
        self.wrapping_func = wrapping_function_call(self.f) if wants_wrapping else f

    def __call__(self, *args):
        def type_for_arg(a):
            if isinstance(a, float):
                return type_model.Float64
            if isinstance(a, bool):
                return type_model.Bool
            if isinstance(a, int):
                return type_model.Int64
            if isinstance(a, WrappedObject):
                return a._object_type.reference

            return type_model.FreePythonObjectReference(a)

        def arg_for_arg(a):
            if isinstance(a, (float,int, bool)):
                return a
            if isinstance(a, WrappedObject):
                return a._object_ptr
            return None

        f_target = self.runtime.convert(self.wrapping_func, [type_for_arg(a) for a in args])

        f_callable = self.runtime.native_ptr_for_typed_target(f_target)

        output_type = f_target.output_type

        if is_simple_type(output_type):
            simple_result = f_callable(*[arg_for_arg(a) for a in args])
            if isinstance(output_type, python_to_native_ast.CompileTimeType):
                return output_type.python_object_representation
            return simple_result

        res = f_callable(*[arg_for_arg(a) for a in args])

        if res is None:
            return res

        return WrappedObject(self.runtime, res, output_type.value_type)

class WrappedObject(object):
    def __init__(self, runtime, object_ptr, object_type):
        self._runtime = runtime
        self._object_ptr = object_ptr
        self._object_type = object_type

    def __getattr__(self, attr):
        return self._runtime._pointer_attribute_func(attr)(self)

    def __len__(self):
        return self._runtime._len_fun(self)

    def __call__(self, *args):
        return self._runtime._call_func(self, *args)

    def __getitem__(self, a):
        return self._runtime._getitem_func(self, a)

    def __setitem__(self, a, val):
        return self._runtime._setitem_func(self, a, val)

    def __repr__(self):
        return "WrappedObject(t=%s,p=%s)" % (self._object_type, self._object_ptr)

    def __del__(self):
        self._runtime._free_func(self)


_singleton = [None]

class Runtime:
    @staticmethod
    def singleton():
        if _singleton[0] is None:
            _singleton[0] = Runtime()
        return _singleton[0]

    def __init__(self):
        self.compiler = llvm_compiler.Compiler()
        self.converter = python_to_native_ast.Converter()
        self.functions_by_name = {}

        def call_func(f, *args):
            return util.ref_if_ref(f(*args))

        def getitem_func(f, a):
            return util.ref_if_ref(f[a])

        def setitem_func(f, a, b):
            f[a] = b

        def len_func(f):
            return len(f)

        def free_func(f):
            f_ptr = util.addr(f)
            util.in_place_destroy(f_ptr)
            util.free(f_ptr)

        self._call_func = WrappedFunction(self, call_func)
        self._getitem_func = WrappedFunction(self, getitem_func)
        self._setitem_func = WrappedFunction(self, setitem_func)
        self._len_fun = WrappedFunction(self, len_func)
        self._free_func = WrappedFunction(self, free_func, wants_wrapping=False)

        self._pointer_attribute_funcs = {}

    def _pointer_attribute_func(self, attr):
        if attr not in self._pointer_attribute_funcs:
            getter = util.attribute_getter(attr)
            self._pointer_attribute_funcs[attr] = \
                WrappedFunction(self, lambda p: util.ref_if_ref(getter(p)))

        return self._pointer_attribute_funcs[attr]

    def wrap(self, f):
        return WrappedFunction(self, f)

    def convert(self, f, argtypes):
        f_target = self.converter.convert(f, argtypes)

        functions = self.converter.extract_new_function_definitions()

        for k,v in self.compiler.add_functions(functions).iteritems():
            self.functions_by_name[k] = v

        return f_target

    def native_ptr_for_typed_target(self, target):
        return self.functions_by_name[target.name]


