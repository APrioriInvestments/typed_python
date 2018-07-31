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
import nativepython

Int8 = type_model.Int8

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

@util.typefun
def ResultOrException(T):
    class ResultOrException(type_model.cls):
        def __types__(cls):
            cls.types.p = T.pointer
            cls.types.e = nativepython.lib.exception.InFlightException.pointer

        def __init__(self, p, e):
            self.p = T.pointer(p)
            self.e = nativepython.lib.exception.InFlightException.pointer(e)

    return ResultOrException

def roe_teardown(roe, T):
    roe_p = ResultOrException(T).pointer(roe)

    if roe_p.p:
        util.in_place_destroy(roe_p.p)
        util.free(roe_p.p)
    else:
        nativepython.lib.exception.exception_teardown(roe_p.e)
    util.free(roe)

def roe_extract_ptr(roe):
    roe_p = ResultOrException(Int8).pointer(roe)

    p = roe_p.p
    util.free(roe_p)
    return p

def roe_get_exception(roe):
    roe_p = ResultOrException(Int8).pointer(roe)

    e = roe_p.e
    util.free(roe_p)
    return e

def roe_is_exception(roe):
    roe_p = ResultOrException(Int8).pointer(roe)
    return int(roe_p.e) != 0

def roe_get_as(roe, T):
    roe_p = ResultOrException(T).pointer(roe)
    return roe_p.p[0]

def to_ptr(i):
    T = util.typeof(i).nonref_type
    res = T.pointer(util.malloc(T.sizeof))
    util.in_place_new(res, i)
    return res

def wrapping_function_call(f):
    def new_f(*in_args):
        args = util.map_struct(in_args, dereferenceRefHeldAsPointer)

        output_type = util.typeof(f(*args)).nonref_type
        output_type_raw = util.typeof(f(*args))

        if output_type_raw.is_ref and not is_simple_type(output_type):
            if output_type_raw.is_ref_to_temp:
                #this function creates a new object and returns it to us
                raw_ptr = output_type.pointer(util.malloc(output_type.sizeof))

                util.in_place_new(raw_ptr, f(*args))

                return raw_ptr
            else:
                #this function is returning an internal reference
                ref_as_ptr =  ReferenceHeldAsPointer(output_type)(util.addr(f(*args)))

                reftype = util.typeof(ref_as_ptr).nonref_type

                raw_ptr = reftype.pointer(util.malloc(reftype.sizeof))

                util.in_place_new(raw_ptr, ref_as_ptr)

                return raw_ptr
        else:
            sz = output_type.sizeof
            if sz == 0:
                sz = 1

            #this function creates a new object and returns it to us by value
            raw_ptr = output_type.pointer(util.malloc(sz))

            util.in_place_new(raw_ptr, f(*args))

            return raw_ptr

    def exception_catcher(*args):
        out_t = util.typeof(new_f(*args)).nonref_type.value_type

        roe = ResultOrException(out_t)(0,0)

        try:
            roe.p = new_f(*args)
        except nativepython.lib.exception.InFlightException.pointer as e_ptr:
            roe.e = e_ptr

        return to_ptr(roe)

    exception_catcher.__name__ = "<wrappedouter f=%s>" % str(f)
    new_f.__name__ = "<wrappedinner f=%s>" % str(f)
    return exception_catcher


class SimpleFunction(object):
    def __init__(self, runtime, f, wants_wrapping=True):
        self.runtime = runtime
        self.f = f

    def __call__(self, *args):
        def type_for_arg(a):
            if isinstance(a, float):
                return type_model.Float64
            if isinstance(a, bool):
                return type_model.Bool
            if isinstance(a, int):
                return type_model.Int64
            if isinstance(a, RuntimeObject):
                if a._object_ptr is None:
                    return a._object_type
                else:
                    return a._object_type.reference

            return type_model.FreePythonObjectReference(a)

        def arg_for_arg(a):
            if isinstance(a, (float,int, bool)):
                return a
            if isinstance(a, RuntimeObject):
                return a._object_ptr
            return None

        f_target = self.runtime.convert(self.f, [type_for_arg(a) for a in args])

        f_callable = self.runtime.native_ptr_for_typed_target(f_target)

        return f_callable(*[arg_for_arg(a) for a in args])

class WrappedFunction(object):
    def __init__(self, runtime, f):
        self.runtime = runtime
        self.f = f
        self.wrapping_func = wrapping_function_call(self.f)

    def __call__(self, *args):
        def type_for_arg(a):
            if isinstance(a, float):
                return type_model.Float64
            if isinstance(a, bool):
                return type_model.Bool
            if isinstance(a, int):
                return type_model.Int64
            if isinstance(a, RuntimeObject):
                if a._object_ptr is None:
                    return a._object_type
                else:
                    return a._object_type.reference

            return type_model.FreePythonObjectReference(a)

        def arg_for_arg(a):
            if isinstance(a, (float,int, bool)):
                return a
            if isinstance(a, RuntimeObject):
                return a._object_ptr
            return None

        f_target = self.runtime.convert(self.wrapping_func, [type_for_arg(a) for a in args])

        f_callable = self.runtime.native_ptr_for_typed_target(f_target)

        output_type = f_target.output_type

        output_type_normal = output_type.value_type.element_types[0][1].value_type

        roe = f_callable(*[arg_for_arg(a) for a in args])

        if self.runtime._roe_is_exception(roe):
            raise RuntimeException(self.runtime._roe_get_exception(roe))

        if is_simple_type(output_type_normal):
            simple_result = self.runtime._roe_get_as(roe, output_type_normal)
            self.runtime._roe_teardown(roe, output_type_normal)

            if isinstance(output_type_normal, python_to_native_ast.CompileTimeType):
                return output_type_normal.python_object_representation
            
            return simple_result

        res = self.runtime._roe_extract_ptr(roe)

        return RuntimeObject(self.runtime, res, output_type_normal)

class RuntimeException(Exception):
    def __init__(self, in_flight_exception):
        Exception.__init__(self)

        self.in_flight_exception = in_flight_exception

class RuntimeObject(object):
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
        return "RuntimeObject(t=%s,p=%s)" % (self._object_type, self._object_ptr)

    def __del__(self):
        if self._object_ptr is not None:
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

        self._return = WrappedFunction(self, lambda x:x)
        self._call_func = WrappedFunction(self, call_func)
        self._getitem_func = WrappedFunction(self, getitem_func)
        self._setitem_func = WrappedFunction(self, setitem_func)
        self._len_fun = WrappedFunction(self, len_func)

        self._free_func = SimpleFunction(self, free_func)
        self._roe_is_exception = SimpleFunction(self, roe_is_exception)
        self._roe_get_as = SimpleFunction(self, roe_get_as)
        self._roe_teardown = SimpleFunction(self, roe_teardown)
        self._roe_extract_ptr = SimpleFunction(self, roe_extract_ptr)
        self._roe_get_exception = SimpleFunction(self, roe_get_exception)

        self._pointer_attribute_funcs = {}

    def _pointer_attribute_func(self, attr):
        if attr not in self._pointer_attribute_funcs:
            getter = util.attribute_getter(attr)
            self._pointer_attribute_funcs[attr] = \
                WrappedFunction(self, lambda p: util.ref_if_ref(getter(p)))

        return self._pointer_attribute_funcs[attr]

    def wrap(self, f):
        if isinstance(f, (int,float,bool,RuntimeObject)):
            return f
        
        return RuntimeObject(self, None, type_model.FreePythonObjectReference(f))


    def convert(self, f, argtypes):
        f_target = self.converter.convert(f, argtypes)

        functions = self.converter.extract_new_function_definitions()

        for k,v in self.compiler.add_functions(functions).items():
            self.functions_by_name[k] = v

        return f_target

    def native_ptr_for_typed_target(self, target):
        return self.functions_by_name[target.name]


