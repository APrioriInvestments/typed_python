#   Copyright 2018 Braxton Mckee
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

valid_primitive_types = (str,int,bool,float,bytes,type(None))

import types
import nativepython.python.inspect_override as inspect

from typed_python.hash import sha_hash

_null_hash = sha_hash(None)

def IsType(t):
    if t in valid_primitive_types:
        return True

    if isinstance(t, valid_primitive_types):
        return True

    return hasattr(t, "__typed_python_type__")

def IsTypeFilter(t):
    if IsType(t):
        return True

    return hasattr(t, "__typed_python_type_filter__")

def IsTypeFilterTuple(t):
    return isinstance(t, tuple) and all(IsTypeFilter(e) for e in t)

def TypeToString(t):
    if isinstance(t, (int, str, float, type(None), bool, bytes)):
        return repr(t)
    if hasattr(t,'__qualname__'):
        return t.__qualname__

    return str(t)

def IsTypeFilterDict(t):
    return (
        isinstance(t, dict) 
            and all(IsTypeFilter(v) for v in t.values()) 
            and all(isinstance(k,str) for k in t.keys())
        )

def toTuple(t):
    if not isinstance(t, (tuple,list)):
        return t
    return tuple(toTuple(x) for x in t)

class UndefinedBehaviorException(Exception):
    """For errors so bad that compiled code would be expected to abort!"""
    pass

def MakeMetaclassFunction(name, **keywords):
    def TypeFunction(func):
        """Decorator that makes regular functions into memoized TypeFunctions."""
        lookup = {}

        def preprocessHook(args, kwargs):
            return (args,kwargs)

        class MetaMetaClass(type):
            def __instancecheck__(self, instance):
                if not hasattr(instance, '__typed_python_metaclass__'):
                    return False

                return instance.__typed_python_metaclass__ is MetaClass

        class MetaClass(metaclass=MetaMetaClass):
            def __new__(self, *args, **kwargs):
                args = toTuple(args)
                kwargs = toTuple(sorted(kwargs.items()))

                args,kwargs = MetaClass.preprocessHook(args,kwargs)

                lookup_key = (args, kwargs)

                if lookup_key not in lookup:
                    lookup[lookup_key] = func(*args, **dict(kwargs))
                    lookup[lookup_key].__typed_python_type__ = True
                    lookup[lookup_key].__typed_python_metaclass__ = MetaClass

                return lookup[lookup_key]

        MetaClass.__name__ = func.__name__
        MetaClass.preprocessHook = preprocessHook

        return MetaClass #return actual_func
    
    TypeFunction.__name__ = name

    return TypeFunction

TypeFunction = MakeMetaclassFunction("TypeFunction", __typed_python_type__ = True)
Memoized = MakeMetaclassFunction("Memoized")

def TypeConvert(type_filter, value, allow_construct_new=False):
    res = TryTypeConvert(type_filter, value, allow_construct_new)
    if res is None:
        raise TypeError("Can't convert %s to %s" % (value, str(type_filter)))
    return res[0]

def TryTypeConvert(type_filter, value, allow_construct_new=False):
    if isinstance(type_filter, valid_primitive_types):
        if isinstance(value, type(type_filter)) and value == type_filter:
            return (value,)

        return None

    if type_filter in valid_primitive_types:
        if isinstance(value, type_filter):
            return (value,)

        if allow_construct_new:
            if isinstance(value, (float, int)) and type_filter in (float, int, bool):
                return (type_filter(value),)

        return None

    return type_filter.__typed_python_try_convert_instance__(value, allow_construct_new)

@Memoized
def OneOf(*args):
    assert args, "Need at least one option for OneOf to make sense"

    assert IsTypeFilterTuple(args)

    args = set(args)

    class OneOf:
        __typed_python_type_filter__ = True

        options = args
        
        def __new__(cls):
            raise TypeError("OneOf is a TypeFilter, not an actual Type, so you can't instantiate it.")

        def __init__(self, *args, **kwargs):
            raise TypeError("OneOf is a TypeFilter, not an actual Type, so you can't instantiate it.")

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            for o in OneOf.options:
                res = TryTypeConvert(o, value, allow_construct_new)
                if res:
                    return res
            return None

    OneOf.__qualname__ = "OneOf(" + ",".join(TypeToString(x) for x in sorted(args, key=str)) + ")"

    return OneOf

class Any:
    __typed_python_type_filter__ = True

    def __init__(self, *args, **kwargs):
        raise TypeError("Any is a TypeFilter, not an actual Type, so you can't instantiate it.")

    @staticmethod
    def __typed_python_try_convert_instance__(value, allow_construct_new):
        return (value,)

@TypeFunction
def ListOf(t):
    assert IsTypeFilter(t)
    
    class ListOf:
        ElementType = t

        def __init__(self, iterable = ()):
            self.__contents__ = [TypeConvert(t, x) for x in iterable]

        def __getitem__(self, x):
            return self.__contents__[x]

        def __setitem__(self, ix, x):
            self.__contents__[ix] = TypeConvert(t, x)

        def append(self, x):
            self.__contents__.append(TypeConvert(t, x))

        def __len__(self):
            return len(self.__contents__)

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if not isinstance(value, ListOf):
                return None
            return (value,)

    ListOf.__qualname__ = "ListOf(%s)" % str(t)

    return ListOf

@TypeFunction
def Dict(K,V):
    assert IsTypeFilter(K)
    assert IsTypeFilter(V)

    class Dict:
        KeyType=K
        ValueType=V

        def __init__(self, iterable = ()):
            self.__contents__ = {TypeConvert(K, k): TypeConvert(V,v) for k,v in iterable}

        def __getitem__(self, k):
            return self.__contents__[TypeConvert(K,k)]

        def __setitem__(self, k, v):
            self.__contents__[TypeConvert(K,k)] = TypeConvert(V, v)

        def __contains__(self, k):
            return TypeConvert(K, k) in self.__contents__
        
        def __delitem__(self, k):
            del self.__contents__[TypeConvert(K,k)]

        def __iter__(self):
            return self.__contents__.__iter__()

        def __repr__(self):
            return repr(self.__contents__)

        def __str__(self):
            return str(self.__contents__)

        def pop(self, k):
            return self.__contents__.pop(TypeConvert(K,k))

        def __len__(self):
            return len(self.__contents__)

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if not isinstance(value, Dict):
                return None
            return (value,)

    Dict.__qualname__ = "Dict(%s->%s)" % (K.__name__, V.__name__)

    return Dict

@TypeFunction
def ConstDict(K,V):
    assert IsTypeFilter(K)
    assert IsTypeFilter(V)
    
    class ConstDict:
        KeyType=K
        ValueType=V

        def __init__(self, iterable = ()):
            self._sha_hash_cache = None

            if isinstance(iterable, ConstDict):
                self.__contents__ = dict(iterable.__contents__)
            elif isinstance(iterable, dict):
                self.__contents__ = {TypeConvert(K, k): TypeConvert(V,v) for k,v in iterable.items()}
            else:
                self.__contents__ = {TypeConvert(K, k): TypeConvert(V,v) for k,v in iterable}
        
        def __sha_hash__(self):
            if self._sha_hash_cache is None:
                base_hash = sha_hash(len(self.__contents__))

                for k,v in sorted(self.__contents__.items()):
                    base_hash = base_hash + sha_hash(k) + sha_hash(v)

                self._sha_hash_cache = base_hash

            return self._sha_hash_cache

        def __eq__(self, other):
            if not isinstance(other, ConstDict):
                x = TryTypeConvert(ConstDict, other, allow_construct_new=True)

                if x is None:
                    return NotImplemented

                return self == x[0]

            return self.__contents__ == other.__contents__

        def __hash__(self):
            return hash(self.__sha_hash__())

        def __getitem__(self, k):
            return self.__contents__[TypeConvert(K,k)]

        def get(self, k, other=None):
            return self.__contents__.get(TypeConvert(K, k), other)

        def __len__(self):
            return len(self.__contents__)

        def __add__(self, other):
            other = TypeConvert(ConstDict, other)

            res = ConstDict(self)
            res.__contents__.update(other.__contents__)

            return res

        def __repr__(self):
            return repr(self.__contents__)

        def __str__(self):
            return str(self.__contents__)

        def __contains__(self, k):
            return TypeConvert(K, k) in self.__contents__

        def items(self):
            return self.__contents__.items()

        def keys(self):
            return self.__contents__.keys()

        def values(self):
            return self.__contents__.values()

        def __sub__(self, other):
            res = ConstDict(self)

            for k in other:
                k = TypeConvert(K, k)
                if k in res.__contents__:
                    del res.__contents__[k]

            return res

        def __iter__(self):
            return self.__contents__.__iter__()

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if isinstance(value, ConstDict):
                return (value,)

            if allow_construct_new or isinstance(value, dict):
                try:
                    res = list(value.items())
                except:
                    return None

                members = []

                for k,v in res:
                    converted_k = TryTypeConvert(K, k, allow_construct_new)
                    converted_v = TryTypeConvert(V, v, allow_construct_new)

                    if converted_k is None or converted_v is None:
                        return None

                    members.append((converted_k[0], converted_v[0]))

                return (ConstDict(members),)

            return None

    ConstDict.__qualname__ = "ConstDict(%s->%s)" % (TypeToString(K), TypeToString(V))

    return ConstDict

@TypeFunction
def TupleOf(t):
    assert IsTypeFilter(t)
    
    class TupleOf:
        ElementType = t

        def __init__(self, iterable = ()):
            self._sha_hash_cache = None
            self.__contents__ = tuple(TypeConvert(t, x) for x in iterable)
        
        def __sha_hash__(self):
            if self._sha_hash_cache is None:
                base_hash = sha_hash(len(self.__contents__))
            
                for k in self.__contents__:
                    base_hash = base_hash + sha_hash(k)

                self._sha_hash_cache = base_hash

            return self._sha_hash_cache

        def __getitem__(self, x):
            return self.__contents__[x]

        def __len__(self):
            return len(self.__contents__)

        def __repr__(self):
            return repr(self.__contents__)

        def __str__(self):
            return str(self.__contents__)

        def index(self, other):
            return self.__contents__.index(TypeConvert(t,other))

        def __add__(self, other):
            res = TupleOf(())
            res.__contents__ = self.__contents__ + tuple(TypeConvert(t, x) for x in other)
            return res

        def __eq__(self, other):
            if not isinstance(other, TupleOf):
                x = TryTypeConvert(TupleOf, other, allow_construct_new=True)
                if x is None:
                    return NotImplemented
                return self == x[0]

            return self.__contents__ == other.__contents__

        def __hash__(self):
            return hash(self.__contents__)

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if isinstance(value, TupleOf):
                return (value,)

            if allow_construct_new or isinstance(value, tuple):
                try:
                    res = list(value)
                except:
                    return None

                members = []

                for val in res:
                    converted = TryTypeConvert(t, val, allow_construct_new)
                    if converted is None:
                        return None
                    members.append(converted[0])

                return (TupleOf(members),)

            return None

    TupleOf.__qualname__ = "TupleOf(%s)" % str(t)

    return TupleOf

@TypeFunction
def Kwargs(**kwargs):
    assert IsTypeFilterDict(kwargs)
    
    kwargs_sorted = sorted(kwargs)

    class Kwargs:
        ElementTypes = kwargs
        ElementNames = kwargs_sorted

        def __init__(self, iterable=None, **starargs):
            if iterable is None and starargs:
                iterable = starargs

            assert len(iterable) == len(kwargs)

            self.__contents__ = {k: TypeConvert(kwargs[k], iterable[k]) for k in kwargs_sorted}

        def __getitem__(self, x):
            return self.__contents__[x]

        def __len__(self):
            return len(self.__contents__)

        def __iter__(self):
            return iter(self.__contents__)

        def items(self):
            return self.__contents__.items()

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if not isinstance(value, Kwargs):
                return None
            return (value,)

    Kwargs.__qualname__ = "Kwargs(" + ",".join("%s=%s" % (k,v) for k,v in sorted(kwargs.items())) + ")"

    return Kwargs

@TypeFunction
def NamedTuple(*namesAndTypes, **kwargs):
    for n in namesAndTypes:
        assert isinstance(n, tuple)
        assert len(n) == 2
        assert IsTypeFilter(n[1]), n[1]
        assert isinstance(n[0], str)

    names = [n[0] for n in namesAndTypes]

    assert len(set(names)) == len(names), "names need to be unique"

    types = [n[1] for n in namesAndTypes]

    nameLookup = { names[i]:i for i in range(len(names)) }

    class NamedTuple_:
        ElementTypes = types
        ElementNames = names
        ElementNamesAndTypes = namesAndTypes
        NameLookup = nameLookup

        def __init__(self, iterable=None, **kwargs):
            self._sha_hash_cache = None

            if iterable is not None:
                assert len(iterable) == len(namesAndTypes)

                self.__contents__ = tuple(TypeConvert(types[i], iterable[i]) for i in range(len(namesAndTypes)))
            else:
                assert len(kwargs) == len(namesAndTypes)
                self.__contents__ = tuple(TypeConvert(types[i], kwargs[names[i]]) for i in range(len(namesAndTypes)))

        def __getitem__(self, x):
            return self.__contents__[x]

        def __getattr__(self, x):
            if x not in nameLookup:
                raise AttributeError(x)
            return self.__contents__[nameLookup[x]]

        def __len__(self):
            return len(self.__contents__)

        def __add__(self, other):
            out_type = NamedTuple(NamedTuple_.ElementNamesAndTypes + type(other).ElementNamesAndTypes)
            return out_type(self.__contents__ + other.__contents__)

        def __str__(self):
            return repr(self)

        def __sha_hash__(self):
            if self._sha_hash_cache is None:
                base_hash = sha_hash(len(self.__contents__))

                for k in self.__contents__:
                    base_hash = base_hash + sha_hash(k)

                self._sha_hash_cache = base_hash

            return self._sha_hash_cache

        def __repr__(self):
            return "(%s)" % (",".join("%s=%s" % (NamedTuple_.ElementNames[i], self.__contents__[i]) 
                        for i in range(len(self.__contents__))))

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if isinstance(value, NamedTuple_):
                return (value,)

            if allow_construct_new:
                if isinstance(value, dict):
                    if len(value) != len(types):
                        return None

                    members = []

                    for ix in range(len(types)):
                        if names[ix] not in value:
                            return None

                        converted = TryTypeConvert(types[ix], value[names[ix]], allow_construct_new)
                        if converted is None:
                            return None

                        members.append(converted[0])

                    return (NamedTuple_(members),)
                else:
                    try:
                        res = list(value.items())
                    except:
                        return None

                    if len(res) != len(types):
                        return None

                    members = []

                    for ix in range(len(types)):
                        converted = TryTypeConvert(types[ix], res[ix], allow_construct_new)
                        if converted is None:
                            return None
                        members.append(converted[0])

                    return (NamedTuple_(members),)

            return None

    NamedTuple_.__qualname__ = "NamedTuple(" + ",".join("%s=%s" % (k,TypeToString(v)) for k,v in sorted(namesAndTypes)) + ")"

    return NamedTuple_

def _named_tuple_preprocess_hook(args, kwargs):
    """Allow NamedTuple to accept kwargs, which we'll sort (since we need a canonical ordering)"""
    return args + kwargs, ()

NamedTuple.preprocessHook = _named_tuple_preprocess_hook
def _named_tuple_new(**kwargs):
    T = NamedTuple(**{k: type(v) for k,v in kwargs.items()})
    
    return T(**kwargs)

NamedTuple.New = staticmethod(_named_tuple_new)

@TypeFunction
def Tuple(*args):
    assert IsTypeFilterTuple(args), args
    
    class Tuple:
        ElementTypes = args

        def __init__(self, iterable):
            assert len(iterable) == len(args)
            self._sha_hash_cache = None
            self.__contents__ = tuple(TypeConvert(args[i], iterable[i]) for i in range(len(args)))

        def __sha_hash__(self):
            if self._sha_hash_cache is None:
                base_hash = sha_hash(len(self.__contents__))

                for k in self.__contents__:
                    base_hash = base_hash + sha_hash(k)

                self._sha_hash_cache = base_hash

            return self._sha_hash_cache

        def __eq__(self, other):
            if not isinstance(other, Tuple):
                x = TryTypeConvert(Tuple, other, allow_construct_new=True)
                if x is None:
                    return NotImplemented
                return self == x[0]

            return self.__contents__ == other.__contents__

        def __hash__(self):
            return hash(self.__contents__)

        def __getitem__(self, x):
            return self.__contents__[x]

        def __len__(self):
            return len(self.__contents__)

        def __repr__(self):
            return repr(self.__contents__)

        def __str__(self):
            return str(self.__contents__)

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if isinstance(value, Tuple):
                return (value,)

            if allow_construct_new or isinstance(value, tuple):
                try:
                    value = tuple(value)
                except:
                    return None

                if len(value) == len(args):
                    new_elts = []
                    for i in range(len(args)):
                        converted = TryTypeConvert(args[i], value[i], allow_construct_new=True)
                        if converted is None:
                            return None
                        new_elts.append(converted[0])
                    return (Tuple(new_elts),)
            return None

    Tuple.__qualname__ = "Tuple(" + ",".join(TypeToString(x) for x in args) + ")"

    return Tuple

@TypeFunction
def Function(return_type, args):
    assert IsTypeFilter(return_type)
    assert IsTypeFilterTuple(args), args

    class Function:
        ReturnType = return_type
        ArgTypes = args

        def __init__(self, f):
            self.__func__ = f

        def __typed_python_matches__(self, *args, **kwargs):
            if kwargs:
                return None, None

            if len(args) != len(Function.ArgTypes):
                return None, None

            converted_args = []
            for ix, t in enumerate(Function.ArgTypes):
                res = TryTypeConvert(t, args[ix])
                if res:
                    converted_args.append(res[0])
                else:
                    return None, None

            return tuple(converted_args), {}

        def __call__(self, *args, **kwargs):
            args, kwargs = self.__typed_python_matches__(*args, **kwargs)

            if args is None:
                raise TypeError()

            result = self.__func__(*args, **kwargs)

            return TypeConvert(Function.ReturnType, result, allow_construct_new=True)

        def addTerm(self, term):
            T = OverloadedFunction(
                (type(self), type(term))
                )

            return T((self,term))

        def overload(self, other):
            return self.addTerm(TypedFunction(other))


    return Function

def annotationToTypeFilter(t):
    if t is None:
        return Any
    assert IsTypeFilter(t)
    return t


@TypeFunction
def OverloadedFunction(term_types):
    assert IsTypeFilterTuple(term_types)

    class OverloadedFunction_:
        TermTypes = term_types

        def __init__(self, terms):
            self.__terms__ = terms

        def addTerm(self, term):
            T = OverloadedFunction(
                term_types + (type(term),)
                )

            return T(self.__terms__ + (term,))

        def __call__(self, *args, **kwargs):
            for term in self.__terms__:
                a,k = term.__typed_python_matches__(*args, **kwargs)
                if a is not None:
                    return TypeConvert(type(term).ReturnType, term.__func__(*a,**k))

            raise TypeError()

        def overload(self, other):
            return self.addTerm(TypedFunction(f))

    return OverloadedFunction_

def TypedFunction(f, wrapper=None):
    if isinstance(f, types.FunctionType):
        spec = inspect.getfullargspec(f)
        if spec.varargs or spec.varkw or spec.defaults or spec.kwonlyargs or spec.kwonlydefaults:
            raise Exception("Don't know how to handle %s which uses some features I don't support yet" % f)

        arg_types = []
        for argname in spec.args:
            arg_types.append(
                annotationToTypeFilter(spec.annotations.get(argname, None))
                )

        return_type = annotationToTypeFilter(spec.annotations.get('return'))

        return Function(return_type, arg_types)(f if not wrapper else wrapper(f))

    raise Exception("Don't know how to type %s" % f)

class ClassMetaNamespace:
    def __init__(self):
        self.ns = {}
        self.order = []

    def __getitem__(self, k):
        return self.ns[k]

    def __setitem__(self, k, v):
        if k not in self.ns:
            self.order.append(k)

        special = ("__init__", "__constructor__", "__assign__", "__copy_constructor__", "__destructor__")

        def initToConstructor(initFun):
            def __constructor__(self, *args, **kwargs):
                type(self).__typed_python_class_pre_init__(self)
                initFun(self, *args, **kwargs)
            return __constructor__

        if isinstance(v, types.FunctionType) and (k[:2] != "__" or k in special):
            if k == '__init__':
                k = '__constructor__'
                v = TypedFunction(v, initToConstructor)
            else:
                v = TypedFunction(v)

            if k in self.ns:
                if isinstance(self.ns[k], (OverloadedFunction, Function)):
                    self.ns[k] = self.ns[k].addTerm(v)
                else:
                    self.ns[k] = v
            else:
                self.ns[k] = v
        else:
            assert k not in self.ns, "Cannot define a class member twice unless its a function overload."
            self.ns[k] = v

    def __contains__(self, k):
        return self.ns[k]

def wrapFunction(name, f):
    def inner(*args, **kwargs):
        return f(*args, **kwargs)
    inner.__name__ = name
    return inner

class ClassMeta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        return ClassMetaNamespace()

    def __new__(cls, name, bases, namespace, **kwds):
        res = dict(namespace.ns)

        members = {}
        internal_members = {}

        for k in res:
            if isinstance(res[k], (OverloadedFunction, Function)):
                res[k] = wrapFunction(k, res[k])
            elif IsTypeFilter(res[k]) and k[:2] != "__":
                if isinstance(res[k], Internal):
                    internal_members[k] = res[k].ElementType
                    members[k] = res[k].ElementType
                else:
                    members[k] = res[k]

        #pop out the user's definitions of their functions. We don't want them in the
        #actual class.
        constructor_fun = res.pop('__constructor__', None)
        copy_constructor_fun = res.pop('__copy_constructor__', None)
        assign_fun = res.pop('__assign__', None)
        destroy_fun = res.pop('__destructor__', None)

        res['__typed_python_class_members__'] = members
        res['__typed_python_class_internal_members__'] = internal_members

        def __pre_init__(self):
            for m,t in members.items():
                if self.__dict__[m] is not Uninitialized:
                    raise UndefinedBehaviorException("Variable %s is already initialized" % m)

                self.__dict__[m] = _construct(t, self.__typed_python_class_context__[0], self.__typed_python_class_context__[1] + (m,))

        def __destructor__(self):
            try:
                destroy_fun(self)
            except:
                #throwing exceptions in destructors is bad
                self.__typed_python_class_context__ = None

                try:
                    for k,v in internal_members.items():
                        object.__getattribute__(self, k).__destructor__()
                    return
                except:
                    #throwing doubly is an abort situation!
                    raise UndefinedBehaviorException("Exception thrown in destructor exception handling routine. This would crash in c++.")

            self.__typed_python_class_context__ = None

            for k,v in internal_members.items():
                object.__getattribute__(self, k).__destructor__()

        def __constructor__(self):
            __pre_init__(self)

        def __copy_constructor__(self, other):
            for m,t in members.items():
                if self.__dict__[m] is not Uninitialized:
                    raise UndefinedBehaviorException("Variable %s is already initialized" % m)

                self.__dict__[m] = _copy_construct(t, getattr(other, m), self.__typed_python_class_context__[0], self.__typed_python_class_context__[1] + (m,))

        def __assign__(self, other):
            for m,t in members.items():
                _assign(self.__dict__, m, getattr(other, m))

        res['__typed_python_class_pre_init__'] = __pre_init__
        res['__destructor__'] = __destructor__
        res['__constructor__'] = constructor_fun or __constructor__
        res['__copy_constructor__'] = copy_constructor_fun or __copy_constructor__
        res['__assign__'] = assign_fun or __assign__

        return type.__new__(cls, name, bases, res, **kwds)

class Uninitialized:
    pass

@Memoized
def Internal(t):
    assert isinstance(t, ClassMeta)

    class Internal:
        ElementType = t
        
    Internal.__name__ = "Internal(%s)" % repr(t)

    return Internal

def _construct(cls, context_owner, context_path, *args, **kwargs):
    if isinstance(cls, ClassMeta):
        res = object.__new__(cls)
        res.__typed_python_class_context__ = (context_owner, context_path)

        for m in cls.__typed_python_class_members__:
            res.__dict__[m] = Uninitialized
        
        res.__constructor__(*args, **kwargs)
        
        return res

    if not args and not kwargs:
        return cls()

    if len(args) == 1 and not kwargs:
        return TypeConvert(cls, args[0])

    raise TypeError("Can't construct a %s with args %s/%s" % (cls, args, kwargs))

def _copy_construct(cls, otherVal, context_owner, context_path):
    if isinstance(cls, ClassMeta):
        res = object.__new__(cls)
        res.__typed_python_class_context__ = (context_owner, context_path)

        for m in cls.__typed_python_class_members__:
            res.__dict__[m] = Uninitialized
        
        res.__copy_constructor__(otherVal)

        return res

    return TypeConvert(cls, otherVal)

def _assign(container, key, val):
    elt = container[key]

    if isinstance(elt, Class):
        elt.__assign__(val)
    else:
        container[key] = val

class init:
    def __init__(self, inst):
        self.__inst = inst

    def __getattr__(self, memberName):
        t = type(self.__inst).__typed_python_class_members__.get(memberName,None)
        if not t:
            raise TypeError("Can't initialize non-member " + memberName)

        def call(*args, **kwargs):
            assert self.__inst.__dict__.get(memberName) is Uninitialized
            self.__inst.__dict__[memberName] = _construct(
                t, 
                self.__inst.__typed_python_class_context__[0], 
                self.__inst.__typed_python_class_context__[1] + (memberName,),
                *args,
                **kwargs
                )

        return call


class Class(object, metaclass=ClassMeta):
    """Base class for all strongly-typed classes."""
    __typed_python_type__ = True

    def __new__(cls, *args, **kwargs):
        res = object.__new__(cls)

        res.__typed_python_class_context__ = (res, ())
        for m in cls.__typed_python_class_members__:
            res.__dict__[m] = Uninitialized
        
        res.__constructor__(*args, **kwargs)

        return res

    def __setattr__(self, k, v):
        if k[:2] == '__':
            return object.__setattr__(self, k, v)

        if self.__typed_python_class_context__ is None:
            raise UndefinedBehaviorException("Instance was deleted already!")

        if self.__dict__.get(k) is Uninitialized:
            raise UndefinedBehaviorException("Can't assign to an uninitialized value")

        int_type = type(self).__typed_python_class_internal_members__.get(k, None)
        if int_type is not None:
            self.__dict__[k].__assign__(TypeConvert(int_type,v))
            return

        t = type(self).__typed_python_class_members__.get(k, None)

        if t is not None:
            v = TypeConvert(t, v)
            return object.__setattr__(self, k, v)

        raise AttributeError(k)

    def __getattribute__(self, attr):
        if attr[:2] == "__":
            return object.__getattribute__(self, attr)

        if object.__getattribute__(self, '__typed_python_class_context__') is None:
            raise UndefinedBehaviorException("Instance was deleted already!")

        return object.__getattribute__(self, attr)

    @classmethod
    def __typed_python_try_convert_instance__(cls, value, allow_construct_new):
        if not isinstance(value, cls):
            return None
        return (value,)


@TypeFunction
def PackedArray(t):
    assert IsTypeFilter(t)
    
    class PackedArray:
        ElementType = t

        def __init__(self, iterable = ()):
            self.__contents__ = [_copy_construct(t, val, self, (ix,)) for ix,val in enumerate(iterable)]

        def __getitem__(self, ix):
            if ix < 0 or ix >= len(self.__contents__):
                raise IndexError()
            return self.__contents__[ix]

        def __setitem__(self, ix, x):
            if ix < 0 or ix >= len(self.__contents__):
                raise IndexError()
            _assign(self.__contents__, ix, TypeConvert(t, x))

        def append(self, x):
            self.__contents__.append(TypeConvert(t, x))

        def resize(self, ix, argument=None):
            #resize the array.
            if ix > len(self.__contents__):
                if argument is None:
                    while len(self.__contents__) < ix:
                        self.__contents__.append(_construct(t, self, (ix,)))
                else:
                    while len(self.__contents__) < ix:
                        self.__contents__.append(_copy_construct(t, argument, self, (ix,)))
            else:
                if isinstance(t, ClassMeta):
                    for x in self.__contents__[ix:]:
                        x.__destructor__()

                self.__contents__ = self.__contents__[:ix]

        def ptr(self, ix):
            assert ix >= 0 and ix < len(self.__contents__)
            return Pointer(t)(None, self, ix, _PrivateGuard)

        def __len__(self):
            return len(self.__contents__)

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if not isinstance(value, PackedArray):
                return None
            return (value,)

    PackedArray.__qualname__ = "PackedArray(%s)" % str(t)

    return PackedArray

class _PrivateGuard:
    """Class used as a token to provide access to Pointer construction.

    Intended to prevent users from accidentally constructing Pointer
    objects directly.
    """
    pass

def ptr(x):
    assert isinstance(x, Class)

    return Pointer(type(x))(x, _PrivateGuard)

@TypeFunction
def Pointer(t):
    assert IsTypeFilter(t)

    class Pointer:
        ElementType = t

        def __init__(self, value=None, container=None, offset=None, internal_guard = None):
            """Initialize a Pointer. Either value or (container, offset) must be populated.

            if value is populated, then it's a class instance.
            if container is populated, then this is a member of a container.
            if container is not populated, but offset is something other than zero, then this
                is illegal pointer arithmetic!
            """
            if value is None and container is None and offset is None:
                #this constructs the null pointer
                self.__value__ = None
                self.__container__ = None
                self.__offset__ = None
                return

            assert internal_guard is _PrivateGuard, "only internal code should call this."


            self.__value__ = value
            self.__container__ = container
            self.__offset__ = offset

        def __add__(self, offset):
            new_offset = (self.__offset__ or 0) + offset
            
            if new_offset == 0 and self.__container__ is None:
                new_offset = 0

            return Pointer(self.__value__, self.__container__, new_offset, _PrivateGuard)

        def get(self):
            if self.__offset__ is not None and self.__container__ is None:
                raise UndefinedBehaviorException("Invalid Pointer Dereference")

            if self.__value__ is not None:
                return self.__value__
            else:
                if self.__offset__ < 0 or self.__offset__ >= len(self.__container__):
                    raise UndefinedBehaviorException("Invalid Pointer Dereference")

                return self.__container__[self.__offset__]

        def set(self, arg):
            if self.__offset__ is not None and self.__container__ is None:
                raise UndefinedBehaviorException("Invalid Pointer Dereference")

            if self.__value__ is not None:
                self.__value__.__assign__(arg)
            else:
                if self.__offset__ < 0 or self.__offset__ >= len(self.__container__):
                    raise UndefinedBehaviorException("Invalid Pointer Dereference")

                if isinstance(t, ClassMeta):
                    self.__container__[self.__offset__].__assign__(arg)
                else:
                    #this is a container of non-classes, so we can just plug the value in
                    self.__container__[self.__offset__] = TypeConvert(t, arg)

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if not isinstance(value, Pointer):
                return None
            return (value,)

    Pointer.__qualname__ = "Pointer" + repr(t)

    return Pointer

class StackGetter:
    def __init__(self, stack):
        self.stack = stack

    def __getitem__(self, ix):
        if self.stack._destroyed:
            raise UndefinedBehaviorException("Access already destroyed stack variable.")
        return self.stack._variables[ix]

    def __setitem__(self, ix, v):
        if self.stack._destroyed:
            raise UndefinedBehaviorException("Access already destroyed stack variable.")
        self.stack._variables[ix] = v

    def __len__(self):
        return len(self.stack._variables)

class Stack:
    def __init__(self):
        self._variables = []
        self._active = False
        self._destroyed = False
        self._getter = StackGetter(self)

    def __enter__(self):
        self._active = True
        return self

    def __exit__(self, *args):
        self._destroyed = True

        for v in reversed(self._variables):
            if isinstance(v, Class):
                v.__destructor__()

    def __call__(self, t):
        assert IsTypeFilter(t)

        assert self._active
        assert not self._destroyed

        intended_count = len(self._variables)

        def init(*args, **kwargs):
            assert len(self._variables) == intended_count

            self._variables.append(_construct(t, self._getter, (len(self._variables),), *args, **kwargs))

            return Pointer(t)(None, self._getter, intended_count, _PrivateGuard)

        return init

