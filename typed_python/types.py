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

from typed_python.hash import sha_hash

def IsType(t):
    if t in valid_primitive_types:
        return True

    if isinstance(t, valid_primitive_types):
        return True

    return hasattr(t, "__typed_python_type__")

def IsTypeFilter(t):
    if IsType(t):
        return True

    if isinstance(t, OneOf):
        return True

    return False

def TypeFunction(func):
    """Decorator that makes regular functions into memoized TypeFunctions."""
    lookup = {}

    class MetaMetaClass(type):
        def __instancecheck__(self, instance):
            if not hasattr(instance, '__typed_python_metaclass__'):
                return False

            return instance.__typed_python_metaclass__ is MetaClass

    class MetaClass(metaclass=MetaMetaClass):
        def __new__(self, *args, **kwargs):
            lookup_key = (TypeFunction, func.__name__, tuple(args), tuple(sorted(kwargs.items())))

            if lookup_key not in lookup:
                lookup[lookup_key] = func(*args, **kwargs)
                lookup[lookup_key].__typed_python_type__ = True
                lookup[lookup_key].__typed_python_metaclass__ = MetaClass

            return lookup[lookup_key]

    return MetaClass #return actual_func

def TypeConvert(type_filter, value, allow_construct_new=False):
    """Return a converted value, or throw an exception"""
    if type_filter in valid_primitive_types:
        if isinstance(value, type_filter):
            return value
        raise TypeError("Can't convert %s to %s" % (value, type_filter))

    res = type_filter.__typed_python_try_convert_instance__(value, allow_construct_new)
    if res:
        return res[0]
    raise TypeError("Can't convert %s to %s" % (value, type_filter))

def TryTypeConvert(type_filter, value, allow_construct_new=False):
    if isinstance(type_filter, valid_primitive_types):
        if isinstance(value, type(type_filter)) and value == type_filter:
            return (value,)
        return None

    if type_filter in valid_primitive_types:
        if isinstance(value, type_filter):
            return (value,)
        else:
            return None

    return type_filter.__typed_python_try_convert_instance__(value, allow_construct_new)

@TypeFunction
def OneOf(*args):
    class OneOf:
        options = args
        
        def __init__(self, *args, **kwargs):
            raise Exception("OneOf is a TypeFilter, not an actual Type, so you can't instantiate it.")

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            for o in OneOf.options:
                res = TryTypeConvert(o, value, allow_construct_new)
                if res:
                    return res
            return None

    OneOf.__name__ = "OneOf" + repr(args)

    return OneOf

@TypeFunction
def ListOf(t):
    assert IsTypeFilter(t)

    class ListOf:
        T = t

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

    ListOf.__name__ == "ListOf(%s)" % t.__name__

    return ListOf

@TypeFunction
def Dict(K,V):
    raise NotImplementedError()

@TypeFunction
def ConstDict(K,V):
    raise NotImplementedError()

@TypeFunction
def Kwargs(**kwargs):
    raise NotImplementedError()

@TypeFunction
def TupleOf(t):
    assert IsTypeFilter(t)

    class TupleOf:
        __typed_python_type__ = True
        __typed_python_hash__ = None
        T = t

        def __init__(self, iterable = ()):
            self.__contents__ = tuple(TypeConvert(t, x) for x in iterable)

        def __getitem__(self, x):
            return self.__contents__[x]

        def __len__(self):
            return len(self.__contents__)

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if isinstance(value, TupleOf):
                return (value,)

            if allow_construct_new:
                try:
                    res = list(value)
                except:
                    return None

                members = []

                for val in res:
                    converted = TryTypeConvert(t, val)
                    if converted is None:
                        return None
                    members.append(converted[0])

                return (TupleOf(members),)

            return None

    TupleOf.__name__ == "TupleOf(%s)" % t.__name__

    return TupleOf

@TypeFunction
def Tuple(*args):
    for a in args:
        assert IsTypeFilter(a)

    class Tuple:
        __typed_python_type__ = True
        __typed_python_hash__ = None
        T = t

        def __init__(self, iterable):
            assert len(iterable) == len(args)

            self.__contents__ = tuple(TypeConvert(iterable[i], args[i]) for i in xrange(len(args)))

        def __getitem__(self, x):
            return self.__contents__[x]

        def __len__(self):
            return len(self.__contents__)

        @staticmethod
        def __typed_python_try_convert_instance__(value, allow_construct_new):
            if not isinstance(value, Tuple):
                return None
            return (value,)

    Tuple.__name__ == "Tuple" + repr(args)

    return Tuple
