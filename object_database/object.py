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

#singleton object that clients should never see

from object_database.view import _cur_view

from typed_python.hash import sha_hash

from typed_python import Alternative, OneOf, TupleOf, ConstDict, TypeConvert, Tuple, Kwargs

from types import FunctionType

class DatabaseObject(object):
    __typed_python_type__ = True
    __types__ = None
    _database = None

    def __ne__(self, other):
        return not (self==other)
        
    def __eq__(self, other):
        if not isinstance(other, DatabaseObject):
            return False
        if not self._database is other._database:
            return False
        if not type(self) is type(other):
            return False
        return self._identity == other._identity

    def __hash__(self):
        return hash(self._identity)

    @classmethod
    def __typed_python_try_convert_instance__(cls, value, allow_construct_new):
        if isinstance(value, cls):
            return (value,)
        
        return None

    def __init__(self, identity):
        object.__init__(self)

        assert isinstance(identity, str), type(identity)
        
        self.__dict__['_identity'] = identity

    @classmethod
    def New(cls, **kwds):
        if not hasattr(_cur_view, "view"):
            raise Exception("Please create new objects from within a transaction.")

        if _cur_view.view._db is not cls._database:
            raise Exception("Please create new objects from within a transaction created on the same database as the object.")

        return _cur_view.view._new(cls, kwds)

    def __repr__(self):
        return type(self).__qualname__ + "(" + self._identity[:8] + ")"

    @classmethod
    def lookupOne(cls, **kwargs):
        return cls._database.current_transaction().indexLookupOne(cls, **kwargs or {' exists': True})

    @classmethod
    def lookupAll(cls, **kwargs):
        return cls._database.current_transaction().indexLookup(cls, **kwargs or {' exists': True})

    @classmethod
    def lookupAny(cls, **kwargs):
        return cls._database.current_transaction().indexLookupAny(cls, **kwargs or {' exists': True})

    def exists(self):
        if not hasattr(_cur_view, "view"):
            raise Exception("Please access properties from within a view or transaction.")

        if _cur_view.view._db is not type(self)._database:
            raise Exception("Please access properties from within a view or transaction created on the same database as the object.")

        return _cur_view.view._exists(self, type(self).__qualname__, self._identity)

    def __getattr__(self, name):
        if name[:1] == "_":
            raise AttributeError(name)

        return self.get_field(name)

    def get_field(self, name):
        if name not in self.__types__:
            raise AttributeError("Object of type %s has no field '%s'" % (type(self).__qualname__, name))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access properties from within a view or transaction.")

        if _cur_view.view._db is not type(self)._database:
            raise Exception("Please access properties from within a view or transaction created on the same database as the object.")
        
        return _cur_view.view._get(type(self).__qualname__, self._identity, name, self.__types__[name])

    def __setattr__(self, name, val):
        if name not in self.__types__:
            raise AttributeError("Database object of type %s has no attribute %s" % (type(self).__qualname__, name))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access properties from within a view or transaction.")

        if _cur_view.view._db is not type(self)._database:
            raise Exception("Please access properties from within a view or transaction created on the same database as the object.")

        coerced_val = TypeConvert(self.__types__[name], val, allow_construct_new=True)

        _cur_view.view._set(self, type(self).__qualname__, self._identity, name, self.__types__[name], coerced_val)

    def delete(self):
        _cur_view.view._delete(self, type(self).__qualname__, self._identity, self.__types__.keys())

    @classmethod
    def _define(cls, **types):
        assert cls.__types__ is None, "already defined"
        assert isinstance(types, dict)

        cls.__types__ = types
        
        return cls

    @classmethod
    def to_json(cls, obj):
        return obj.__dict__['_identity']

    @classmethod
    def from_json(cls, obj):
        assert isinstance(obj, str), obj

        return cls(obj)

    def __sha_hash__(self):
        return sha_hash(self._identity) + sha_hash(type(self).__qualname__)

class Indexed:
    def __init__(self, obj):
        assert isinstance(obj, (type, FunctionType))
        self.obj = obj

class Index:
    def __init__(self, *names):
        self.names = names

    def __call__(self, instance):
        return tuple(getattr(instance,x) for x in self.names)

