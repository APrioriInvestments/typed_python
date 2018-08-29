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

from object_database.object import DatabaseObject, Index, Indexed
from types import FunctionType
from typed_python import *
import inspect

TypeDefinition = NamedTuple(fields=TupleOf(str), indices=TupleOf(str))
SchemaDefinition = ConstDict(str, TypeDefinition)

class Schema:
    """A collection of types that can be used to access data in a database."""
    def __init__(self, name):
        self._name = name
        self._types = {}
        #class -> indexname -> fun(object->value)
        self._indices = {}
        self._indexTypes = {}
        self._frozen = False

    def toDefinition(self):
        return SchemaDefinition({
            tname: self.typeToDef(t) for tname, t in self._types.items()
            })

    def typeToDef(self, t):
        return TypeDefinition(fields = tuple(t.__types__.keys()) + (" exists",), indices= tuple(self._indices.get(t,{}).keys()))

    @property
    def name(self):
        return self._name

    def freeze(self):
        if not self._frozen:
            for tname, t in self._types.items():
                if issubclass(t, DatabaseObject) and t.__types__ is None:
                    raise Exception("Database subtype %s is not defined." % tname)

            self._frozen = True

    def __setattr__(self, typename, val):
        if typename[:1] == "_":
            self.__dict__[typename] = val
            return
        
        assert not self._frozen, "Schema is already frozen."

        self._types[typename] = val

    def __getattr__(self, typename):
        assert '.' not in typename

        if typename[:1] == "_":
            return self.__dict__[typename]

        if typename not in self._types:
            if self._frozen:
                raise AttributeError(typename)

            class cls(DatabaseObject):
                pass

            cls.__qualname__ = typename
            cls.__schema__ = self

            self._types[typename] = cls
            self._indices[cls] = {" exists": lambda e: True}
            self._indexTypes[cls] = {" exists": bool}

        return self._types[typename]

    def _addIndex(self, type, prop, fun = None, index_type = None):
        assert issubclass(type, DatabaseObject)

        if type not in self._indices:
            self._indices[type] = {}
            self._indexTypes[type] = {}

        if fun is None:
            fun = lambda o: getattr(o, prop)
            index_type = type.__types__[prop]
        else:
            if index_type is None:
                spec = inspect.getfullargspec(fun)
                index_type = spec.annotations.get('return', None)

        self._indices[type][prop] = fun
        self._indexTypes[type][prop] = index_type

    def define(self, cls):
        assert cls.__name__[:1] != "_", "Illegal to use _ for first character in database classnames."
        assert not self._frozen, "Schema is already frozen"

        t = getattr(self, cls.__name__)
        
        types = {}
        
        for name, val in cls.__dict__.items():
            if name[:2] != '__' and isinstance(val, type):
                types[name] = val
            elif name[:2] != '__' and isinstance(val, Indexed):
                if isinstance(val.obj, type):
                    types[name] = val.obj

        t._define(**types)

        for name, val in cls.__dict__.items():
            if isinstance(val, Index):
                self._addIndex(t, name, val, Tuple(*tuple(types[k] for k in val.names)))

            if name[:2] != '__' and isinstance(val, Indexed):
                if isinstance(val.obj, FunctionType):
                    self._addIndex(t, name, val.obj)
                    setattr(t, name, val.obj)
                else:
                    self._addIndex(t, name)
            elif (not name.startswith("__") or name in ["__str__", "__repr__"]):
                if isinstance(val, (FunctionType, staticmethod)):
                    setattr(t, name, val)

        return t
