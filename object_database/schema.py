#   Coyright 2017-2019 Nativepython Authors
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

import object_database._types as _types

from object_database.object import Index, Indexed
from types import FunctionType
from typed_python import ConstDict, NamedTuple, Tuple, TupleOf, serialize


ObjectId = int
FieldId = int
ObjectFieldId = NamedTuple(objId=int, fieldId=int, isIndexValue=bool)
IndexValue = bytes
IndexId = NamedTuple(fieldId=int, indexValue=IndexValue)
DatabaseObjectBase = NamedTuple(_identity=int)

TypeDefinition = NamedTuple(fields=TupleOf(str), indices=TupleOf(str))
SchemaDefinition = ConstDict(str, TypeDefinition)

FieldDefinition = NamedTuple(schema=str, typename=str, fieldname=str)


def SubscribeLazilyByDefault(t):
    t.__object_database_lazy_subscription__ = True
    return t


def indexValueFor(type, value, serializationContext=None):
    return serialize(type, value, serializationContext)


class Schema:
    """A collection of types that can be used to access data in a database."""

    def __init__(self, name):
        self._name = name

        # Map: typename:str -> cls
        self._types = {}

        # set of typename that still need definition
        self._undefinedTypes = set()

        # Map: typename:str -> type
        # contains types we have defined on the schema that are not
        # DatabaseObject types.
        self._supportingTypes = {}

        # class -> indexname -> tuple(str)
        self._indices = {}

        # class -> indexname -> tuple(str)
        self._index_types = {}

        # class -> fieldname -> type
        self._field_types = {}

        self._frozen = False

        # Map: cls -> original_cls
        self._types_to_original = {}

    def toDefinition(self):
        return SchemaDefinition({
            tname: self.typeToDef(t)
            for tname, t in self._types.items()
            if getattr(t, "__is_database_object_type__", False)
        })

    def __repr__(self):
        return "Schema(%s)" % self.name

    def lookupFullyQualifiedTypeByName(self, name):
        if not name.startswith(self.name + "."):
            return None
        return self._types.get(name[len(self.name)+1:])

    def typeToDef(self, t):
        return TypeDefinition(
            fields=sorted(self._field_types[t.__name__]),
            indices=sorted(self._indices[t.__name__])
        )

    def getType(self, t):
        return self._types.get(t)

    def fieldType(self, typename, fieldname):
        """Return the type of the field named 'fieldname' in 'typename'.

        If the field or type is unknown, return None.
        """
        return self._field_types.get(typename, {}).get(fieldname, None)

    def indexType(self, typename, fieldname):
        """Return the type of the field named 'fieldname' in 'typename'.

        If the field or type is unknown, return None.
        """
        return self._index_types.get(typename, {}).get(fieldname, None)

    @property
    def name(self):
        return self._name

    def freeze(self):
        if not self._frozen:
            assert not self._undefinedTypes, "Still need definitions for %s" % (", ".join(self._undefinedTypes))
            self._frozen = True

    def __setattr__(self, typename, val):
        if typename.startswith("_"):
            self.__dict__[typename] = val
            return

        assert not self._frozen, "Schema is already frozen."

        assert isinstance(val, type)

        self._supportingTypes[typename] = val

    def __getattr__(self, typename):
        assert '.' not in typename

        if typename.startswith("_"):
            return self.__dict__[typename]

        if typename in self._supportingTypes:
            return self._supportingTypes[typename]

        if typename not in self._types:
            if self._frozen:
                raise AttributeError(typename)

            cls = _types.createDatabaseObjectType(self, typename)

            self._types[typename] = cls
            self._indices[typename] = {}
            self._index_types[typename] = {}
            self._field_types[typename] = {}

            self._undefinedTypes.add(typename)

        return self._types[typename]

    def define(self, cls):
        typename = cls.__name__

        assert not typename.startswith("_"), "Illegal to use _ for first character in database classnames."
        assert not self._frozen, "Schema is already frozen"

        # get a type stub
        t = getattr(self, typename)

        # add the canonical ' exists' property, which we use under the hood to indicate
        # existence/deletion of an object.
        self._field_types[typename][' exists'] = bool
        self._indices[typename][' exists'] = (' exists',)
        self._index_types[typename][' exists'] = bool

        t.addField(' exists', bool)
        t.addIndex(' exists', (' exists',))

        # make sure it's not defined yet
        assert typename in self._undefinedTypes, f"Type {typename} is not undefined."
        self._undefinedTypes.discard(typename)

        self._types_to_original[t] = cls

        # compute baseClasses in order to collect the type's attributes but filter out
        # object and the DatabaseObjectBase
        baseClasses = [x for x in cls.__mro__ if x not in (object, DatabaseObjectBase)]

        properBaseClasses = [self._types_to_original.get(b, b) for b in baseClasses]

        # Collect the type's attributes and populate the actual type object
        # Map: name -> type
        classMembers = {}

        for base in reversed(properBaseClasses):
            for name, val in base.__dict__.items():
                classMembers[name] = val

        for name, val in classMembers.items():
            isMagic = name[:2] == "__"

            if isinstance(val, type) and not isMagic:
                t.addField(name, val)
                self._field_types[typename][name] = val
            elif isinstance(val, Indexed):
                t.addField(name, val.obj)
                t.addIndex(name, (name,))
                self._field_types[typename][name] = val.obj
                self._indices[typename][name] = (name,)
                self._index_types[typename][name] = val.obj
            elif isinstance(val, Index):
                # do this in a second pass
                pass
            elif isinstance(val, property):
                t.addProperty(name, val.fget, val.fset)
            elif isinstance(val, staticmethod):
                t.addStaticMethod(name, val.__func__)
            elif isinstance(val, FunctionType):
                t.addMethod(name, val)

        for name, val in classMembers.items():
            if isinstance(val, Index):
                t.addIndex(name, tuple(val.names))
                assert len(val.names)

                self._indices[typename][name] = tuple(val.names)

                if len(val.names) > 1:
                    self._index_types[typename][name] = Tuple(*[self._field_types[typename][fieldname] for fieldname in val.names])
                else:
                    self._index_types[typename][name] = self._field_types[typename][val.names[0]]

        t.finalize()

        if hasattr(cls, '__object_database_lazy_subscription__'):
            t.markLazyByDefault()

        return t
