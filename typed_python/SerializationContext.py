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

from typed_python._types import serialize, deserialize

_builtin_value_to_name = {id(v):k for k,v in __builtins__.items() if isinstance(v,type)}
_builtin_name_to_value = {k:v for k,v in __builtins__.items() if isinstance(v,type)}

class SerializationContext(object):
    """Represents a collection of types with well-specified names that we can use to serialize objects."""
    def __init__(self, nameToObject=None, plugin=None):
        super().__init__()
        self.nameToObject = nameToObject or {}
        self.plugin = plugin
        self.objToName = {}

        #take the shortest name for each object in case of ambiguity
        for k,v in self.nameToObject.items():
            if id(v) not in self.objToName or len(k) < len(self.objToName[id(v)]):
                self.objToName[id(v)] = k

    def withPlugin(self, plugin):
        return SerializationContext(self.nameToObject, plugin)

    def nameForObject(self, t):
        if id(t) in _builtin_value_to_name:
            return _builtin_value_to_name[id(t)]

        return self.objToName.get(id(t))

    def objectFromName(self, name):
        if name in _builtin_name_to_value:
            return _builtin_name_to_value[name]

        return self.nameToObject.get(name)

    def serialize(self, instance, plugin=None):
        if plugin:
            self = self.withPlugin(plugin)
        return serialize(object, instance, self)

    def deserialize(self, bytes, plugin=None):
        if plugin:
            self = self.withPlugin(plugin)
        return deserialize(object, bytes, self)

    def representationFor(self, inst):
        if self.plugin:
            return self.plugin.representationFor(inst)
        return None

    def setInstanceStateFromRepresentation(self, instance, representation):
        self.plugin.setInstanceStateFromRepresentation(instance, representation)

class SerializationPlugin(object):
    def representationFor(self, inst):
        """Return an alternative representation for an object.

        If no alternative representation is desired, return None.

        Otherwise, return a tuple
            (type, representation)
        where 'type' will be used to construct a new empty object, which
        will then be passed to 'setInstanceStateFromRepresentation' along with 'representation'.
        """
        return None

    def setInstanceStateFromRepresentation(self, instance, representation):
        """Fill out an instance from its representation. """
        raise NotImplemented()