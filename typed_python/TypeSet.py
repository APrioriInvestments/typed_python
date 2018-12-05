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

_builtin_value_to_name = {v:k for k,v in __builtins__.items() if isinstance(v,type)}
_builtin_name_to_value = {k:v for k,v in __builtins__.items() if isinstance(v,type)}


class TypeSet(object):
    """Represents a collection of types with well-specified names that we can use to serialize objects."""
    def __init__(self, nameToType=None, plugin=None):
        super().__init__()
        self.nameToType = nameToType or {}
        self.plugin = plugin
        self.typeToName = {v:k for k,v in self.nameToType.items()}

    def withPlugin(self, plugin):
        return TypeSet(self.nameToType, plugin)

    def nameForType(self, t):
        if t in _builtin_value_to_name:
            return _builtin_value_to_name[t]

        return self.typeToName.get(t)

    def typeFromName(self, name):
        if name in _builtin_name_to_value:
            return _builtin_name_to_value[name]

        return self.nameToType.get(name)

    def serialize(self, instance, plugin=None):
        return serialize(object, instance, self.withPlugin(plugin))

    def deserialize(self, bytes, plugin=None):
        return deserialize(object, bytes, self.withPlugin(plugin))

    @staticmethod
    def fromModules(modules):
        nameToType = {}

        for modulename, module in modules.items():
            for membername, member in module.__dict__.items():
                if isinstance(member, type):
                    nameToType[modulename + "." + membername] = member

        return TypeSet(nameToType)

    def representationFor(self, inst):
        if self.plugin:
            return self.plugin.representationFor(inst)
        return inst

    def fromRepresentation(self, originalType, representation):
        return self.plugin.fromRepresentation(originalType, representation)

class SerializationPlugin(object):
    def representationFor(self, inst):
        """Return an alternative representation for an object.

        If the instance is returned, serialize it as is.

        The object will be memoized in the serialized stream, but if
        there is a sequence of objects that memoize to re
        """
        return inst

    def fromRepresentation(self, originalType, representation):
        """return an instance given an original type and a representation produced by the plugin.

        By default, there's no implementation because we never produce any alternative representations.
        """

        raise NotImplemented()