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
from typed_python.python_ast import convertFunctionToAlgebraicPyAst, evaluateFunctionPyAst, Expr, Statement
from types import FunctionType

def createEmptyFunction(ast):
    return evaluateFunctionPyAst(ast)

_builtin_name_to_value = {".builtin." + k:v for k,v in __builtins__.items() if isinstance(v,type)}
_builtin_name_to_value[".builtin.createEmptyFunction"] = createEmptyFunction
_builtin_name_to_value[".ast.Expr.Lambda"] = Expr.Lambda
_builtin_name_to_value[".ast.Statement.FunctionDef"] = Statement.FunctionDef

_builtin_value_to_name = {id(v):k for k,v in _builtin_name_to_value.items()}

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
            rep = self.plugin.representationFor(inst)
            if rep is not None:
                return rep

        if isinstance(inst, FunctionType):
            print("Serializing ", inst)
            representation = {}
            representation["qualname"] = inst.__qualname__
            representation["name"] = inst.__name__
            representation["module"] = inst.__module__
            representation["freevars"] = {k:v for k,v in inst.__globals__.items() if k in inst.__code__.co_names}

            for ix, x in enumerate(inst.__code__.co_freevars):
                representation["freevars"][x] = inst.__closure__[ix].cell_contents

            args = (convertFunctionToAlgebraicPyAst(inst),)

            return (createEmptyFunction, args, representation)

        return None

    def setInstanceStateFromRepresentation(self, instance, representation):
        if self.plugin:
            if self.plugin.setInstanceStateFromRepresentation(instance, representation):
                return True

        if isinstance(instance, FunctionType):
            instance.__globals__.update(representation['freevars'])
            instance.__name__ = representation['name']
            instance.__qualname__ = representation['qualname']

            return True

        return False


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
        return False