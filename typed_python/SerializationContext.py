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
from typed_python.hash import sha_hash
from types import FunctionType
import numpy
import lz4.frame

_reconstruct = numpy.array([1,2,3]).__reduce__()[0]
_ndarray = numpy.ndarray

def createEmptyFunction(ast):
    return evaluateFunctionPyAst(ast)

_builtin_name_to_value = {
    ".builtin." + k: v for k, v in __builtins__.items()
    if isinstance(v, type) or 'builtin_function_or_method' in str(type(v))
}
_builtin_name_to_value[".builtin.createEmptyFunction"] = createEmptyFunction
_builtin_name_to_value[".builtin._reconstruct"] = _reconstruct
_builtin_name_to_value[".builtin._ndarray"] = _ndarray
_builtin_name_to_value[".builtin.dtype"] = numpy.dtype
_builtin_name_to_value[".ast.Expr.Lambda"] = Expr.Lambda
_builtin_name_to_value[".ast.Statement.FunctionDef"] = Statement.FunctionDef

_builtin_value_to_name = {id(v):k for k,v in _builtin_name_to_value.items()}


class SerializationContext(object):
    """Represents a collection of types with well-specified names that we can use to serialize objects."""
    def __init__(self, nameToObject=None):
        super().__init__()

        self.nameToObject = nameToObject or {}
        self.objToName = {}

        for k in self.nameToObject:
            assert isinstance(k, str), (
                "nameToObject keys must be strings (This one was not: {})"
                .format(k)
            )

        assert '' not in self.nameToObject, (
            "Empty object/type name not allowed: {}"
            .format(self.nameToObject[''])
        )

        # take the lexically lowest name, so that we're not dependent on ordering.
        for k,v in self.nameToObject.items():
            if id(v) not in self.objToName or k < self.objToName[id(v)]:
                self.objToName[id(v)] = k

        self.numpyCompressionEnabled = True
        self.encodeLineInformationForCode = True

    def withoutLineInfoEncoded(self):
        res = SerializationContext(self.nameToObject)
        res.encodeLineInformationForCode = False
        return res

    def nameForObject(self, t):
        ''' Return a name(string) for an input object t, or None if not found. '''
        tid = id(t)
        res = self.objToName.get(tid)

        if res is not None:
            return res
        else:
            return _builtin_value_to_name.get(tid)

    def objectFromName(self, name):
        ''' Return an object for an input name(string), or None if not found. '''
        res = self.nameToObject.get(name)

        if res is not None:
            return res
        else:
            return _builtin_name_to_value.get(name)

    def sha_hash(self, o):
        return sha_hash(self.serialize(o))

    def serialize(self, instance):
        return serialize(object, instance, self)

    def deserialize(self, bytes):
        return deserialize(object, bytes, self)

    def representationFor(self, inst):
        ''' Return the representation of a given instance or None.

            For certain types, we want to special-case how they are serialized.
            For those types, we return a representation object and for other
            types we return None.

            @param inst: an instance to be serialized
            @return a representation object or None
        '''
        if isinstance(inst, numpy.ndarray):
            result = inst.__reduce__()
            #compress the numpy data
            if self.numpyCompressionEnabled:
                result = (result[0], result[1], result[2][:-1] + (lz4.frame.compress(result[2][-1]),))
            return result

        if isinstance(inst, numpy.dtype):
            return (numpy.dtype, (str(inst),), None)

        if isinstance(inst, FunctionType):
            representation = {}
            representation["qualname"] = inst.__qualname__
            representation["name"] = inst.__name__
            representation["module"] = inst.__module__
            representation["freevars"] = {k:v for k,v in inst.__globals__.items() if k in inst.__code__.co_names}

            for ix, x in enumerate(inst.__code__.co_freevars):
                representation["freevars"][x] = inst.__closure__[ix].cell_contents

            args = (convertFunctionToAlgebraicPyAst(inst, keepLineInformation=self.encodeLineInformationForCode),)

            return (createEmptyFunction, args, representation)

        return None

    def setInstanceStateFromRepresentation(self, instance, representation):
        if isinstance(instance, numpy.dtype):
            return True

        if isinstance(instance, _ndarray):
            if self.numpyCompressionEnabled:
                representation = representation[:-1] + (lz4.frame.decompress(representation[-1]),)
            instance.__setstate__(representation)
            return True

        if isinstance(instance, FunctionType):
            instance.__globals__.update(representation['freevars'])
            instance.__name__ = representation['name']
            instance.__qualname__ = representation['qualname']

            return True

        return False

