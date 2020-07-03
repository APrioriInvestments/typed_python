#   Copyright 2017-2019 typed_python Authors
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

from typed_python._types import serialize, deserialize, Type, Alternative, NamedTuple
from typed_python import _types

from typed_python.python_ast import (
    convertFunctionToAlgebraicPyAst,
    evaluateFunctionPyAst,
    evaluateFunctionDefWithLocalsInCells
)
from typed_python.hash import sha_hash
from typed_python.type_function import ConcreteTypeFunction
from types import FunctionType, ModuleType, CodeType, BuiltinFunctionType
from _thread import LockType, RLock
import numpy
import datetime
import lz4.frame
import importlib


def createEmptyFunction(ast):
    return evaluateFunctionPyAst(ast, stripAnnotations=True)


def createFunctionWithLocalsAndGlobals(code, globals, closure):
    if globals is None:
        globals = {}
    if closure is None:
        closure = ()
    return _types.buildPyFunctionObject(code, globals, closure)


def astToCodeObject(ast, freevars):
    return evaluateFunctionDefWithLocalsInCells(
        ast,
        globals={},
        locals={var: None for var in freevars},
        stripAnnotations=True
    ).__code__


DEFAUL_TNAME_TO_OVERRIDE = {
    ".builtin.lock": LockType,
    ".builtin.rlock": RLock,
}


class SerializationContext(object):
    """Represents a collection of types with well-specified names that we can use to serialize objects."""
    def __init__(self, nameToObjectOverride=None, compressionEnabled=True, encodeLineInformationForCode=True, objectToNameOverride=None):
        super().__init__()

        self.nameToObjectOverride = dict(nameToObjectOverride or DEFAUL_TNAME_TO_OVERRIDE)
        self.objectToNameOverride = (
            dict(objectToNameOverride)
            if objectToNameOverride is not None else
            {id(v): n for n, v in self.nameToObjectOverride.items()}
        )
        self.compressionEnabled = compressionEnabled
        self.encodeLineInformationForCode = encodeLineInformationForCode

    def compress(self, bytes):
        if self.compressionEnabled:
            res = lz4.frame.compress(bytes)
            return res
        else:
            return bytes

    def decompress(self, bytes):
        if self.compressionEnabled:
            return lz4.frame.decompress(bytes)
        else:
            return bytes

    def withoutLineInfoEncoded(self):
        if not self.encodeLineInformationForCode:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=self.compressionEnabled,
            encodeLineInformationForCode=False,
            objectToNameOverride=self.objectToNameOverride
        )

    def withoutCompression(self):
        if not self.compressionEnabled:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=False,
            encodeLineInformationForCode=self.encodeLineInformationForCode,
            objectToNameOverride=self.objectToNameOverride
        )

    def withCompression(self):
        if self.compressionEnabled:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=True,
            encodeLineInformationForCode=self.encodeLineInformationForCode,
            objectToNameOverride=self.objectToNameOverride
        )

    def nameForObject(self, t):
        ''' Return a name(string) for an input object t, or None if not found. '''
        if id(t) in self.objectToNameOverride:
            return self.objectToNameOverride[id(t)]

        if isinstance(t, ConcreteTypeFunction):
            name = t._concreteTypeFunction.__module__ + "." + t._concreteTypeFunction.__name__
            if self.objectFromName(name) is t:
                return name
            return None

        if isinstance(t, dict) and '__name__' in t and isinstance(t['__name__'], str):
            maybeModule = self.objectFromName('.modules.' + t['__name__'])
            if maybeModule is not None and t is getattr(maybeModule, '__dict__', None):
                return ".module_dict." + t['__name__']

        if isinstance(t, (FunctionType, BuiltinFunctionType)) or isinstance(t, type) and issubclass(t, _types.Function):
            mname = getattr(t, "__typed_python_module__", t.__module__)
            qualname = getattr(t, "__typed_python_qualname__", t.__qualname__)
            fname = t.__name__

            if mname is not None:
                if fname == qualname:
                    name = mname + "." + fname
                else:
                    name = '.fun_in_class.' + mname + "." + qualname

                o = self.objectFromName(name)

                if o is t:
                    return name

                if type(o) is t:
                    return ".typeof." + name

        elif isinstance(t, type) and issubclass(t, Alternative):
            mname = t.__typed_python_module__
            fname = t.__name__

            if self.objectFromName(mname + "." + fname) is t:
                return mname + "." + fname

        elif isinstance(t, ModuleType):
            if self.objectFromName(".modules." + t.__name__) is t:
                return ".modules." + t.__name__

        elif isinstance(t, type):
            fname = t.__name__

            if hasattr(t, "__typed_python_module__"):
                mname = getattr(t, "__typed_python_module__", None)
                if self.objectFromName(mname + "." + fname) is t:
                    return mname + "." + fname

            if hasattr(t, "__module__"):
                mname = getattr(t, "__module__", None)
                if self.objectFromName(mname + "." + fname) is t:
                    return mname + "." + fname

        return None

    def objectFromName(self, name):
        ''' Return an object for an input name(string), or None if not found. '''
        if name in self.nameToObjectOverride:
            return self.nameToObjectOverride[name]

        if name.startswith(".typeof."):
            res = self.objectFromName(name[8:])
            if res is not None:
                return type(res)
            return None

        if name.startswith(".fun_in_class."):
            items = name[14:].split(".")
            if len(items) < 3:
                return

            fname = items[-1]
            classname = items[-2]
            moduleName = ".".join(items[:-2])

            try:
                module = importlib.import_module(moduleName)
            except ImportError:
                return None

            clsObj = getattr(module, classname, None)

            if clsObj is None:
                return None

            return getattr(clsObj, fname, None)

        if name.startswith(".module_dict."):
            res = self.objectFromName(".modules." + name[13:])

            if res is None:
                return None

            return res.__dict__

        if name.startswith(".modules."):
            try:
                return importlib.import_module(name[9:])
            except ImportError:
                return None

        names = name.rsplit(".", 1)

        if len(names) != 2:
            return None

        module, objName = name.rsplit(".", 1)

        try:
            return getattr(importlib.import_module(module), objName, None)
        except ImportError:
            return None

    def sha_hash(self, o):
        return sha_hash(self.serialize(o))

    def serialize(self, instance, serializeType=object):
        return serialize(serializeType, instance, self)

    def deserialize(self, bytes, serializeType=object):
        return deserialize(serializeType, bytes, self)

    def representationFor(self, inst):
        ''' Return the representation of a given instance or None.

            For certain types, we want to special-case how they are serialized.
            For those types, we return a representation object and for other
            types we return None.

            The representation consists of a tuple (factory, args, representation).

            During reconstruction, we call factory(*args) to produce an emnpty
            'skeleton' object, and then call `setInstanceStateFromRepresentation`
            with the resulting object and the 'representation'. The values returned
            for  'factory' and 'args' may not have circular dependencies with the current
            object - we deserialize those first, call factory(*args) to get the
            resulting object, and that object gets returned to any objects inside of
            'representation' that have references to the original object.

            @param inst: an instance to be serialized
            @return a representation object or None
        '''
        if isinstance(inst, type):
            if (
                # is this a regular python class
                not issubclass(inst, Type)
                # or is it a subclass of a NamedTuple
                or (issubclass(inst, NamedTuple) and inst.__bases__[0] != NamedTuple)
            ):
                # this is a regular class
                classMembers = {}

                for name, memb in inst.__dict__.items():
                    getset_descriptor = type(type.__dict__['__doc__'])
                    wrapper_descriptor = type(object.__repr__)

                    if name != "__dict__" and not isinstance(memb, (wrapper_descriptor, getset_descriptor)):
                        classMembers[name] = memb

                return (type, (inst.__name__, inst.__bases__, {}), classMembers)

        if isinstance(inst, property):
            return (property, (inst.fget, inst.fset, inst.fdel), None)

        if isinstance(inst, staticmethod):
            return (staticmethod, (inst.__func__,), None)

        if isinstance(inst, classmethod):
            return (classmethod, (inst.__func__,), None)

        if isinstance(inst, numpy.ndarray):
            return inst.__reduce__()

        if isinstance(inst, numpy.number):
            return inst.__reduce__() + (None,)

        if isinstance(inst, numpy.dtype):
            return (numpy.dtype, (str(inst),), None)

        if isinstance(inst, datetime.datetime):
            return inst.__reduce__() + (None,)

        if isinstance(inst, datetime.date):
            return inst.__reduce__() + (None,)

        if isinstance(inst, datetime.time):
            return inst.__reduce__() + (None,)

        if isinstance(inst, datetime.timedelta):
            return inst.__reduce__() + (None,)

        if isinstance(inst, datetime.tzinfo):
            return inst.__reduce__() + (None,)

        if isinstance(inst, CodeType):
            pyast = convertFunctionToAlgebraicPyAst(inst, keepLineInformation=self.encodeLineInformationForCode)

            return (astToCodeObject, (pyast, inst.co_freevars), {})

        if isinstance(inst, FunctionType):
            representation = {}
            representation["qualname"] = inst.__qualname__
            representation["name"] = inst.__name__
            representation["module"] = inst.__module__
            representation["annotations"] = inst.__annotations__
            representation["defaults"] = inst.__defaults__
            representation["kwdefaults"] = inst.__kwdefaults__

            globalsToUse = None

            if self.nameForObject(inst.__globals__) is not None:
                globalsToUse = inst.__globals__
            else:
                all_names = set()

                def walkCodeObject(code):
                    all_names.update(code.co_names)
                    # there are 'code' objects for embedded list comprehensions.
                    for c in code.co_consts:
                        if type(c) is type(code):
                            walkCodeObject(c)

                walkCodeObject(inst.__code__)

                globalsToUse = {k: v for k, v in inst.__globals__.items() if k in all_names}

            localsInCells = {}
            for ix, x in enumerate(inst.__code__.co_freevars):
                localsInCells[x] = inst.__closure__[ix].cell_contents

            args = (
                inst.__code__,
                globalsToUse,
                inst.__closure__
            )

            return (createFunctionWithLocalsAndGlobals, args, representation)

        return None

    def setInstanceStateFromRepresentation(self, instance, representation):
        if isinstance(instance, property):
            return True

        if isinstance(instance, staticmethod):
            return True

        if isinstance(instance, classmethod):
            return True

        if isinstance(instance, CodeType):
            return True

        if isinstance(instance, datetime.datetime):
            return True

        if isinstance(instance, datetime.date):
            return True

        if isinstance(instance, datetime.time):
            return True

        if isinstance(instance, datetime.timedelta):
            return True

        if isinstance(instance, datetime.tzinfo):
            return True

        if isinstance(instance, numpy.dtype):
            return True

        if isinstance(instance, numpy.number):
            return True

        if isinstance(instance, numpy.ndarray):
            instance.__setstate__(representation)
            return True

        if isinstance(instance, FunctionType):
            instance.__name__ = representation['name']
            instance.__qualname__ = representation['qualname']
            instance.__module__ = representation['module']
            instance.__annotations__ = representation.get('annotations', {})
            instance.__kwdefaults__ = representation.get('kwdefaults', {})
            instance.__defaults__ = representation.get('defaults', ())

            return True

        if isinstance(instance, type):
            if representation is not None:
                for k, v in representation.items():
                    setattr(instance, k, v)
            return True

        if isinstance(instance, ModuleType):
            return True

        return False
