#   Copyright 2017-2020 typed_python Authors
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
from typed_python.type_function import ConcreteTypeFunction, reconstructTypeFunctionType, isTypeFunctionType
from types import FunctionType, ModuleType, CodeType, BuiltinFunctionType
from _thread import LockType, RLock
import numpy
import sys
import datetime
import lz4.frame
import importlib


_badModuleCache = set()


def createEmptyFunction(ast):
    return evaluateFunctionPyAst(ast, stripAnnotations=True)


def createFunctionWithLocalsAndGlobals(code, globals):
    if globals is None:
        globals = {}
    return _types.buildPyFunctionObject(code, globals, ())


def astToCodeObject(ast, freevars):
    return evaluateFunctionDefWithLocalsInCells(
        ast,
        globals={},
        locals={var: None for var in freevars},
        stripAnnotations=True
    ).__code__


def capture(x):
    return lambda: x


CellType = type(capture(10).__closure__[0])


DEFAULT_NAME_TO_OVERRIDE = {
    ".builtin.lock": LockType,
    ".builtin.rlock": RLock,
    ".builtin.cell": CellType,
    ".builtin.builtin_function_or_method": BuiltinFunctionType,
    ".builtin.code_type": CodeType,
    ".builtin.module_type": ModuleType,
    ".builtin.function_type": FunctionType,
}


class SerializationContext:
    """Represents a collection of types with well-specified names that we can use to serialize objects."""
    def __init__(
        self,
        nameToObjectOverride=None,
        compressionEnabled=True,
        encodeLineInformationForCode=True,
        objectToNameOverride=None,
        internalizeTypeGroups=True,
        serializeFunctionsAsNonAst=False
    ):
        super().__init__()

        self.nameToObjectOverride = dict(nameToObjectOverride or DEFAULT_NAME_TO_OVERRIDE)
        self.objectToNameOverride = (
            dict(objectToNameOverride)
            if objectToNameOverride is not None else
            {id(v): n for n, v in self.nameToObjectOverride.items()}
        )
        self.compressionEnabled = compressionEnabled
        self.encodeLineInformationForCode = encodeLineInformationForCode
        self.internalizeTypeGroups = internalizeTypeGroups
        self.serializeFunctionsAsNonAst = serializeFunctionsAsNonAst

    def addNamedObject(self, name, obj):
        self.nameToObjectOverride[name] = obj
        self.objectToNameOverride[id(obj)] = name

    def dropNamedObject(self, name):
        objId = id(self.nameToObjectOverride[name])

        del self.nameToObjectOverride[name]
        del self.objectToNameOverride[objId]

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

    def withoutFunctionsSerializedAsNonAst(self):
        """Just serialize a function's code, not its AST (from source).

        this means we can't deserialize it, but is good for hashing.
        """
        if self.serializeFunctionsAsNonAst:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=self.compressionEnabled,
            encodeLineInformationForCode=self.encodeLineInformationForCode,
            objectToNameOverride=self.objectToNameOverride,
            internalizeTypeGroups=self.internalizeTypeGroups,
            serializeFunctionsAsNonAst=True
        )

    def withoutInternalizingTypeGroups(self):
        """Make sure we fully deserialize types.

        This means we don't place them into the main type memo
        by identityHash, which is really only useful for testing.
        """
        if not self.internalizeTypeGroups:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=self.compressionEnabled,
            encodeLineInformationForCode=self.encodeLineInformationForCode,
            objectToNameOverride=self.objectToNameOverride,
            internalizeTypeGroups=False,
            serializeFunctionsAsNonAst=self.serializeFunctionsAsNonAst
        )

    def withoutLineInfoEncoded(self):
        if not self.encodeLineInformationForCode:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=self.compressionEnabled,
            encodeLineInformationForCode=False,
            objectToNameOverride=self.objectToNameOverride,
            internalizeTypeGroups=self.internalizeTypeGroups,
            serializeFunctionsAsNonAst=self.serializeFunctionsAsNonAst
        )

    def withoutCompression(self):
        if not self.compressionEnabled:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=False,
            encodeLineInformationForCode=self.encodeLineInformationForCode,
            objectToNameOverride=self.objectToNameOverride,
            internalizeTypeGroups=self.internalizeTypeGroups,
            serializeFunctionsAsNonAst=self.serializeFunctionsAsNonAst
        )

    def withCompression(self):
        if self.compressionEnabled:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=True,
            encodeLineInformationForCode=self.encodeLineInformationForCode,
            objectToNameOverride=self.objectToNameOverride,
            internalizeTypeGroups=self.internalizeTypeGroups,
            serializeFunctionsAsNonAst=self.serializeFunctionsAsNonAst
        )

    def nameForObject(self, t):
        ''' Return a name(string) for an input object t, or None if not found. '''
        if id(t) in self.objectToNameOverride:
            return self.objectToNameOverride[id(t)]

        if isinstance(t, type) and hasattr(t, '__typed_python_category__') and t.__typed_python_category__ == 'ConcreteAlternative':
            tName = self.nameForObject(t.Alternative)
            if tName is not None:
                return ".alt." + tName + ":" + str(t.Index)

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

            if getattr(t, "__typed_python_category__", None) == 'Value':
                return None

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

        if name.startswith(".alt."):
            altName, altIndex = name[5:].rsplit(":")
            try:
                altIndex = int(altIndex)
            except ValueError:
                return None

            alt = self.objectFromName(altName)
            if alt is None:
                return None

            return alt.__typed_python_alternatives__[altIndex]

        if name.startswith(".fun_in_class."):
            items = name[14:].split(".")
            if len(items) < 3:
                return

            fname = items[-1]
            classname = items[-2]
            moduleName = ".".join(items[:-2])

            try:
                if moduleName in sys.modules:
                    module = sys.modules[moduleName]
                else:
                    if moduleName in _badModuleCache:
                        return None

                    module = importlib.import_module(moduleName)
            except ImportError:
                _badModuleCache.add(moduleName)
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
                if name[9:] in _badModuleCache:
                    return None

                if name[9:] in sys.modules:
                    return sys.modules[name[9:]]
                return importlib.import_module(name[9:])
            except ImportError:
                _badModuleCache.add(name[9:])
                return None

        names = name.rsplit(".", 1)

        if len(names) != 2:
            return None

        moduleName, objName = name.rsplit(".", 1)

        try:
            if moduleName in _badModuleCache:
                return None

            if moduleName in sys.modules:
                module = sys.modules[moduleName]
            else:
                module = importlib.import_module(moduleName)

            return getattr(module, objName, None)

        except ImportError:
            _badModuleCache.add(moduleName)
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
            # only serialize Class and Alternative objects from type functions.
            # otherwise, we'll end up changing how we serialize things like 'int',
            # if they ever make their way into a type function.
            if getattr(inst, '__typed_python_category__', None) in ('Class', 'Alternative'):
                funcArgsAndKwargs = isTypeFunctionType(inst)

                if funcArgsAndKwargs is not None:
                    typeFunc = funcArgsAndKwargs[0]
                    # we can't use this methodology for type funcs that are not named because
                    # they have a memo inside them
                    if self.nameForObject(typeFunc) is not None:
                        return (reconstructTypeFunctionType, funcArgsAndKwargs, reconstructTypeFunctionType)

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
            return (property, (None, None, None), (inst.fget, inst.fset, inst.fdel))

        if isinstance(inst, staticmethod):
            return (staticmethod, (None,), inst.__func__)

        if isinstance(inst, classmethod):
            return (classmethod, (None,), inst.__func__)

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
            if self.serializeFunctionsAsNonAst:
                # serialize only enough to hash this
                return (
                    astToCodeObject,
                    (inst.co_freevars, inst.co_code, inst.co_names, inst.co_consts, inst.co_varnames,
                     inst.co_filename if self.encodeLineInformationForCode else None,
                     inst.co_firstlineno if self.encodeLineInformationForCode else None),
                    None
                )
            else:
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
            representation["closure"] = inst.__closure__

            globalsToUse = None

            if self.nameForObject(inst.__globals__) is not None:
                globalsToUse = inst.__globals__
            else:
                globalsToUse = {}

                all_names = set(['__builtins__'])

                def walkCodeObject(code):
                    all_names.update(code.co_names)
                    # there are 'code' objects for embedded list comprehensions.
                    for c in code.co_consts:
                        if type(c) is type(code):
                            walkCodeObject(c)

                walkCodeObject(inst.__code__)

                representation['globals'] = {k: v for k, v in inst.__globals__.items() if k in all_names}

            args = (
                inst.__code__,
                globalsToUse
            )

            return (createFunctionWithLocalsAndGlobals, args, representation)

        return None

    def setInstanceStateFromRepresentation(self, instance, representation):
        if representation is reconstructTypeFunctionType:
            return True

        if isinstance(instance, property):
            _types.setPropertyGetSetDel(instance, representation[0], representation[1], representation[2])
            return True

        if isinstance(instance, staticmethod):
            _types.setClassOrStaticmethod(instance, representation)
            return True

        if isinstance(instance, classmethod):
            _types.setClassOrStaticmethod(instance, representation)
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

            if 'globals' in representation:
                instance.__globals__.update(representation['globals'])

            _types.setFunctionClosure(instance, representation['closure'])

            return True

        if isinstance(instance, type):
            if representation is not None:
                for k, v in representation.items():
                    setattr(instance, k, v)
            return True

        if isinstance(instance, ModuleType):
            return True

        return False
