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
    cacheAstForCode,
)
from typed_python.hash import sha_hash
from typed_python.compiler.runtime_lock import runtimeLock
from typed_python.type_function import TypeFunction, reconstructTypeFunctionType, isTypeFunctionType
from types import FunctionType, ModuleType, CodeType, MethodType, BuiltinFunctionType
from _thread import LockType, RLock
import abc
import sys
import importlib
import threading
import types
import traceback
import logging


_badModuleCache = set()


def createFunctionWithLocalsAndGlobals(code, globals):
    if globals is None:
        globals = {}
    return _types.buildPyFunctionObject(code, globals, ())


def buildCodeObject(
    ast,
    co_argcount,
    co_kwonlyargcount,
    co_nlocals,
    co_stacksize,
    co_flags,
    co_code,
    co_consts,
    co_names,
    co_varnames,
    co_freevars,
    co_cellvars,
    co_filename,
    co_name,
    co_firstlineno,
    co_lnotab,
    co_posonlyargcount=0
):
    if sys.version_info.minor < 8 and co_posonlyargcount:
        raise Exception(
            "tried to deserialize a code object from a future version "
            "of python that uses positional-only arguments."
        )

    codeObj = types.CodeType(
        co_argcount,
        *([] if sys.version_info.minor < 8 else [co_posonlyargcount]),
        co_kwonlyargcount,
        co_nlocals,
        co_stacksize,
        co_flags,
        co_code,
        co_consts,
        co_names,
        co_varnames,
        co_filename,
        co_name,
        co_firstlineno,
        co_lnotab,
        co_freevars,
        co_cellvars,
    )

    cacheAstForCode(codeObj, ast)

    return codeObj


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
    ".builtin.method_type": MethodType,
    ".builtin.function_type": FunctionType,
    ".builtin.ellipsis_type": type(...),
    ".builtin.ellipsis": ...,
    ".builtin.NotImplemented": NotImplemented
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
        serializeFunctionGlobalsAsIs=False,
        serializeHashSequence=False
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
        self.serializeFunctionGlobalsAsIs = serializeFunctionGlobalsAsIs
        self.serializeHashSequence = serializeHashSequence

    def addNamedObject(self, name, obj):
        self.nameToObjectOverride[name] = obj
        self.objectToNameOverride[id(obj)] = name

    def dropNamedObject(self, name):
        objId = id(self.nameToObjectOverride[name])

        del self.nameToObjectOverride[name]
        del self.objectToNameOverride[objId]

    def withFunctionGlobalsAsIs(self):
        """When serializing a function, don't replace its globals dict.

        By default, when we serialize a function we only serialize the globals
        it depends on. This prevents us from having to carry large parts of a codebase
        around for no reason, because most use-cases for serialization don't
        require us to be able to modify the object that the function's original
        globals were tied to.

        If, however, you want to retain the original structure of the function
        including its reference to its unnamed parent module, then you can
        modify the context in this way.
        """
        if self.serializeFunctionGlobalsAsIs:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=self.compressionEnabled,
            encodeLineInformationForCode=self.encodeLineInformationForCode,
            objectToNameOverride=self.objectToNameOverride,
            internalizeTypeGroups=self.internalizeTypeGroups,
            serializeFunctionGlobalsAsIs=True,
            serializeHashSequence=self.serializeHashSequence
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
            serializeFunctionGlobalsAsIs=self.serializeFunctionGlobalsAsIs,
            serializeHashSequence=self.serializeHashSequence
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
            serializeFunctionGlobalsAsIs=self.serializeFunctionGlobalsAsIs,
            serializeHashSequence=self.serializeHashSequence
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
            serializeFunctionGlobalsAsIs=self.serializeFunctionGlobalsAsIs,
            serializeHashSequence=self.serializeHashSequence
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
            serializeFunctionGlobalsAsIs=self.serializeFunctionGlobalsAsIs,
            serializeHashSequence=self.serializeHashSequence
        )

    def withSerializeHashSequence(self):
        if self.serializeHashSequence:
            return self

        return SerializationContext(
            nameToObjectOverride=self.nameToObjectOverride,
            compressionEnabled=self.compressionEnabled,
            encodeLineInformationForCode=self.encodeLineInformationForCode,
            objectToNameOverride=self.objectToNameOverride,
            internalizeTypeGroups=self.internalizeTypeGroups,
            serializeFunctionGlobalsAsIs=self.serializeFunctionGlobalsAsIs,
            serializeHashSequence=True
        )

    def nameForObject(self, t):
        ''' Return a name(string) for an input object t, or None if not found. '''
        if id(t) in self.objectToNameOverride:
            return self.objectToNameOverride[id(t)]

        if isinstance(t, type) and hasattr(t, '__typed_python_category__') and t.__typed_python_category__ == 'ConcreteAlternative':
            tName = self.nameForObject(t.Alternative)
            if tName is not None:
                return ".alt." + tName + ":" + str(t.Index)

        if isinstance(t, type) and issubclass(t, TypeFunction):
            if len(t.MRO) == 2:
                name = t.__module__ + "." + t.__name__
                if self.objectFromName(name, allowImport=False) is t:
                    return name
                return None

        if isinstance(t, dict) and '__name__' in t and isinstance(t['__name__'], str):
            maybeModule = self.objectFromName('.modules.' + t['__name__'], allowImport=False)
            if maybeModule is not None and t is getattr(maybeModule, '__dict__', None):
                return ".module_dict." + t['__name__']

        if isinstance(t, (FunctionType, BuiltinFunctionType)) or isinstance(t, type) and issubclass(t, _types.Function):
            mname = getattr(t, "__module__", t.__module__)
            qualname = getattr(t, "__qualname__", t.__qualname__)
            fname = t.__name__

            if mname is not None:
                if fname == qualname:
                    name = mname + "." + fname
                else:
                    name = '.fun_in_class.' + mname + "." + qualname

                o = self.objectFromName(name, allowImport=False)

                if o is t:
                    return name

                if type(o) is t:
                    return ".typeof." + name

        elif isinstance(t, type) and issubclass(t, Alternative):
            mname = t.__module__
            fname = t.__name__

            if self.objectFromName(mname + "." + fname, allowImport=False) is t:
                return mname + "." + fname

        elif isinstance(t, ModuleType):
            if self.objectFromName(".modules." + t.__name__, allowImport=False) is t:
                return ".modules." + t.__name__

        elif isinstance(t, type):
            fname = t.__name__

            if getattr(t, "__typed_python_category__", None) == 'Value':
                return None

            if hasattr(t, "__module__"):
                mname = getattr(t, "__module__", None)
                if self.objectFromName(mname + "." + fname, allowImport=False) is t:
                    return mname + "." + fname

        return None

    def objectFromName(self, name, allowImport=True):
        """Return an object for an input name(string), or None if not found.

        If allowImport is False, then don't import any new modules. This is used
        when we are checking if an object is a given named object. If we have
        a reference to an object, then we clearly don't need to import the module
        that contains it to see if it has that name.
        """
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
                with runtimeLock:
                    if moduleName in sys.modules:
                        module = sys.modules[moduleName]
                    else:
                        if moduleName in _badModuleCache or not allowImport:
                            return None

                        module = importlib.import_module(moduleName)
            except ModuleNotFoundError:
                _badModuleCache.add(moduleName)
                return None
            except ImportError:
                _badModuleCache.add(moduleName)
                logging.error("Failed to import module %s:\n%s", moduleName, traceback.format_exc())
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
                with runtimeLock:
                    if name[9:] in _badModuleCache:
                        return None

                    if name[9:] in sys.modules:
                        return sys.modules[name[9:]]

                    if not allowImport:
                        return None

                    return importlib.import_module(name[9:])
            except ModuleNotFoundError:
                _badModuleCache.add(name[9:])
                return None
            except ImportError:
                _badModuleCache.add(name[9:])
                logging.error("Failed to import module %s:\n%s", name[9:], traceback.format_exc())
                return None

        names = name.rsplit(".", 1)

        if len(names) != 2:
            return None

        moduleName, objName = name.rsplit(".", 1)

        try:
            if moduleName in _badModuleCache:
                return None

            with runtimeLock:
                if moduleName in sys.modules:
                    module = sys.modules[moduleName]
                else:
                    if not allowImport:
                        return None

                    module = importlib.import_module(moduleName)

            return getattr(module, objName, None)
        except ModuleNotFoundError:
            _badModuleCache.add(moduleName)
            return None
        except ImportError:
            _badModuleCache.add(moduleName)
            logging.error("Failed to import module %s:\n%s", moduleName, traceback.format_exc())
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
        if type(inst) in (tuple, list, dict, set, str, int, bool, float):
            return None

        if isinstance(inst, CellType):
            return None

        if isinstance(inst, types.ModuleType) and inst is not sys.modules.get(inst.__name__):
            # this is an 'unnamed' module
            return (types.ModuleType, (inst.__name__,), inst.__dict__)

        if isinstance(inst, LockType):
            return (threading.Lock, ())

        if isinstance(inst, RLock):
            return (threading.RLock, ())

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
                typeConstructorNamespace = {}

                for name, memb in inst.__dict__.items():
                    getset_descriptor = type(type.__dict__['__doc__'])
                    wrapper_descriptor = type(object.__repr__)

                    if (
                        name not in ("__dict__", '__slotnames__')
                        and not isinstance(memb, (wrapper_descriptor, getset_descriptor))
                    ):
                        classMembers[name] = memb

                # filter out weird class members introduced by 'abc'
                if issubclass(inst, abc.ABC):
                    classMembers = {k: v for k, v in classMembers.items() if not k.startswith("_abc")}

                return (type(inst), (inst.__name__, inst.__bases__, typeConstructorNamespace), classMembers)

        if isinstance(inst, property):
            return (property, (None, None, None), (inst.fget, inst.fset, inst.fdel))

        if isinstance(inst, staticmethod):
            return (staticmethod, (None,), inst.__func__)

        if isinstance(inst, MethodType):
            # create a 'dummy' method which we'll fill out below
            return (MethodType, (lambda: None, 0), (inst.__self__, inst.__func__))

        if isinstance(inst, classmethod):
            return (classmethod, (None,), inst.__func__)

        if isinstance(inst, CodeType):
            pyast = convertFunctionToAlgebraicPyAst(
                inst,
                keepLineInformation=self.encodeLineInformationForCode
            )

            return (
                buildCodeObject,
                (
                    pyast,
                    inst.co_argcount,
                    inst.co_kwonlyargcount,
                    inst.co_nlocals,
                    inst.co_stacksize,
                    inst.co_flags,
                    inst.co_code,
                    inst.co_consts,
                    inst.co_names,
                    inst.co_varnames,
                    inst.co_freevars,
                    inst.co_cellvars,
                    inst.co_filename if self.encodeLineInformationForCode else "",
                    inst.co_name,
                    inst.co_firstlineno if self.encodeLineInformationForCode else 0,
                    inst.co_lnotab if sys.version_info.minor < 10 else inst.co_linetable
                ) + (
                    () if sys.version_info.minor < 8 or inst.co_posonlyargcount == 0 else
                    (inst.co_posonlyargcount,)
                ),
                {}
            )

        if isinstance(inst, FunctionType):
            representation = {}
            representation["qualname"] = inst.__qualname__
            representation["name"] = inst.__name__
            if self.encodeLineInformationForCode:
                representation["module"] = inst.__module__
            else:
                representation["module"] = ""

            representation["annotations"] = inst.__annotations__
            representation["defaults"] = inst.__defaults__
            representation["kwdefaults"] = inst.__kwdefaults__
            representation["closure"] = inst.__closure__

            if hasattr(inst, '__isabstractmethod__'):
                representation['__isabstractmethod__'] = inst.__isabstractmethod__

            globalsToUse = None

            if self.nameForObject(inst.__globals__) is not None or self.serializeFunctionGlobalsAsIs:
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

                # build the set of names that are actually used.
                # clients are allowed to put references to submodules (e.g. "lxml.etree")
                # indicating that if we use 'lxml' we also need to import 'lxml.etree'
                representation['globals'] = {
                    k: v for k, v in sorted(inst.__globals__.items()) if k.split(".")[0] in all_names
                }

            args = (
                inst.__code__,
                globalsToUse
            )

            return (createFunctionWithLocalsAndGlobals, args, representation)

        if not isinstance(inst, type) and hasattr(type(inst), '__reduce_ex__'):
            res = inst.__reduce_ex__(4)

            # pickle supports a protocol where __reduce__ can return a string
            # giving a global name. We'll already find that separately, so we
            # don't want to handle it here. We ought to look at this in more detail
            # however
            if isinstance(res, str):
                return None

            return res

        if not isinstance(inst, type) and hasattr(type(inst), '__reduce__'):
            res = inst.__reduce__()

            # pickle supports a protocol where __reduce__ can return a string
            # giving a global name. We'll already find that separately, so we
            # don't want to handle it here. We ought to look at this in more detail
            # however
            if isinstance(res, str):
                return None

            return res

        return None

    def setInstanceStateFromRepresentation(
        self, instance, representation=None, itemIt=None, kvPairIt=None, setStateFun=None
    ):
        if representation is reconstructTypeFunctionType:
            return

        if isinstance(instance, types.MethodType):
            _types.setMethodObjectInternals(instance, representation[0], representation[1])
            return

        if isinstance(instance, (LockType, RLock)):
            return

        if isinstance(instance, types.ModuleType):
            # note that we can't simply copy the contents of the dict into
            # the module dict because functions will have references to the
            # dict itself as their 'globals', and copying directly would break
            # that relationship.
            _types.setModuleDict(instance, representation)
            return

        if isinstance(instance, property):
            _types.setPropertyGetSetDel(instance, representation[0], representation[1], representation[2])
            return

        if isinstance(instance, staticmethod):
            _types.setClassOrStaticmethod(instance, representation)
            return

        if isinstance(instance, classmethod):
            _types.setClassOrStaticmethod(instance, representation)
            return

        if isinstance(instance, CodeType):
            return

        if isinstance(instance, FunctionType):
            instance.__name__ = representation['name']
            instance.__qualname__ = representation['qualname']
            instance.__module__ = representation['module']
            instance.__annotations__ = representation.get('annotations', {})
            instance.__kwdefaults__ = representation.get('kwdefaults', {})
            instance.__defaults__ = representation.get('defaults', ())

            if '__isabstractmethod__' in representation:
                instance.__isabstractmethod__ = representation['__isabstractmethod__']

            if 'globals' in representation:
                _types.setFunctionGlobals(instance, representation['globals'])

            _types.setFunctionClosure(instance, representation['closure'])

            return

        if isinstance(instance, type):
            # set class members
            if representation is not None:
                for k, v in representation.items():
                    setattr(instance, k, v)
            return

        if isinstance(instance, ModuleType):
            return

        if setStateFun is not None:
            setStateFun(instance, representation)
        elif hasattr(type(instance), '__setstate__') and representation is not None:
            type(instance).__setstate__(instance, representation)
        elif representation is not None:
            instance.__dict__.update(representation)

        if itemIt:
            instance.extend(itemIt)

        if kvPairIt:
            for key, val in kvPairIt:
                instance[key] = val
