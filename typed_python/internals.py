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

from types import FunctionType, ModuleType

import _thread
import threading

import typed_python
import inspect


# some 'types' (threading.Lock, threading.RLock) aren't really types, they're
# functions that produce some internal type. This contains the map from the
# factory to the type that we actually expect instances to hold.
_nonTypesAcceptedAsTypes = {threading.Lock: _thread.LockType, threading.RLock: _thread.RLock}


class CellAccess:
    "A singleton object for use in ClosureVariableBinding representation"
    pass


class UndefinedBehaviorException(BaseException):
    """An unsafe operation with known undefined behavior was performed.

    This Exception is deliberately not a subclass of Exception because by
    default it should not be caught by normal exception handlers. In compiled
    code, the operation that raised this exception is likely to segfault.
    """


# needed by the C api
object = object
type = type


class Final:
    """Mixin to make a class type 'Final'.

    Final classes can't be subclassed, but generate faster code because
    we don't have to look up method dispatch in the vtable.
    """

    pass


def closurePassingType(x):
    """Determine the type we'll use to represent 'x' in a closure."""
    if isinstance(x, (type, ModuleType, FunctionType)):
        return typed_python.Value(x)

    return type(x)


class Member:
    """Describe a data member of a Class object.

    A Class definition usually looks something like:

        class MyClass(Class):
            aMember = Member(int)
            anotherMember = Member(int)

    where 'Member' is this class.  Members always have a type which
    describes which kinds of values can go into the class. Subclasses
    are guaranteed to also have this type for this member name.

    Members may be marked 'nonempty' by passing the 'nonempty=True' flag.
    This means that they cannot be deleted as attributes, they are
    always initialized when the object is constructed, and attribute
    access will always succeed. This also improves performance of
    compiled code because we don't have to check a flag to see if the
    value is populated.
    """

    def __init__(self, t, default_value=None, nonempty=False):
        self._type = t
        self._default_value = default_value
        self._nonempty = nonempty

        if self._default_value is not None:
            assert isinstance(self._default_value, self._type)

    @property
    def type(self):
        if getattr(self._type, "__typed_python_category__", None) == "Forward":
            return self._type.get()
        return self._type

    @property
    def defaultValue(self):
        return self._default_value

    @property
    def isNonempty(self):
        return self._nonempty

    def __eq__(self, other):
        if not isinstance(other, Member):
            return False
        return self.type == other.type


class ClassMetaNamespace:
    def __init__(self):
        self.ns = {}
        self.order = []

    def __getitem__(self, k):
        return self.ns[k]

    def __setitem__(self, k, v):
        self.ns[k] = v
        self.order.append((k, v))

    def get(self, k, default):
        return self.ns.get(k, default)


magicMethodTypes = {
    "__init__": type(None),
    "__repr__": str,
    "__str__": str,
    "__bool__": bool,
    "__bytes__": bytes,
    "__contains__": bool,
    "__float__": float,
    "__int__": int,
    "__len__": int,
    "__lt__": bool,
    "__gt__": bool,
    "__le__": bool,
    "__ge__": bool,
    "__eq__": bool,
    "__ne__": bool,
    "__hash__": int,
    "__setattr__": type(None),
    "__delattr__": type(None),
    "__setitem__": type(None),
    "__delitem__": type(None),
}


def makeFunctionType(
    name, f, classname=None, ignoreAnnotations=False, assumeClosuresGlobal=False, returnTypeOverride=None
):
    if isinstance(f, typed_python._types.Function):
        if assumeClosuresGlobal:
            if typed_python.bytecount(type(f).ClosureType):
                # we need to build the equivalent function with global closures
                assert len(f.overloads) == 1, "Can't do this for multiple overloads yet"

                res = makeFunctionType(
                    name, f.extractPyFun(0), classname, ignoreAnnotations, assumeClosuresGlobal
                )

                assert typed_python.bytecount(res.ClosureType) == 0

                if f.isEntrypoint:
                    # reapply the entrypoint flag
                    res = type(res().withEntrypoint(True))

                if f.isNocompile:
                    # reapply the entrypoint flag
                    res = type(res().withNocompile(True))

                return res

        return type(f)

    if isinstance(f, type) and issubclass(f, typed_python._types.Function):
        return f

    spec = inspect.getfullargspec(f)

    def getAnn(argname):
        """ Return the annotated type for the given argument or None. """
        if ignoreAnnotations:
            return None

        if argname not in spec.annotations:
            return None
        else:
            ann = spec.annotations.get(argname)
            if ann is None:
                return type(None)
            else:
                return ann

    def getDefault(idx: int):
        """ Return the default value for a positional argument given its index. """
        if spec.defaults is not None:
            if idx >= len(spec.args) - len(spec.defaults):
                default = (spec.defaults[idx - (len(spec.args) - len(spec.defaults))],)
            else:
                default = None
        else:
            default = None

        return default

    arg_types = []
    for i, argname in enumerate(spec.args):
        default = getDefault(i)

        arg_types.append((argname, getAnn(argname), default, False, False))

    return_type = None

    if "return" in spec.annotations and not ignoreAnnotations:
        ann = spec.annotations.get("return")
        if ann is None:
            ann = type(None)
        return_type = ann

    if classname is not None and name in magicMethodTypes:
        tgtType = magicMethodTypes[name]

        if return_type is None:
            return_type = tgtType
        elif return_type != tgtType:
            raise Exception(f"{name} must return {tgtType.__name__}")

    if returnTypeOverride is not None:
        return_type = returnTypeOverride

    if spec.varargs is not None:
        arg_types.append((spec.varargs, getAnn(spec.varargs), None, True, False))

    if spec.kwonlyargs:
        raise Exception("Keyword only args not supported yet")
        # for arg in spec.kwonlyargs:
        #     arg_types.append((arg, getAnn(arg), (spec.kwonlydefaults.get(arg),), False, False))

    if spec.varkw is not None:
        arg_types.append((spec.varkw, getAnn(spec.varkw), None, False, True))

    if classname is not None:
        qualname = classname + "." + name
    else:
        qualname = name

    res = typed_python._types.Function(name, qualname, return_type, f, tuple(arg_types), assumeClosuresGlobal)

    return res


class ClassMetaclass(type):
    @classmethod
    def __prepare__(cls, *args, **kwargs):
        return ClassMetaNamespace()

    def __new__(cls, name, bases, namespace, **kwds):
        if not bases:
            return type.__new__(cls, name, bases, namespace.ns, **kwds)

        classCell = namespace.ns.get('__classcell__')

        members = []
        isFinal = Final in bases

        bases = [x for x in bases if x is not typed_python._types.Class and x is not Final]

        memberFunctions = {}
        staticFunctions = {}
        classMembers = {}
        classMethods = {}
        properties = {}

        if "__name__" in kwds:
            name = kwds["__name__"]

        for eltName, elt in namespace.order:
            if eltName == '__classcell__':
                pass
            elif eltName == '__name__':
                name = elt
            elif isinstance(elt, Member):
                members.append((eltName, elt._type, elt._default_value, elt._nonempty))
                classMembers[eltName] = elt
            elif isinstance(elt, property):
                properties[eltName] = makeFunctionType(eltName, elt.fget, assumeClosuresGlobal=True)
            elif isinstance(elt, classmethod):
                if eltName in staticFunctions:
                    raise TypeError(f"method {eltName} can't be both a staticmethod and a classmethod")

                if eltName not in classMethods:
                    classMethods[eltName] = makeFunctionType(
                        eltName, elt.__func__, assumeClosuresGlobal=True
                    )
                else:
                    classMethods[eltName] = typed_python._types.Function(
                        classMethods[eltName],
                        makeFunctionType(eltName, elt.__func__, assumeClosuresGlobal=True),
                    )
            elif isinstance(elt, staticmethod):
                if eltName in classMethods:
                    raise TypeError(f"method {eltName} can't be both a staticmethod and a classmethod")

                if eltName not in staticFunctions:
                    staticFunctions[eltName] = makeFunctionType(
                        eltName, elt.__func__, assumeClosuresGlobal=True
                    )
                else:
                    staticFunctions[eltName] = typed_python._types.Function(
                        staticFunctions[eltName],
                        makeFunctionType(eltName, elt.__func__, assumeClosuresGlobal=True),
                    )
            elif (
                isinstance(elt, FunctionType)
                or isinstance(elt, typed_python._types.Function)
                or isinstance(elt, type)
                and issubclass(elt, typed_python._types.Function)
            ):
                if eltName == '__typed_python_template__':
                    classMembers[eltName] = elt
                elif eltName not in memberFunctions:
                    memberFunctions[eltName] = makeFunctionType(
                        eltName, elt, classname=name, assumeClosuresGlobal=True
                    )
                else:
                    memberFunctions[eltName] = typed_python._types.Function(
                        memberFunctions[eltName],
                        makeFunctionType(eltName, elt, classname=name, assumeClosuresGlobal=True),
                    )
            else:
                classMembers[eltName] = elt

        res = typed_python._types.Class(
            name,
            tuple(bases),
            isFinal,
            tuple(members),
            tuple(memberFunctions.items()),
            tuple(staticFunctions.items()),
            tuple(properties.items()),
            tuple(classMembers.items()),
            tuple(classMethods.items()),
        )

        if classCell is not None:
            typed_python._types.setPyCellContents(classCell, res)

        return res

    def __subclasscheck__(cls, subcls):
        if cls is subcls:
            return True

        if getattr(subcls, "__typed_python_category__", None) != "Class":
            return False

        if cls is typed_python._types.Class:
            return True

        return cls in subcls.MRO

    def __instancecheck__(cls, instance):
        if type(instance) is cls:
            return True

        if getattr(type(instance), "__typed_python_category__", None) != "Class":
            return False

        if cls is typed_python._types.Class:
            return True

        return cls in type(instance).MRO


def Function(f, assumeClosuresGlobal=False, returnTypeOverride=None):
    """Turn a normal python function into a 'typed_python.Function' which obeys type restrictions."""
    return makeFunctionType(
        f.__name__, f, assumeClosuresGlobal=assumeClosuresGlobal, returnTypeOverride=returnTypeOverride
    )(f)


class FunctionOverloadArg:
    def __init__(self, name, defaultVal, typeFilter, isStarArg, isKwarg):
        """Initialize a single argument descriptor in a FunctionOverload

        Args:
            name (str) the actual name of the argument in the function
            defaultVal - None or a tuple with one element containing the python value
                specified as the default value for this argument
            isStarArg (bool) - if True, then this is a '*arg', of which there can be
                at most one.
            isKwarg (bool) - if True, then this is a '**kwarg' of which there can be
                at most one at the end of the signature.
        """
        self.name = name
        self.defaultValue = defaultVal
        self._typeFilter = typeFilter
        self.isStarArg = isStarArg
        self.isKwarg = isKwarg

    @property
    def typeFilter(self):
        if getattr(self._typeFilter, "__typed_python_category__", None) == "Forward":
            return self._typeFilter.get()
        return self._typeFilter

    def __repr__(self):
        res = f"{self.name}"
        if self.typeFilter is not None:
            res += f": {self.typeFilter}"

        if self.defaultValue is not None:
            res += " = " + str(self.defaultValue[0])
        if self.isKwarg:
            res = "**" + res
        if self.isStarArg:
            res = "*" + res

        return res


class FunctionOverload:
    def __init__(self, functionTypeObject, index, code, funcGlobalsInCells, closureVarLookups, returnType, signatureFunction, methodOf):
        """Initialize a FunctionOverload.

        Args:
            functionTypeObject - a _types.Function type object representing the function
            index - the index within the _types.Function sequence of overloads we represent
            code - the code object for the function we're wrapping
            funcGlobals - the globals for the function we're wrapping
            funcGlobalsInCells - a dict of cells that also act like globals
            closureVarLookups - a dict from str to a list of ClosureVariableBindingSteps indicating
                how each function's closure variables are found in the closure of the
                actual function.
            returnType - the return type annotation, or None if None provided. (if None was
                specified, that would be the NoneType)
            signatureFunction - None, or the user-provided callable that determines the
                return type as a function of the types of the arguments to the function.
            methodOf - if we are a method of a class, the class object
        """
        self.functionTypeObject = functionTypeObject
        self.index = index
        self.closureVarLookups = closureVarLookups
        self.functionCode = code
        self.funcGlobalsInCells = funcGlobalsInCells
        self.returnType = returnType
        self.signatureFunction = signatureFunction
        self._realizedGlobals = None
        self.methodOf = methodOf
        self.args = ()

    @property
    def functionGlobals(self):
        return self.functionTypeObject.extractOverloadGlobals(self.index)

    @staticmethod
    def extractGlobalNamesFromCode(codeObj):
        res = set()
        res.update(codeObj.co_names)

        for const in codeObj.co_consts:
            if isinstance(const, type(codeObj)):
                res.update(FunctionOverload.extractGlobalNamesFromCode(const))

        return res

    @property
    def realizedGlobals(self):
        """Merge the 'functionGlobals' and the set of globals in 'cells' into a single dict."""
        if self._realizedGlobals is None:
            res = {}

            globalNames = set(self.extractGlobalNamesFromCode(self.functionCode))

            for name in self.functionGlobals:
                if name.split(".")[0] in globalNames:
                    res[name] = self.functionGlobals[name]

            for varname, cell in self.funcGlobalsInCells.items():
                res[varname] = cell.cell_contents

            if self.methodOf is not None:
                res['__class__'] = self.methodOf.Class

            self._realizedGlobals = res

        return self._realizedGlobals

    @property
    def name(self):
        return self.functionTypeObject.__name__

    def minPositionalCount(self):
        for i in range(len(self.args)):
            a = self.args[i]
            if a.defaultValue or a.isStarArg or a.isKwarg:
                return i
        return len(self.args)

    def maxPositionalCount(self):
        for i in range(len(self.args)):
            a = self.args[i]
            if a.isStarArg:
                return None
        return len(self.args)

    def addArg(self, name, defaultVal, typeFilter, isStarArg, isKwarg):
        self.args = self.args + (FunctionOverloadArg(name, defaultVal, typeFilter, isStarArg, isKwarg),)

    def __str__(self):
        if self.methodOf:
            return "FunctionOverload(%s, returns %s, %s)" % (self.methodOf.Class.__name__, self.returnType, self.args)
        return "FunctionOverload(returns %s, %s)" % (self.returnType, self.args)

    def _installNativePointer(self, fp, returnType, argumentTypes):
        typed_python._types.installNativeFunctionPointer(
            self.functionTypeObject,
            self.index,
            fp,
            returnType,
            tuple(argumentTypes)[len(self.closureVarLookups):],
        )


class DisableCompiledCode:
    def __init__(self):
        pass

    def __enter__(self):
        typed_python._types.disableNativeDispatch()

    def __exit__(self, *args):
        typed_python._types.enableNativeDispatch()

    @staticmethod
    def isDisabled():
        return not typed_python._types.isDispatchEnabled()


def makeNamedTuple(**kwargs):
    return typed_python._types.NamedTuple(**{k: type(v) for k, v in kwargs.items()})(kwargs)


def isCompiled():
    """Returns True if we're in compiled code, False otherwise."""
    return False


def checkOneOfType(x):
    """Instruct compiler to fork based on the type of 'x'.

    If variable 'x' is a OneOf type, and you write

        checkOneOfType(x)

    Then the compiler will generate code that forks based on the type of
    'x' and compiles the remainder of the current sequence of statements
    based on that type.

    This can make code involving 'OneOf' substantially faster by lifting
    the check of the type of 'x' to the top of a block of statements,
    but at the cost of producing larger code.

    Note: Placing several 'checkOneOfType' calls in a row can produce
    absurdly large code, since we fork on all the possible combined types
    of OneOf.
    """
    return x


def checkType(x, *types):
    """Instruct compiler to fork based on whether 'x' is one of the types in 'types'

    Specifically, if 'x' is known to the compiler as Base, and you wish to check
    if its one of Child1, Child2 (which are subclasses of Base), you may write

        checkType(x, Child1, Child2)

    Then the compiler will generate code that checks the subsequent code based on
    whether 'x' is one of these types.

    In interpreted code, this is a no-op.
    """
    return x


def typeKnownToCompiler(x):
    """Returns the type object that the compiler knows for 'x'

    This can be used to check whether compiled code is fully understanding
    the type structure of what you've written.

    This returns None in the interpeter.
    """
    return None


def localVariableTypesKnownToCompiler():
    """Return all the types of local variables known to the compiler at this point.

    Note that the actual values themselves won't be visible to the compiler - you'll
    be getting a handle to a Dict(str, object) that contains the wrapper values that
    you can use to print diagnostics for debugging your code.

    Returns:
        None from the interpreter. a dict containing the local variables
        and their types as known to the compiler from the compiler.
    """
    return None


def Held(T):
    if not issubclass(T, typed_python._types.Class):
        raise Exception(f"{T} is not a Class")

    return T.HeldClass
