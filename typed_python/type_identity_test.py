#   Copyright 2022 typed_python Authors
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

import threading
import pytest
import tempfile
import os

import typed_python
from typed_python.test_util import evaluateExprInFreshProcess, callFunctionInFreshProcess
from typed_python import (
    UInt64, UInt32,
    ListOf, TupleOf, Tuple, NamedTuple, Dict, OneOf, Forward, compilerHash,
    Entrypoint, Class, Member, Final, TypeFunction, SerializationContext,
    Function, NotCompiled
)

from typed_python.hash import Hash

from typed_python._types import (
    prepareArgumentToBePassedToCompiler,
    recursiveTypeGroup,
    getCodeGlobalDotAccesses,
    typesAndObjectsVisibleToCompilerFrom,
    checkForHashInstability,
    resetCompilerVisibleObjectHashCache,
    typeWalkRecord
)


class A:
    pass


class B:
    pass


@Entrypoint
def fModuleLevel(x):
    return gModuleLevel(x)


@Entrypoint
def gModuleLevel(x):
    return fModuleLevel(x)


def looksAtFilename():
    return __file__


def looksAtFilename2():
    return typed_python.type_identity_test.__file__


def checkHash(filesToWrite, expression):
    """Check the hash of a piece of python code using a subprocess.

    Args:
        filesToWrite = a dictionary from filename to the actual file contents to write.
            note that you need to provide __init__.py for any submodules you create.
        expression - the expression to evaluate (assume we've imported all the modules)

    Returns:
        a bytes object containing the sha-hash of module.thingToGrab.
    """
    return Hash(evaluateExprInFreshProcess(filesToWrite, f"compilerHash({expression})"))


def returnSerializedValue(filesToWrite, expression, printComments=False):
    return evaluateExprInFreshProcess(
        filesToWrite,
        f"SerializationContext({{}}).serialize({expression})",
        printComments=printComments
    )


def test_identity_ignores_function_file_accesses():
    # make sure these functions succeed
    assert looksAtFilename()
    assert looksAtFilename2()

    walk1 = typeWalkRecord(looksAtFilename)
    walk2 = typeWalkRecord(looksAtFilename2)

    assert '__file__' not in walk1
    assert '__file__' not in walk2


def test_identities_of_basic_types_different():
    assert compilerHash(int) != compilerHash(float)
    assert compilerHash(TupleOf(int)) != compilerHash(TupleOf(float))


def test_object_graph_instability_is_noticed():
    class C:
        pass

    typesAndObjectsVisibleToCompilerFrom(C)

    C.f = staticmethod(lambda: 10)

    with pytest.raises(Exception):
        typesAndObjectsVisibleToCompilerFrom(C)

    resetCompilerVisibleObjectHashCache()


def test_object_graph_instability_is_noticed_globally():
    class C:
        pass

    typesAndObjectsVisibleToCompilerFrom(C)

    C.f = staticmethod(lambda: 10)

    assert "staticmethod" in checkForHashInstability()

    resetCompilerVisibleObjectHashCache()


def test_identity_of_function_with_annotation():
    def f(x: int):
        pass

    @Entrypoint
    def g(x: int):
        return f(x)

    compilerHash(f)
    compilerHash(NotCompiled(f))

    hashInstability = checkForHashInstability()

    if hashInstability is not None:
        print(hashInstability)
        raise Exception("hash instability found")


moduleLevelThreadLocal = threading.local()


@NotCompiled
def functionAccessingModuleLevelThreadLocal():
    return hasattr(moduleLevelThreadLocal, "anything")


def test_identity_of_function_accessing_thread_local():
    print(typesAndObjectsVisibleToCompilerFrom(type(functionAccessingModuleLevelThreadLocal)))

    compilerHash(functionAccessingModuleLevelThreadLocal)
    moduleLevelThreadLocal.anything = True
    compilerHash(functionAccessingModuleLevelThreadLocal)

    hashInstability = checkForHashInstability()

    if hashInstability is not None:
        print(hashInstability)
        raise Exception("hash instability found")


def test_identity_of_method_descriptors():
    assert compilerHash(ListOf(int).append) != compilerHash(ListOf(float).append)
    assert compilerHash(ListOf(int).append) != compilerHash(ListOf(int).extend)


def test_class_and_held_class_in_group():
    class C(Class):
        pass

    H = C.HeldClass

    assert H in recursiveTypeGroup(H)
    assert C in recursiveTypeGroup(H)

    assert H in recursiveTypeGroup(C)
    assert C in recursiveTypeGroup(C)


def test_identity_of_register_types():
    assert isinstance(compilerHash(UInt64), bytes)
    assert len(compilerHash(UInt64)) == 20

    assert compilerHash(UInt64) != compilerHash(UInt32)


def test_identity_of_list_of():
    assert compilerHash(ListOf(int)) != compilerHash(ListOf(float))
    assert compilerHash(ListOf(int)) == compilerHash(ListOf(int))
    assert compilerHash(ListOf(int)) != compilerHash(TupleOf(int))


def test_identity_of_named_tuple_and_tuple():
    assert compilerHash(NamedTuple(x=int)) != compilerHash(NamedTuple(x=float))
    assert compilerHash(NamedTuple(x=int)) == compilerHash(NamedTuple(x=int))
    assert compilerHash(NamedTuple(x=int)) != compilerHash(Tuple(float))

    assert compilerHash(NamedTuple(x=int)) != compilerHash(NamedTuple(y=int))
    assert compilerHash(NamedTuple(x=int, y=float)) != compilerHash(NamedTuple(y=float, x=int))


def test_identity_of_dict():
    assert compilerHash(Dict(int, float)) != compilerHash(Dict(int, int))
    assert compilerHash(Dict(int, float)) != compilerHash(Dict(float, int))


def test_identity_of_oneof():
    assert compilerHash(OneOf(None, int)) != compilerHash(OneOf(None, float))


def test_identity_of_recursive_types():
    X = Forward("X")
    X = X.define(TupleOf(OneOf(int, X)))

    X2 = Forward("X")
    X2 = X2.define(TupleOf(OneOf(int, X2)))

    X3 = Forward("X")
    X3 = X3.define(TupleOf(OneOf(float, X3)))

    assert compilerHash(X2) == compilerHash(X)
    assert compilerHash(X3) != compilerHash(X)


def test_identity_of_recursive_types_2():
    X = Forward("X")
    X = X.define(TupleOf(OneOf(int, TupleOf(X))))

    compilerHash(X)


def test_identity_of_recursive_types_produced_same_way():
    def make(name, T):
        X = Forward(name)
        return X.define(TupleOf(OneOf(T, X)))

    assert compilerHash(make("X", int)) == compilerHash(make("X", int))
    assert compilerHash(make("X", int)) != compilerHash(make("X", float))
    assert compilerHash(make("X", int)) != compilerHash(make("X2", int))


def test_identity_of_lambda_functions():
    @Entrypoint
    def makeAdder(a):
        return lambda x: x + a

    # these two have the same closure type
    assert makeAdder(10).ClosureType == makeAdder(11).ClosureType
    assert compilerHash(type(makeAdder(10))) == compilerHash(type(makeAdder(10)))
    assert compilerHash(type(makeAdder(10))) == compilerHash(type(makeAdder(11)))

    # these two are different
    assert compilerHash(type(makeAdder(10))) != compilerHash(type(makeAdder(10.5)))


def test_checkHash_works():
    assert checkHash({"x.py": "A = TupleOf(int)\n"}, 'x.A') == Hash(compilerHash(TupleOf(int)))


def test_mutually_recursive_group_basic():
    assert recursiveTypeGroup(TupleOf(int)) == [TupleOf(int)]

    X = Forward("X")
    X = X.define(TupleOf(OneOf(int, X)))

    assert recursiveTypeGroup(X) == [X, OneOf(int, X)]


def test_mutually_recursive_group_through_functions_in_closure():
    @Entrypoint
    def f(x):
        return g(x)

    @Entrypoint
    def g(x):
        return f(x)

    gType = type(prepareArgumentToBePassedToCompiler(g))
    fType = gType.overloads[0].closureVarLookups['f'][0]

    assert recursiveTypeGroup(gType) == [gType, fType]


def test_mutually_recursive_group_through_functions_at_module_level():
    assert set(recursiveTypeGroup(type(gModuleLevel))) == set([
        fModuleLevel, type(fModuleLevel), gModuleLevel, type(gModuleLevel)
    ])

    assert set(recursiveTypeGroup(gModuleLevel)) == set([
        fModuleLevel, type(fModuleLevel), gModuleLevel, type(gModuleLevel)
    ])


def test_recursive_group_of_function_values():
    @Entrypoint
    def f(x):
        return g(x)

    @Entrypoint
    def g(x):
        return f(x)

    assert recursiveTypeGroup(f)


def test_checkHash_lambdas_stable():
    contents = {"x.py": "@Entrypoint\ndef f(x):\n    return x + 1\n"}

    h1 = checkHash(contents, 'type(x.f)')

    for passIx in range(4):
        assert h1 == checkHash(contents, 'type(x.f)')


def test_checkHash_lambdas_hash_code_correctly():
    contents1 = {"x.py": "@Entrypoint\ndef f(x):\n    return x + 1\n"}
    contents2 = {"x.py": "@Entrypoint\ndef f(x):\n    return x + 2\n"}

    assert checkHash(contents1, 'type(x.f)') != checkHash(contents2, 'type(x.f)')


def test_checkHash_mutable_global_constants():
    contents1 = {"x.py": "G=Dict(int, int)({1:2})\n@Entrypoint\ndef g(x):\n    return G[x]\n"}
    contents2 = {"x.py": "G=Dict(int, int)({1:3})\n@Entrypoint\ndef g(x):\n    return G[x]\n"}
    contents3 = {"x.py": "G=Dict(int, float)({1:3.0})\n@Entrypoint\ndef g(x):\n    return G[x]\n"}

    assert checkHash(contents1, 'type(x.g)') == checkHash(contents2, 'type(x.g)')
    assert checkHash(contents1, 'type(x.g)') != checkHash(contents3, 'type(x.g)')


def test_checkHash_lambdas_hash_dependent_functions_correctly():
    contents1 = {"x.py": "@Entrypoint\ndef g(x):\n    return x + 1\n@Entrypoint\ndef f(x):\n    return g(x)\n"}
    contents2 = {"x.py": "@Entrypoint\ndef g(x):\n    return x + 2\n@Entrypoint\ndef f(x):\n    return g(x)\n"}

    assert checkHash(contents1, 'type(x.f)') != checkHash(contents2, 'type(x.f)')


def test_checkHash_lambdas_hash_mutually_recursive_correctly():
    contents1 = {"x.py": "@Entrypoint\ndef g(x):\n    return f(x + 1)\n@Entrypoint\ndef f(x):\n    return g(x)\n"}
    contents2 = {"x.py": "@Entrypoint\ndef g(x):\n    return f(x + 2)\n@Entrypoint\ndef f(x):\n    return g(x)\n"}

    assert checkHash(contents1, 'type(x.f)') != checkHash(contents2, 'type(x.f)')


def test_checkHash_class_member_access():
    contents1 = {"x.py": "class C:\n    x=1\n@Entrypoint\ndef g(x):\n    return C.x\n"}
    contents2 = {"x.py": "class C:\n    x=2\n@Entrypoint\ndef g(x):\n    return C.x\n"}

    assert not checkHash(contents1, 'x.C').isPoison()
    assert checkHash(contents1, 'type(x.g)') != checkHash(contents2, 'type(x.g)')


def test_checkHash_function_body():
    contents1 = {"x.py": "def f(x): return x + 1\n@Entrypoint\ndef g(x):\n    return f(x)\n"}
    contents2 = {"x.py": "def f(x): return x + 2\n@Entrypoint\ndef g(x):\n    return f(x)\n"}

    assert checkHash(contents1, 'type(x.g)') != checkHash(contents2, 'type(x.g)')


def test_checkHash_function_arg_types():
    contents1 = {"x.py": "@Entrypoint\ndef g(x: int):\n    return f(x)\n"}
    contents2 = {"x.py": "@Entrypoint\ndef g(x: float):\n    return f(x)\n"}

    assert checkHash(contents1, 'type(x.g)') != checkHash(contents2, 'type(x.g)')


def test_checkHash_function_arg_default_vals():
    contents1 = {"x.py": "@Entrypoint\ndef g(x=1):\n    return f(x)\n"}
    contents2 = {"x.py": "@Entrypoint\ndef g(x=2):\n    return f(x)\n"}

    assert checkHash(contents1, 'type(x.g)') != checkHash(contents2, 'type(x.g)')


def test_checkHash_function_arg_default_vals_string():
    contents1 = {"x.py": "@Entrypoint\ndef g(x='1'):\n    return f(x)\n"}
    contents2 = {"x.py": "@Entrypoint\ndef g(x='2'):\n    return f(x)\n"}

    assert checkHash(contents1, 'type(x.g)') != checkHash(contents2, 'type(x.g)')


def test_hash_of_oneof():
    oneOfs = [
        OneOf(None, 1),
        OneOf(None, '1'),
        OneOf(None, UInt32(1)),
        OneOf(None, TupleOf(int)([1])),
        OneOf(None, NamedTuple(x=int)(x=1)),
        OneOf(None, Tuple(int)((1,))),
    ]

    hashes = set([compilerHash(t) for t in oneOfs])
    assert len(hashes) == len(oneOfs)


def test_identityHash_of_none():
    assert not Hash(compilerHash(type(None))).isPoison()


def test_identityHash_of_a_typefunction():
    def L(t):
        return ListOf(t)

    L1 = TypeFunction(L)
    L2 = TypeFunction(L)

    L2(int)

    assert compilerHash(L1) == compilerHash(L2)


def test_hash_of_TP_produced_lambdas_with_different_closure_types():
    @Entrypoint
    def returnIt(x):
        def f():
            return x
        return f

    returnIt(1)
    returnIt(2)
    returnIt(2.0)

    assert compilerHash(returnIt(1)) == compilerHash(returnIt(2))
    assert compilerHash(returnIt(1)) != compilerHash(returnIt(2.0))


def test_hash_of_native_lambdas_with_different_closure_types():
    def returnIt(x):
        def f():
            return x
        return f

    t1 = prepareArgumentToBePassedToCompiler(Entrypoint(returnIt(1)))
    t2 = prepareArgumentToBePassedToCompiler(Entrypoint(returnIt(2)))
    t3 = prepareArgumentToBePassedToCompiler(Entrypoint(returnIt(2.0)))

    assert compilerHash(t1) == compilerHash(t2)
    assert compilerHash(t1) != compilerHash(t3)


def test_checkHash_type_functions():
    contents1 = {"x.py": "@TypeFunction\ndef L(t):\n    return ListOf(t)\n\n@Entrypoint\ndef f(x):\n    return L(type(x))()"}
    contents2 = {"x.py": "@TypeFunction\ndef L(t):\n    return TupleOf(t)\n@Entrypoint\ndef f(x):\n    return L(type(x))()"}
    contents3 = {"x.py": "@TypeFunction\ndef L(t):\n    return ListOf(t)\nL(int)\n@Entrypoint\ndef f(x):\n    return L(type(x))()"}

    assert checkHash(contents1, 'type(x.f)') != checkHash(contents2, 'type(x.f)')
    assert checkHash(contents1, 'type(x.f)') == checkHash(contents3, 'type(x.f)')


def test_checkHash_mutually_recursive_function_bodies():
    contents1 = {"x.py": "def f(x): return g(x + 1)\ndef g(x):\n    return f(x)\n@Entrypoint\ndef h(x):\n    return f(x)"}
    contents2 = {"x.py": "def f(x): return g(x + 2)\ndef g(x):\n    return f(x)\n@Entrypoint\ndef h(x):\n    return f(x)"}

    assert checkHash(contents1, 'type(x.h)') != checkHash(contents2, 'type(x.h)')


def test_checkHash_methods():
    contents1 = {"x.py": "class N(Class):\n    def f(self): return 1\n"}
    contents2 = {"x.py": "class N(Class):\n    def f(self): return 2\n"}

    assert checkHash(contents1, 'x.N.f') != checkHash(contents2, 'x.N.f')


def test_checkHash_methods_on_class():
    contents1 = {"x.py": "class N(Class):\n    def f(self): return 1\n"}
    contents2 = {"x.py": "class N(Class):\n    def f(self): return 2\n"}

    assert checkHash(contents1, 'x.N') != checkHash(contents2, 'x.N')


def test_checkHash_methods_on_empty_python_class():
    contents1 = {"x.py": "class N1:\n    pass\n"}
    contents2 = {"x.py": "class N2:\n    pass\n"}

    assert checkHash(contents1, 'x.N1') != checkHash(contents2, 'x.N2')


def test_checkHash_methods_on_python_class():
    contents1 = {"x.py": "class N:\n    def f(self): return 1\n"}
    contents2 = {"x.py": "class N:\n    def f(self): return 2\n"}

    assert checkHash(contents1, 'x.N') != checkHash(contents2, 'x.N')


def test_checkHash_methods_on_named_tuple_subclass():
    contents1 = {"x.py": "class N(NamedTuple()):\n    def f(self): return 1\n"}
    contents2 = {"x.py": "class N(NamedTuple()):\n    def f(self): return 2\n"}

    assert checkHash(contents1, 'x.N') != checkHash(contents2, 'x.N')


FUNCMAKER = """
@Entrypoint
def makeAdder(x):
    def f(y):
        return y + x
    return f
"""


def test_checkHash_references_to_typed_free_objects():
    contents1 = {"x.py": FUNCMAKER + "A = makeAdder(1)\ndef f(x):\n    return A(x)"}
    contents2 = {"x.py": FUNCMAKER + "A = makeAdder(2)\ndef f(x):\n    return A(x)"}
    contents3 = {"x.py": FUNCMAKER + "A = makeAdder(1.0)\ndef f(x):\n    return A(x)"}

    assert checkHash(contents1, 'x.f') == checkHash(contents2, 'x.f')
    assert checkHash(contents1, 'x.f') != checkHash(contents3, 'x.f')


def test_hash_of_builtins():
    assert not Hash(compilerHash(isinstance)).isPoison()


def test_hash_of_classObj():
    class C(Class, Final):
        x = Member(int)

    assert not Hash(compilerHash(Member(int))).isPoison()
    assert not Hash(compilerHash(C)).isPoison()


MODULE = {
    'x.py':
    """

aTypedGlobal = Dict(int, int)()

def addToTypedGlobalMaker():
    def addToTypedGlobal():
        aTypedGlobal[len(aTypedGlobal)] = 1
        return len(aTypedGlobal)
    return addToTypedGlobal

def C():
    class SomeOtherRandomClass:
        def __init__(self):
            self.x = x

        @staticmethod
        def staticM():
            return

        @property
        def aProp(self):
            return 10

    return SomeOtherRandomClass

def S():
    class SomeRandomClass(Class, Final):
        x=Member(int)
        def f(self, y):
            return self.x+y
    return SomeRandomClass

def RecursiveClass():
    class RecursiveClass(Class, Final):
        def f(self, y):
            return RecursiveClass
    return RecursiveClass

def MakeA():
    return Alternative("A", A1={})

NT = NamedTuple(aNameUnlikelyToShowUpAnywhereElse=int)


def deserializeTwiceAndCall(rep):
    v1 = SerializationContext().deserialize(rep)
    v2 = SerializationContext().deserialize(rep)

    return (v1(), v2())


def deserializeAndReturnHash(rep):
    return compilerHash(SerializationContext().deserialize(rep))


def deserializeTwiceAndConfirmEquivalent(rep):
    v1 = SerializationContext().deserialize(rep)
    v2 = SerializationContext().deserialize(rep)

    return v1 is v2
"""
}


def test_repeated_deserialize_externally_defined_named_tuple():
    ser = returnSerializedValue(MODULE, 'x.NT')

    assert SerializationContext().deserialize(ser) is SerializationContext().deserialize(ser)


def test_repeated_deserialize_externally_defined_class_is_stable():
    ser = returnSerializedValue(MODULE, 'x.S()')

    assert SerializationContext().deserialize(ser) is SerializationContext().deserialize(ser)


def test_repeated_deserialize_externally_defined_alternative_is_stable():
    ser = returnSerializedValue(MODULE, 'x.MakeA()')

    assert SerializationContext().deserialize(ser) is SerializationContext().deserialize(ser)


def test_repeated_deserialize_externally_defined_anonymous_classes():
    ser = returnSerializedValue(MODULE, 'x.C()')

    assert evaluateExprInFreshProcess(
        MODULE,
        f'x.deserializeTwiceAndConfirmEquivalent({repr(ser)})'
    )


def test_serialization_of_anonymous_functions_preserves_references():
    ser = returnSerializedValue(MODULE, 'x.addToTypedGlobalMaker()')

    vals = evaluateExprInFreshProcess(
        MODULE,
        f'x.deserializeTwiceAndCall({repr(ser)})'
    )

    # this checks that the second copy of the function we deserialized in the process
    # also refers to the same of 'aTypedGlobal' as the first one, which it should inherity
    # through its closure.
    assert vals == (1, 2)


def test_hash_stability():
    idHash = evaluateExprInFreshProcess({
        'x.py': 'from typed_python.compiler.native_compiler.native_ast import NamedCallTarget\n'
    }, 'compilerHash(x.NamedCallTarget)')
    ser = returnSerializedValue({
        'x.py': 'from typed_python.compiler.native_compiler.native_ast import NamedCallTarget\n'
    }, 'x.NamedCallTarget', printComments=True)

    idHashDeserialized = evaluateExprInFreshProcess(
        MODULE,
        f'x.deserializeAndReturnHash({repr(ser)})'
    )

    assert idHash == idHashDeserialized


def test_deserialize_external_recursive_class():
    ser = returnSerializedValue(MODULE, 'x.RecursiveClass()')

    # this shouldn't throw
    evaluateExprInFreshProcess(
        MODULE,
        f'x.deserializeAndReturnHash({repr(ser)})'
    )


def test_dot_accesses():
    def f():
        return typed_python._types

    assert getCodeGlobalDotAccesses(f.__code__) == [['__module_hash__'], ['typed_python', '_types']]

    def f2():
        typed_python
        return typed_python._types

    assert getCodeGlobalDotAccesses(f2.__code__) == [['__module_hash__'], ['typed_python'], ['typed_python', '_types']]

    def f3():
        return typed_python.f()

    import dis
    dis.dis(f3)

    assert getCodeGlobalDotAccesses(f3.__code__) == [['__module_hash__'], ['typed_python', 'f']]


def test_identity_of_entrypointed_functions():
    def f():
        return 0

    assert compilerHash(Function(f)) != compilerHash(Entrypoint(f))


def test_identity_of_singleton_classes():
    assert compilerHash(A) != compilerHash(B)


def test_type_walk_for_named_tuple_subclass():
    class N(NamedTuple()):
        def f(self):
            return 0

    print(typeWalkRecord(N))


<<<<<<< HEAD
def test_module_hash_magic_value():
    with tempfile.TemporaryDirectory() as tempDir:

        def makeFun(mh):
            """Produce a dummy function with __module_hash__ of 'mh'

            Note that we need to produce this function with 'backing code' so that
            the AST system can find it and serialize it.
            """
            globalsDict = {}
            fname = os.path.join(tempDir, "code_" + mh + ".py")

            pyCode = (
                f"from typed_python import Function\n"
                f"__module_hash__ = '{mh}'\n"
                f"@Function\n"
                f"def f(x):\n"
                f"    return x\n"
            )

            with open(fname, "w") as f:
                f.write(pyCode)

            exec(compile(pyCode, fname, "exec"), globalsDict)
            return globalsDict['f']

        def makeFunIH(mh):
            return identityHash(type(makeFun(mh)))

        # check that the identity hash depends on the module hash
        assert identityHash(type(makeFun('A'))) == identityHash(type(makeFun('A')))
        assert identityHash(type(makeFun('A'))) != identityHash(type(makeFun('B')))

        # functions should have this in their globals
        f = makeFun('A')
        assert '__module_hash__' in f.overloads[0].functionGlobals
        assert '__module_hash__' in f.overloads[0].realizedGlobals

        # even if we execute this in another process, we should get the same function back
        # and it should have a __module_hash__ in its globals. We have to be careful about
        # this because we need the serializer to understand that __module_hash__ is special
        # and that the function implicitly references it.
        fFromOtherProcess = callFunctionInFreshProcess(makeFun, ('C',))
        assert fFromOtherProcess.overloads[0].functionGlobals['__module_hash__'] == 'C'
        assert fFromOtherProcess.overloads[0].realizedGlobals['__module_hash__'] == 'C'

        # check that the identity hash we loaded is the same one we would get from a
        # subprocess reading it.
        assert (
            identityHash(type(fFromOtherProcess))
            == callFunctionInFreshProcess(makeFunIH, ('C',))
        )


def test_module_hash_magic_value_on_untyped_function_preserved_by_serialization():
    with tempfile.TemporaryDirectory() as tempDir:

        def makeFun(mh):
            """Produce a dummy function with __module_hash__ of 'mh'

            Note that we need to produce this function with 'backing code' so that
            the AST system can find it and serialize it.
            """
            globalsDict = {}
            fname = os.path.join(tempDir, "code_" + mh + ".py")

            pyCode = (
                f"from typed_python import Entrypoint\n"
                f"__module_hash__ = '{mh}'\n"
                f"def f(x):\n"
                f"    return x\n"
            )

            with open(fname, "w") as f:
                f.write(pyCode)

            exec(compile(pyCode, fname, "exec"), globalsDict)
            return globalsDict['f']

        fFromOtherProcess = callFunctionInFreshProcess(makeFun, ('C',))
        assert fFromOtherProcess.__globals__['__module_hash__'] == 'C'


@TypeFunction
def RefsAValueInEntrypointedStaticmethod(i):
    class C:
        @staticmethod
        @Entrypoint
        def f():
            return i
    return C


@TypeFunction
def RefsAValueInStaticmethod(i):
    class C:
        @staticmethod
        def f():
            return i
    return C


@TypeFunction
def RefsAValueInStaticmethodOnClass(i):
    class C(Class):
        @staticmethod
        def f():
            return i
    return C


def test_type_function_identity_referencing_int_in_function_only():
    assert RefsAValueInEntrypointedStaticmethod(1).f() == 1
    assert RefsAValueInEntrypointedStaticmethod(2).f() == 2

    assert (
        identityHash(RefsAValueInStaticmethod(1))
        != identityHash(RefsAValueInStaticmethod(2))
    )

    assert (
        identityHash(RefsAValueInEntrypointedStaticmethod(1))
        != identityHash(RefsAValueInEntrypointedStaticmethod(2))
    )

    assert (
        identityHash(RefsAValueInStaticmethodOnClass(1))
        != identityHash(RefsAValueInStaticmethodOnClass(2))
    )


def test_recursive_type_groups_separate():
    class C:
        pass

    assert recursiveTypeGroup(C, "compiler", False) is None
    assert recursiveTypeGroup(C, "identity", False) is None

    g1 = recursiveTypeGroup(C, "compiler", True)
    assert g1
    assert recursiveTypeGroup(C, "identity", False) is None

    g2 = recursiveTypeGroup(C, "identity", True)
    assert g2

    print(g1, g2)
