#   Copyright 2020 typed_python Authors
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

import tempfile
import sys
import subprocess
import os
import pytest

from typed_python import (
    UInt64, UInt32,
    ListOf, TupleOf, Tuple, NamedTuple, Dict, OneOf, Forward, identityHash,
    Entrypoint
)

from typed_python._types import (
    prepareArgumentToBePassedToCompiler,
    recursiveTypeGroup
)


@Entrypoint
def fModuleLevel(x):
    return gModuleLevel(x)


@Entrypoint
def gModuleLevel(x):
    return fModuleLevel(x)


def checkHash(filesToWrite, expression):
    """Check the hash of a piece of python code using a subprocess.

    Args:
        filesToWrite = a dictionary from filename to the actual file contents to write.
            note that you need to provide __init__.py for any submodules you create.
        expression - the expression to evaluate (assume we've imported all the modules)

    Returns:
        a bytes object containing the sha-hash of module.thingToGrab.
    """
    with tempfile.TemporaryDirectory() as tf:
        # write all the files out
        for fname, contents in filesToWrite.items():
            fullname = os.path.join(tf, fname)
            dirname = os.path.dirname(fullname)

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(fullname, "w") as f:
                f.write("from typed_python import *\n" + contents)

        namesToImport = [
            fname[:-3].replace("/", ".") for fname in filesToWrite if '__init__' not in fname
        ]

        # we're returning a 'repr' of a bytes object. the 'eval'
        # turns it back into a python bytes object so we can compare it.
        return eval(
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "".join(f"import {modname};" for modname in namesToImport) +
                    f"from typed_python import identityHash; print(repr(identityHash({expression})))"
                ],
                cwd=tf
            )
        )


if sys.platform != "darwin":
    def test_identity_of_register_types():
        assert isinstance(identityHash(UInt64), bytes)
        assert len(identityHash(UInt64)) == 20

        assert identityHash(UInt64) != identityHash(UInt32)

    def test_identity_of_list_of():
        assert identityHash(ListOf(int)) != identityHash(ListOf(float))
        assert identityHash(ListOf(int)) == identityHash(ListOf(int))
        assert identityHash(ListOf(int)) != identityHash(TupleOf(int))

    def test_identity_of_named_tuple_and_tuple():
        assert identityHash(NamedTuple(x=int)) != identityHash(NamedTuple(x=float))
        assert identityHash(NamedTuple(x=int)) == identityHash(NamedTuple(x=int))
        assert identityHash(NamedTuple(x=int)) != identityHash(Tuple(float))

        assert identityHash(NamedTuple(x=int)) != identityHash(NamedTuple(y=int))
        assert identityHash(NamedTuple(x=int, y=float)) != identityHash(NamedTuple(y=float, x=int))

    def test_identity_of_dict():
        assert identityHash(Dict(int, float)) != identityHash(Dict(int, int))
        assert identityHash(Dict(int, float)) != identityHash(Dict(float, int))

    def test_identity_of_oneof():
        assert identityHash(OneOf(None, int)) != identityHash(OneOf(None, float))

    def test_identity_of_recursive_types():
        X = Forward("X")
        X = X.define(TupleOf(OneOf(int, X)))

        X2 = Forward("X")
        X2 = X2.define(TupleOf(OneOf(int, X2)))

        X3 = Forward("X")
        X3 = X3.define(TupleOf(OneOf(float, X3)))

        assert identityHash(X2) == identityHash(X)
        assert identityHash(X3) != identityHash(X)

    def test_identity_of_recursive_types_2():
        X = Forward("X")
        X = X.define(TupleOf(OneOf(int, TupleOf(X))))

        identityHash(X)

    def test_identity_of_recursive_types_produced_same_way():
        def make(name, T):
            X = Forward(name)
            return X.define(TupleOf(OneOf(T, X)))

        assert identityHash(make("X", int)) == identityHash(make("X", int))
        assert identityHash(make("X", int)) != identityHash(make("X", float))
        assert identityHash(make("X", int)) != identityHash(make("X2", int))

    def test_identity_of_lambda_functions():
        @Entrypoint
        def makeAdder(a):
            return lambda x: x + a

        # these two have the same closure type
        assert makeAdder(10).ClosureType == makeAdder(11).ClosureType
        assert identityHash(type(makeAdder(10))) == identityHash(type(makeAdder(10)))
        assert identityHash(type(makeAdder(10))) == identityHash(type(makeAdder(11)))

        # these two are different
        assert identityHash(type(makeAdder(10))) != identityHash(type(makeAdder(10.5)))

    def test_checkHash_works():
        assert checkHash({"x.py": "A = TupleOf(int)\n"}, 'x.A') == identityHash(TupleOf(int))

    def test_mutually_recursive_group_basic():
        assert recursiveTypeGroup(TupleOf(int)) == [TupleOf(int)]

        X = Forward("X")
        X = X.define(TupleOf(OneOf(int, X)))

        assert recursiveTypeGroup(X) == [OneOf(int, X), X]

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
        assert recursiveTypeGroup(type(gModuleLevel)) == [
            type(fModuleLevel), type(gModuleLevel)
        ]

    @pytest.mark.skip("We can't make 'value' objects out of functions yet.")
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

    def test_checkHash_lambdas_hash_dependent_functions_correctly():
        contents1 = {"x.py": "@Entrypoint\ndef g(x):\n    return x + 1\n@Entrypoint\ndef f(x):\n    return g(x)\n"}
        contents2 = {"x.py": "@Entrypoint\ndef g(x):\n    return x + 2\n@Entrypoint\ndef f(x):\n    return g(x)\n"}

        assert checkHash(contents1, 'type(x.f)') != checkHash(contents2, 'type(x.f)')

    def test_checkHash_lambdas_hash_mutually_recursive_correctly():
        contents1 = {"x.py": "@Entrypoint\ndef g(x):\n    return f(x + 1)\n@Entrypoint\ndef f(x):\n    return g(x)\n"}
        contents2 = {"x.py": "@Entrypoint\ndef g(x):\n    return f(x + 2)\n@Entrypoint\ndef f(x):\n    return g(x)\n"}

        assert checkHash(contents1, 'type(x.f)') != checkHash(contents2, 'type(x.f)')
