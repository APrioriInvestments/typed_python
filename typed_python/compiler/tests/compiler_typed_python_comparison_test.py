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

from typed_python import (
    TupleOf, ListOf, Dict,
    UInt8, UInt16, UInt32, UInt64,
    Int8, Int16, Int32, Int64,
    Float32, Float64, Tuple
)
from typed_python import Entrypoint

import unittest
import collections


def pyTypeFor(T):
    """Return the python interpreter type that T is supposed to behave like"""
    if T in (float, bool, int, type(None), str, bytes):
        return T

    if T in (UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float32, Float64):
        return int

    if hasattr(T, '__typed_python_category__'):
        if T.__typed_python_category__ == "ListOf":
            return list
        if T.__typed_python_category__ == "TupleOf":
            return tuple
        if T.__typed_python_category__ == "Tuple":
            return tuple
        if T.__typed_python_category__ == "Dict":
            return dict

    assert False, f"No pyType for {T}"


# the types we will test.
types = [
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    UInt8, UInt16, UInt32, UInt64,
    Int8, Int16, Int32,
    Float32,
    ListOf(int),
    ListOf(str),
    TupleOf(int),
    TupleOf(str),
    Dict(int, int),
    Dict(str, str),
    Tuple(int, int, int),
    Tuple(str, int, str)
]


def instancesOf(T):
    """Produce some instances of type T"""
    if T is type(None):  # noqa
        return [None]

    if T is bool:
        return [True, False]

    if T in (int, float, Int8, Int16, Int32, Float32):
        return [T(x) for x in [-2, -1, 0, 1, 2]]

    if T in (UInt8, UInt16, UInt32, UInt64):
        return [T(x) for x in [0, 1, 2]]

    if T is str:
        return ['', 'a', 'b', 'ab', 'ba']

    if T is bytes:
        return [b'', b'a', b'b', b'ab', b'ba']

    if T is ListOf(int):
        return [ListOf(int)(), ListOf(int)([1]), ListOf(int)([2]), ListOf(int)([1, 2]), ListOf(int)([2, 1])]
    if T is ListOf(str):
        return [ListOf(str)(), ListOf(str)(['a']), ListOf(str)(['b']), ListOf(str)(['a', 'b']), ListOf(str)(['b', 'a'])]

    if T is TupleOf(int):
        return [TupleOf(int)(), TupleOf(int)([1]), TupleOf(int)([2]), TupleOf(int)([1, 2]), TupleOf(int)([2, 1])]
    if T is TupleOf(str):
        return [TupleOf(str)(), TupleOf(str)(['a']), TupleOf(str)(['b']), TupleOf(str)(['a', 'b']), TupleOf(str)(['b', 'a'])]

    if T is Tuple(int, int, int):
        return [T((1, 2, 3)), T((2, 2, 3)), T((3, 2, 1))]
    if T is Tuple(str, int, str):
        return [T(('1', 2, '3')), T(('2', 2, '3')), T(('3', 2, '1'))]

    if T is Dict(int, int):
        return [T({1: 2}), T({3: 4}), T({1: 2, 3: 4})]
    if T is Dict(str, str):
        return [T({'1': '2'}), T({'3': '4'}), T({'1': '2', '3': '4'})]

    assert False, f"Can't make instances of {T}"


def toPyForm(x):
    """Return the python form of a value 'x', which it should behave like."""
    if x is None:
        return None

    return pyTypeFor(type(x))(x)


binary_operations = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'floordiv': lambda a, b: a // b,
    'truediv': lambda a, b: a / b,
    'mul': lambda a, b: a * b,
    'pow': lambda a, b: a ** b,
    'bitxor': lambda a, b: a ^ b,
    'bitor': lambda a, b: a | b,
    'bitand': lambda a, b: a & b,
    'lshift': lambda a, b: a << b,
    'rshift': lambda a, b: a >> b,
    'eq': lambda a, b: a == b,
    'neq': lambda a, b: a != b,
    'lt': lambda a, b: a < b,
    'gt': lambda a, b: a > b,
    'lte': lambda a, b: a <= b,
    'gte': lambda a, b: a >= b,
}


unary_operations = {
    'neg': lambda a: -a,
    'pos': lambda a: +a,
    'inv': lambda a: ~a
}


_specializedCache = {}


def compiled(f):
    if f not in _specializedCache:
        _specializedCache[f] = Entrypoint(f)
    return _specializedCache[f]


def sanitize(x):
    x = repr(x)
    if len(x) > 50:
        return x[:50] + "..."
    return x


class TestTypedPythonAgainstCompiler(unittest.TestCase):
    """Systematically compare typed_python, the interpreter, and the compiler.

    We rely on two main invariants. First, the compiler should cause
    us to produce the same outputs as we would get when we run typed_python code
    in the interpreter.  Second, typed_python types are intended to work like their
    untyped counterparts: as long as all the datatypes are representable in equivalent
    forms between the untyped and typed versions, adding the typing shouldn't change
    the outcome.

    This test suite attempts to systematically verify that that is true. Because
    some functionality is not implemented, we provide functions to suppress errors
    when something doesn't work yet.
    """

    def callOrException(self, f, *args):
        try:
            return f(*args)
        except Exception:
            return 'Exception'

    def test_binary_operations(self):
        # systematically check that all binary operations work as expected
        for binop, impl in binary_operations.items():
            for T1 in types:
                for T2 in types:
                    categories = collections.defaultdict(int)

                    for i1 in instancesOf(T1):
                        for i2 in instancesOf(T2):
                            # print(binop, T1, T2, i1, i2)

                            pyI1 = toPyForm(i1)
                            pyI2 = toPyForm(i2)

                            res = self.callOrException(impl, pyI1, pyI2)
                            typed_res = self.callOrException(impl, i1, i2)
                            compiled_res = self.callOrException(compiled(impl), i1, i2)

                            categories[self.categorizeResults(res, typed_res, compiled_res)] += 1

                    if list(categories) != ["OK"]:
                        print(binop, T1, T2, categories)

    def categorizeResults(self, interpreter, typed_python, compiler):
        # print(sanitize(interpreter), sanitize(typed_python), sanitize(compiler))

        if interpreter == 'Exception' and typed_python != 'Exception':
            return 'TypedPythonShouldThrow'

        if interpreter != 'Exception' and typed_python == 'Exception':
            return 'TypedPythonThrows'

        if interpreter != "Exception" and typed_python != "Exception":
            if pyTypeFor(type(typed_python)) != type(interpreter):
                return "TypedPythonWrongTypeCategory"

            if pyTypeFor(type(typed_python))(typed_python) != interpreter:
                return "TypedPythonWrongValue"

        if typed_python == 'Exception' and compiler != 'Exception':
            return 'CompilerShouldThrow'

        if typed_python != 'Exception' and compiler == 'Exception':
            return 'CompilerThrows'

        if typed_python != "Exception" and compiler != "Exception":
            if type(typed_python) != type(compiler):
                return "CompilerWrongType"

            if typed_python != compiler:
                return "CompilerWrongValue"

        return "OK"
