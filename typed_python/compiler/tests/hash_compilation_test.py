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
    Tuple, Float32,
    UInt64, UInt32, UInt16, UInt8, Int32, Int16, Int8
)

from typed_python import Entrypoint
import unittest
import numpy


@Entrypoint
def compiledHash(x):
    return hash(x)


class TestHashCompilation(unittest.TestCase):
    def test_hashes_equivalent_integers(self):
        someIntegers = []

        for bits in range(64):
            someIntegers.append(2**bits - 1)
            someIntegers.append(2**bits)

        for intType in [bool, UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32]:
            for intVal in someIntegers:
                self.assertEqual(hash(intType(intVal)), int(compiledHash(intType(intVal))), (intType, intVal, intType(intVal)))

    def test_hashes_equivalent_floats(self):
        # we have to do this through tuples because if we 'hash' a normal python str,
        # we use python's hash function which is not the same as the typed_python one.
        # we are not trying to replicate python's hash function exactly
        someFloats = [-1.0, -.5, 0.0, .5, 1.0, 1.5, 100.0, 1e10, 1e100, numpy.nan, numpy.inf]

        for fType in [Float32, float]:
            for fVal in someFloats:
                tup = Tuple(fType)((fVal,))

                self.assertEqual(hash(tup), compiledHash(tup))

    def test_hash_tuples(self):
        for valueMaker in [
            lambda: Tuple()(()),
            lambda: Tuple(int)((1,)),
            lambda: Tuple(int, int)((1, 2)),
        ]:
            self.assertEqual(hash(valueMaker()), compiledHash(valueMaker()))

    def test_hash_strings_and_bytes(self):
        # we have to do this through tuples because if we 'hash' a normal python str,
        # we use python's hash function which is not the same as the typed_python one.
        # we are not trying to replicate python's hash function exactly
        for valueMaker in [
            lambda: Tuple(str)(("",)),
            lambda: Tuple(str)(("a",)),
            lambda: Tuple(str)(("abcdef this is a long string",)),
            lambda: Tuple(bytes)((b"",)),
            lambda: Tuple(bytes)((b"asdf",)),
            lambda: Tuple(bytes)((b"asdf2",))
        ]:
            self.assertEqual(hash(valueMaker()), compiledHash(valueMaker()))
