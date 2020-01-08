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

from typed_python import OneOf, TupleOf

from typed_python import Entrypoint
import unittest


class TestIsinstanceCompilation(unittest.TestCase):
    def test_basic_isinstance(self):
        @Entrypoint
        def isInt(x):
            return isinstance(x, int)

        self.assertTrue(isInt(0))
        self.assertFalse(isInt(1.0))
        self.assertFalse(isInt("hi"))

    def test_isinstance_with_oneof(self):
        @Entrypoint
        def isIntOneOf(x: OneOf(int, float)):
            return isinstance(x, int)

        self.assertTrue(isIntOneOf(0))
        self.assertFalse(isIntOneOf(1.0))

    def test_isinstance_with_oneof_and_str(self):
        @Entrypoint
        def isStrOneOf(x: OneOf(int, str)):
            return isinstance(x, str)

        self.assertFalse(isStrOneOf(0))
        self.assertTrue(isStrOneOf("1.0"))

    def test_isinstance_with_value(self):
        @Entrypoint
        def isIntValue(x: OneOf(0, 1.0)):
            return isinstance(x, int)

        self.assertTrue(isIntValue(0))
        self.assertFalse(isIntValue(1.0))

    def test_isinstance_complex(self):
        @Entrypoint
        def isTupleOfInt(x):
            return isinstance(x, TupleOf(int))

        self.assertFalse(isTupleOfInt(0))
        self.assertFalse(isTupleOfInt((1, 2)))
        self.assertTrue(isTupleOfInt(TupleOf(int)((1, 2))))

    def test_isinstance_typeof_none(self):
        @Entrypoint
        def isNone(x):
            return isinstance(x, type(None))

        self.assertFalse(isNone(0))
        self.assertTrue(isNone(None))
