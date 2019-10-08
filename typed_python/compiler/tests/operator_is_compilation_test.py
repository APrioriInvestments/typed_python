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

from typed_python import OneOf
from typed_python import Entrypoint
import unittest


class TestOperatorIsCompilation(unittest.TestCase):
    def test_none_is_none(self):
        @Entrypoint
        def f(x):
            return x is None

        self.assertTrue(f(None))
        self.assertFalse(f(1))

    def test_one_is_not_one(self):
        @Entrypoint
        def f(x):
            return x is 1  # noqa

        self.assertFalse(f(None))
        self.assertFalse(f(1))

    def test_true_is_not_false(self):
        @Entrypoint
        def testIs(x, y):
            return x is y

        @Entrypoint
        def testIsNot(x, y):
            return x is not y

        self.assertTrue(testIs(None, None))
        self.assertTrue(testIsNot(None, 1))
        self.assertTrue(testIsNot(None, True))
        self.assertTrue(testIsNot(None, False))
        self.assertTrue(testIsNot(None, "hi"))

        # we diverge from normal python here. The only things where 'is' works are
        # instances of classes (pointer semantics), None, True, False, and types.
        self.assertFalse(testIs(1, 1))
        self.assertTrue(testIsNot(1, 1))

        # check boolean semantics
        self.assertTrue(testIs(True, True))
        self.assertFalse(testIsNot(True, True))

        self.assertTrue(testIs(False, False))
        self.assertFalse(testIsNot(False, False))

        self.assertFalse(testIs(True, False))
        self.assertTrue(testIsNot(True, False))

        self.assertFalse(testIs(False, True))
        self.assertTrue(testIsNot(False, True))

    def test_types_are_themselves(self):
        trueLambdas = [
            lambda: type is type,
            lambda: type is not int,
            lambda: int is int,
            lambda: int is not float
        ]

        falseLambda = [
            lambda: type is not type,
            lambda: type is int,
            lambda: int is float,
            lambda: int is not int
        ]

        for tl in trueLambdas:
            self.assertTrue(Entrypoint(tl)())

        for fl in falseLambda:
            self.assertFalse(Entrypoint(fl)())

    def test_oneof_and_is(self):
        @Entrypoint
        def testIs(x: OneOf(None, bool), y: OneOf(None, bool)):
            return x is y

        vals = [None, True, False]
        for v1 in vals:
            for v2 in vals:
                self.assertEqual(testIs(v1, v2), v1 is v2)

    def test_type_is(self):
        @Entrypoint
        def testSameType(x, y):
            return type(x) is type(y)

        vals = [None, True, False, 1, 2, "hi", b"bye"]

        for v1 in vals:
            for v2 in vals:
                self.assertEqual(testSameType(v1, v2), type(v1) is type(v2))
