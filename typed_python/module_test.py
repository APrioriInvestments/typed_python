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

from typed_python import Module, TupleOf, Class, Member, OneOf
import unittest


class ModuleTest(unittest.TestCase):
    def test_definining_members(self):
        m = Module("M")
        m.Int = int

        self.assertEqual(m.Int, int)

        m.Y = int
        m.T = TupleOf(m.Y)

        self.assertEqual(m.T, TupleOf(int))

    def test_defining_classes(self):
        m = Module("M")

        @m.define
        class X(Class):
            y = Member(OneOf(None, m.Y))

        @m.define
        class Y(Class):
            x = Member(OneOf(None, m.X))

        x = X(y=Y(x=None))
        y = Y(x=x)

        self.assertTrue(y.x.y.x is None)

    def test_cant_assign_lower_case(self):
        m = Module("M")
        with self.assertRaises(AttributeError):
            m.int = int

        with self.assertRaises(AttributeError):
            m._int = int

    def test_freezing_prevents_defining(self):
        m = Module("M")
        m.Int = int

        m.freeze()
        with self.assertRaises(Exception):
            m.Float = float

    def test_freezing_fails_if_undefined(self):
        m = Module("M")
        m.T = TupleOf(m.I)

        with self.assertRaisesRegex(Exception, "M.I is not defined yet"):
            m.freeze()
