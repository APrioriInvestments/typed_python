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

from typed_python import OneOf, ListOf, Int32, Int64, Float64
from typed_python.compiler.runtime import Entrypoint
import unittest


class TestCompilingTypeOperations(unittest.TestCase):
    def test_can_make_new_types(self):
        @Entrypoint
        def f(x):
            result = ListOf(OneOf(None, type(x)))()
            result.append(x)
            return result

        self.assertEqual(
            type(f(10)).ElementType,
            OneOf(None, int)
        )

    def test_stringification_of_type(self):
        @Entrypoint
        def f(x):
            return str(x)

        def check(T):
            self.assertEqual(f(T), str(T))

        for typ in [
            str, int, Int32, Int64, Float64, bool, float,
            type(None), ListOf(int), ListOf(OneOf(int, float))
        ]:
            check(typ)

    def test_type_of_list_of_int(self):
        def f(x):
            return type(x).ElementType is Int64

        self.assertTrue(f(ListOf(int)()))
        self.assertTrue(Entrypoint(f)(ListOf(int)()))
