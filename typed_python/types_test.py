#   Copyright 2018 Braxton Mckee
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

from typed_python.hash import sha_hash
from typed_python.types import TypeFunction, ListOf, OneOf

import unittest

class TypesTests(unittest.TestCase):
    def test_type_function_memoization(self):
        int_list_1 = ListOf(int)
        int_list_2 = ListOf(int)
        str_list = ListOf(str)

        self.assertTrue(int_list_1 is int_list_2)
        self.assertTrue(int_list_1 is not str_list)

        self.assertTrue(isinstance(int_list_1, ListOf))
        self.assertFalse(isinstance(int_list_1, OneOf))

        self.assertTrue(int_list_1.T is int)
        self.assertTrue(str_list.T is str)

    def test_list_type(self):
        int_list = ListOf(int)()

        int_list.append(10)

        self.assertEqual(len(int_list), 1)
        self.assertEqual(int_list[0], 10)

        with self.assertRaises(TypeError):
            int_list[0] = 1.1

    def test_list_of_oneof_type(self):
        l = ListOf(OneOf(int, str))()

        l.append(10)
        l.append("hi")

        self.assertEqual(len(l), 2)
        self.assertEqual(l[0], 10)
        self.assertEqual(l[1], "hi")

        l[0] = "hi2"
        l[1] = 10

        with self.assertRaises(TypeError):
            l[0] = 1.1
        


