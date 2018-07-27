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
from typed_python.types import  TypeFunction, ListOf, OneOf, Dict, \
                                ConstDict, TypedFunction, Class, PackedArray, \
                                Pointer

import unittest


class BasicTypedClass(Class):
    x = int

    def __init__(self):
        self.x = 0

    def __init__(self, y: int):
        self.x = y

    def f(self, z: int) -> int:
        return self.x + z

    def f(self, z: int, z2: int) -> int:
        return self.x + z + z2

    def f(self, z: int, z2: int, z3: int) -> str:
        return self.x + z + z2 + z3

class TypesTests(unittest.TestCase):
    def test_type_function_memoization(self):
        int_list_1 = ListOf(int)
        int_list_2 = ListOf(int)
        str_list = ListOf(str)

        self.assertTrue(int_list_1 is int_list_2)
        self.assertTrue(int_list_1 is not str_list)

        self.assertTrue(isinstance(int_list_1, ListOf))
        
        self.assertTrue(int_list_1.ElementType is int)
        self.assertTrue(str_list.ElementType is str)

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

    def test_dict_type(self):
        l = Dict(int, int)()
        
        l[10] = 20
        l[20] = 30

        self.assertEqual(len(l), 2)
        self.assertEqual(l[10], 20)
        self.assertEqual(l[20], 30)

        with self.assertRaises(TypeError):
            l[0] = 1.1

        with self.assertRaises(TypeError):
            l[1.1] = 0

        with self.assertRaises(TypeError):
            del l[1.1]

        
        self.assertTrue(10 in l)
        
        del l[10]

        self.assertTrue(10 not in l)
        self.assertTrue(20 in l)
        self.assertTrue(30 not in l)

    def test_const_dicts(self):
        l = ConstDict(int, int)({10:10, 20:20})
        l2 = ConstDict(int, int)({10:100, 30:300})

        l3 = l + l2

        self.assertTrue(10 in l)
        self.assertTrue(10 in l2)
        self.assertTrue(10 in l3)

        
    def test_types_with_none(self):
        l = ListOf(OneOf(int, None, "hi"))()

        l.append(10)
        l.append(None)
        l.append("hi")

        self.assertEqual(len(l), 3)
        self.assertEqual(l[0], 10)
        self.assertEqual(l[1], None)
        self.assertEqual(l[2], "hi")

        l[0] = None

        with self.assertRaises(TypeError):
            l[0] = "hi2"
        
        with self.assertRaises(TypeError):
            l[0] = 1.1

    def test_compound_list_types(self):
        l = ListOf(OneOf(ListOf(int), ListOf(float)))()

        l.append(ListOf(int)())
        l.append(ListOf(float)())

        with self.assertRaises(TypeError):
            l.append(ListOf(None))
        
    def test_function_types(self):
        @TypedFunction
        def f(x: int):
            return x + 1

        self.assertEqual(f(20),21)

        with self.assertRaises(TypeError):
            f("hi")

    def test_function_types_bad_return_val(self):
        @TypedFunction
        def good(x: int) -> int:
            return x + 1

        @TypedFunction
        def bad(x: int) -> str:
            return x + 1

        self.assertEqual(good(20),21)
        
        with self.assertRaises(TypeError):
            bad(20)

    def test_function_overload_dispatch(self):
        @TypedFunction
        def f_first(x: int) -> int:
            return x + 1

        @f_first.overload
        def f(x: float) -> str:
            return str(x)

        self.assertEqual(f(20),21)
        self.assertEqual(f(20.1),"20.1")
        
        with self.assertRaises(TypeError):
            f("hi")
        
    def test_typed_classes(self):
        c = BasicTypedClass()
        self.assertEqual(c.x, 0)

        c = BasicTypedClass(1)
        self.assertEqual(c.x, 1)

        self.assertEqual(c.f(10), 11)
        self.assertEqual(c.f(10, 20), 31)

        with self.assertRaises(TypeError):
            c.f(10,20,30)

        c.x = c.x + 1
        self.assertEqual(c.x, 2)


        with self.assertRaises(TypeError):
            c.x = "hi"

    def test_packed_arrays_of_integers(self):
        i = PackedArray(int)()

        i.append(10)
        self.assertEqual(i[0], 10)
        self.assertEqual(len(i), 1)

        i.append(20)

        self.assertIsInstance(i.ptr(0), Pointer)
        self.assertIsInstance(i.ptr(0), Pointer(int))

        i.ptr(0).set(30)

        self.assertEqual(i[0], 30)
        self.assertEqual(i[1], 20)

        i.ptr(1).set(40)
        self.assertEqual(i[1], 40)

        (i.ptr(0) + 1).set(50)
        self.assertEqual(i[1], 50)

        with self.assertRaises(Exception):
            i.ptr(2)

        #this is always OK
        i.ptr(0) + 2

        with self.assertRaises(Exception):
            (i.ptr(0) + 2).set(50)


        i.resize(0)

        self.assertEqual(len(i), 0)

    def test_packed_arrays_of_classes(self):
        i = PackedArray(BasicTypedClass)()

        i.resize(10)

        self.assertEqual(len(i), 10)
        self.assertIsInstance(i[0], BasicTypedClass)

        for ix in range(10):
            i[ix].x = ix

        i[0] = i[5]

        self.assertEqual(i[0].x, 5)

        i[0].x = 10
        self.assertEqual(i[0].x, 10)
        self.assertEqual(i[5].x, 5)

        