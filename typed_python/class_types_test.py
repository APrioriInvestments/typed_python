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
import unittest
from typed_python import Int8, NoneType, TupleOf, OneOf, Tuple, NamedTuple, \
    ConstDict, Alternative, serialize, deserialize, Value, Class, Member, _types

class Interior(Class):
    x = Member(int)
    y = Member(int)

class Exterior(Class):
    x = Member(int)
    i = Member(Interior)
    iTup = Member(NamedTuple(x=Interior, y=Interior))

class NativeClassTypesTests(unittest.TestCase):
    def test_class(self):
        with self.assertRaises(TypeError):
            class A(Class):
                x = Member((1,2,3))

        class A(Class):
            x = Member(int)

            y = int #not a member. Just a value.

            def f(self):
                return 10

        self.assertEqual(_types.bytecount(A), 8)
        self.assertTrue(A.y is int)

        a = A()

        with self.assertRaises(AttributeError):
            a.not_an_attribute

        self.assertEqual(a.x, 0)

        a.x = 10

        self.assertEqual(a.x, 10)

    def test_class_holding_class(self):
        e = Exterior()

        #'anI' is a reference to an internal element of 'e'. 'anI' will keep 'e' alive.
        anI = e.i
        anI.x = 10
        self.assertEqual(e.i.x, 10)

        #verify we get proper referencing to underlying python objects
        anI2 = e.iTup.x
        anI2.x = 10
        self.assertEqual(e.iTup.x.x, 10)

    def test_class_stringification(self):
        i = Interior()

        self.assertEqual(Interior.__qualname__, "Interior")
        self.assertEqual(str(Interior()), "Interior(x=0, y=0)")

    def test_class_functions_work(self):
        class C(Class):
            x = Member(int)

            def setX(self, x):
                self.x = x

        c = C()

        c.setX(20)

        self.assertEqual(c.x, 20)

    def test_class_functions_return_types(self):
        class C(Class):
            def returnsInt(self, x) -> int:
                return x

            def returnsFloat(self, x) -> float:
                return x


        c = C()

        c.returnsInt(20)

        with self.assertRaises(TypeError):
            #this should throw because we cannot convert a string to an int
            c.returnsInt("hi")

        #this should throw because we are happy to convert int to float
        c.returnsFloat(1)

    def test_class_function_dispatch_on_arity(self):
        class C(Class):
            def f(self):
                return 0
            def f(self, i):
                return 1
            def f(self, i, i2):
                return 2
            def f(self, i, i2, *args):
                return 2 + len(args)
            def f(self, i, i2, *args):
                return 2 + len(args)
        
        c = C()

        self.assertEqual(c.f(), 0)
        self.assertEqual(c.f(1), 1)
        self.assertEqual(c.f(1, 2), 2)
        self.assertEqual(c.f(1, 2, 3), 3)

    def test_class_function_exceptions(self):
        class C(Class):
            def g(self, a,b):
                assert False
            def f(self, a,b):
                return 1

        c = C()
        with self.assertRaises(Exception):
            c.g(1,2)


    def test_class_function_sends_args_to_right_place(self):
        def g(a,b,c=10, *args, d=20, **kwargs):
            return (a,b,args,kwargs)
        
        class C(Class):
            def g(self, a,b, c=10, *args, d=20, **kwargs):
                return (a,b,args,kwargs)

        c = C()

        def assertSame(takesCallable):
            self.assertEqual(
                takesCallable(c.g),
                takesCallable(g)
                )

        assertSame(lambda formOfG: formOfG(1,2))
        assertSame(lambda formOfG: formOfG(1,2,3))
        assertSame(lambda formOfG: formOfG(1,b=2))
        assertSame(lambda formOfG: formOfG(a=1,b=2))
        assertSame(lambda formOfG: formOfG(a=1,b=2,c=20))
        assertSame(lambda formOfG: formOfG(a=1,b=2,c=20,d=30))
        assertSame(lambda formOfG: formOfG(1,2,3,4))
        assertSame(lambda formOfG: formOfG(1,2,3,4,5,6))
        assertSame(lambda formOfG: formOfG(1,2,3,4,5,6,d=10,q=20))
        

    def test_class_function_type_dispatch(self):
        class C(Class):
            def f(self, a: float):
                return float
            def f(self, a):
                return "any"

            def f(self, *args: int):
                return "int list"
            def f(self, *args: str):
                return "string list"
            def f(self, **kwargs: TupleOf(int)):
                return "named tuple of ints"

        self.assertEqual(C().f(1), float)
        self.assertEqual(C().f("asdf"), "any")
        self.assertEqual(C().f("asdf", "asdf2"), "string list")
        self.assertEqual(C().f(1,2), "int list")
        with self.assertRaises(TypeError):
            C().f(1,"hi")
        self.assertEqual(C().f(x=(1,2)), "named tuple of ints")
        

        
