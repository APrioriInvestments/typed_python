#   Copyright 2017-2020 typed_python Authors
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
from typed_python import NamedTuple, ListOf, _types, OneOf, TupleOf


class NamedTupleTests(unittest.TestCase):
    def test_named_tuple(self):
        t = NamedTuple(a=int, b=int)

        with self.assertRaisesRegex(AttributeError, "object has no attribute"):
            t().asdf

        with self.assertRaisesRegex(AttributeError, "immutable"):
            t().a = 1

        self.assertEqual(t()[0], 0)
        self.assertEqual(t().a, 0)
        self.assertEqual(t()[1], 0)

        self.assertEqual(t(a=1, b=2).a, 1)
        self.assertEqual(t(a=1, b=2).b, 2)

    def test_error_message_when_assigning_bad_attributes(self):
        T = NamedTuple(a=int)

        class A(T):
            pass

        with self.assertRaisesRegex(AttributeError, "'NamedTuple\\(a=int\\)' object has no attribute 'z'"):
            T().z = 20

        with self.assertRaisesRegex(AttributeError, "'A' object has no attribute 'z'"):
            A().z = 20

        with self.assertRaisesRegex(
            AttributeError,
            "Cannot set attributes on instance of type 'NamedTuple\\(a=int\\)' because it is immutable"
        ):
            T().a = 20

        with self.assertRaisesRegex(AttributeError, "Cannot set attributes on instance of type 'A' because it is immutable"):
            A().a = 20

    def test_named_tuple_construction(self):
        t = NamedTuple(a=int, b=int)

        self.assertEqual(t(a=10).a, 10)
        self.assertEqual(t(a=10).b, 0)
        self.assertEqual(t(a=10, b=2).a, 10)
        self.assertEqual(t(a=10, b=2).b, 2)
        self.assertEqual(t({'a': 10, 'b': 2}).a, 10)
        self.assertEqual(t({'a': 10, 'b': 2}).b, 2)

        self.assertEqual(t({'b': 2}).a, 0)
        self.assertEqual(t({'b': 2}).b, 2)

        with self.assertRaises(TypeError):
            t({'c': 10})
        with self.assertRaises(TypeError):
            t(c=10)

    def test_named_tuple_construction_bad_args(self):
        T = NamedTuple(a=int, b=str)

        with self.assertRaisesRegex(TypeError, "member named 'c'"):
            T(c=100)

        with self.assertRaisesRegex(TypeError, "member 'a'"):
            T(a="hi")

    def test_named_tuple_str(self):
        t = NamedTuple(a=str, b=str)

        self.assertEqual(t(a='1', b='2').a, '1')
        self.assertEqual(t(a='1', b='2').b, '2')

        self.assertEqual(t(b='2').a, '')
        self.assertEqual(t(b='2').b, '2')
        self.assertEqual(t().a, '')
        self.assertEqual(t().b, '')

    def test_named_tuple_subclass(self):
        class X(NamedTuple(x=int)):
            pass

        self.assertEqual(
            str(X(x=10)),
            str(NamedTuple(x=int)(x=10))
        )

    def test_named_tuple_subclass_magic_methods(self):
        hashCalled = []
        hashWantsException = False

        class X(NamedTuple(x=int, y=int)):
            def __str__(self):
                return "str override"

            def __repr__(self):
                return "repr override"

            def __call__(self):
                return "call implemented"

            def __hash__(self):
                if hashWantsException:
                    raise Exception("exception")
                hashCalled.append(self)
                return self.x

        self.assertEqual(repr(X()), "repr override")
        self.assertEqual(str(X()), "str override")
        self.assertEqual(X()(), "call implemented")

        B = NamedTuple(x=X)

        self.assertEqual(str(X()), "str override")
        self.assertEqual(repr(X()), "repr override")
        self.assertEqual(str(B()), "(x=repr override,)")

        self.assertEqual(
            hash(B(x=X(x=1, y=2))),
            hash(B(x=X(x=1, y=3)))
        )

        self.assertEqual(len(hashCalled), 2)

        # check that the internal state machine for processing
        # exceptions is correct.
        hashWantsException = True

        with self.assertRaises(Exception):
            hash(B())

        hashWantsException = False
        hash(B())

    def test_named_tuple_from_dict(self):
        N = NamedTuple(x=int, y=str, z=OneOf(None, "hihi"))
        self.assertEqual(N().x, 0)
        self.assertEqual(N().y, "")
        self.assertEqual(N().z, None)

        self.assertEqual(N({}).x, 0)
        self.assertEqual(N({}).y, "")
        self.assertEqual(N({}).z, None)

        self.assertEqual(N({'x': 20}).x, 20)
        self.assertEqual(N({'x': 20, 'y': "30"}).y, "30")
        self.assertEqual(N({'y': "30", 'x': 20}).y, "30")
        self.assertEqual(N({'z': "hihi"}).z, "hihi")

        with self.assertRaises(Exception):
            N({'r': 'hi'})
            N({'y': 'hi', 'z': "not hihi"})
            N({'a': 0, 'b': 0, 'c': 0, 'd': 0})

    def test_named_tuple_comparison(self):
        N = NamedTuple(x=OneOf(None, int), y=OneOf(None, int))

        class S(N):
            pass

        self.assertEqual(N(x=1, y=2), N(x=1, y=2))
        self.assertNotEqual(N(x=1, y=2), N(x=1, y=3))
        self.assertFalse(N(x=1, y=2) == N(x=1, y=3))

        self.assertEqual(S(x=1, y=2), S(x=1, y=2))
        self.assertNotEqual(S(x=1, y=2), S(x=1, y=3))
        self.assertFalse(S(x=1, y=2) == S(x=1, y=3))

    def test_named_tuple_replacing_argument_errors(self):
        N = NamedTuple(a=int, b=str)
        n = N(a=10, b='20')

        with self.assertRaises(ValueError) as context:
            n.replacing(10)

        self.assertTrue("Only keyword arguments are allowed." in str(context.exception), str(context.exception))

        with self.assertRaises(ValueError) as context:
            n.replacing()

        self.assertTrue("No arguments provided." in str(context.exception), str(context.exception))

        with self.assertRaises(ValueError) as context:
            n.replacing(a=1, b='xx', c=2)

        self.assertTrue("Argument 'c' is not in the tuple definition." in str(context.exception), str(context.exception))

    def test_named_tuple_replacing_function(self):
        N = NamedTuple(a=int, b=str)

        n1 = N(a=1, b='xx')
        n1_copy = n1

        self.assertIsInstance(n1, N)
        self.assertIsInstance(n1.a, int)
        self.assertIsInstance(n1.b, str)
        self.assertEqual(n1.a, 1)
        self.assertEqual(n1.b, 'xx')

        n2 = n1.replacing(a=2)

        self.assertIsInstance(n2, N)

        self.assertTrue(n1 is n1_copy)
        self.assertTrue(n1 == n1_copy)

        self.assertFalse(n1 is n2)
        self.assertIsInstance(n2.a, int)
        self.assertIsInstance(n2.b, str)
        self.assertEqual(n2.a, 2)
        self.assertEqual(n2.b, 'xx')

        n3 = n2.replacing(b='yy', a=3)

        self.assertIsInstance(n3, N)

        self.assertFalse(n2 is n3)
        self.assertIsInstance(n3.a, int)
        self.assertIsInstance(n3.b, str)
        self.assertEqual(n3.a, 3)
        self.assertEqual(n3.b, 'yy')

    def test_named_tuple_replacing_refcount(self):
        N = NamedTuple(x=ListOf(int))
        aList = ListOf(int)([1, 2, 3])

        self.assertEqual(_types.refcount(aList), 1)
        nt = N().replacing(x=aList)
        self.assertEqual(nt.x, aList)
        self.assertEqual(_types.refcount(aList), 2)
        nt = None
        self.assertEqual(_types.refcount(aList), 1)

    def test_named_tuple_replacing_subclass(self):
        class NTSubclass(NamedTuple(x=int, y=str)):
            def f(self, y):
                return self.replacing(y=y)

        nt2 = NTSubclass(x=10, y="hi").replacing(x=20)

        self.assertIsInstance(nt2, NTSubclass)
        self.assertIsInstance(nt2.f("a string"), NTSubclass)
        self.assertEqual(nt2.f("a string").y, "a string")

    def test_construct_named_tuple_from_tuple(self):
        NT = NamedTuple(x=int, y=str)

        self.assertEqual(NT((1, "2")), NT(x=1, y="2"))

    def test_repr_of_string_in_named_tuple(self):
        NT = NamedTuple(x=str)

        self.assertEqual(repr(NT(x="asdf\nbsdf")), '(x="asdf\\nbsdf",)')
        self.assertEqual(repr(NT(x="asdf\rbsdf")), '(x="asdf\\rbsdf",)')
        self.assertEqual(repr(NT(x="asdf\tbsdf")), '(x="asdf\\tbsdf",)')
        self.assertEqual(repr(NT(x="asdf\x12bsdf")), '(x="asdf\\x12bsdf",)')

    def test_subclassing(self):
        BaseTuple = NamedTuple(x=int, y=float)

        class NTSubclass(BaseTuple):
            def f(self):
                return self.x + self.y

            def __repr__(self):
                return "ASDF"

        inst = NTSubclass(x=10, y=20)

        self.assertTrue(isinstance(inst, BaseTuple))
        self.assertTrue(isinstance(inst, NTSubclass))
        self.assertTrue(type(inst) is NTSubclass)

        self.assertEqual(repr(inst), "ASDF")
        self.assertNotEqual(repr(BaseTuple(inst)), "ASDF")

        self.assertEqual(inst.x, 10)
        self.assertEqual(inst.f(), 30)

        TupleOfSubclass = TupleOf(NTSubclass)

        instTup = TupleOfSubclass((inst, BaseTuple(x=20, y=20.0)))

        self.assertTrue(isinstance(instTup[0], NTSubclass))
        self.assertTrue(isinstance(instTup[1], NTSubclass))
        self.assertEqual(instTup[0].f(), 30)
        self.assertEqual(instTup[1].f(), 40)

        self.assertEqual(BaseTuple(inst).x, 10)

        self.assertTrue(OneOf(None, NTSubclass)(None) is None)
        self.assertTrue(OneOf(None, NTSubclass)(inst) == inst)
