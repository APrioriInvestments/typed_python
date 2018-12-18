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

import numpy
import threading
import time
import unittest

from typed_python import (
    Int8, NoneType, TupleOf, OneOf, Tuple, NamedTuple, Int64, Float64,
    String, Bool, Bytes, ConstDict, Alternative, serialize, deserialize,
    Value, Class, Member, _types, TypedFunction, SerializationContext
)


def ping_pong(serialization_context, obj):
    return serialization_context.deserialize(
        serialization_context.serialize(obj)
    )


class TypesSerializationTest(unittest.TestCase):

    def check_idempotence(self, ser_ctx, obj):
        self.assertEqual(obj, ping_pong(ser_ctx, obj))

    def test_serialize_core_python_objects(self):
        ts = SerializationContext()


        self.check_idempotence(ts, 10)
        self.check_idempotence(ts, 10.5)
        self.check_idempotence(ts, None)
        self.check_idempotence(ts, True)
        self.check_idempotence(ts, False)
        self.check_idempotence(ts, "a string")
        self.check_idempotence(ts, b"some bytes")
        self.check_idempotence(ts, (1,2,3))
        self.check_idempotence(ts, {"key":"value"})
        self.check_idempotence(ts, {"key":"value", "key2": "value2"})
        self.check_idempotence(ts, [1,2,3])
        self.check_idempotence(ts, int)
        self.check_idempotence(ts, object)
        self.check_idempotence(ts, type)

    def test_serialize_recursive_list(self):
        ts = SerializationContext()

        def check_reclist(size):
            init = list(range(size))
            reclist = list(init)
            reclist.append(reclist)
            alt_reclist = ping_pong(ts, reclist)

            for i in range(size):
                self.assertEqual(init[i], alt_reclist[i])
                self.assertEqual(reclist[i], alt_reclist[i])
            self.assertIs(alt_reclist[size], alt_reclist)

        for i in range(4):
            check_reclist(i)

    def test_serialize_recursive_dict(self):
        ts = SerializationContext()

        d = {}
        d[0] = d

        d_alt = ping_pong(ts, d)
        self.assertIs(d_alt[0], d_alt)

    def test_serialize_memoizes_tuples(self):
        ts = SerializationContext()

        l = (1,2,3)
        for i in range(100):
            l = (l,l)
            self.assertTrue(len(ts.serialize(l)) < (i+1) * 100)

    def test_serialize_objects(self):
        class AnObject:
            def __init__(self, o):
                self.o = o

        ts = SerializationContext({'O': AnObject})

        o = AnObject(123)

        o2 = ping_pong(ts, o)

        self.assertIsInstance(o2, AnObject)
        self.assertEqual(o2.o, 123)

    def test_serialize_recursive_object(self):
        class AnObject:
            def __init__(self, o):
                self.o = o

        ts = SerializationContext({'O': AnObject})

        o = AnObject(None)
        o.o = o

        o2 = ping_pong(ts, o)
        self.assertIs(o2.o, o2)

    def test_serialize_primitive_native_types(self):
        ts = SerializationContext()
        for t in [Int64, Float64, Bool, NoneType, String, Bytes]:
            self.assertIs(ping_pong(ts, t()), t())

    def test_serialize_primitive_compound_types(self):
        class A:
            pass

        B = Alternative("B", X={'a': A})

        ts = SerializationContext({'A': A, 'B': B})

        for t in [  ConstDict(int, float),
                    NamedTuple(x=int, y=str),
                    TupleOf(bool),
                    Tuple(int,int,bool),
                    OneOf(int,float),
                    OneOf(1,2,3,"hi",b"goodbye"),
                    TupleOf(NamedTuple(x=int)),
                    TupleOf(object),
                    TupleOf(A),
                    TupleOf(B)
                    ]:
            self.assertIs(ping_pong(ts, t), t)

    def test_serialize_functions(self):
        def f():
            return 10

        ts = SerializationContext({'f': f})
        self.assertIs(ping_pong(ts, f), f)

    def test_serialize_alternatives(self):
        A = Alternative("A", X={'a': int}, Y={'a': lambda: A})

        ts = SerializationContext({'A': A})
        self.assertIs(ping_pong(ts, A.X), A.X)

    def test_serialize_lambdas(self):
        ts = SerializationContext()

        def check(f, args):
            self.assertEqual(f(*args), ping_pong(ts, f)(*args))

        y = 20

        def f(x):
            return x + 1

        def f2(x):
            return x + y

        check(f, (10,))
        check(f2, (10,))
        check(lambda x:x+1, (10,))
        check(lambda x:(x,True, False), (10,))
        check(lambda x:(x,"hi"), (10,))
        check(lambda x:(x,None), (10,))
        check(lambda x:x+y, (10,))

    def test_serialize_class_instance(self):
        class A:
            def __init__(self, x):
                self.x = x

            def f(self):
                return b"an embedded string"

        ts = SerializationContext({'A':A})
        serialization = ts.serialize(A(10))

        self.assertTrue(b'an embedded string' not in serialization)

        anA = ts.deserialize(serialization)

        self.assertEqual(anA.x, 10)

        anA2 = deserialize(A, serialize(A, A(10), ts), ts)
        self.assertEqual(anA2.x, 10)

    def test_serialize_and_numpy(self):
        x = numpy.ones(10000)
        ts = SerializationContext()

        self.assertTrue(numpy.all(x == ts.deserialize(ts.serialize(x))))

        sizeCompressed = len(ts.serialize(x))

        ts.numpyCompressionEnabled=False

        self.assertTrue(numpy.all(x == ts.deserialize(ts.serialize(x))))

        sizeNotCompressed = len(ts.serialize(x))

        self.assertTrue(sizeNotCompressed > sizeCompressed * 2, (sizeNotCompressed, sizeCompressed))

    def test_serialize_and_numpy_with_dicts(self):
        x = numpy.ones(10000)
        ts = SerializationContext()

        self.assertTrue(numpy.all(ping_pong(ts, {'a': x, 'b': x})['a'] == x))

    def test_serialize_and_threads(self):
        x = numpy.ones(10000)

        class A:
            def __init__(self, x):
                self.x=x

        ts = SerializationContext({'A':A})

        OK = []
        def thread():
            t0 = time.time()
            while time.time() - t0 < 1.0:
                ping_pong(ts, A(10))
            OK.append(True)


        threads = [threading.Thread(target=thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(OK), len(threads))

    def test_serialize_named_tuple(self):
        X = NamedTuple(x=int)
        ts = SerializationContext()
        self.check_idempotence(ts, X(x=20))


    def test_serialize_named_tuple_subclass(self):
        class X(NamedTuple(x=int)):
            def f(self):
                return self.x

        ts = SerializationContext({'X':X})

        self.assertIs(ping_pong(ts, X), X)

        self.assertTrue(ts.serialize(X(x=20)) != ts.serialize(X(x=21)))

        self.check_idempotence(ts, X(x=20))

    def test_bad_serialization_context(self):
        with self.assertRaises(AssertionError):
            ts = SerializationContext({'': int})

        with self.assertRaises(AssertionError):
            ts = SerializationContext({b'': int})

    def test_serialization_context_queries(self):
        sc = SerializationContext({
            'X': False,
            'Y': True,
        })
        self.assertIs(sc.objectFromName('X'), False)
        self.assertIs(sc.nameFromObject(False), 'X')
        self.assertIs(sc.objectFromName('Y'), True)
        self.assertIs(sc.nameFromObject(True), 'Y')
