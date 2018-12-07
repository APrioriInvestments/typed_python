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
import numpy
import time
import threading
from typed_python import Int8, NoneType, TupleOf, OneOf, Tuple, NamedTuple, Int64, Float64, String, \
    Bool, Bytes, ConstDict, Alternative, serialize, deserialize, Value, Class, Member, _types, TypedFunction, SerializationContext

class TypesSerializationTest(unittest.TestCase):
    def test_serialize_core_python_objects(self):
        ts = SerializationContext()

        self.assertEqual(ts.deserialize(ts.serialize(10)), 10)
        self.assertEqual(ts.deserialize(ts.serialize(10.5)), 10.5)
        self.assertEqual(ts.deserialize(ts.serialize(None)), None)
        self.assertEqual(ts.deserialize(ts.serialize(True)), True)
        self.assertEqual(ts.deserialize(ts.serialize(False)), False)
        self.assertEqual(ts.deserialize(ts.serialize("a string")), "a string")
        self.assertEqual(ts.deserialize(ts.serialize(b"some bytes")), b"some bytes")
        self.assertEqual(ts.deserialize(ts.serialize((1,2,3))), (1,2,3))
        self.assertEqual(ts.deserialize(ts.serialize({"key":"value"})), {"key":"value"})
        self.assertEqual(ts.deserialize(ts.serialize({"key":"value", "key2": "value2"})), {"key":"value", "key2": "value2"})
        self.assertEqual(ts.deserialize(ts.serialize([1,2,3])), [1,2,3])
        self.assertEqual(ts.deserialize(ts.serialize([1,2,3])), [1,2,3])
        self.assertEqual(ts.deserialize(ts.serialize(int)), int)
        self.assertEqual(ts.deserialize(ts.serialize(object)), object)
        self.assertEqual(ts.deserialize(ts.serialize(type)), type)

    def test_serialize_recursive_list(self):
        ts = SerializationContext()

        l = []
        l.append(l)

        l_alt = ts.deserialize(ts.serialize(l))
        self.assertIs(l_alt[0], l_alt)

    def test_serialize_recursive_dict(self):
        ts = SerializationContext()

        d = {}
        d[0] = d

        d_alt = ts.deserialize(ts.serialize(d))
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

        o2 = ts.deserialize(ts.serialize(o))

        self.assertIsInstance(o2, AnObject)
        self.assertEqual(o2.o, 123)

    def test_serialize_recursive_object(self):
        class AnObject:
            def __init__(self, o):
                self.o = o

        ts = SerializationContext({'O': AnObject})

        o = AnObject(None)
        o.o = o

        o2 = ts.deserialize(ts.serialize(o))
        self.assertIs(o2.o, o2)

    def test_serialize_primitive_native_types(self):
        ts = SerializationContext()
        for t in [Int64, Float64, Bool, NoneType, String, Bytes]:
            self.assertIs(ts.deserialize(ts.serialize(t())), t())

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
            self.assertIs(ts.deserialize(ts.serialize(t)), t)

    def test_serialize_functions(self):
        def f():
            return 10

        ts = SerializationContext({'f': f})
        self.assertIs(ts.deserialize(ts.serialize(f)), f)

    def test_serialize_alternatives(self):
        A = Alternative("A", X={'a': int}, Y={'a': lambda: A})

        ts = SerializationContext({'A': A})
        self.assertIs(ts.deserialize(ts.serialize(A.X)), A.X)

    def test_serialize_lambdas(self):
        ts = SerializationContext()

        def check(f, args):
            self.assertEqual(f(*args), ts.deserialize(ts.serialize(f))(*args))

        y = 20

        def f(x):
            return x + 1

        def f2(x):
            return x + y

        check(f, (10,))
        check(f2, (10,))
        check(lambda x:x+1, (10,))
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

        self.assertTrue(numpy.all(ts.deserialize(ts.serialize({'a': x, 'b': x}))['a'] == x))

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
                ts.deserialize(ts.serialize(A(10)))
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
        self.assertEqual(ts.deserialize(ts.serialize(X(x=20))), X(x=20))


    def test_serialize_named_tuple_subclass(self):
        class X(NamedTuple(x=int)):
            def f(self):
                return self.x

        ts = SerializationContext({'X':X})

        self.assertIs(ts.deserialize(ts.serialize(X)), X)

        self.assertTrue(ts.serialize(X(x=20)) != ts.serialize(X(x=21)))

        self.assertEqual(ts.deserialize(ts.serialize(X(x=20))), X(x=20))
