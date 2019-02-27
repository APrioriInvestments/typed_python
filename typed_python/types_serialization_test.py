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

import sys
import os
import importlib
import typed_python.ast_util as ast_util
import threading
import time
import unittest
import numpy
import datetime
import pytz
import gc
import tempfile
import typed_python.dummy_test_module as dummy_test_module

from typed_python.Codebase import Codebase
from typed_python.test_util import currentMemUsageMb

from typed_python import (
    NoneType, TupleOf, ListOf, OneOf, Tuple, NamedTuple, Int64, Float64,
    String, Bool, Bytes, ConstDict, Alternative, serialize, deserialize,
    Dict, SerializationContext
)

module_level_testfun = dummy_test_module.testfunction


class C:
    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class D(C):
    def __init__(self, arg):
        pass


class E(C):
    def __getinitargs__(self):
        return ()


class H(object):
    pass

# Hashable mutable key


class K(object):
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        # Shouldn't support the recursion itself
        return K, (self.value,)


import __main__  # noqa
__main__.C = C
C.__module__ = "__main__"
__main__.D = D
D.__module__ = "__main__"
__main__.E = E
E.__module__ = "__main__"
__main__.H = H
H.__module__ = "__main__"
__main__.K = K
K.__module__ = "__main__"


class myint(int):
    def __init__(self, x):
        self.str = str(x)


class initarg(C):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __getinitargs__(self):
        return self.a, self.b


class metaclass(type):
    pass


class use_metaclass(object, metaclass=metaclass):
    pass


class pickling_metaclass(type):
    def __eq__(self, other):
        return (type(self) == type(other) and
                self.reduce_args == other.reduce_args)

    def __reduce__(self):
        return (create_dynamic_class, self.reduce_args)


def create_dynamic_class(name, bases):
    result = pickling_metaclass(name, bases, dict())
    result.reduce_args = (name, bases)
    return result


def create_data():
    c = C()
    c.foo = 1
    c.bar = 2
    # TODO: add support for complex numbers
    # x = [0, 1, 2.0, 3.0+0j]
    x = [0, 1, 2.0]
    # Append some integer test cases at cPickle.c's internal size
    # cutoffs.
    uint1max = 0xff
    uint2max = 0xffff
    int4max = 0x7fffffff
    x.extend([1, -1,
              uint1max, -uint1max, -uint1max-1,
              uint2max, -uint2max, -uint2max-1,
              int4max, -int4max, -int4max-1])
    y = ('abc', 'abc', c, c)
    x.append(y)
    x.append(y)
    x.append(5)
    return x

# Test classes for newobj


class MyInt(int):
    sample = 1


class MyFloat(float):
    sample = 1.0


class MyComplex(complex):
    sample = 1.0 + 0.0j


class MyStr(str):
    sample = "hello"


class MyUnicode(str):
    sample = "hello \u1234"


class MyBytes(bytes):
    sample = b"hello"


class MyTuple(tuple):
    sample = (1, 2, 3)


class MyList(list):
    sample = [1, 2, 3]


class MyDict(dict):
    sample = {"a": 1, "b": 2}


class MySet(set):
    sample = {"a", "b"}


class MyFrozenSet(frozenset):
    sample = frozenset({"a", "b"})


myclasses = [MyInt, MyFloat,
             MyComplex,
             MyStr, MyUnicode,
             MyTuple, MyList, MyDict, MySet, MyFrozenSet]


REDUCE_A = 'reduce_A'


class AAA(object):
    def __reduce__(self):
        return str, (REDUCE_A,)


sc = SerializationContext({
    'initarg': initarg,
    'C': C,
    'D': D,
    'E': E,
    'H': H,
    'K': K,
    'MyInt': MyInt,
    'MyFloat': MyFloat,
    'MyComplex': MyComplex,
    'MyStr': MyStr,
    'MyUnicode': MyUnicode,
    'MyBytes': MyBytes,
    'MyTuple': MyTuple,
    'MyList': MyList,
    'MyDict': MyDict,
    'MySet': MySet,
    'MyFrozenSet': MyFrozenSet,
    'use_metaclass': use_metaclass,
    'metaclass': metaclass,
    'pickling_metaclass': pickling_metaclass,
    'AAA': AAA,
})


def ping_pong(obj, serialization_context=None):
    serialization_context = serialization_context or SerializationContext()
    s = serialization_context.serialize(obj)
    return serialization_context.deserialize(s)


class TypesSerializationTest(unittest.TestCase):
    def assert_is_copy(self, obj, objcopy, msg=None):
        """Utility method to verify if two objects are copies of each others.
        """
        if msg is None:
            msg = "{!r} is not a copy of {!r}".format(obj, objcopy)
        self.assertEqual(obj, objcopy, msg=msg)
        self.assertIs(type(obj), type(objcopy), msg=msg)
        if hasattr(obj, '__dict__'):
            if isinstance(obj.__dict__, dict):
                self.assertDictEqual(obj.__dict__, objcopy.__dict__, msg=msg)
            self.assertIsNot(obj.__dict__, objcopy.__dict__, msg=msg)
        if hasattr(obj, '__slots__'):
            self.assertListEqual(obj.__slots__, objcopy.__slots__, msg=msg)
            for slot in obj.__slots__:
                self.assertEqual(
                    hasattr(obj, slot), hasattr(objcopy, slot), msg=msg)
                self.assertEqual(getattr(obj, slot, None),
                                 getattr(objcopy, slot, None), msg=msg)

    def check_idempotence(self, obj, ser_ctx=None):
        ser_ctx = ser_ctx or SerializationContext()

        self.assert_is_copy(obj, ping_pong(obj, ser_ctx))

    def test_serialize_core_python_objects(self):
        self.check_idempotence(0)
        self.check_idempotence(10)
        self.check_idempotence(-10)
        self.check_idempotence(-0.0)
        self.check_idempotence(0.0)
        self.check_idempotence(10.5)
        self.check_idempotence(-10.5)
        self.check_idempotence(None)
        self.check_idempotence(True)
        self.check_idempotence(False)
        self.check_idempotence("")
        self.check_idempotence("a string")
        self.check_idempotence(b"")
        self.check_idempotence(b"some bytes")

        self.check_idempotence(())
        self.check_idempotence((1,))
        self.check_idempotence((1, 2, 3))
        self.check_idempotence({})
        self.check_idempotence({"key": "value"})
        self.check_idempotence({"key": "value", "key2": "value2"})
        self.check_idempotence([])
        self.check_idempotence([1, 2, 3])
        self.check_idempotence(set())
        self.check_idempotence({1, 2, 3})
        self.check_idempotence(frozenset())
        self.check_idempotence(frozenset({1, 2, 3}))

        self.check_idempotence(int)
        self.check_idempotence(object)
        self.check_idempotence(type)

    def test_serialize_python_dict(self):
        d = {1: 2, 3: '4', '5': 6, 7.0: b'8'}
        self.check_idempotence(d)

    def test_serialize_recursive_list(self):

        def check_reclist(size):
            init = list(range(size))
            reclist = list(init)
            reclist.append(reclist)
            alt_reclist = ping_pong(reclist)

            for i in range(size):
                self.assertEqual(init[i], alt_reclist[i])
                self.assertEqual(reclist[i], alt_reclist[i])
            self.assertIs(alt_reclist[size], alt_reclist)

        for i in range(4):
            check_reclist(i)

    def test_serialize_memoizes_tuples(self):
        ts = SerializationContext()

        lst = (1, 2, 3)
        for i in range(100):
            lst = (lst, lst)
            self.assertTrue(len(ts.serialize(lst)) < (i+1) * 100)

    def test_serialize_objects(self):
        class AnObject:
            def __init__(self, o):
                self.o = o

        ts = SerializationContext({'O': AnObject})

        o = AnObject(123)

        o2 = ping_pong(o, ts)

        self.assertIsInstance(o2, AnObject)
        self.assertEqual(o2.o, 123)

    def test_serialize_recursive_object(self):
        class AnObject:
            def __init__(self, o):
                self.o = o

        ts = SerializationContext({'O': AnObject})

        o = AnObject(None)
        o.o = o

        o2 = ping_pong(o, ts)
        self.assertIs(o2.o, o2)

    def test_serialize_primitive_native_types(self):
        for t in [Int64, Float64, Bool, NoneType, String, Bytes]:
            self.assertIs(ping_pong(t), t)

    def test_serialize_primitive_compound_types(self):
        class A:
            pass

        B = Alternative("B", X={'a': A})

        ts = SerializationContext({'A': A, 'B': B})

        for t in [  ConstDict(int, float),
                    NamedTuple(x=int, y=str),
                    TupleOf(bool),
                    Tuple(int, int, bool),
                    OneOf(int, float),
                    OneOf(1, 2, 3, "hi", b"goodbye"),
                    TupleOf(NamedTuple(x=int)),
                    TupleOf(object),
                    TupleOf(A),
                    TupleOf(B)
                    ]:
            self.assertIs(ping_pong(t, ts), t)

    def test_serialize_functions(self):
        def f():
            return 10

        ts = SerializationContext({'f': f})
        self.assertIs(ping_pong(f, ts), f)

    def test_serialize_alternatives(self):
        A = Alternative("A", X={'a': int}, Y={'a': lambda: A})

        ts = SerializationContext({'A': A})
        self.assertIs(ping_pong(A.X, ts), A.X)

    def test_serialize_lambdas(self):

        def check(f, args):
            self.assertEqual(f(*args), ping_pong(f)(*args))

        y = 20

        def f(x):
            return x + 1

        def f2(x):
            return x + y

        check(f, (10,))
        check(f2, (10,))
        check(lambda x: x+1, (10,))
        check(lambda x: (x, True, False), (10,))
        check(lambda x: (x, "hi"), (10,))
        check(lambda x: (x, None), (10,))
        check(lambda x: x+y, (10,))

    def test_serialize_class_instance(self):
        class A:
            def __init__(self, x):
                self.x = x

            def f(self):
                return b"an embedded string"

        ts = SerializationContext({'A': A})
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

        ts.compressionEnabled = False

        self.assertTrue(numpy.all(x == ts.deserialize(ts.serialize(x))))

        sizeNotCompressed = len(ts.serialize(x))

        self.assertTrue(sizeNotCompressed > sizeCompressed * 2, (sizeNotCompressed, sizeCompressed))

    def test_serialize_and_numpy_with_dicts(self):
        x = numpy.ones(10000)

        self.assertTrue(numpy.all(ping_pong({'a': x, 'b': x})['a'] == x))

    def test_serialize_and_threads(self):
        class A:
            def __init__(self, x):
                self.x = x

        ts = SerializationContext({'A': A})

        OK = []

        def thread():
            t0 = time.time()
            while time.time() - t0 < 1.0:
                ping_pong(A(10), ts)
            OK.append(True)

        threads = [threading.Thread(target=thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(OK), len(threads))

    def test_serialize_named_tuple(self):
        X = NamedTuple(x=int)
        self.check_idempotence(X(x=20))

    def test_serialize_named_tuple_subclass(self):
        class X(NamedTuple(x=int)):
            def f(self):
                return self.x

        ts = SerializationContext({'X': X})

        self.assertIs(ping_pong(X, ts), X)

        self.assertTrue(ts.serialize(X(x=20)) != ts.serialize(X(x=21)))

        self.check_idempotence(X(x=20), ts)

    def test_bad_serialization_context(self):
        with self.assertRaises(AssertionError):
            SerializationContext({'': int})

        with self.assertRaises(AssertionError):
            SerializationContext({b'': int})

    def test_serialization_context_queries(self):
        sc = SerializationContext({
            'X': False,
            'Y': True,
        })
        self.assertIs(sc.objectFromName('X'), False)
        self.assertIs(sc.nameForObject(False), 'X')
        self.assertIs(sc.objectFromName('Y'), True)
        self.assertIs(sc.nameForObject(True), 'Y')

    def test_serializing_dicts_in_loop(self):
        self.serializeInLoop(lambda: 1)
        self.serializeInLoop(lambda: {})
        self.serializeInLoop(lambda: {1: 2})
        self.serializeInLoop(lambda: {1: {2: 3}})

    def test_serializing_tuples_in_loop(self):
        self.serializeInLoop(lambda: ())
        self.serializeInLoop(lambda: (1, 2, 3))
        self.serializeInLoop(lambda: (1, 2, (3, 4,), ((5, 6), (((6,),),))))

    def test_serializing_lists_in_loop(self):
        self.serializeInLoop(lambda: [])
        self.serializeInLoop(lambda: [1, 2, 3, 4])
        self.serializeInLoop(lambda: [1, 2, [3, 4, 5], [6, [[[[]]]]]])

    def test_serializing_objects_in_loop(self):
        class X:
            def __init__(self, a=None, b=None, c=None):
                self.a = a
                self.b = b
                self.c = c
        c = SerializationContext({'X': X})

        self.serializeInLoop(lambda: X(a=X(), b=[1, 2, 3], c=X(a=X())), context=c)

    def test_serializing_numpy_arrays_in_loop(self):
        self.serializeInLoop(lambda: numpy.array([]))
        self.serializeInLoop(lambda: numpy.array([1, 2, 3]))
        self.serializeInLoop(lambda: numpy.array([[1, 2, 3], [2, 3, 4]]))
        self.serializeInLoop(lambda: numpy.ones(2000))

    def test_serializing_anonymous_recursive_named_tuples(self):
        NT = NamedTuple(x=OneOf(int, float), y=OneOf(int, lambda: NT))

        nt = NT(x=10, y=NT(x=20, y=2))

        nt_ponged = ping_pong(nt)

        self.assertEqual(nt_ponged.y.x, 20)

    def test_serializing_named_tuples_in_loop(self):
        NT = NamedTuple(x=OneOf(int, float), y=OneOf(int, lambda: NT))

        context = SerializationContext({'NT': NT})

        self.serializeInLoop(lambda: NT(x=10, y=NT(x=20, y=2)), context=context)

    def test_serializing_tuple_of_in_loop(self):
        TO = TupleOf(int)

        context = SerializationContext({'TO': TO})

        self.serializeInLoop(lambda: TO((1, 2, 3, 4, 5)), context=context)

    def test_serializing_alternatives_in_loop(self):
        AT = Alternative("AT", X={'x': int, 'y': float}, Y={'x': int, 'y': lambda: AT})

        context = SerializationContext({'AT': AT})

        self.serializeInLoop(lambda: AT, context=context)
        self.serializeInLoop(lambda: AT.Y, context=context)
        self.serializeInLoop(lambda: AT.X(x=10, y=20), context=context)

    def test_inject_exception_into_context(self):
        NT = NamedTuple()

        context = SerializationContext({'NT': NT})
        context2 = SerializationContext({'NT': NT})

        def throws(*args):
            raise Exception("Test Exception")

        context.nameForObject = throws
        context2.objectFromName = throws

        with self.assertRaisesRegex(Exception, "Test Exception"):
            context.serialize(NT)

        data = context2.serialize(NT)
        with self.assertRaisesRegex(Exception, "Test Exception"):
            context2.deserialize(data)

    def serializeInLoop(self, objectMaker, context=None):
        context = context or SerializationContext({})
        memUsage = currentMemUsageMb()

        t0 = time.time()

        while time.time() - t0 < .25:
            data = context.serialize(objectMaker())
            context.deserialize(data)

        gc.collect()
        self.assertLess(currentMemUsageMb() - memUsage, 1.0)

    ##########################################################################
    # The Tests below are  Adapted from pickletester.py in cpython/Lib/test

    def test_serialize_roundtrip_equality(self):
        expected = create_data()
        got = ping_pong(expected, sc)
        self.assert_is_copy(expected, got)

    def test_serialize_recursive_tuple_and_list(self):
        t = ([],)
        t[0].append(t)

        x = ping_pong(t)
        self.assertIsInstance(x, tuple)
        self.assertEqual(len(x), 1)
        self.assertIsInstance(x[0], list)
        self.assertEqual(len(x[0]), 1)
        self.assertIs(x[0][0], x)

    def test_serialize_recursive_dict(self):
        d = {}
        d[1] = d

        x = ping_pong(d)
        self.assertIsInstance(x, dict)
        self.assertEqual(list(x.keys()), [1])
        self.assertIs(x[1], x)

    def test_serialize_recursive_dict_key(self):
        d = {}
        k = K(d)
        d[k] = 1

        x = ping_pong(d, sc)
        self.assertIsInstance(x, dict)
        self.assertEqual(len(x.keys()), 1)
        self.assertIsInstance(list(x.keys())[0], K)
        self.assertIs(list(x.keys())[0].value, x)

    def test_serialize_recursive_set(self):
        y = set()
        k = K(y)
        y.add(k)

        x = ping_pong(y, sc)
        self.assertIsInstance(x, set)
        self.assertEqual(len(x), 1)
        self.assertIsInstance(list(x)[0], K)
        self.assertIs(list(x)[0].value, x)

    def test_serialize_recursive_inst(self):
        i = C()
        i.attr = i

        x = ping_pong(i, sc)
        self.assertIsInstance(x, C)
        self.assertEqual(dir(x), dir(i))
        self.assertIs(x.attr, x)

    def test_serialize_recursive_multi(self):
        lst = []
        d = {1: lst}
        i = C()
        i.attr = d
        lst.append(i)

        x = ping_pong(lst, sc)
        self.assertIsInstance(x, list)
        self.assertEqual(len(x), 1)
        self.assertEqual(dir(x[0]), dir(i))
        self.assertEqual(list(x[0].attr.keys()), [1])
        self.assertTrue(x[0].attr[1] is x)

    def check_recursive_collection_and_inst(self, factory):
        h = H()
        y = factory([h])
        h.attr = y

        x = ping_pong(y, sc)
        self.assertIsInstance(x, type(y))
        self.assertEqual(len(x), 1)
        self.assertIsInstance(list(x)[0], H)
        self.assertIs(list(x)[0].attr, x)

    def test_serialize_recursive_list_and_inst(self):
        self.check_recursive_collection_and_inst(list)

    def test_serialize_recursive_tuple_and_inst(self):
        self.check_recursive_collection_and_inst(tuple)

    def test_serialize_recursive_dict_and_inst(self):
        self.check_recursive_collection_and_inst(dict.fromkeys)

    def test_serialize_recursive_set_and_inst(self):
        self.check_recursive_collection_and_inst(set)

    def test_serialize_recursive_frozenset_and_inst(self):
        self.check_recursive_collection_and_inst(frozenset)

    def test_serialize_base_type_subclass(self):
        with self.assertRaises(TypeError):
            sc.serialize(MyInt())

        with self.assertRaises(TypeError):
            sc.serialize(MyFloat())

        with self.assertRaises(TypeError):
            sc.serialize(MyComplex())

        with self.assertRaises(TypeError):
            sc.serialize(MyStr())

        with self.assertRaises(TypeError):
            sc.serialize(MyUnicode())

        with self.assertRaises(TypeError):
            sc.serialize(MyBytes())

        with self.assertRaises(TypeError):
            sc.serialize(MyTuple())

        with self.assertRaises(TypeError):
            sc.serialize(MyList())

        with self.assertRaises(TypeError):
            sc.serialize(MyDict())

        with self.assertRaises(TypeError):
            sc.serialize(MySet())

        with self.assertRaises(TypeError):
            sc.serialize(MyFrozenSet())

    # THIS FAILS
    @unittest.skip
    def test_serialize_unicode_1(self):
        endcases = ['', '<\\u>', '<\\\u1234>', '<\n>',
                    '<\\>', '<\\\U00012345>',
                    # surrogates
                    '<\udc80>']

        for u in endcases:
            print("u = {}".format(u))
            u2 = ping_pong(u)
            self.assert_is_copy(u, u2)

    def test_serialize_unicode_high_plane(self):
        t = '\U00012345'

        t2 = ping_pong(t)
        self.assert_is_copy(t, t2)

    def test_serialize_bytes(self):
        for s in b'', b'xyz', b'xyz'*100:
            s2 = ping_pong(s)
            self.assert_is_copy(s, s2)

        for s in [bytes([i]) for i in range(256)]:
            s2 = ping_pong(s)
            self.assert_is_copy(s, s2)

        for s in [bytes([i, i]) for i in range(256)]:
            s2 = ping_pong(s)
            self.assert_is_copy(s, s2)

    def test_serialize_ints(self):
        n = sys.maxsize
        while n:
            for expected in (-n, n):
                n2 = ping_pong(expected)
                self.assert_is_copy(expected, n2)
            n = n >> 1

    # FAILS
    @unittest.skip
    def test_serialize_long(self):
        # 256 bytes is where LONG4 begins.
        for nbits in 1, 8, 8*254, 8*255, 8*256, 8*257:
            nbase = 1 << nbits
            for npos in nbase-1, nbase, nbase+1:
                for n in npos, -npos:
                    got = ping_pong(n)
                    self.assert_is_copy(n, got)
        # Try a monster.
        nbase = int("deadbeeffeedface", 16)
        nbase += nbase << 1000000
        for n in nbase, -nbase:
            got = ping_pong(n)
            # assert_is_copy is very expensive here as it precomputes
            # a failure message by computing the repr() of n and got,
            # we just do the check ourselves.
            self.assertIs(type(got), int)
            self.assertEqual(n, got)

    def test_serialize_float(self):
        test_values = [0.0, 4.94e-324, 1e-310, 7e-308, 6.626e-34, 0.1, 0.5,
                       3.14, 263.44582062374053, 6.022e23, 1e30]
        test_values = test_values + [-x for x in test_values]

        for value in test_values:
            got = ping_pong(value)
            self.assert_is_copy(value, got)

    def test_serialize_numpy_float(self):
        deserializedVal = ping_pong(numpy.float64(1.0))

        self.assertEqual(deserializedVal, 1.0)
        self.assertIsInstance(deserializedVal, numpy.float64)

    # FAILS
    @unittest.skip
    def test_serialize_reduce(self):
        inst = AAA()
        loaded = ping_pong(inst, sc)
        self.assertEqual(loaded, REDUCE_A)

    # FAILS with: TypeError: tp_new threw an exception
    @unittest.skip
    def test_serialize_getinitargs(self):
        inst = initarg(1, 2)
        loaded = ping_pong(inst)
        self.assert_is_copy(inst, loaded)

    def test_serialize_metaclass(self):
        a = use_metaclass()
        b = ping_pong(a, sc)
        self.assertEqual(a.__class__, b.__class__)

    # Didn't even bother
    @unittest.skip
    def test_serialize_dynamic_class(self):
        import copyreg
        a = create_dynamic_class("my_dynamic_class", (object,))
        copyreg.pickle(pickling_metaclass, pickling_metaclass.__reduce__)

        b = ping_pong(a)
        self.assertEqual(a, b)
        self.assertIs(type(a), type(b))

    # FAILS with: TypeError: Classes derived from `tuple` cannot be serialized
    @unittest.skip
    def test_serialize_structseq(self):
        import time
        import os

        t = time.localtime()
        u = ping_pong(t)
        self.assert_is_copy(t, u)
        if hasattr(os, "stat"):
            t = os.stat(os.curdir)
            u = ping_pong(t)
            self.assert_is_copy(t, u)
        if hasattr(os, "statvfs"):
            t = os.statvfs(os.curdir)
            u = ping_pong(t)
            self.assert_is_copy(t, u)

    # FAILS
    @unittest.skip
    def test_serialize_ellipsis(self):
        u = ping_pong(...)
        self.assertIs(..., u)

    # FAILS
    @unittest.skip
    def test_serialize_notimplemented(self):
        u = ping_pong(NotImplemented)
        self.assertIs(NotImplemented, u)

    # FAILS
    @unittest.skip
    def test_serialize_singleton_types(self):
        # Issue #6477: Test that types of built-in singletons can be pickled.
        singletons = [None, ..., NotImplemented]
        for singleton in singletons:
            u = ping_pong(type(singleton))
            self.assertIs(type(singleton), u)

    def test_serialize_many_puts_and_gets(self):
        # Test that internal data structures correctly deal with lots of
        # puts/gets.
        keys = ("aaa" + str(i) for i in range(100))
        large_dict = dict((k, [4, 5, 6]) for k in keys)
        obj = [dict(large_dict), dict(large_dict), dict(large_dict)]

        loaded = ping_pong(obj)
        self.assert_is_copy(obj, loaded)

    # FAILS with: AssertionError: 'bar' is not 'bar'
    @unittest.skip
    def test_serialize_attribute_name_interning(self):
        # Test that attribute names of pickled objects are interned when
        # unpickling.
        x = C()
        x.foo = 42
        x.bar = "hello"

        y = ping_pong(x, sc)
        x_keys = sorted(x.__dict__)
        y_keys = sorted(y.__dict__)
        for x_key, y_key in zip(x_keys, y_keys):
            self.assertIs(x_key, y_key)

    def test_serialize_large_pickles(self):
        # Test the correctness of internal buffering routines when handling
        # large data.
        data = (1, min, b'xy' * (30 * 1024), len)
        loaded = ping_pong(data, sc)
        self.assertEqual(len(loaded), len(data))
        self.assertEqual(loaded, data)

    def test_serialize_nested_names(self):
        global Nested

        class Nested:
            class A:
                class B:
                    class C:
                        pass

        sc = SerializationContext({
            'Nested': Nested,
            'Nested.A': Nested.A,
            'Nested.A.B': Nested.A.B,
            'Nested.A.B.C': Nested.A.B.C
        })

        for obj in [Nested.A, Nested.A.B, Nested.A.B.C]:
            with self.subTest(obj=obj):
                unpickled = ping_pong(obj, sc)
                self.assertIs(obj, unpickled)

    def test_serialize_lambdas(self):
        sc = SerializationContext({})

        with tempfile.TemporaryDirectory() as tf:
            fpath = os.path.join(tf, "weird_serialization_test.py")
            with open(fpath, "w") as f:
                f.write("def f(x):\n    return x + 1\n")
            sys.path.append(tf)

            m = importlib.import_module('weird_serialization_test')

            # verify we can serialize this
            deserialized_f = sc.deserialize(sc.serialize(m.f))

            self.assertEqual(deserialized_f(10), 11)

        assert not os.path.exists(fpath)

        ast_util.clearAllCaches()

        # at this point, the backing data for serialization is not there
        # and also, the cache is cleared.
        deserialized_f_2 = sc.deserialize(sc.serialize(deserialized_f))

        self.assertEqual(deserialized_f_2(10), 11)

    def test_serialize_result_of_decorator(self):
        sc = SerializationContext({})

        def decorator(f):
            def addsOne(x):
                return f(x) + 1

            return addsOne

        @decorator
        def g(x):
            return x + 1

        g2 = sc.deserialize(sc.serialize(g))

        self.assertEqual(g2(10), g(10))

    def test_serialize_modules(self):
        codebase = Codebase._FromModule(dummy_test_module)
        sc = codebase.serializationContext

        self.assertIn('.modules.pytz', sc.nameToObject)

        pytz = dummy_test_module.pytz
        self.assertIs(pytz, sc.deserialize(sc.serialize(pytz)))

    def test_serialize_lambdas_with_references_in_list_comprehensions(self):
        codebase = Codebase._FromModule(dummy_test_module)
        sc = codebase.serializationContext

        # note that it matters that the 'module_level_testfun' is at the module level,
        # because that induces a freevar in a list-comprehension code object
        def f():
            return [module_level_testfun() for _ in range(1)][0]

        self.assertEqual(f(), "testfunction")

        self.assertEqual(sc.deserialize(sc.serialize(f))(), "testfunction")

    def test_serialize_large_lists(self):
        x = SerializationContext({})

        lst = ListOf(ListOf(int))()

        lst.resize(100)
        for sublist in lst:
            sublist.resize(1000000)

        t0 = time.time()
        l2 = x.deserialize(x.serialize(lst))
        print(time.time() - t0, " to roundtrip")

        self.assertEqual(lst, l2)

    def test_serialize_large_numpy_arrays(self):
        x = SerializationContext({})

        a = numpy.arange(100000000)
        a2 = x.deserialize(x.serialize(a))

        self.assertTrue(numpy.all(a == a2))

    def test_serialize_datetime_objects(self):
        x = SerializationContext({})

        d = datetime.date.today()
        d2 = x.deserialize(x.serialize(d))
        self.assertEqual(d, d2, (d, type(d)))

        d = datetime.datetime.now()
        d2 = x.deserialize(x.serialize(d))
        self.assertEqual(d, d2, (d, type(d)))

        d = datetime.timedelta(days=1)
        d2 = x.deserialize(x.serialize(d))
        self.assertEqual(d, d2, (d, type(d)))

        d = datetime.datetime.now().time()
        d2 = x.deserialize(x.serialize(d))
        self.assertEqual(d, d2, (d, type(d)))

        d = pytz.timezone("America/New_York")
        d2 = x.deserialize(x.serialize(d))
        self.assertEqual(d, d2, (d, type(d)))

        d = pytz.timezone("America/New_York").localize(datetime.datetime.now())
        d2 = x.deserialize(x.serialize(d))
        self.assertEqual(d, d2, (d, type(d)))

    def test_serialize_dict(self):
        x = SerializationContext({})

        d = Dict(str, str)()
        d["hi"] = "hi"
        d["a"] = "a"

        d2 = x.deserialize(x.serialize(d))

        self.assertEqual(d, d2)

    def test_serialize_recursive_dict(self):
        D = Dict(str, OneOf(str, lambda: D))
        x = SerializationContext({"D": D})

        d = D()

        d["hi"] = "bye"
        d["recurses"] = d

        d2 = x.deserialize(x.serialize(d))

        self.assertEqual(d2['recurses']['recurses']['hi'], 'bye')

    def test_serialize_dict_doesnt_leak(self):
        d = Dict(int, int)({i: i+1 for i in range(100)})
        x = SerializationContext({})

        usage = currentMemUsageMb()
        for _ in range(20000):
            x.deserialize(x.serialize(d))

        self.assertLess(currentMemUsageMb(), usage+1)
