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

import sys
import os
import importlib
import typed_python.compiler.python_ast_util as python_ast_util
import threading
import textwrap
import time
import unittest
import numpy
import numpy.linalg
import lz4
import lz4.frame
import datetime
import pytest
import pytz
import gc
import pprint
import tempfile
import types
import typed_python.dummy_test_module as dummy_test_module

from typed_python.compiler.native_ast import Expression, NamedCallTarget
from typed_python.test_util import currentMemUsageMb

from typed_python import (
    TupleOf, ListOf, OneOf, Tuple, NamedTuple, Class,
    Member, ConstDict, Alternative, serialize, deserialize,
    Dict, Set, SerializationContext, EmbeddedMessage,
    serializeStream, deserializeStream, decodeSerializedObject,
    Forward, Final, Function, Entrypoint, TypeFunction, PointerTo
)

from typed_python._types import refcount, isRecursive, identityHash, buildPyFunctionObject, setFunctionClosure

module_level_testfun = dummy_test_module.testfunction


@Entrypoint
def moduleLevelEntrypointedFunction(x):
    return x + 1


ModuleLevelAlternative = Alternative(
    "ModuleLevelAlternative",
    X={'a': int},
    Y={'b': float}
)


class ModuleLevelNormalClass:
    def method(self):
        pass


class ModuleLevelNamedTupleSubclass(NamedTuple(x=int)):
    def f(self):
        return self.x


class ModuleLevelClass(Class, Final):
    def f(self):
        return "HI!"


def moduleLevelIdentityFunction(x):
    return x


ModuleLevelRecursiveForward = Forward("ModuleLevelRecursiveForward")
ModuleLevelRecursiveForward = ModuleLevelRecursiveForward.define(
    ConstDict(int, OneOf(None, ModuleLevelRecursiveForward))
)


moduleLevelDict = Dict(int, int)()


def moduleLevelDictGetter(x):
    def f():
        return (moduleLevelDict, x)
    return f


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


import __main__  # noqa: E402
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


@TypeFunction
def FancyClass(T):
    class FancyClass_(Class, Final):
        __name__ = "FancyClass(" + T.__name__ + ")"

        def f(self):
            return 1

    return FancyClass_


def ping_pong(obj, serialization_context=None):
    serialization_context = serialization_context or SerializationContext()
    s = serialization_context.withoutCompression().serialize(obj)
    try:
        return serialization_context.withoutCompression().deserialize(s)
    except Exception:
        print("FAILED TO DECODE:")
        print(s)
        print(pprint.PrettyPrinter(indent=2).pprint(decodeSerializedObject(s)))
        raise


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
        self.check_idempotence([])
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

        self.check_idempotence(TupleOf(int))
        self.check_idempotence(TupleOf(int)([0x08]))

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

    def test_serialize_stream_integers(self):
        for someInts in [(1, 2), TupleOf(int)((1, 2)), [1, 2]]:
            self.assertEqual(
                serializeStream(int, someInts),
                b"".join([serialize(int, x) for x in someInts])
            )

            self.assertEqual(
                deserializeStream(int, serializeStream(int, someInts)),
                TupleOf(int)(someInts)
            )

    def test_serialize_stream_complex(self):
        T = OneOf(None, float, str, int, ListOf(int))

        for items in [
                (1, 2),
                ("hi", None, 10, [1, 2, 3, 4]),
                ()]:
            self.assertEqual(
                serializeStream(T, items),
                b"".join([serialize(T, x) for x in items])
            )

            self.assertEqual(
                deserializeStream(T, serializeStream(T, items)),
                TupleOf(T)(items)
            )

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
        for t in [int, float, bool, type(None), str, bytes]:
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

    def test_serialize_functions_basic(self):
        def f():
            return 10

        ts = SerializationContext({'f': f})
        self.assertIs(ping_pong(f, ts), f)

    def test_serialize_alternatives_as_types(self):
        A = Forward("A")
        A = A.define(Alternative("A", X={'a': int}, Y={'a': A}))

        ts = SerializationContext({'A': A})
        self.assertIs(ping_pong(A, ts), A)
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

    def test_serializing_anonymous_recursive_types(self):
        NT = Forward("NT")
        NT = NT.define(TupleOf(OneOf(int, NT)))
        NT2 = ping_pong(NT)

        # verify we can construct these objects
        nt2 = NT2((1, 2, 3))
        NT2((nt2, 2))

    def test_serializing_named_tuples_in_loop(self):
        NT = Forward("NT")
        NT = NT.define(NamedTuple(x=OneOf(int, float), y=OneOf(int, TupleOf(NT))))

        context = SerializationContext({'NT': NT})

        self.serializeInLoop(lambda: NT(x=10, y=(NT(x=20, y=2),)), context=context)

    def test_serializing_tuple_of_in_loop(self):
        TO = TupleOf(int)

        context = SerializationContext({'TO': TO})

        self.serializeInLoop(lambda: TO((1, 2, 3, 4, 5)), context=context)

    def test_serializing_alternatives_in_loop(self):
        AT = Forward("AT")
        AT = AT.define(Alternative("AT", X={'x': int, 'y': float}, Y={'x': int, 'y': AT}))

        context = SerializationContext({'AT': AT}).withoutCompression()

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
        # this test fails on macos for some reason
        if sys.platform == "darwin":
            return

        context = context or SerializationContext()
        memUsage = currentMemUsageMb()

        t0 = time.time()

        data = context.serialize(objectMaker())

        while time.time() - t0 < .25:
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

    def test_serialize_unicode_1(self):
        endcases = ['', '<\\u>', '<\\\u1234>', '<\n>',
                    '<\\>', '<\\\U00012345>']

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

    @pytest.mark.skip(reason="it fails")
    def test_serialize_reduce(self):
        inst = AAA()
        loaded = ping_pong(inst, sc)
        self.assertEqual(loaded, REDUCE_A)

    @pytest.mark.skip(reason="fails with: tp_new threw an exception")
    def test_serialize_getinitargs(self):
        inst = initarg(1, 2)
        loaded = ping_pong(inst)
        self.assert_is_copy(inst, loaded)

    def test_serialize_metaclass(self):
        a = use_metaclass()
        b = ping_pong(a, sc)
        self.assertEqual(a.__class__, b.__class__)

    @pytest.mark.skip(reason="Didn't even bother")
    def test_serialize_dynamic_class(self):
        import copyreg
        a = create_dynamic_class("my_dynamic_class", (object,))
        copyreg.pickle(pickling_metaclass, pickling_metaclass.__reduce__)

        b = ping_pong(a)
        self.assertEqual(a, b)
        self.assertIs(type(a), type(b))

    @pytest.mark.skip(reason="fails with: TypeError: Classes derived from `tuple` cannot be serialized")
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

    @pytest.mark.skip(reason="fails")
    def test_serialize_ellipsis(self):
        u = ping_pong(...)
        self.assertIs(..., u)

    @pytest.mark.skip(reason="fails")
    def test_serialize_notimplemented(self):
        u = ping_pong(NotImplemented)
        self.assertIs(NotImplemented, u)

    @pytest.mark.skip(reason="fails")
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

    @pytest.mark.skip(reason="fails with: AssertionError: 'bar' is not 'bar'")
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

    def test_serialize_lambdas_more(self):
        sc = SerializationContext()

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

        python_ast_util.clearAllCaches()

        # at this point, the backing data for serialization is not there
        # and also, the cache is cleared.
        deserialized_f_2 = sc.deserialize(sc.serialize(deserialized_f))

        self.assertEqual(deserialized_f_2(10), 11)

    def test_serialize_result_of_decorator(self):
        sc = SerializationContext()

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
        sc = SerializationContext()
        self.assertIs(pytz, sc.deserialize(sc.serialize(pytz)))

    def test_serialize_submodules(self):
        sc = SerializationContext()

        self.assertEqual(
            sc.deserialize(sc.serialize(numpy.linalg)),
            numpy.linalg
        )

        self.assertEqual(
            sc.deserialize(sc.serialize(lz4.frame)),
            lz4.frame
        )

    def test_serialize_lambdas_with_references_in_list_comprehensions(self):
        sc = SerializationContext()

        # note that it matters that the 'module_level_testfun' is at the module level,
        # because that induces a freevar in a list-comprehension code object
        def f():
            return [module_level_testfun() for _ in range(1)][0]

        self.assertEqual(f(), "testfunction")

        self.assertEqual(sc.deserialize(sc.serialize(f))(), "testfunction")

    def test_serialize_large_lists(self):
        x = SerializationContext()

        lst = ListOf(ListOf(int))()

        lst.resize(100)
        for sublist in lst:
            sublist.resize(1000000)

        t0 = time.time()
        l2 = x.deserialize(x.serialize(lst))
        print(time.time() - t0, " to roundtrip")

        self.assertEqual(lst, l2)

    def test_serialize_large_numpy_arrays(self):
        x = SerializationContext()

        a = numpy.arange(100000000)
        a2 = x.deserialize(x.serialize(a))

        self.assertTrue(numpy.all(a == a2))

    def test_serialize_datetime_objects(self):
        x = SerializationContext()

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
        x = SerializationContext()

        d = Dict(str, str)()
        d["hi"] = "hi"
        d["a"] = "a"

        d2 = x.deserialize(x.serialize(d))

        self.assertEqual(d, d2)

    def test_serialize_set(self):
        x = SerializationContext()

        s = Set(str)()
        self.assertEqual(s, x.deserialize(x.serialize(s)))

        s.add("hi")
        self.assertEqual(s, x.deserialize(x.serialize(s)))

        s.add("bye")
        self.assertEqual(s, x.deserialize(x.serialize(s)))

        s.clear()
        self.assertEqual(s, x.deserialize(x.serialize(s)))

    def test_serialize_recursive_dict_more(self):
        D = Forward("D")
        D = D.define(Dict(str, OneOf(str, D)))
        x = SerializationContext({"D": D})

        d = D()

        d["hi"] = "bye"
        d["recurses"] = d

        d2 = x.deserialize(x.serialize(d))

        self.assertEqual(d2['recurses']['recurses']['hi'], 'bye')

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_serialize_dict_doesnt_leak(self):
        T = Dict(int, int)
        d = T({i: i+1 for i in range(100)})
        x = SerializationContext()

        assert not isRecursive(T)

        usage = currentMemUsageMb()
        for _ in range(20000):
            x.deserialize(x.serialize(d))

        self.assertLess(currentMemUsageMb(), usage+1)

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_serialize_array_doesnt_leak(self):
        d = numpy.ones(1000000)
        x = SerializationContext()

        x.deserialize(x.serialize(d))

        usage = currentMemUsageMb()

        for passIx in range(30):
            x.deserialize(x.serialize(d))

        self.assertLess(currentMemUsageMb(), usage+2)

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_deserialize_set_doesnt_leak(self):
        s = set(range(1000000))
        x = SerializationContext()

        x.deserialize(x.serialize(s))

        usage = currentMemUsageMb()

        for _ in range(10):
            x.deserialize(x.serialize(s))
            print(currentMemUsageMb())

        self.assertLess(currentMemUsageMb(), usage+1)

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_deserialize_tuple_doesnt_leak(self):
        s = tuple(range(1000000))
        x = SerializationContext()

        x.deserialize(x.serialize(s))

        usage = currentMemUsageMb()

        for _ in range(10):
            x.deserialize(x.serialize(s))
            print(currentMemUsageMb())

        self.assertLess(currentMemUsageMb(), usage+1)

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_deserialize_list_doesnt_leak(self):
        s = list(range(1000000))
        x = SerializationContext()

        x.deserialize(x.serialize(s))

        usage = currentMemUsageMb()

        for _ in range(10):
            x.deserialize(x.serialize(s))
            print(currentMemUsageMb())

        self.assertLess(currentMemUsageMb(), usage+1)

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_deserialize_class_doesnt_leak(self):
        class C(Class, Final):
            x = Member(int)

            def f(self, x=10):
                return 10

        x = SerializationContext()

        msg = x.serialize(C)
        x.deserialize(msg)

        usage = currentMemUsageMb()

        for passIx in range(1000):
            x.deserialize(msg)

        self.assertLess(currentMemUsageMb(), usage+.5)

    def test_serialize_named_tuples_with_extra_fields(self):
        T1 = NamedTuple(x=int)
        T2 = NamedTuple(x=int, y=float, z=str)

        self.assertEqual(
            deserialize(T2, serialize(T1, T1(x=10))),
            T2(x=10, y=0.0, z="")
        )

    def test_serialize_listof(self):
        T = ListOf(int)

        aList = T()
        aPopulatedList = T([1, 2, 3])

        self.assertEqual(aList, deserialize(T, serialize(T, aList)))
        self.assertEqual(refcount(deserialize(T, serialize(T, aList))), 1)

        self.assertEqual(aPopulatedList, deserialize(T, serialize(T, aPopulatedList)))
        self.assertEqual(refcount(deserialize(T, serialize(T, aPopulatedList))), 1)

    def test_serialize_classes(self):
        class AClass(Class, Final):
            x = Member(int)
            y = Member(float)

        T = Tuple(AClass, AClass)

        a = AClass(x=10, y=20.0)

        a2, a2_copy = deserialize(T, serialize(T, (a, a)))

        self.assertEqual(a2.x, 10)
        a2.x = 300
        self.assertEqual(a2_copy.x, 300)

        a2_copy = None

        self.assertEqual(refcount(a2), 1)

    def test_embedded_messages(self):
        T = NamedTuple(x=TupleOf(int))
        T_with_message = NamedTuple(x=EmbeddedMessage)
        T_with_two_messages = NamedTuple(x=EmbeddedMessage, y=EmbeddedMessage)
        T2 = NamedTuple(x=TupleOf(int), y=TupleOf(int))

        t = T(x=(1, 2, 3, 4))
        tm = deserialize(T_with_message, serialize(T, t))
        tm2 = T_with_two_messages(x=tm.x, y=tm.x)
        t2 = deserialize(T2, serialize(T_with_two_messages, tm2))

        self.assertEqual(t2.x, t.x)
        self.assertEqual(t2.y, t.x)

    def test_serialize_untyped_classes(self):
        sc = SerializationContext()

        class B(object):
            def __init__(self, x):
                self.x = x

            def g(self):
                return self.x

        class C(B):
            def f(self):
                return self.x + 10

        C2 = sc.deserialize(sc.serialize(C))

        self.assertEqual(C2(20).f(), 30)
        self.assertEqual(C2(20).g(), 20)

    def test_serialize_functions_with_return_types(self):
        sc = SerializationContext()

        @Function
        def f(x) -> int:
            return x

        f2 = sc.deserialize(sc.serialize(f))
        self.assertEqual(f2(10.5), 10)

    def test_serialize_functions_with_annotations(self):
        sc = SerializationContext()

        B = int
        C = 10

        @Function
        def f(x: B = C) -> B:
            return x

        f2 = sc.deserialize(sc.serialize(f))
        self.assertEqual(f2(10.5), 10)

        with self.assertRaises(TypeError):
            f2("hi")

        self.assertEqual(f(), f2())

    def test_serialize_typed_classes(self):
        sc = SerializationContext()

        class B(Class):
            x = Member(int)

            def f(self, y) -> int:
                return self.x + y

        class C(B, Final):
            y = Member(int)

            def g(self):
                return self.x + self.y

        B2 = sc.deserialize(sc.serialize(B))

        self.assertEqual(
            B2(x=10).f(20),
            B(x=10).f(20)
        )

        C2 = sc.deserialize(sc.serialize(C))

        self.assertEqual(
            C2(x=10, y=30).f(20),
            C(x=10, y=30).f(20)
        )

        self.assertEqual(
            C2(x=10, y=30).g(),
            C(x=10, y=30).g()
        )

    def test_serialize_recursive_typed_classes(self):
        sc = SerializationContext()

        B = Forward("B")

        @B.define
        class B(Class, Final):
            x = Member(int)

            def f(self, y) -> int:
                return self.x + y

            def getSelf(self) -> B:
                return self

        B2 = sc.deserialize(sc.serialize(B))

        assert identityHash(B) == identityHash(B2)

        instance = B2()

        self.assertTrue(isinstance(instance.getSelf(), B))
        self.assertTrue(isinstance(instance.getSelf(), B2))

        B3 = sc.deserialize(sc.serialize(B2))

        instance2 = B3()

        self.assertTrue(isinstance(instance2.getSelf(), B))
        self.assertTrue(isinstance(instance2.getSelf(), B2))
        self.assertTrue(isinstance(instance2.getSelf(), B3))

    def test_serialize_functions_with_cells(self):
        def fMaker():
            @Entrypoint
            def g(x):
                return x + 1

            @Entrypoint
            def f(x):
                return g(x)

            return f

        sc = SerializationContext()

        f = fMaker()

        f2 = sc.deserialize(sc.serialize(f))

        self.assertEqual(f2(10), 11)

    def test_reserialize_functions(self):
        sc = SerializationContext({'Entrypoint': Entrypoint})

        with tempfile.TemporaryDirectory() as tf:
            fpath = os.path.join(tf, "module.py")

            with open(fpath, "w") as f:
                someText = textwrap.dedent("""
                from typed_python import Forward, Class, Final, Member, Entrypoint

                @Entrypoint
                def g(x):
                    return x + 1

                @Entrypoint
                def f(x):
                    return g(x)

                def getH():
                    y = (lambda x: x + 3, lambda x: x + 4)

                    @Entrypoint
                    def h(x):
                        return (y[0](x), y[1](x))

                    return h
                """)

                f.write(someText)

            code = compile(someText, fpath, "exec")

            moduleVals = {}

            exec(code, moduleVals)

            f = moduleVals['f']
            getH = moduleVals['getH']

            f2 = sc.deserialize(sc.serialize(f))
            getH2 = sc.deserialize(sc.serialize(getH))

            self.assertEqual(f2(10), 11)

        # clear the ast's filesystem cache.
        python_ast_util.clearAllCaches()

        # make sure we can still serialize 'f' itself
        sc.serialize(f)
        sc.serialize(getH())

        # now the directory is deleted. When we reserialize it we shouldn't
        # need it because it should be stashed in the ast cache.
        h = getH2()
        f3 = sc.deserialize(sc.serialize(f2))
        h2 = sc.deserialize(sc.serialize(h))

        self.assertEqual(f3(10), 11)
        self.assertEqual(h2(10), (13, 14))

    def test_serialize_unnamed_classes_retains_identity(self):
        sc = SerializationContext()

        class B:
            def f(self):
                return B

        B2 = sc.deserialize(sc.serialize(B))

        assert B2 is B
        assert B2().f() is B2
        assert B().f() is B2

    def test_serialize_unnamed_typed_classes_retains_identity(self):
        sc = SerializationContext()

        class B(Class):
            def f(self) -> object:
                return B

        B2 = sc.deserialize(sc.serialize(B))

        assert B2 is B
        assert B2().f() is B2
        assert B().f() is B2

    def test_serialize_lambda_preserves_identity_hash(self):
        sc = SerializationContext()

        def aFunction(self, x):
            sys
            return 10

        aFunction2 = sc.deserialize(sc.serialize(aFunction))

        assert identityHash(aFunction) == identityHash(aFunction2)

    def test_serialize_subclasses(self):
        sc = SerializationContext()

        class B(Class):
            x = Member(int)

        class C1(B, Final):
            f = Member(float)

        class C2(B, Final):
            b = Member(B)

        aList = ListOf(B)()

        aList.append(B(x=10))
        aList.append(C1(x=20, f=30.5))
        aList.append(C2(x=30, b=aList[0]))

        aList2 = sc.deserialize(sc.serialize(aList))

        B2 = type(aList2[0])
        C12 = type(aList2[1])
        C22 = type(aList2[2])

        self.assertTrue(issubclass(C12, B2))
        self.assertTrue(issubclass(C22, B2))

        self.assertEqual(aList2[0].x, 10)
        self.assertEqual(aList2[1].x, 20)
        self.assertEqual(aList2[2].x, 30)
        self.assertEqual(aList2[2].b.x, 10)

        # verify that the reference in aList2[2].b points at aList2[0]
        aList2[2].b.x = 100
        self.assertEqual(aList2[0].x, 100)

    def test_serialize_subclasses_multiple_views(self):
        sc = SerializationContext()

        class B(Class):
            x = Member(int)

        class C1(B):
            x2 = Member(int)

        class C2(C1):
            x3 = Member(int)

        class C3(C2):
            x4 = Member(int)

        c = C3()
        t = Tuple(C3, C1, C2, B)((c, c, c, c))

        t = sc.deserialize(sc.serialize(t))

        t[0].x4 = 2
        for e in t:
            self.assertEqual(e.x4, 2)

    def test_serialize_classes_with_staticmethods_and_properties(self):
        sc = SerializationContext()

        class B:
            @staticmethod
            def f(x):
                return x + 1

            @property
            def p(self):
                return 11

        B2 = sc.deserialize(sc.serialize(B))

        self.assertEqual(B2.f(10), 11)
        self.assertEqual(B().p, 11)

    def test_roundtrip_serialization_of_functions_with_annotations(self):
        T = int

        def f() -> T:
            return 1

        sc = SerializationContext()
        f2 = sc.deserialize(sc.serialize(f))
        self.assertEqual(f2(), 1)

        f2Typed = Function(f2)
        self.assertEqual(f2Typed.overloads[0].returnType, int)

    def test_roundtrip_serialization_of_functions_with_defaults(self):
        def f(x=10, *, y=20):
            return x + y

        sc = SerializationContext()
        f2 = sc.deserialize(sc.serialize(f))
        self.assertEqual(f2(), 30)

    def test_roundtrip_serialization_of_functions_with_closures(self):
        F = int

        @Function
        def f():
            return float(moduleLevelIdentityFunction(F(1)))

        def fWrapper():
            return f()

        sc = SerializationContext()
        f2 = sc.deserialize(sc.serialize(f))

        self.assertEqual(f2(), 1)

        fWrapper2 = sc.deserialize(sc.serialize(fWrapper))
        self.assertEqual(fWrapper2(), 1)

    def test_serialize_many_large_equivalent_strings(self):
        sc = SerializationContext()

        def f(x):
            return " " * x + "hi" * x

        someStrings = [f(1000) for _ in range(100)]
        someStrings2 = [f(1000) for _ in range(101)]

        # we memoize strings, so this should be cheap
        self.assertLess(
            len(sc.serialize(someStrings2)) - len(sc.serialize(someStrings)),
            20
        )

    def test_serialize_class_with_classmethod(self):
        class ClassWithClassmethod(Class, Final):
            @classmethod
            def ownName(cls):
                return str(cls)

        sc = SerializationContext()

        ClassWithClassmethod2 = sc.deserialize(sc.serialize(ClassWithClassmethod))

        self.assertEqual(
            ClassWithClassmethod2.ownName(),
            ClassWithClassmethod.ownName(),
        )

    def test_serialize_class_with_nontrivial_signatures(self):
        N = NamedTuple(x=int, y=float)

        class ClassWithStaticmethod(Class, Final):
            @staticmethod
            def hi(x: ListOf(N)):
                return len(x)

        sc = SerializationContext()

        ClassWithStaticmethod2 = sc.deserialize(sc.serialize(ClassWithStaticmethod))

        lst = ListOf(N)()
        lst.resize(2)

        self.assertEqual(
            ClassWithStaticmethod2.hi(lst),
            ClassWithStaticmethod.hi(lst),
        )

    def test_serialize_class_simple(self):
        sc = SerializationContext()
        self.assertTrue(
            sc.deserialize(sc.serialize(C)) is C
        )

    def test_serialize_unnamed_alternative(self):
        X = Alternative("X", A={}, B={'x': int})

        sc = SerializationContext()

        self.assertTrue(
            sc.deserialize(sc.serialize(X)).B(x=2).x == 2
        )

    def test_serialize_mutually_recursive_unnamed_forwards_alternatives(self):
        X1 = Forward("X1")
        X2 = Forward("X2")

        X1 = X1.define(Alternative("X1", A={}, B={'x': X2}))
        X2 = X2.define(Alternative("X2", A={}, B={'x': X1}))

        sc = SerializationContext()
        sc.deserialize(sc.serialize(X1))

    def test_serialize_mutually_recursive_unnamed_forwards_tuples(self):
        X1 = Forward("X1")
        X2 = Forward("X2")

        X1 = X1.define(TupleOf(OneOf(int, X2)))
        X2 = X2.define(TupleOf(OneOf(float, X1)))

        self.assertTrue(isRecursive(X1))
        self.assertTrue(isRecursive(X2))

        self.assertIs(X1.ElementType.Types[1], X2)
        self.assertIs(X2.ElementType.Types[1], X1)

        sc = SerializationContext().withoutCompression()
        sc.deserialize(sc.serialize(X1))

    def test_serialize_named_alternative(self):
        self.assertEqual(
            ModuleLevelAlternative.__typed_python_module__,
            "typed_python.types_serialization_test"
        )

        sc = SerializationContext()

        self.assertIs(
            sc.deserialize(sc.serialize(ModuleLevelAlternative)),
            ModuleLevelAlternative
        )

    def test_serialize_unnamed_recursive_alternative(self):
        X = Forward("X")
        X = X.define(
            Alternative("X", A={}, B={'x': int}, C={'anX': X})
        )

        sc = SerializationContext()

        self.assertTrue(
            sc.deserialize(sc.serialize(X)).B(x=2).x == 2
        )

    def test_serialize_module_level_class(self):
        sc = SerializationContext().withoutCompression()

        self.assertIs(sc.deserialize(sc.serialize(ModuleLevelClass)), ModuleLevelClass)

        self.assertIn(
            b'typed_python.types_serialization_test.ModuleLevelClass',
            sc.serialize(ModuleLevelClass),
        )

    def test_serialize_unnamed_subclass_of_named_tuple(self):
        class SomeNamedTuple(NamedTuple(x=int)):
            def f(self):
                return self.x

        sc = SerializationContext()

        self.assertEqual(
            sc.deserialize(sc.serialize(SomeNamedTuple))(x=10).f(),
            10
        )

        self.assertEqual(
            sc.deserialize(sc.serialize(SomeNamedTuple(x=10))).f(),
            10
        )

    def test_serialize_named_subclass_of_named_tuple(self):
        sc = SerializationContext()

        self.assertEqual(
            ModuleLevelNamedTupleSubclass.__module__,
            "typed_python.types_serialization_test"
        )

        self.assertEqual(
            ModuleLevelNamedTupleSubclass.__name__,
            "ModuleLevelNamedTupleSubclass"
        )

        self.assertIs(
            sc.deserialize(sc.serialize(ModuleLevelNamedTupleSubclass)),
            ModuleLevelNamedTupleSubclass
        )

        self.assertIs(
            type(sc.deserialize(sc.serialize(ModuleLevelNamedTupleSubclass()))),
            ModuleLevelNamedTupleSubclass
        )

        self.assertIs(
            type(sc.deserialize(sc.serialize([ModuleLevelNamedTupleSubclass()]))[0]),
            ModuleLevelNamedTupleSubclass
        )

        self.assertIs(
            sc.deserialize(sc.serialize(ModuleLevelNamedTupleSubclass(x=10))).f(),
            10
        )

    def test_serialize_builtin_tp_functions(self):
        sc = SerializationContext()

        for thing in [
            TupleOf, ListOf, OneOf, Tuple, NamedTuple, Class,
            Member, ConstDict, Alternative, serialize, deserialize,
            Dict, Set, SerializationContext, EmbeddedMessage,
            serializeStream, deserializeStream, decodeSerializedObject,
            Forward, Final, Function, Entrypoint
        ]:
            self.assertIs(
                sc.deserialize(sc.serialize(thing)), thing
            )

    def test_serialize_methods_on_named_classes(self):
        sc = SerializationContext()

        m1 = ModuleLevelNormalClass.method
        m2 = sc.deserialize(sc.serialize(m1))

        assert sc.nameForObject(m1) is not None

        print(m1, m2)

        self.assertIs(m1, m2)

    def test_serialize_entrypointed_modulelevel_functions(self):
        sc = SerializationContext()

        self.assertIs(
            type(moduleLevelEntrypointedFunction),
            sc.deserialize(sc.serialize(type(moduleLevelEntrypointedFunction)))
        )

        self.assertIs(
            type(moduleLevelEntrypointedFunction),
            type(sc.deserialize(sc.serialize(moduleLevelEntrypointedFunction)))
        )

    def test_serialize_entrypointed_modulelevel_class_functions(self):
        sc = SerializationContext()

        self.assertIs(
            type(ModuleLevelClass.f),
            sc.deserialize(sc.serialize(type(ModuleLevelClass.f)))
        )

        self.assertIs(
            type(ModuleLevelClass.f),
            type(sc.deserialize(sc.serialize(ModuleLevelClass.f)))
        )

    def test_serialize_type_function(self):
        sc = SerializationContext()

        self.assertIs(
            FancyClass(int),
            sc.deserialize(sc.serialize(FancyClass(int)))
        )

    def test_serialize_module_level_recursive_forward(self):
        sc = SerializationContext()

        self.assertIs(
            ModuleLevelRecursiveForward,
            sc.deserialize(sc.serialize(ModuleLevelRecursiveForward))
        )

    def test_serialize_reference_to_module_level_constant(self):
        sc = SerializationContext()

        getter = sc.deserialize(sc.serialize(moduleLevelDictGetter(10)))

        assert getter()[0] is moduleLevelDict

    def test_serialize_type_with_reference_to_self_through_closure(self):
        @Entrypoint
        def f(x):
            if x < 0:
                return 0
            return x + C.anF(x-1)

        class C:
            anF = f

        assert f(100) == sum(range(101))

        # C and 'f' are mutually recursive
        sc.deserialize(sc.serialize(C))

    def test_serialize_cell_type(self):
        sc = SerializationContext().withoutInternalizingTypeGroups()

        def f():
            return sc

        cellType = type(f.__closure__[0])

        assert sc.deserialize(sc.serialize(cellType)) is cellType

    def test_serialize_self_referencing_class(self):
        sc = SerializationContext().withoutCompression().withoutInternalizingTypeGroups()

        def g(x):
            return 10

        @TypeFunction
        def C(T):
            class C_(Class, Final):
                @Entrypoint
                def s(self):
                    return C_

                @Entrypoint
                def g(self):
                    return g(10)

            return C_

        C1 = C(int)
        C2 = sc.deserialize(sc.serialize(C1))

        c1 = C1()
        c2 = C2()

        assert c2.g() == 10
        assert c2.s() is C2

        # this should dispatch but we can't assume which compiled version
        # of the code we'll get, so we cant check identity of C2
        assert c1.g() == 10

    def test_serialize_self_referencing_class_through_tuple(self):
        sc = SerializationContext().withoutCompression().withoutInternalizingTypeGroups()

        def g(x):
            return 10

        @TypeFunction
        def C(T):
            class C_(Class, Final):
                @Function
                def s(self):
                    return tup[0]

                @Function
                def g(self):
                    return g(10)

            # this fails because when we serliaze mutually recursive python objects
            # we don't understand all the kinds of objects we can walk
            tup = (C_, 10)

            return C_

        C1 = C(int)
        C2 = sc.deserialize(sc.serialize(C1))

        c1 = C1()
        c2 = C2()

        assert c2.g() == 10
        assert c2.s() is C2

        assert c1.g() == 10
        assert c1.s() is C1

    def test_names_of_builtin_alternatives(self):
        sc = SerializationContext().withoutCompression().withoutInternalizingTypeGroups()

        assert sc.nameForObject(Expression) is not None
        assert sc.nameForObject(Expression.Load) is not None

        assert b'Store' not in sc.serialize(Expression)
        assert b'Store' not in sc.serialize(Expression.Load)

        sc.deserialize(sc.serialize(Expression))
        sc.deserialize(sc.serialize(Expression.Load))

        sc.deserialize(sc.serialize(NamedCallTarget))

    def test_badly_named_module_works(self):
        sc = SerializationContext()

        assert sc.objectFromName(".modules.NOT.A.REAL.MODULE") is None

    def test_can_serialize_nullptrs(self):
        x = PointerTo(int)()

        sc = SerializationContext()

        assert sc.deserialize(sc.serialize(type(x))) == type(x)
        assert sc.deserialize(sc.serialize(x)) == x

    def test_can_serialize_nested_function_references(self):
        sc = SerializationContext().withoutInternalizingTypeGroups()

        def lenProxy(x):
            return len(x)

        otherGlobals = {'__builtins__': __builtins__}

        lenProxyWithNonstandardGlobals = buildPyFunctionObject(
            lenProxy.__code__,
            otherGlobals,
            ()
        )

        assert lenProxy("asdf") == 4
        assert lenProxyWithNonstandardGlobals("asdf") == 4

        lenProxyDeserialized = sc.deserialize(sc.serialize(lenProxyWithNonstandardGlobals))

        assert lenProxyDeserialized("asdf") == 4

    @pytest.mark.skipif('sys.platform=="darwin"')
    def test_set_closure_doesnt_leak(self):
        def makeFunWithClosure(x):
            def f():
                return x
            return f

        aFun = makeFunWithClosure(10)

        mem = currentMemUsageMb()
        for i in range(1000000):
            setFunctionClosure(aFun, makeFunWithClosure(20).__closure__)

        assert currentMemUsageMb() < mem + 1

    def test_serialize_without_line_info_doesnt_have_path(self):
        def aFun():
            return 10

        sc = SerializationContext().withoutCompression().withoutLineInfoEncoded()

        assert b'types_serialization_test' not in sc.serialize(aFun.__code__)
        assert b'types_serialization_test' not in sc.serialize(aFun.__code__)

    def test_serialize_builtin_type_objects(self):
        s = SerializationContext()

        def check(x):
            assert s.deserialize(s.serialize(x)) is x

        check(types.BuiltinFunctionType)
        check(types.FunctionType)
        check(types.ModuleType)
