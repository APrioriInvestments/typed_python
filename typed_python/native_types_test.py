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

from typed_python._types import Int8, NoneType, TupleOf, OneOf, Tuple, NamedTuple, ConstDict, Alternative
import typed_python._types as _types

import unittest
import traceback
import time
import numpy
import sys

def typeFor(t):
    assert not isinstance(t, list), t
    return type(t)

def typeForSeveral(t):
    ts = set(typeFor(a) for a in t)
    if len(ts) == 1:
        return list(ts)[0]
    return OneOf(*ts)

def makeTupleOf(*args):
    if not args:
        return TupleOf(int)()
    return TupleOf(typeForSeveral(args))(args)

def makeNamedTuple(**kwargs):
    if not kwargs:
        return NamedTuple()()
    return NamedTuple(**{k:typeFor(v) for k,v in kwargs.items()})(**kwargs)

def makeTuple(*args):
    if not args:
        return Tuple()()
    return Tuple(*[typeFor(v) for v in args])(args)

def makeDict(d):
    if not d:
        return ConstDict(int,int)()

    return ConstDict(typeForSeveral(d.keys()), typeForSeveral(d.values()))(d)

def makeAlternative(severalDicts):
    types = list(
        set(
            tuple(
                (k,typeFor(v)) for k,v in ntDict.items()
                )
            for ntDict in severalDicts
            )
        )

    alt = Alternative("Alt", **{
        "a_%s" % i: dict(types[i]) for i in range(len(types))
        })

    res = []
    for thing in severalDicts:
        did = False
        for i in range(len(types)):
            try:
                res.append(getattr(alt,"a_%s" % i)(**thing))
                did = True
            except:
                pass

            if did:
                break
    assert len(res) == len(severalDicts)

    return res

def choice(x):
    #numpy.random.choice([1,(1,2)]) blows up because it looks 'multidimensional'
    #so we have to pick from a list of indices
    if not isinstance(x,list):
        x = list(x)
    return x[numpy.random.choice(list(range(len(x))))]

class RandomValueProducer:
    def __init__(self):
        self.levels = {0: [b'1', b'', '2', '', 0, 1, 0.0, 1.0, None, False, True]}

    def addEvenly(self, levels, count):
        for level in range(1, levels+1):
            self.addValues(level, count)

    def all(self):
        res = []
        for valueList in self.levels.values():
            res.extend(valueList)
        return res 

    def addValues(self, level, count, sublevels = None):
        assert level > 0

        if sublevels is None:
            sublevels = list(range(level))
        sublevels = [x for x in sublevels if x in self.levels]

        assert sublevels

        def picker():
            whichLevel = choice(sublevels)
            try:
                return choice(self.levels[whichLevel])
            except:
                print(self.levels[whichLevel])
                raise

        for _ in range(count):
            val = self.randomValue(picker)
            if not isinstance(val,list):
                val = [val]    
            self.levels.setdefault(level, []).extend(val)

    def randomValue(self, picker):
        def randomTuple():
            return makeTuple(*[picker() for i in range(choice([0,1,2,3,4]))])

        def randomNamedTupleDict():
            return {"x_%s" % i: picker() for i in range(choice([0,1,2,3,4]))}

        def randomNamedTuple():
            return makeNamedTuple(**randomNamedTupleDict())

        def randomDict():
            return makeDict({picker():picker() for i in range(choice([0,1,2,3,4]))})

        def randomTupleOf():
            return makeTupleOf(*[picker() for i in range(choice([0,1,2,3,4]))])

        def randomAlternative():
            return makeAlternative([randomNamedTupleDict() for i in range(choice([1,2,3,4]))])

        return choice([randomTuple,randomNamedTuple,randomDict,randomTupleOf,randomAlternative,picker])()

    def pickRandomly(self):
        return choice(self.levels[choice(list(self.levels))])

class NativeTypesTests(unittest.TestCase):
    def test_objects_are_singletons(self):
        self.assertTrue(Int8() is Int8())
        self.assertTrue(NoneType() is NoneType())

    def test_object_bytecounts(self):
        self.assertEqual(NoneType().bytecount(), 0)
        self.assertEqual(Int8().bytecount(), 1)

    def test_type_stringification(self):
        for t in ['Int8', 'NoneType']:
            self.assertEqual(str(getattr(_types,t)()), "<class '%s'>" % t)

    def test_tuple_of(self):
        tupleOfInt = TupleOf(int)
        i = tupleOfInt(())
        i = tupleOfInt((1,2,3))

        self.assertEqual(len(i), 3)
        self.assertEqual(tuple(i), (1,2,3))

        for x in range(10):
            self.assertEqual(tuple(tupleOfInt(tuple(range(x)))), tuple(range(x)))

    def test_tuple_of_tuple_of(self):
        tupleOfInt = TupleOf(int)
        tupleOfTupleOfInt = TupleOf(tupleOfInt)

        pyVersion = (1,2,3),(1,2,3,4)
        nativeVersion = tupleOfTupleOfInt(pyVersion)

        self.assertEqual(len(nativeVersion), 2)
        self.assertEqual(len(nativeVersion[0]), 3)
        self.assertEqual(tuple(tuple(x) for x in nativeVersion), pyVersion)

        bigTup = tupleOfInt(list(range(1000)))

        t0 = time.time()
        t = (bigTup,bigTup,bigTup,bigTup,bigTup)
        for i in range(1000000):
            tupleOfTupleOfInt(t)
        print(time.time() - t0, " to do 1mm")

        #like 5mm/sec
        self.assertTrue(time.time() - t0 < 1.0)

    def test_tuple_of_various_things(self):
        for thing, typeOfThing in [("hi", str), (b"somebytes", bytes), 
                                   (1.0, float), (2, int),
                                   (None, type(None))
                                   ]:
            tupleType = TupleOf(typeOfThing)
            t = tupleType((thing,))
            self.assertTrue(type(t[0]) is typeOfThing)
            self.assertEqual(t[0], thing)

    def test_one_of(self):
        o = OneOf(None, str)

        self.assertEqual(o("hi"), "hi")
        self.assertTrue(o(None) is None)

    def test_one_of_conversion_failure(self):
        o = OneOf(None, str)

        with self.assertRaises(Exception):
            o(b"bytes")
        
    def test_one_of_in_tuple(self):
        t = Tuple(OneOf(None, str), str)

        self.assertEqual(t(("hi","hi2"))[0], "hi")
        self.assertEqual(t(("hi","hi2"))[1], "hi2")
        self.assertEqual(t((None,"hi2"))[1], "hi2")
        self.assertEqual(t((None,"hi2"))[0], None)
        with self.assertRaises(TypeError):
            t((None,None))
        with self.assertRaises(IndexError):
            t((None,"hi2"))[2]
        
    def test_one_of_composite(self):
        t = OneOf(TupleOf(str), TupleOf(float))

        self.assertIsInstance(t((1.0,2.0)), TupleOf(float))
        self.assertIsInstance(t(("1.0","2.0")), TupleOf(str))

        with self.assertRaises(TypeError):
            t((1.0,"2.0"))

    def test_named_tuple(self):
        t = NamedTuple(a=int, b=int)

        with self.assertRaises(AttributeError):
            t().asdf

        self.assertEqual(t()[0], 0)
        self.assertEqual(t().a, 0)
        self.assertEqual(t()[1], 0)

        self.assertEqual(t(a=1,b=2).a, 1)
        self.assertEqual(t(a=1,b=2).b, 2)

    def test_named_tuple_str(self):
        t = NamedTuple(a=str, b=str)

        self.assertEqual(t(a='1',b='2').a, '1')
        self.assertEqual(t(a='1',b='2').b, '2')

        self.assertEqual(t(b='2').a, '')
        self.assertEqual(t(b='2').b, '2')
        self.assertEqual(t().a, '')
        self.assertEqual(t().b, '')

    def test_tuple_of_string_perf(self):
        t = NamedTuple(a=str, b=str)

        t0 = time.time()
        for i in range(1000000):
            t(a="a", b="b").a

        print("Took ", time.time() - t0, " to do 1mm")
        self.assertTrue(time.time() - t0 < 1.0)

    def test_comparisons_in_one_of(self):
        t = OneOf(None, float)
        
        def map(x):
            if x is None:
                return -1000000.0
            else:
                return x

        lt = lambda a,b: map(a) < map(b)    
        le = lambda a,b: map(a) <= map(b)   
        eq = lambda a,b: map(a) == map(b)   
        ne = lambda a,b: map(a) != map(b)
        gt = lambda a,b: map(a) > map(b)    
        ge = lambda a,b: map(a) >= map(b)
        
        funcs = [lt,le,eq,ne,gt,ge]
        ts = [None,1.0,2.0,3.0]

        for f in funcs:
            for t1 in ts:
                for t2 in ts:
                    self.assertTrue(f(t1,t2) is f(t(t1),t(t2)))
        
    def test_comparisons_equivalence(self):
        t = TupleOf(OneOf(None, str, bytes, float, int, bool, TupleOf(int)),)

        def lt(a,b): return a < b    
        def le(a,b): return a <= b   
        def eq(a,b): return a == b   
        def ne(a,b): return a != b
        def gt(a,b): return a > b    
        def ge(a,b): return a >= b
        
        funcs = [lt,le,eq,ne,gt,ge]

        tgroups = [
            [1.0,2.0,3.0],
            [1,2,3],
            [True,False],
            ["a","b","ab","bb","ba","aaaaaaa","","asdf"],
            ["1","2","3","12","13","23","24","123123", "0", ""],
            [b"a",b"b",b"ab",b"bb",b"ba",b"aaaaaaa",b"",b"asdf"],
            [(1,2),(1,2,3),(),(1,1),(1,)]
            ]

        for ts in tgroups:
            for f in funcs:
                for t1 in ts:
                    for t2 in ts:
                        self.assertTrue(f(t1,t2) is f(t((t1,)),t((t2,))), 
                            (f, t1,t2, f(t1,t2), f(t((t1,)),t((t2,))))
                            )

    def test_const_dict(self):
        t = ConstDict(str,str)

        self.assertEqual(len(t()), 0)
        self.assertEqual(len(t({})), 0)
        self.assertEqual(len(t({'a':'b'})), 1)
        self.assertEqual(t({'a':'b'})['a'], 'b')
        self.assertEqual(t({'a':'b','b':'c'})['b'], 'c')

    def test_const_dict_mixed(self):
        t = ConstDict(str,int)
        self.assertTrue(t({"a":10})["a"] == 10)

        t = ConstDict(int, str)
        self.assertTrue(t({10:"a"})[10] == "a")

    def test_const_dict_comparison(self):
        t = ConstDict(str,str)

        self.assertEqual(t({'a':'b'}), t({'a':'b'}))
        self.assertLess(t({}), t({'a':'b'}))

    def test_const_dict_lookup(self):
        for type_to_use, vals in [
                    (int, list(range(20))), 
                    (bytes, [b'1', b'2', b'3', b'4', b'5'])
                    ]:
            t = ConstDict(type_to_use, type_to_use)

            for _ in range(10):
                ks = list(vals)
                vs = list(vals)

                numpy.random.shuffle(ks)
                numpy.random.shuffle(vs)

                py_d = {}
                for i in range(len(ks)):
                    py_d[ks[i]] = vs[i]

                typed_d = t(py_d)

                for k in py_d:
                    self.assertEqual(py_d[k], typed_d[k])

                last_k = None
                for k in typed_d:
                    assert last_k is None or k > last_k, (k,last_k)
                    last_k = k

    def test_const_dict_lookup_time(self):
        int_dict = ConstDict(int, int)

        d = int_dict({k:k for k in range(1000000)})

        for k in range(1000000):
            self.assertTrue(k in d)
            self.assertTrue(d[k] == k)

    def test_const_dict_of_dict(self):
        int_dict = ConstDict(int, int)
        int_dict_2 = ConstDict(int_dict,int_dict)

        d = int_dict({1:2})
        d2 = int_dict({1:2,3:4})

        big = int_dict_2({d:d2})

        self.assertTrue(d in big)
        self.assertTrue(d2 not in big)
        self.assertTrue(big[d] == d2)

    def test_dict_hash_perf(self):
        str_dict = ConstDict(str, str)

        s = str_dict({'a' * 1000000: 'b' * 1000000})

        t0 = time.time()
        for k in range(1000000):
            hash(s)
        print(time.time() - t0, " to do 1mm")
        self.assertTrue(time.time() - t0 < 1.0)

    def test_const_dict_str_perf(self):
        t = ConstDict(str,str)

        t0 = time.time()
        for i in range(100000):
            t({str(k): str(k+1) for k in range(10)})

        print("Took ", time.time() - t0, " to do 1mm")
        self.assertTrue(time.time() - t0 < 1.0)

    def test_const_dict_int_perf(self):
        t = ConstDict(int,int)

        t0 = time.time()
        for i in range(100000):
            t({k:k+1 for k in range(10)})

        print("Took ", time.time() - t0, " to do 1mm")
        self.assertTrue(time.time() - t0 < 1.0)

    def test_const_dict_iter_int(self):
        t = ConstDict(int,int)

        aDict = t({k:k+1 for k in range(100)})
        for k in aDict:
            self.assertEqual(aDict[k], k+1)

    def test_const_dict_iter_str(self):
        t = ConstDict(str,str)

        aDict = t({str(k):str(k+1) for k in range(100)})
        for k in aDict:
            self.assertEqual(aDict[str(k)], str(int(k)+1))

    def test_alternatives(self):
        alt = Alternative(
            "Alt",
            child_ints={'x': int, 'y': int},
            child_strings={'x': str, 'y': str}
            )

        self.assertTrue(issubclass(alt.child_ints, alt))
        self.assertTrue(issubclass(alt.child_strings, alt))

        a = alt.child_ints(x=10,y=20)
        self.assertTrue(isinstance(a, alt))
        self.assertTrue(isinstance(a, alt.child_ints))

        self.assertEqual(a.x, 10)
        self.assertEqual(a.y, 20)
        self.assertTrue(a.matches.child_ints)
        self.assertFalse(a.matches.child_strings)

    def test_alternatives_perf(self):
        alt = Alternative(
            "Alt",
            child_ints={'x': int, 'y': int},
            child_strings={'x': str, 'y': str}
            )

        t0 = time.time()
        
        for i in range(1000000):
            a = alt.child_ints(x=10,y=20)
            a.matches.child_ints
            a.x

        print("took ", time.time() - t0, " to do 1mm")
        self.assertTrue(time.time() - t0 < 2.0)

    def test_object_hashing_and_equality(self):
        for _ in range(100):
            producer = RandomValueProducer()
            producer.addEvenly(20, 2)

            values = producer.all()

            for v1 in values:
                for v2 in values:
                    if v1 == v2:
                        self.assertEqual(hash(v1), hash(v2))
                        if type(v1) is type(v2):
                            self.assertEqual(repr(v1), repr(v2), (v1, v2, type(v1),type(v2)))

            values = sorted([makeTuple(v) for v in values])

            for i in range(len(values)-1):
                self.assertTrue(values[i] <= values[i+1])
                self.assertTrue(values[i+1] >= values[i])
