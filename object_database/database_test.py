#   Coyright 2017-2019 Nativepython Authors
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

from typed_python import Alternative, TupleOf, OneOf, ConstDict
from typed_python.SerializationContext import SerializationContext

from object_database.schema import Indexed, Index, Schema, SubscribeLazilyByDefault
from object_database.core_schema import core_schema
from object_database.view import RevisionConflictException, DisconnectedException, ObjectDoesntExistException
from object_database.database_connection import DatabaseConnection
from object_database.tcp_server import TcpServer
from object_database.inmem_server import InMemServer
from object_database.persistence import InMemoryPersistence, RedisPersistence
from object_database.util import configureLogging, genToken
from object_database.test_util import currentMemUsageMb
from object_database.RedisTestHelper import RedisTestHelper
from object_database.reactor import Reactor

import object_database
import object_database.messages as messages
import queue
import unittest
import tempfile
import numpy
import os
import traceback
import threading
import random
import time
import ssl


class BlockingCallback:
    def __init__(self):
        self.callbackArgs = queue.Queue()
        self.is_released = queue.Queue()

    def callback(self, arg=None):
        self.callbackArgs.put(arg)
        self.is_released.get(timeout=1.0)

    def waitForCallback(self, timeout):
        return self.callbackArgs.get(timeout=timeout)

    def releaseCallback(self):
        self.is_released.put(True)


expr = Alternative(
    "Expr",
    Constant={'value': int},
    # Add = {'l': expr, 'r': expr},
    # Sub = {'l': expr, 'r': expr},
    # Mul = {'l': expr, 'r': expr}
)

schema = Schema("test_schema")
schema.expr = expr


@schema.define
class Root:
    obj = OneOf(None, schema.Object)
    k = int


@schema.define
class Object:
    k = Indexed(expr)
    other = OneOf(None, schema.Object)

    @property
    def otherK(self):
        if self.other is not None:
            return self.other.k


@schema.define
class DeletedThingWithInit:
    pass


@schema.define
class ThingWithInit:
    x = int
    y = float
    z = str

    def __init__(self, x=0, y=123):
        self.x = x
        self.y = y

    def __del__(self):
        DeletedThingWithInit()


@schema.define
class ThingWithThrowingInit:
    x = int

    def __init__(self):
        self.x = 10
        raise Exception("ThingWithThrowingInit")


@schema.define
class ThingWithInitHoldingOdbRef:
    x = Indexed(Root)

    def __init__(self):
        self.x = Root()


@schema.define
class ThingWithInitAndInitializableRef:
    x = Indexed(str)

    def __init__(self):
        self.x = "Hi"


@schema.define
class ThingWithDicts:
    x = ConstDict(str, bytes)


@schema.define
class Counter:
    k = Indexed(int)
    x = int

    def f(self):
        return self.k + 1

    def __str__(self):
        return "Counter(k=%s)" % self.k


@schema.define
class ObjectWithManyIndices:
    x0 = Indexed(int)
    x1 = Indexed(int)
    x2 = Indexed(int)
    x3 = Indexed(int)
    x4 = Indexed(int)
    x5 = Indexed(int)
    x6 = Indexed(int)
    x7 = Indexed(int)
    x8 = Indexed(int)
    x9 = Indexed(int)


@schema.define
class StringIndexed:
    name = Indexed(str)


@schema.define
class ThingWithObjectIndex:
    value = Indexed(object)
    name = str

    name_and_value = Index('name', 'value')


class ObjectDatabaseTests:
    @classmethod
    def setUpClass(cls):
        configureLogging("database_test")
        cls.PERFORMANCE_FACTOR = 1.0 if os.environ.get('TRAVIS_CI', None) is None else 2.0

    def test_object_indices(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            z = ThingWithObjectIndex(value=None)
            z.value = z

        with db.transaction():
            self.assertEqual(ThingWithObjectIndex.lookupAny(value=z), z)
            z.value = "hello"
            z.name = "name"
            self.assertEqual(ThingWithObjectIndex.lookupAny(value=z), None)
            self.assertEqual(ThingWithObjectIndex.lookupAny(value="hello"), z)
            self.assertEqual(ThingWithObjectIndex.lookupAny(name_and_value=("name", "hello")), z)

    def test_assigning_dicts(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            z = ThingWithDicts()
            z.x = {'a': b'b'}

        with db.transaction():
            z2 = ThingWithDicts()
            z2.x = z.x

    def test_construct_with_init(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            x = ThingWithInit()
            y = ThingWithInit(1, y=1000)
            z = ThingWithInitAndInitializableRef() # noqa

        with db.view():
            self.assertEqual(x.x, 0)
            self.assertEqual(y.x, 1)

            self.assertEqual(x.y, 123)
            self.assertEqual(y.y, 1000)

            self.assertEqual(x.z, "")
            self.assertEqual(y.z, "")

    def test_construct_with_throwing_init(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            y = Counter()
            y.delete()

        with self.assertRaisesRegex(Exception, "ThingWithThrowingInit"):
            with db.transaction():
                ThingWithThrowingInit()

    def test_indices_independent(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            x = ObjectWithManyIndices()

        with db.transaction() as t:
            self.assertTrue(x.exists())
            x.x0 = x.x0 + 1

            # exists, and x.x0, but not x1 through x9
            self.assertEqual(len(t.getFieldReads()), 2)

    def test_create_and_delete_is_no_op(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction() as t:
            x = Object()

            self.assertEqual(len(t.getFieldWrites()), 3)

            x.delete()

            self.assertEqual(len(t.getFieldWrites()), 0)

    def test_construct_with_indexed_init(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            r = ThingWithInitHoldingOdbRef()

        with db.view():
            self.assertTrue(ThingWithInitHoldingOdbRef.lookupOne(x=r.x) == r)
            self.assertEqual(ThingWithInitHoldingOdbRef.lookupAll(), (r,))

    def test_destructors(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            x = ThingWithInit()
            x.delete()

        with db.view():
            self.assertEqual(len(ThingWithInit.lookupAll()), 0)
            self.assertEqual(len(DeletedThingWithInit.lookupAll()), 1)

    def test_subscribe_excluding(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)
        db2.subscribeToSchema(schema, excluding=[ThingWithDicts])

        with db1.transaction():
            t = ThingWithDicts()

        db2.flush()
        with db2.view():
            self.assertFalse(t.exists())

        db2.subscribeToType(ThingWithDicts)

        with db2.view():
            self.assertTrue(t.exists())

    def test_can_convert_numpy_int(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            c = Counter(x=numpy.int64(10))
            c.k = numpy.int64(20)

        with db.view():
            self.assertEqual(c.k, 20)

    def test_subscribe_to_objects(self):
        db1 = self.createNewDb()
        db1.subscribeToSchema(schema)

        db2 = self.createNewDb()

        with db1.transaction():
            someThings = [Counter(k=i) for i in range(10)]

        db2.subscribeToObjects(someThings[::2])

        with db2.view():
            for i in range(10):
                if i % 2 == 0:
                    self.assertTrue(someThings[i].exists())
                else:
                    self.assertFalse(someThings[i].exists())

    def test_serialization_contexts(self):
        db = self.createNewDb()

        class ArbitraryBaseClass:
            def __init__(self, x):
                self.x = x

        class ArbitrarySubclass(ArbitraryBaseClass):
            def __init__(self, x, y):
                super().__init__(x)
                self.y = y

        db.setSerializationContext(SerializationContext({'ABC': ArbitraryBaseClass, 'SUB': ArbitrarySubclass}))

        schema = Schema("test_schema")

        @schema.define
        class HoldsArbitrary:
            holding = ArbitraryBaseClass

        @schema.define
        class HoldsObject:
            holding = object

        db.subscribeToSchema(schema)

        with db.transaction():
            x = HoldsArbitrary(holding=ArbitraryBaseClass(10))
            self.assertEqual(x.holding.x, 10)

        with db.transaction():
            self.assertEqual(x.holding.x, 10)
            self.assertIsInstance(x.holding, ArbitraryBaseClass)
            x.holding = ArbitrarySubclass(10, 20)

        with db.transaction():
            self.assertEqual(x.holding.x, 10)
            self.assertEqual(x.holding.y, 20)
            self.assertIsInstance(x.holding, ArbitrarySubclass)

        with self.assertRaises(Exception):
            with db.transaction():
                x.holding = "hi"

        with db.transaction():
            x = HoldsObject(holding="hi")
            x.holding = 10

        with db.transaction():
            self.assertEqual(x.holding, 10)

    def test_reading_many_python_objects_from_many_threads(self):
        # this test simply verifies that we don't segfault when we do this.
        # we need to verify that multiple threads writing into the view background
        # don't corrupt its state. view internals are protected by the gil, except
        # that deserialization can release the GIL, allowing other threads (namely the
        # pump loop) to write into the view, and also read values from it.

        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class HoldsObject:
            holding = object

        db.subscribeToSchema(schema)

        elapsed = 1

        aTup = TupleOf(int)(range(100))

        t0 = time.time()

        def reader():
            while time.time() - t0 < elapsed:
                with db.view():
                    for h in HoldsObject.lookupAll():
                        h.holding

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()

        while time.time() - t0 < elapsed:
            for i in range(10):
                with db.transaction():
                    h = HoldsObject(holding=aTup)
            with db.transaction():
                for h in HoldsObject.lookupAll():
                    h.delete()

        for t in threads:
            t.join()

    def test_disconnecting(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)
        db.disconnect()

        with self.assertRaises(DisconnectedException):
            with db.view():
                pass

        usage = currentMemUsageMb(residentOnly=False)

        for i in range(100):
            db = self.createNewDb()
            db.subscribeToSchema(schema)
            db.flush()
            db.disconnect()

        usage = currentMemUsageMb(residentOnly=False)

        for i in range(500):
            db = self.createNewDb()
            db.subscribeToSchema(schema)
            db.flush()
            db.disconnect()
            self.assertLess(currentMemUsageMb(residentOnly=False) - usage, 100)

    def test_disconnecting_is_immediate(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(core_schema)
        db2.subscribeToSchema(core_schema)

        with db2.view():
            assert db1.connectionObject.exists()
        with db1.view():
            assert db2.connectionObject.exists()

        db2Connection = db2.connectionObject
        db2.disconnect()

        self.assertTrue(
            db1.waitForCondition(
                lambda: not db2Connection.exists(),
                timeout=2.0*self.PERFORMANCE_FACTOR)
        )

    def checkCallbackTriggersLazyLoad(self, callback, shouldExist=True):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        loadedIDs = queue.Queue()
        self.server._lazyLoadCallback = loadedIDs.put

        with db.transaction():
            c = Counter(k=2, x=3)

        db2 = self.createNewDb()
        db2.subscribeToSchema(schema, lazySubscription=True)

        with db2.view():
            # lookup in the index doesn't dirty the object because we have to load
            # the index values when we first subscribe
            self.assertEqual(Counter.lookupAll(k=2), (c,))

        with db2.transaction():
            callback(c)

        self.assertEqual(loadedIDs.get_nowait(), c._identity)

        # at this point, the value is loaded
        with db2.view():
            if shouldExist:
                self.assertEqual(c.x, 3)
            else:
                self.assertFalse(c.exists())

        with self.assertRaises(queue.Empty):
            loadedIDs.get_nowait()

    def test_lazy_subscriptions_read(self):
        self.checkCallbackTriggersLazyLoad(lambda c: self.assertEqual(c.k, 2))

    def test_lazy_subscriptions_write(self):
        self.checkCallbackTriggersLazyLoad(lambda c: setattr(c, 'k', 20))

    def test_lazy_subscriptions_exists(self):
        self.checkCallbackTriggersLazyLoad(lambda c: c.exists())

    def test_lazy_subscriptions_delete(self):
        self.checkCallbackTriggersLazyLoad(lambda c: c.delete(), shouldExist=False)

    def test_lazy_by_default(self):
        s = Schema("test")

        @s.define
        @SubscribeLazilyByDefault
        class Lazy:
            x = int

        self.assertTrue(Lazy.isLazyByDefault())
        self.assertFalse(Counter.isLazyByDefault())

        lazyObjects = []

        for passIx in range(10):
            db1 = self.createNewDb()
            db1.subscribeToType(Lazy)

            with db1.transaction():
                lazyObjects.append(Lazy(x=len(lazyObjects)))

                for i in range(len(lazyObjects)):
                    self.assertTrue(lazyObjects[i].exists())
                    self.assertEqual(lazyObjects[i].x, i)

            db2 = self.createNewDb()
            db2.subscribeToType(Lazy, lazySubscription=False)

            with db2.view():
                for i in range(len(lazyObjects)):
                    self.assertTrue(lazyObjects[i].exists())
                    self.assertEqual(lazyObjects[i].x, i)

    # this fails because 'lazy' objects are incorrectly implemented. we should
    # probably just get rid of the idea entirely and make everything 'lazy' at
    # some level (forcing the server to keep old tid's around)
    def DISABLEDtest_lazy_priors(self):
        s = Schema("test")

        @s.define
        @SubscribeLazilyByDefault
        class Lazy:
            x = int

        db1 = self.createNewDb()
        db1.subscribeToType(Lazy)

        with db1.transaction():
            lazy = Lazy(x=1)

        db2 = self.createNewDb()
        db2.subscribeToType(Lazy)

        v = db2.view()

        with db1.transaction():
            lazy.x = 2

        db2.flush()
        with v:
            self.assertEqual(lazy.x, 1)

        with db2.view():
            self.assertEqual(lazy.x, 2)

    def test_lazy_objects_visible_in_own_transaction(self):
        db = self.createNewDb()
        db.subscribeToType(Object, lazySubscription=True)

        with db.transaction():
            o = Object()
            self.assertTrue(o.exists())

    def test_unsubscribed_objects_visible_in_own_transaction(self):
        db = self.createNewDb()
        db.subscribeToNone(Object)

        with db.transaction():
            o = Object()
            self.assertTrue(o.exists())

    def test_methods(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            counter = Counter()
            counter.k = 2
            self.assertEqual(counter.f(), 3)
            self.assertEqual(str(counter), "Counter(k=2)")

    def test_property_object(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            counter = Object(k=expr.Constant(value=10))

            counter2 = Object(other=counter, k=expr.Constant(value=0))

            self.assertEqual(counter2.otherK, counter.k)

    def test_identity_transfer(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()
            root2 = Root.fromIdentity(root._identity)

            root.obj = Object(k=expr.Constant(value=23))
            self.assertEqual(root2.obj.k.value, 23)

    def test_adding_fields_to_type(self):
        schema = Schema("schema")

        @schema.define
        class Test:
            i = int

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            t = Test(i=1)

        schema2 = Schema("schema")
        @schema2.define
        class Test:
            i = int
            k = int

        db2 = self.createNewDb()
        db2.subscribeToSchema(schema2)

        t = Test.fromIdentity(t._identity)

        with db2.view():
            self.assertEqual(t.i, 1)
            self.assertEqual(t.k, 0)

    def test_subclassing(self):
        schema = Schema("schema")

        @schema.define
        class Test:
            i = int

            def f(self):
                return 1

            def g(self):
                return 2

        @schema.define
        class SubclassTesting(Test):
            y = int

            def g(self):
                return 3

            def h(self):
                return 4

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            t = Test(i=1)
            t2 = SubclassTesting(i=2, y=3)

            self.assertEqual(t.f(), 1)
            self.assertEqual(t.g(), 2)

            self.assertEqual(t2.f(), 1)
            self.assertEqual(t2.g(), 3)
            self.assertEqual(t2.h(), 4)
            self.assertEqual(t2.y, 3)

    def test_many_subscriptions(self):
        OK = []
        FINISHED = []
        count = 10
        threadCount = 10

        def worker(index):
            db = self.createNewDb()

            indices = list(range(count))
            numpy.random.shuffle(indices)

            for i in indices:
                db.subscribeToIndex(Counter, k=i)

                with db.transaction():
                    Counter(k=i, x=index)

            FINISHED.append(True)

            db.waitForCondition(
                lambda: len(FINISHED) == threadCount,
                10.0*self.PERFORMANCE_FACTOR
            )
            db.flush()

            with db.view():
                actuallyVisible = len(Counter.lookupAll())

            if actuallyVisible != count * threadCount:
                print("TOTAL is ", actuallyVisible, " != ", count*threadCount)
            else:
                OK.append(True)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(threadCount)]
        for t in threads:
            t.daemon = True
            t.start()
        for t in threads:
            t.join()

        db1 = self.createNewDb()
        db1.subscribeToSchema(schema)
        with db1.view():
            self.assertEqual(len(Counter.lookupAll()), count*threadCount)

        db2 = self.createNewDb()

        for i in range(count):
            db2.subscribeToIndex(Counter, k=i)
        db2.flush()
        with db2.view():
            self.assertEqual(len(Counter.lookupAll()), count*threadCount)

        self.assertEqual(len(OK), 10)

    def test_basic(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()

            self.assertTrue(root.exists())

            self.assertTrue(root.obj is None, root.obj)

            o = Object(k=expr.Constant(value=23))

            root.obj = o

            self.assertTrue(root.obj.k == o.k)
            self.assertTrue(root.obj == o)

        with db.view():
            self.assertTrue(root.exists())
            self.assertEqual(root.obj.k.value, 23)

        db2 = self.createNewDb()
        db2.subscribeToSchema(schema)

        with db2.view():
            self.assertTrue(root.exists())
            self.assertEqual(root.obj.k.value, 23)

    def test_throughput_basic(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()
            root.obj = Object(k=expr.Constant(value=0))

        t0 = time.time()
        while time.time() < t0 + 1.0:
            with db.transaction():
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)

        with db.view():
            self.assertTrue(root.obj.k.value > 500, root.obj.k.value)
            print(root.obj.k.value, "transactions per second")

    def test_throughput_read(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()
            root.obj = Object(k=expr.Constant(value=1))

        t0 = time.time()
        with db.transaction():
            count = 0
            while time.time() < t0 + 1.0:
                for _ in range(100):
                    root.obj
                count = count + 100

        self.assertGreater(count, 500000 / self.PERFORMANCE_FACTOR)
        print(count, " reads per second")  # I get about 3mm on my machine.

    def test_throughput_write_within_view(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            c = Counter()

        t0 = time.time()
        with db.transaction():
            count = 0
            while time.time() < t0 + 1.0:
                for _ in range(100):
                    c.x = c.x + 1
                count = count + 100

        self.assertGreater(count, 500000 / self.PERFORMANCE_FACTOR)

        # I get about 2.8mm on my machine.
        print(count, " in-view writes per second")

    def test_throughput_write_indexed_value_within_view(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            c = Counter()

        t0 = time.time()
        with db.transaction():
            count = 0
            while time.time() < t0 + 1.0:
                for _ in range(100):
                    c.k = c.k + 1
                count = count + 100

        self.assertGreater(count, 50000 / self.PERFORMANCE_FACTOR)

        # I get about 400k on my machine.
        print(count, " in-view writes with on indexed values per second")

    def test_delayed_transactions(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        confirmed = queue.Queue()

        with db.transaction():
            root = Root()
            root.obj = Object(k=expr.Constant(value=0))

        for i in range(1000):
            with db.transaction().onConfirmed(confirmed.put):
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)

        self.assertTrue(confirmed.qsize() < 1000)

        good = 0
        for i in range(1000):
            if confirmed.get().matches.Success:
                good += 1

        self.assertGreater(good, 0)
        self.assertLess(good, 1000)

    def test_exists(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()

            self.assertTrue(root.exists())
            self.assertEqual(root.k, 0)

            root.delete()

            self.assertFalse(root.exists())

            with self.assertRaises(ObjectDoesntExistException):
                root.k

        with db.view():
            self.assertFalse(root.exists())

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.view():
            self.assertFalse(root.exists())

    def test_read_performance(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        objects = {}
        with db.transaction():
            for i in range(1000):
                root = Root()

                e = expr.Constant(value=i)

                root.obj = Object(k=e)

                objects[i] = root

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        t0 = time.time()
        count = 0
        reads = 0
        while time.time() < t0 + 1.0:
            with db.view():
                for i in range(300):
                    count += objects[i].obj.k.value
                    reads += 2

        print(f"Performed {reads/(time.time()-t0)} per second")

    def test_transactions(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()

        views = [db.view()]

        for i in [1, 2, 3]:
            with db.transaction():
                root.obj = Object(k=expr.Constant(value=i))
            views.append(db.view())

        vals = []
        for v in views:
            with v:
                assert root.exists()
                if root.obj is None:
                    vals.append(None)
                else:
                    vals.append(root.obj.k.value)

        self.assertEqual(vals, [None, 1, 2, 3])

    def test_conflicts(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()
            root.obj = Object(k=expr.Constant(value=0))

        for ordering in [0, 1]:
            t1 = db.transaction()
            t2 = db.transaction()

            if ordering:
                t1, t2 = t2, t1

            with t1:
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)

            with self.assertRaises(RevisionConflictException):
                with t2:
                    root.obj.k = expr.Constant(value=root.obj.k.value + 1)

    def test_conflicts_write_then_read(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            c = Counter()

        t1 = db.transaction()
        t2 = db.transaction()

        # these don't conflict because they didn't read from the prior
        # value.
        with t1:
            c.x = 1

        with t2:
            c.x = 2

        # these do conflict because they update indices and we need to
        # know the prior values to compute the index updates correctly
        t1 = db.transaction()
        t2 = db.transaction()

        # these don't conflict because they didn't read from the prior
        # value.
        with t1:
            c.k = 1

        with self.assertRaises(RevisionConflictException):
            with t2:
                c.k = 2

    def test_conflicts_dont_cause_view_leaks(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()
            root.obj = Object(k=expr.Constant(value=0))

        t1 = db.transaction()
        t2 = db.transaction()

        with t1:
            root.obj.k = expr.Constant(value=root.obj.k.value + 1)

        try:
            with t2:
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)
        except RevisionConflictException:
            pass

        for i in range(100):
            with db.transaction():
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)

        self.assertTrue(db._noViewsOutstanding())

    def test_object_versions_robust(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        counters = []
        counter_vals_by_tn = {}
        views_by_tn = {}

        random.seed(123)

        # expect nothing initially
        views_by_tn[db._cur_transaction_num] = db.view()
        counter_vals_by_tn[db._cur_transaction_num] = {}

        # seed the initial state
        with db.transaction():
            for i in range(20):
                counter = Counter()
                counter.k = int(random.random() * 100)
                counters.append(counter)

            counter_vals_by_tn[db._cur_transaction_num + 1] = {c: c.k for c in counters}

        total_writes = 0

        for passIx in range(1000):
            with db.transaction():
                didOne = False
                for subix in range(int(random.random() * 5 + 1)):
                    counter = counters[int(random.random() * len(counters))]

                    if counter.exists():
                        if random.random() < .001:
                            counter.delete()
                        else:
                            counter.k = int(random.random() * 100)
                        total_writes += 1
                        didOne = True

                if didOne:
                    counter_vals_by_tn[db._cur_transaction_num + 1] = {c: c.k for c in counters if c.exists()}

            if didOne:
                views_by_tn[db._cur_transaction_num] = db.view()

            while views_by_tn and random.random() < .5 or len(views_by_tn) > 10:
                # pick a random view and check that it's consistent
                all_tids = list(views_by_tn)
                tid = all_tids[int(random.random() * len(all_tids))]

                with views_by_tn[tid]:
                    for c in counters:
                        if not c.exists():
                            assert c not in counter_vals_by_tn[tid], tid
                        else:
                            self.assertEqual(c.k, counter_vals_by_tn[tid][c])

                del views_by_tn[tid]

            if random.random() < .05 and views_by_tn:
                with db.view():
                    curCounterVals = {c: c.k for c in counters if c.exists()}

                # reset the database
                db = self.createNewDb()
                db.subscribeToSchema(schema)

                with db.view():
                    newCounterVals = {c: c.k for c in counters if c.exists()}

                self.assertEqual(curCounterVals, newCounterVals)

                views_by_tn = {}
                counter_vals_by_tn = {}

        # we may have one or two for connection objects, and we have two values for every indexed thing
        self.assertLess(self.mem_store.storedStringCount(), 203)
        self.assertTrue(total_writes > 500, total_writes)

    def test_flush_db_works(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        counters = []
        with db.transaction():
            for _ in range(10):
                counters.append(Counter(k=1))

        self.assertTrue(self.mem_store.values)

        view = db.view()

        with db.transaction():
            for c in counters:
                c.delete()

        # database doesn't have this
        t0 = time.time()
        while time.time() - t0 < 1.0 and self.mem_store.storedStringCount() >= 2:
            time.sleep(.01)

        self.assertLess(self.mem_store.storedStringCount(), 4)

        # but the view does!
        with view:
            for c in counters:
                self.assertTrue(c.exists())

    def test_indices_lookup_any(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.view():
            self.assertEqual(Counter.lookupAny(), None)

        with db.transaction():
            o1 = Counter(k=20)

        with db.view():
            self.assertEqual(Counter.lookupAny(), o1)

    def test_indices(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.view():
            self.assertEqual(Counter.lookupAll(k=20), ())
            self.assertEqual(Counter.lookupAll(k=30), ())

        with db.transaction():
            o1 = Counter(k=20)

        with db.view():
            self.assertEqual(Counter.lookupAll(k=20), (o1,))
            self.assertEqual(Counter.lookupAll(k=30), ())

        with db.transaction():
            o1.k = 30

        with db.view():
            self.assertEqual(Counter.lookupAll(k=20), ())
            self.assertEqual(Counter.lookupAll(k=30), (o1,))

        with db.transaction():
            o1.delete()

        with db.view():
            self.assertEqual(Counter.lookupAll(k=20), ())
            self.assertEqual(Counter.lookupAll(k=30), ())

    def test_indices_multiple_values(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            k1 = Counter(k=20)
            Counter(k=20)

            self.assertEqual(len(Counter.lookupAll(k=20)), 2)

            k1.k = 30

            self.assertEqual(len(Counter.lookupAll(k=20)), 1)

            k1.k = 20

            self.assertEqual(len(Counter.lookupAll(k=20)), 2)

        with db.transaction():
            self.assertEqual(len(Counter.lookupAll(k=20)), 2)

            k1.k = 30

            self.assertEqual(len(Counter.lookupAll(k=20)), 1)

            k1.k = 20

            self.assertEqual(len(Counter.lookupAll(k=20)), 2)

    def test_indices_across_invocations(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            o = Counter(k=1)
            o.x = 10

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            o = Counter.lookupOne(k=1)
            self.assertEqual(o.x, 10)
            o.k = 2
            o.x = 11

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            o = Counter.lookupOne(k=2)
            o.k = 3
            self.assertEqual(o.x, 11)

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            self.assertFalse(Counter.lookupAny(k=2))

            o = Counter.lookupOne(k=3)
            o.k = 3
            self.assertEqual(o.x, 11)

    def test_new_objects_are_in_index(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            o = Counter()
            self.assertTrue(Counter.lookupAny())
            self.assertTrue(Counter.lookupAll())

        with db.transaction():
            o = Counter()
            self.assertTrue(Counter.lookupAny())
            self.assertEqual(len(Counter.lookupAll()), 2)

        with db.transaction():
            o.delete()
            o = Counter()
            self.assertTrue(Counter.lookupAny())
            self.assertEqual(len(Counter.lookupAll()), 2)

    def test_deletions_visible(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            obs = [Counter() for _ in range(10)]

        with db.transaction():
            while obs:
                # delete a middle object in the list
                obs.pop(len(obs)//2).delete()

                getId = lambda o: o._identity

                # in-view version of this should be correct
                self.assertEqual(
                    sorted(Counter.lookupAll(), key=getId),
                    sorted(obs, key=getId)
                )

    def test_index_consistency(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            x = int
            y = int

            pair = Index('x', 'y')
            single = Index('x')

        db.subscribeToSchema(schema)

        with db.transaction():
            o = Object(x=0, y=0)

        with db.transaction():
            self.assertEqual(Object.lookupOne(pair=(0, 0)), o)
            self.assertEqual(Object.lookupOne(single=0), o)

        t1 = db.transaction()
        t2 = db.transaction()

        with t1.nocommit():
            o.x = 1

        with t2.nocommit():
            o.y = 1

        t1.commit()

        with self.assertRaises(RevisionConflictException):
            t2.commit()

    def test_indices_of_algebraics(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            o1 = Object(k=expr.Constant(value=123))

        with db.view():
            self.assertEqual(Object.lookupAll(k=expr.Constant(value=123)), (o1,))

    def test_frozen_schema(self):
        schema = Schema("test_schema")

        @schema.define
        class Object:
            x = int
            y = int

        schema.freeze()

        with self.assertRaises(AttributeError):
            schema.SomeOtherObject

    def test_freezing_schema_with_undefined_fails(self):
        schema = Schema("test_schema")

        @schema.define
        class Object:
            x = schema.Object2
            y = int

        with self.assertRaises(Exception):
            schema.freeze()

        @schema.define
        class Object2:
            x = int

        schema.freeze()

    def test_index_pairs(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            k = Indexed(int)

            pair_index = Index('k', 'k')

        db.subscribeToSchema(schema)

        with db.transaction():
            o1 = Object(k=10)

        with db.view():
            self.assertEqual(Object.lookupAll(k=10), (o1,))
            self.assertEqual(Object.lookupAll(k=20), ())

            self.assertEqual(Object.lookupAll(pair_index=(10, 10)), (o1,))
            self.assertEqual(Object.lookupAll(pair_index=(10, 11)), ())

            with self.assertRaises(Exception):
                self.assertEqual(Object.lookupAll(pair_index=(10, "hi")), (o1,))

    def test_lookup_in_unsubscribed_index(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            k = Indexed(int)

        db.subscribeToSchema(schema)

        with db.transaction():
            o = Object(k=10)

        db2 = self.createNewDb()

        with self.assertRaises(Exception):
            with db2.view():
                Object.lookupOne(k=10)

        db2.subscribeToSchema(schema)

        with db2.view():
            self.assertEqual(o, Object.lookupOne(k=10))

    def test_indices_update_during_transactions(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            k = Indexed(int)

        db.subscribeToSchema(schema)

        with db.transaction():
            self.assertEqual(Object.lookupAll(k=10), ())
            o1 = Object(k=10)

            self.assertEqual(Object.lookupAll(k=10), (o1,))

            o1.k = 20

            self.assertEqual(Object.lookupAll(k=10), ())
            self.assertEqual(Object.lookupAll(k=20), (o1,))

            o1.delete()

            self.assertFalse(o1.exists())

            self.assertEqual(Object.lookupAll(k=10), ())
            self.assertEqual(Object.lookupAll(k=20), ())

    def test_index_transaction_conflicts(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            k = Indexed(int)

        db.subscribeToSchema(schema)

        with db.transaction():
            o1 = Object(k=10)
            o2 = Object(k=20)
            Object(k=30)

        t1 = db.transaction()
        t2 = db.transaction()

        with t1.nocommit():
            o2.k = len(Object.lookupAll(k=10))

        with t2.nocommit():
            o1.k = 20

        t2.commit()

        with self.assertRaises(RevisionConflictException):
            t1.commit()

    def test_default_constructor_for_list(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            x = TupleOf(int)

        db.subscribeToSchema(schema)

        with db.transaction():
            n = Object()
            self.assertEqual(len(n.x), 0)

    def test_existence_from_nonsubscription(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)
        db2.subscribeToNone(Counter)

        with db2.transaction():
            c = Counter(k=0)

        db1.flush()

        with db1.view():
            self.assertEqual(Counter.lookupAll(), (c,))

    def test_existence_from_nonsubscription_subscribe_after(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db2.subscribeToNone(Counter)

        with db2.transaction():
            c = Counter(k=0)

        db1.flush()

        db1.subscribeToNone(Counter)

        with db1.view():
            self.assertEqual(Counter.lookupAll(), ())

        db1.subscribeToSchema(schema)

        with db1.view():
            self.assertEqual(Counter.lookupAll(), (c,))

    def test_index_subscriptions(self):
        db_all = self.createNewDb()
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db_all.subscribeToSchema(schema)
        with db_all.transaction():
            c0 = Counter(k=0)
            c1 = Counter(k=1)

            c0.x = 20
            c1.x = 30

        db1.subscribeToIndex(Counter, k=0)
        db2.subscribeToIndex(Counter, k=1)

        with db1.view():
            self.assertTrue(c0.exists())
            self.assertEqual(c0.x, 20)
            self.assertFalse(c1.exists())

        with db2.view():
            self.assertTrue(c1.exists())
            self.assertEqual(c1.x, 30)
            self.assertFalse(c0.exists())

        # create a new value in the view and verify it shows up
        with db_all.transaction():
            c2_0 = Counter(k=0)
            c2_1 = Counter(k=1)

        db1.waitForCondition(lambda: c2_0.exists(), 2*self.PERFORMANCE_FACTOR)
        db2.waitForCondition(lambda: c2_1.exists(), 2*self.PERFORMANCE_FACTOR)

        with db2.view():
            self.assertFalse(c2_0.exists())
        with db1.view():
            self.assertFalse(c2_1.exists())

        # now move c2_0 from '0' to '1'. It should show up in db2 and still in db1
        with db_all.transaction():
            c2_0.k = 1

        db1.waitForCondition(lambda: c2_0.exists(), 2*self.PERFORMANCE_FACTOR)
        db2.waitForCondition(lambda: c2_0.exists(), 2*self.PERFORMANCE_FACTOR)

        # now, we should see it get subscribed to in both
        with db_all.transaction():
            c2_0.x = 40

        db1.waitForCondition(lambda: c2_0.x == 40, 2*self.PERFORMANCE_FACTOR)
        db2.waitForCondition(lambda: c2_0.x == 40, 2*self.PERFORMANCE_FACTOR)

        # but if we make a new database connection and subscribe, we won't see it
        db3 = self.createNewDb()
        db3.subscribeToIndex(Counter, k=0)
        db3.flush()

        with db3.view():
            self.assertTrue(not c2_0.exists())
            self.assertTrue(not c2_1.exists())

    def test_implicitly_subscribed_to_objects_we_create(self):
        db1 = self.createNewDb()

        db1.subscribeToNone(Counter)

        with db1.transaction():
            c = Counter(k=1)

        with db1.view():
            self.assertTrue(c.exists())

    def test_create_resubscribe_and_lookup(self):
        db1 = self.createNewDb()

        db1.subscribeToSchema(schema)

        with db1.transaction():
            c = StringIndexed(name="name")

        db2 = self.createNewDb()
        db2.subscribeToSchema(schema)

        with db2.transaction():
            self.assertEqual(StringIndexed.lookupAll(name="name"), (c,))

    def test_adding_while_subscribing_to_index(self):
        self.test_adding_while_subscribing(shouldSubscribeToIndex=True)

    def test_big_transactions(self):
        db1 = self.createNewDb()

        db1.subscribeToSchema(schema)

        with db1.transaction():
            for i in range(21000):
                Counter(k=i, x=i)

        with db1.transaction():
            for i in range(21000):
                Counter.lookupOne(k=i).k = 0

        with db1.transaction():
            self.assertEqual(len(Counter.lookupAll(k=0)), 21000)

    def test_reactor_threaded(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        executed = [0]

        def incrementor():
            with db.transaction():
                for c in Counter.lookupAll():
                    if c.x != 0:
                        executed[0] += 1
                        c.k += c.x
                        c.x = 0

        r1 = Reactor(db, incrementor)
        r1.start()

        try:
            time.sleep(.10)
            self.assertEqual(executed[0], 0)

            with db.transaction():
                c = Counter(k=0, x=100)

            db.waitForCondition(lambda: c.k == 100, 2.0)

            with db.view():
                self.assertEqual(c.k, 100)
                self.assertEqual(c.x, 0)
                self.assertEqual(executed[0], 1)

            with db.transaction():
                c.x += 2

            db.waitForCondition(lambda: c.k == 102, 2.0)

            with db.view():
                self.assertEqual(c.k, 102)
                self.assertEqual(c.x, 0)
                self.assertEqual(executed[0], 2)
        finally:
            r1.stop()
            r1.teardown()

    def test_reactor_with_exception(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            c = Counter(k=0, x=0)

        executed = [0]
        thrown = [0]

        def incrementor():
            shouldThrow = False

            with db.transaction():
                executed[0] += 1
                if c.x:
                    thrown[0] += 1
                    c.x = 0
                    shouldThrow = True

            assert not shouldThrow

        r1 = Reactor(db, incrementor)
        r1.start()

        try:
            for _ in range(10):
                with db.transaction():
                    c.x = 1

                self.assertTrue(db.waitForCondition(lambda: c.x == 0, timeout=1.0))
        finally:
            r1.stop()
            r1.teardown()

        self.assertEqual(thrown[0], 10)
        self.assertGreater(executed[0], thrown[0])

    def test_reactor_with_timestamp_lookup(self):
        # check the semantics of 'curTimestampIsAfter'
        t0 = time.time()
        db = self.createNewDb()

        s = Schema("schema")

        @s.define
        class Thing:
            nextUpdate = float
            timesUpdated = int

        db.subscribeToSchema(s)

        with db.transaction():
            someThings = [Thing(nextUpdate=t0+0.001, timesUpdated=0) for _ in range(10)]

        def incrementor():
            with db.transaction():
                for t in someThings:
                    if Reactor.curTimestampIsAfter(t.nextUpdate):
                        t.nextUpdate = t.nextUpdate + 0.01
                        t.timesUpdated += 1

        def updateCount():
            with db.view():
                return sum([x.timesUpdated for x in Thing.lookupAll()])

        r1 = Reactor(db, incrementor)

        while updateCount() < 1000 and time.time() - t0 < 2.0:
            r1.next(timeout=1.0)

        self.assertTrue(time.time() - t0 < 1.2)

        r1.start()
        self.assertTrue(db.waitForCondition(lambda: sum(x.timesUpdated for x in Thing.lookupAll()) >= 2000, timeout=2.0))
        self.assertTrue(time.time() - t0 < 2.2)
        r1.stop()

    def test_reactor_synchronous(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        def incrementor():
            executed = 0
            with db.transaction():
                for c in Counter.lookupAll():
                    if c.x != 0:
                        executed += 1
                        c.k += c.x
                        c.x = 0

            return executed

        r1 = Reactor(db, incrementor)

        self.assertEqual(r1.next(timeout=.01), 0)
        self.assertIs(r1.next(timeout=.01), object_database.reactor.Timeout)

        with db.transaction():
            c = Counter(k=0, x=100)

        self.assertEqual(r1.next(timeout=.01), 1)
        self.assertEqual(r1.next(timeout=.01), 0)
        self.assertIs(r1.next(timeout=.01), object_database.reactor.Timeout)

        with db.view():
            self.assertEqual(c.k, 100)
            self.assertEqual(c.x, 0)

        with db.transaction():
            c.x += 2

        self.assertEqual(r1.next(timeout=.01), 1)
        self.assertEqual(r1.next(timeout=.01), 0)
        self.assertIs(r1.next(timeout=.01), object_database.reactor.Timeout)

        with db.view():
            self.assertEqual(c.k, 102)
            self.assertEqual(c.x, 0)

        r1.teardown()

    def test_reactor_block_until_true(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        checkCount = [0]
        incrementCount = [0]

        def incrementor():
            incrementCount[0] += 1
            # move '1' from any nonzero 'x' to 'k'
            with db.transaction():
                for c in Counter.lookupAll():
                    if c.x > 0:
                        c.k += 1
                        c.x -= 1

            time.sleep(0.001)

        def checker():
            # Check if any counters exist with x > 0
            checkCount[0] += 1
            with db.view():
                for c in Counter.lookupAll():
                    if c.x > 0:
                        return False
            return True

        incrementor = Reactor(db, incrementor)
        checker = Reactor(db, checker)
        incrementor.start()

        with db.transaction():
            c = Counter(k=0, x=10)

        self.assertFalse(checker.blockUntilTrue(timeout=.00001))
        self.assertTrue(checker.blockUntilTrue(timeout=1.0))

        self.assertTrue(incrementCount[0] >= 10)
        self.assertTrue(checkCount[0] >= 5)

        with db.transaction():
            c.x += 5

        self.assertFalse(checker.blockUntilTrue(timeout=.00001))
        self.assertTrue(checker.blockUntilTrue(timeout=1.0))

        self.assertTrue(incrementCount[0] >= 15)
        self.assertTrue(checkCount[0] >= 8)

    def test_adding_while_subscribing(self, shouldSubscribeToIndex=False):
        pfactor = self.PERFORMANCE_FACTOR
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)

        with db1.transaction():
            # make sure we have values in there.
            for _ in range(10000):
                Counter(k=123, x=-1)

            c1 = Counter(k=123)
            c1.x = 1

        blocker = BlockingCallback()

        self.server._subscriptionBackgroundThreadCallback = blocker.callback

        if shouldSubscribeToIndex:
            subscriptionEvents = db2.subscribeToIndex(Counter, k=123, block=False)
        else:
            subscriptionEvents = db2.subscribeToType(Counter, block=False)

        self.assertEqual(blocker.waitForCallback(pfactor), 0)

        # make a transaction
        with db1.transaction():
            c1.x = 2
            c2 = Counter(k=123)

        blocker.releaseCallback()
        for i in range(1, 101):
            self.assertEqual(blocker.waitForCallback(pfactor), i)
            blocker.releaseCallback()

        self.assertEqual(blocker.waitForCallback(pfactor), "DONE")
        blocker.releaseCallback()

        for e in subscriptionEvents:
            assert e.wait(timeout=2.0*pfactor)

        with db2.transaction():
            # verify we see the write on c1
            self.assertTrue(c1.exists())
            self.assertTrue(c1.x == 2)

            # check we see the creation of c2
            self.assertTrue(c2.exists())

    def test_adding_while_subscribing_and_moving_into_index(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)

        with db1.transaction():
            for _ in range(10000):
                Counter(k=123, x=-1)

            c1 = Counter(k=0)

        blocker = BlockingCallback()

        self.server._subscriptionBackgroundThreadCallback = blocker.callback

        subscriptionEvents = db2.subscribeToIndex(Counter, k=123, block=False)

        for i in range(0, 50):
            self.assertEqual(blocker.waitForCallback(self.PERFORMANCE_FACTOR), i)
            blocker.releaseCallback()

        # even while this is going, we should be able to subscribe to something small
        db3 = self.createNewDb()
        db3.subscribeToIndex(Counter, k=0)
        with db3.view():
            self.assertTrue(c1.exists())

        # make a transaction
        with db1.transaction():
            c1.k = 123

        for i in range(50, 101):
            self.assertEqual(blocker.waitForCallback(self.PERFORMANCE_FACTOR), i)
            blocker.releaseCallback()

        self.assertEqual(blocker.waitForCallback(self.PERFORMANCE_FACTOR), "DONE")
        blocker.releaseCallback()

        for e in subscriptionEvents:
            assert e.wait(timeout=2.0*self.PERFORMANCE_FACTOR)

        with db2.transaction():
            # verify we see the write on c1
            self.assertTrue(c1.exists())

    def test_moving_into_index(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)
        db2.subscribeToIndex(Counter, k=123)

        with db1.transaction():
            c = Counter(k=0)

        db2.flush()
        with db2.view():
            self.assertFalse(c.exists())

        with db1.transaction():
            c.k = 123
            c.x = 100

        db2.flush()
        with db2.view():
            self.assertTrue(c.exists())
            self.assertEqual(c.x, 100)

        with db1.transaction():
            c.k = 0

        db2.flush()
        with db2.view():
            self.assertTrue(c.exists())
            self.assertEqual(c.k, 0)

        with db1.transaction():
            c.x = 101

        db2.flush()
        with db2.view():
            self.assertTrue(c.exists())
            self.assertEqual(c.x, 101)

    def test_multithreading_and_subscribing(self):
        # Verify that if one thread is subscribing and the other is repeatedly looking
        # at indices, that everything works correctly.

        db1 = self.createNewDb()
        db2 = self.createNewDb()

        testSize = 200

        shouldStop = [False]
        isOK = [False]
        lastSeen = [0]

        def readerthread():
            try:
                db1.subscribeToNone(Counter)

                while not shouldStop[0]:
                    with db1.view():
                        ct = 0
                        for x in Counter.lookupAll(k=0):
                            Counter.lookupAny(k=0)
                            ct += 1
                        time.sleep(0.0001)
                        lastSeen[0] = ct

                isOK[0] = True
            except BaseException:
                print("READ THREAD FAILED: ", traceback.format_exc())

        r = threading.Thread(target=readerthread)
        r.daemon = True
        r.start()

        db2.subscribeToNone(Counter)
        counters = []
        with db2.transaction():
            for i in range(testSize):
                c = Counter(k=0, x=i)
                counters.append(c)

        for i in range(testSize):
            db1.subscribeToObject(counters[i])
            time.sleep(0.001)
            print("wrote", i, "of", testSize, "and last saw", lastSeen[0])

        t0 = time.time()
        while lastSeen[0] != testSize and time.time() - t0 < 1.0:
            time.sleep(0.01)

        shouldStop[0] = True
        r.join()

        self.assertEqual(lastSeen[0], testSize)
        self.assertTrue(isOK[0])

    def test_subscription_matching_is_linear(self):
        schemas = []
        dbs = []

        db = self.createNewDb()

        while len(schemas) < 20:
            # make a new schema
            s = Schema("schema_" + str(len(schemas)))

            @s.define
            class Thing:
                x = int

            schemas.append(s)

            # create a new database for this new schema and subscribe in both this one and
            # the main connection
            dbs.append(self.createNewDb())
            dbs[-1].subscribeToSchema(s)
            db.subscribeToSchema(s)

            # create a new object in the schema
            things = []
            for i in range(len(schemas)):
                with dbs[i].transaction():
                    things.append(schemas[i].Thing(x=10))

            # make sure that the main db sees it
            for thing in things:
                db.waitForCondition(lambda: thing.exists(), 10*self.PERFORMANCE_FACTOR)

            # verify the main db sees something quadratic in the number of transactions plus a constant
            self.assertLess(db._messages_received, (len(schemas) + 1) * (len(schemas) + 2) + 8)

            # each database sees two transactions each pass
            for i in range(len(dbs)):
                self.assertTrue(dbs[i]._messages_received < (len(schemas) - i) * 2 + 10)

    def test_transaction_time_constant(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)
        db2.subscribeToSchema(schema)

        times = []
        for i in range(10000):
            t0 = time.time()
            with db1.transaction():
                schema.Root()
            times.append(time.time() - t0)

        m1 = numpy.mean(times[:1000])
        m2 = numpy.mean(times[-1000:])
        self.assertTrue(abs(m2/m1-1) < 1, (m1, m2))

    def test_memory_growth(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)
        db2.subscribeToSchema(schema)

        m0 = currentMemUsageMb()

        for passIx in range(10):
            for i in range(1000):
                with db1.transaction():
                    x = schema.Root()

                with db1.transaction():
                    x.delete()

            self.server._garbage_collect(intervalOverride=.1)
            print(passIx, currentMemUsageMb())
            self.assertLess(currentMemUsageMb() - m0, 10.0)

        db1.flush()
        db2.flush()

        with db1.view():
            assert len(schema.Root.lookupAll()) == 0

        with db2.view():
            assert len(schema.Root.lookupAll()) == 0

        self.assertLess(db1._connection_state.objectCount(), 10)

        self.assertEqual(db1._connection_state.objectCount(), db2._connection_state.objectCount())

        time.sleep(.1)

        self.assertTrue(len(self.server._version_numbers) > 10)

        self.server._garbage_collect(intervalOverride=.1)

        self.assertTrue(len(self.server._version_numbers) < 10)

    def test_max_tid(self):
        schema1 = Schema("schema1")
        schema2 = Schema("schema2")

        @schema1.define
        class T11:
            k = int

        @schema1.define
        class T12:
            k = int

        @schema2.define
        class T2:
            k = int

        db1 = self.createNewDb()

        self.assertEqual(db1.currentTransactionId(), 1)
        for t in [T11, T12, T2]:
            self.assertEqual(db1.currentTransactionIdForType(t), 0)

        db1.subscribeToSchema(schema1)
        db1.subscribeToSchema(schema2)
        with db1.transaction():
            T11(k=0)

        self.assertEqual(db1.currentTransactionId(), 2)
        self.assertEqual(db1.currentTransactionIdForType(T11), 2)
        self.assertEqual(db1.currentTransactionIdForType(T12), 0)
        self.assertEqual(db1.currentTransactionIdForSchema(schema1), 2)
        self.assertEqual(db1.currentTransactionIdForSchema(schema2), 0)

        with db1.transaction():
            T12(k=0)

        self.assertEqual(db1.currentTransactionId(), 3)
        self.assertEqual(db1.currentTransactionIdForType(T11), 2)
        self.assertEqual(db1.currentTransactionIdForType(T12), 3)
        self.assertEqual(db1.currentTransactionIdForSchema(schema1), 3)
        self.assertEqual(db1.currentTransactionIdForSchema(schema2), 0)

    def test_multiple_serialization_contexts(self):
        def make(versionString):
            class Class:
                vs = versionString

                def f(self):
                    return self.vs

            context = SerializationContext({"Class": Class})

            schema = Schema("version_test_schema")

            schema.Class = Class

            @schema.define
            class Object:
                c = OneOf(None, Class)

            return (schema, context)

        db1 = self.createNewDb()

        schema1, context1 = make("v1")
        schema2, context2 = make("v2")

        db1.subscribeToSchema(schema1)
        db1.setSerializationContext(context1)

        with db1.transaction():
            o = schema1.Object(c=schema1.Class())

        with db1.view():
            self.assertEqual(o.c.f(), "v1")

        db1.subscribeToSchema(schema2)
        with db1.view().setSerializationContext(context2):
            self.assertEqual(len(schema2.Object.lookupAll()), 1)
            o2 = schema2.Object.lookupOne()
            self.assertEqual(o2._identity, o._identity)
            self.assertTrue(o.exists())
            self.assertTrue(o2.exists())
            self.assertEqual(o2.c.f(), "v2")


class ObjectDatabaseOverChannelTestsWithRedis(unittest.TestCase, ObjectDatabaseTests):
    @classmethod
    def setUpClass(cls):
        ObjectDatabaseTests.setUpClass()

    def setUp(self):
        self.tempDir = tempfile.TemporaryDirectory()
        self.tempDirName = self.tempDir.__enter__()
        self.auth_token = genToken()

        if hasattr(self, 'redisProcess') and self.redisProcess:
            self.redisProcess.tearDown()

        self.redisProcess = RedisTestHelper(port=1115)

        try:
            self.mem_store = RedisPersistence(port=1115)
            self.server = InMemServer(self.mem_store, self.auth_token)
            self.server._gc_interval = .1
            self.server.start()
        except Exception:
            self.redisProcess.terminate()
            raise

    def createNewDb(self):
        return self.server.connect(self.auth_token)

    def tearDown(self):
        self.server.stop()
        self.redisProcess.tearDown()
        self.redisProcess = None
        self.tempDir.__exit__(None, None, None)

    def test_reboot_against_redis(self):
        db1 = self.createNewDb()
        db1.subscribeToSchema(schema)

        with db1.transaction():
            c = Counter(k=123)

        self.server.stop()
        self.mem_store = RedisPersistence(port=1115)
        self.server = InMemServer(self.mem_store, self.auth_token)
        self.server.start()

        db1.disconnect()

        db1 = self.createNewDb()
        db1.subscribeToSchema(schema)
        with db1.view():
            self.assertTrue(c.exists())
            self.assertEqual(c.k, 123)
            self.assertEqual(list(Counter.lookupAll()), [c])
            self.assertEqual(list(Counter.lookupAll(k=123)), [c])
            self.assertEqual(list(Counter.lookupAll(k=124)), [])

        with db1.transaction():
            c.k = 124

    def test_throughput(self):
        pass

    def test_object_versions_robust(self):
        pass

    def test_flush_db_works(self):
        pass


class ObjectDatabaseOverChannelTestsInMemory(unittest.TestCase, ObjectDatabaseTests):
    @classmethod
    def setUpClass(cls):
        ObjectDatabaseTests.setUpClass()

    def setUp(self):
        self.auth_token = genToken()

        self.mem_store = InMemoryPersistence()
        self.server = InMemServer(self.mem_store, self.auth_token)
        self.server._gc_interval = .1
        self.server.start()
        self.allConnections = []

    def createNewDb(self):
        conn = self.server.connect(self.auth_token)
        self.allConnections.append(conn)
        return conn

    def tearDown(self):
        for c in self.allConnections:
            c.disconnect(block=True)

        self.server.stop()

    def test_connection_without_auth_disconnects(self):
        db = DatabaseConnection(self.server.getChannel())

        old_interval = messages.getHeartbeatInterval()
        messages.setHeartbeatInterval(.25)

        try:
            with self.assertRaises(DisconnectedException):
                db.subscribeToSchema(schema)

        finally:
            messages.setHeartbeatInterval(old_interval)

    def test_heartbeats(self):
        old_interval = messages.getHeartbeatInterval()
        messages.setHeartbeatInterval(.25)

        try:
            db1 = self.createNewDb()
            db2 = self.createNewDb()

            db1.subscribeToSchema(core_schema)
            db2.subscribeToSchema(core_schema)

            db1.flush()

            with db1.view():
                self.assertTrue(len(core_schema.Connection.lookupAll()), 2)

            with db2.view():
                self.assertTrue(len(core_schema.Connection.lookupAll()), 2)

            db1._stopHeartbeating()

            db2.waitForCondition(
                lambda: len(core_schema.Connection.lookupAll()) == 1,
                5.0 * self.PERFORMANCE_FACTOR
            )

            with db2.view():
                self.assertEqual(len(core_schema.Connection.lookupAll()), 1)

            with self.assertRaises(DisconnectedException):
                with db1.view():
                    pass
        finally:
            messages.setHeartbeatInterval(old_interval)

    def test_multithreading_and_cleanup(self):
        # Verify that if one thread is subscribing and the other is repeatedly looking
        # at indices, that everything works correctly.
        db1 = self.createNewDb()
        db1.subscribeToType(Counter)

        db2 = self.createNewDb()
        db2.subscribeToType(Counter)

        shouldStop = [False]
        isOK = []

        threadcount = 4

        def readerthread(db):
            c = None
            while not shouldStop[0]:
                if numpy.random.uniform() < .5:
                    if c is None:
                        with db.transaction():
                            c = Counter(k=0)
                    else:
                        with db.transaction():
                            c.delete()
                            c = None
                else:
                    with db.view():
                        Counter.lookupAny(k=0)

            isOK.append(True)

        threads = [threading.Thread(target=readerthread, args=(db1 if threadcount % 2 else db2,)) for _ in range(threadcount)]
        for t in threads:
            t.start()

        time.sleep(1.0)

        shouldStop[0] = True

        for t in threads:
            t.join()

        self.assertTrue(len(isOK) == threadcount)


class ObjectDatabaseOverSocketTests(unittest.TestCase, ObjectDatabaseTests):
    @classmethod
    def setUpClass(cls):
        ObjectDatabaseTests.setUpClass()

    def setUp(self):
        self.mem_store = InMemoryPersistence()
        self.auth_token = genToken()

        sc = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        sc.load_cert_chain('testcert.cert', 'testcert.key')

        self.server = TcpServer(
            host="localhost", port=8888, mem_store=self.mem_store,
            ssl_context=sc, auth_token=self.auth_token
        )
        self.server._gc_interval = .1
        self.server.start()

    def createNewDb(self, useSecondaryLoop=False):
        db = self.server.connect(self.auth_token, useSecondaryLoop=useSecondaryLoop)
        db.initialized.wait()
        return db

    def tearDown(self):
        self.server.stop()

    def test_very_large_subscriptions(self):
        old_interval = messages.getHeartbeatInterval()
        messages.setHeartbeatInterval(.1)

        try:
            db1 = self.createNewDb()
            db1.subscribeToSchema(schema)

            for ix in range(1, 3):
                with db1.transaction():
                    for i in range(5000):
                        Counter(k=ix, x=i)

            # now there's a lot of stuff in the database

            isDone = [False]
            maxLatency = [None]

            def transactionLatencyTimer():
                while not isDone[0]:
                    t0 = time.time()

                    with db1.transaction():
                        Counter()

                    latency = time.time() - t0
                    maxLatency[0] = max(maxLatency[0] or 0.0, latency)

                    time.sleep(0.01)

            latencyMeasureThread = threading.Thread(target=transactionLatencyTimer)
            latencyMeasureThread.start()

            db2 = self.createNewDb(useSecondaryLoop=True)

            t0 = time.time()

            db2._largeSubscriptionHeartbeatDelay = 10
            db2.subscribeToSchema(schema)
            db2._largeSubscriptionHeartbeatDelay = 0

            subscriptionTime = time.time() - t0

            isDone[0] = True
            latencyMeasureThread.join()

            # verify the properties of the subscription. we shouldn't be disconnected!
            with db2.view():
                self.assertEqual(len(Counter.lookupAll(k=1)), 5000)
                self.assertEqual(len(Counter.lookupAll(k=2)), 5000)
                self.assertEqual(
                    sorted(set([c.x for c in Counter.lookupAll(k=1)])),
                    sorted(range(5000))
                )

            # we should never have had a really long latency
            self.assertTrue(maxLatency[0] < subscriptionTime / 10.0, (maxLatency[0], subscriptionTime))

        finally:
            messages.setHeartbeatInterval(old_interval)
