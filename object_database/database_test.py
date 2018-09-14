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

from typed_python import Alternative, TupleOf, OneOf

from object_database.schema import Indexed, Index, Schema
from object_database.core_schema import core_schema
from object_database.view import RevisionConflictException, DisconnectedException, ObjectDoesntExistException
from object_database.database_connection import TransactionListener, DatabaseConnection
from object_database.tcp_server import TcpServer, connect
from object_database.inmem_server import InMemServer
from object_database.persistence import InMemoryPersistence, RedisPersistence
import object_database.messages as messages
import queue
import unittest
import tempfile
import numpy
import redis
import subprocess
import os
import threading
import random
import time

from object_database.util import configureLogging

configureLogging("test", error=True)

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
        

expr = Alternative("Expr")
expr.define(
    Constant = {'value': int},
    Add = {'l': expr, 'r': expr},
    Sub = {'l': expr, 'r': expr},
    Mul = {'l': expr, 'r': expr}
    )

expr.__str__ = lambda self: (
    "Constant(%s)" % self.value if self.matches.Constant else
    "Add(%s,%s)" % (self.l,self.r) if self.matches.Add else
    "Sub(%s,%s)" % (self.l,self.r) if self.matches.Sub else
    "Mul(%s,%s)" % (self.l,self.r) if self.matches.Mul else "<unknown>"
    )

schema = Schema("test_schema")
schema.expr = expr

@schema.define
class Root:
    obj=OneOf(None, schema.Object)
    k = int

@schema.define
class Object:
    k=Indexed(expr)
    other=OneOf(None, schema.Object)

    @property
    def otherK(self):
        if self.other is not None:
            return self.other.k

@schema.define
class Counter:
    k = Indexed(int)
    x = int

    def f(self):
        return self.k + 1
    
    def __str__(self):
        return "Counter(k=%s)" % self.k

@schema.define
class StringIndexed:
    name = Indexed(str)

class ObjectDatabaseTests:
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
            
            counter2 = Object(other=counter,k=expr.Constant(value=0))

            self.assertEqual(counter2.otherK, counter.k)

    def test_identity_transfer(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)
        
        with db.transaction():
            root = Root()
            root2 = Root.fromIdentity(root._identity)

            root.obj = Object(k=expr.Constant(value=23))
            self.assertEqual(root2.obj.k.value, 23)

    def test_a_many_subscriptions(self):
        OK = []
        FINISHED = []
        count = 10
        threadCount = 10

        def worker(index):
            db = self.createNewDb()

            indices = list(range(count))
            numpy.random.shuffle(indices)

            for i in indices:
                db.subscribeToIndex(Counter,k=i)

                with db.transaction():
                    Counter(k=i, x=index)
            
            FINISHED.append(True)

            db.waitForCondition(lambda: len(FINISHED) == threadCount, 10.0)
            db.flush()

            with db.view():
                actuallyVisible = len(Counter.lookupAll())

            if actuallyVisible != count * threadCount:
                print("TOTAL is ", actuallyVisible, " != ", count*threadCount)
            else:
                OK.append(True)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(threadCount)]
        for t in threads:
            t.daemon=True
            t.start()
        for t in threads:
            t.join()

        db1 = self.createNewDb()
        db1.subscribeToSchema(schema)
        with db1.view():
            self.assertEqual(len(Counter.lookupAll()), count*threadCount)

        db2 = self.createNewDb()

        for i in range(count):
            db2.subscribeToIndex(Counter,k=i)
        db2.flush()
        with db2.view():
            self.assertEqual(len(Counter.lookupAll()), count*threadCount)

        self.assertEqual(len(OK), 10)

    def test_transaction_handlers(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)
        
        didOne = threading.Event()

        def handler(changed):
            didOne.set()

        with TransactionListener(db, handler):
            with db.transaction():
                root = Root()

            didOne.wait()

        assert didOne.isSet()
        
    def test_basic(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)
        
        with db.transaction():
            root = Root()

            self.assertTrue(root.obj is None)

            root.obj = Object(k=expr.Constant(value=23))

        db2 = self.createNewDb()
        db2.subscribeToSchema(schema)
        

        with db2.view():
            self.assertEqual(root.obj.k.value, 23)

    def test_throughput(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)
        
        with db.transaction():
            root = Root()
            root.obj = Object(k=expr.Constant(value=0))

        t0 = time.time()
        while time.time() < t0 + 1.0:
            with db.transaction() as t:
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)
        
        with db.view():
            self.assertTrue(root.obj.k.value > 500, root.obj.k.value)
            print(root.obj.k.value, "transactions per second")
                
    def test_delayed_transactions(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        confirmed = queue.Queue()

        with db.transaction():
            root = Root()
            root.obj = Object(k=expr.Constant(value=0))

        t0 = time.time()

        for i in range(1000):
            with db.transaction().onConfirmed(confirmed.put) as t:
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
            for i in range(100):
                root = Root()

                e = expr.Constant(value=i)
                e = expr.Add(l=e,r=e)
                e = expr.Add(l=e,r=e)
                e = expr.Add(l=e,r=e)

                root.obj = Object(k=e)

                objects[i] = root

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        t0 = time.time()
        count = 0
        steps = 0
        while time.time() < t0 + 1.0:
            with db.transaction() as t:
                for i in range(100):
                    count += objects[i].obj.k.l.r.l.value
                    steps += 1

    def test_transactions(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            root = Root()

        views = [db.view()]

        for i in [1,2,3]:
            with db.transaction():
                root.obj = Object(k=expr.Constant(value=i))
            views.append(db.view())

        vals = []
        for v in views:
            with v:
                if root.obj is None:
                    vals.append(None)
                else:
                    vals.append(root.obj.k.value)

        self.assertEqual(vals, [None, 1,2,3])

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
                t1,t2 = t2,t1

            with t1:
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)

            with self.assertRaises(RevisionConflictException):
                with t2:
                    root.obj.k = expr.Constant(value=root.obj.k.value + 1)
    
    def test_object_versions_robust(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        counters = []
        counter_vals_by_tn = {}
        views_by_tn = {}

        random.seed(123)

        #expect nothing initially
        views_by_tn[db._cur_transaction_num] = db.view()
        counter_vals_by_tn[db._cur_transaction_num] = {}

        #seed the initial state
        with db.transaction() as t:
            for i in range(20):
                counter = Counter(_identity="C_%s" % i)
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
                #pick a random view and check that it's consistent
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

                #reset the database
                db = self.createNewDb()
                db.subscribeToSchema(schema)

                with db.view():
                    newCounterVals = {c: c.k for c in counters if c.exists()}
                
                self.assertEqual(curCounterVals, newCounterVals)

                views_by_tn = {}
                counter_vals_by_tn = {}

        #we may have one or two for connection objects, and we have two values for every indexed thing
        self.assertLess(self.mem_store.storedStringCount(), 203)
        self.assertTrue(total_writes > 500)

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

        #database doesn't have this
        t0 = time.time()
        while time.time() - t0 < 1.0 and self.mem_store.storedStringCount() >= 2:
            time.sleep(.01)

        self.assertLess(self.mem_store.storedStringCount(), 4)

        #but the view does!
        with view:
            for c in counters:
                self.assertTrue(c.exists())

    def test_read_write_conflict(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Counter:
            k = int

        db.subscribeToSchema(schema)

        with db.transaction():
            o1 = Counter()
            o2 = Counter()

        for consistency in [True, False]:
            if consistency:
                t1 = db.transaction().consistency(reads=True)
                t2 = db.transaction().consistency(reads=True)
            else:
                t1 = db.transaction().consistency(none=True)
                t2 = db.transaction().consistency(none=True)

            with t1.nocommit():
                o1.k = o2.k + 1
            
            with t2.nocommit():
                o2.k = o1.k + 1

            t1.commit()

            if consistency:
                with self.assertRaises(RevisionConflictException):
                    t2.commit()
            else:
                t2.commit()
            
    def test_indices(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.view() as v:
            self.assertEqual(Counter.lookupAll(k=20), ())
            self.assertEqual(Counter.lookupAll(k=30), ())

        with db.transaction():
            o1 = Counter(k = 20)

        with db.view() as v:
            self.assertEqual(Counter.lookupAll(k=20), (o1,))
            self.assertEqual(Counter.lookupAll(k=30), ())

        with db.transaction():
            o1.k = 30

        with db.view() as v:
            self.assertEqual(Counter.lookupAll(k=20), ())
            self.assertEqual(Counter.lookupAll(k=30), (o1,))

        with db.transaction():
            o1.delete()

        with db.view() as v:
            self.assertEqual(Counter.lookupAll(k=20), ())
            self.assertEqual(Counter.lookupAll(k=30), ())

    def test_indices_multiple_values(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction() as v:
            k1 = Counter(k=20)
            k2 = Counter(k=20)

            self.assertEqual(len(Counter.lookupAll(k=20)), 2)

            k1.k = 30

            self.assertEqual(len(Counter.lookupAll(k=20)), 1)

            k1.k = 20

            self.assertEqual(len(Counter.lookupAll(k=20)), 2)

        with db.transaction() as v:
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

        with db.transaction() as v:
            o = Counter.lookupOne(k=1)
            self.assertEqual(o.x, 10)
            o.k = 2
            o.x = 11

        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction() as v:
            o = Counter.lookupOne(k=2)
            o.k = 3
            self.assertEqual(o.x, 11)
            
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction() as v:
            self.assertFalse(Counter.lookupAny(k=2))
            
            o = Counter.lookupOne(k=3)
            o.k = 3
            self.assertEqual(o.x, 11)
            
    def test_index_consistency(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            x = int
            y = int

            @Indexed
            def pair(self):
                return (self.x, self.y)

        db.subscribeToSchema(schema)

        with db.transaction():
            o = Object(x=0,y=0)

        t1 = db.transaction()
        t2 = db.transaction()

        with t1.nocommit():
            o.x = 1

        with t2.nocommit():
            o.y = 1

        t1.commit()

        with self.assertRaises(RevisionConflictException):
            t2.commit()

        with self.assertRaises(Exception):
            with db.transaction().consistency(writes=True):
                o.y = 2


    def test_indices_of_algebraics(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            o1 = Object(k=expr.Constant(value=123))

        with db.view() as v:
            self.assertEqual(Object.lookupAll(k=expr.Constant(value=123)), (o1,))

    def test_frozen_schema(self):
        schema = Schema("test_schema")

        @schema.define
        class Object:
            x = int
            y = int

        Object.fromIdentity("hi")

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
        
        
    def test_index_functions(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            k=Indexed(int)

            @Indexed
            def k2(self) -> int:
                return self.k * 2

            pair_index = Index('k', 'k')

        db.subscribeToSchema(schema)

        with db.transaction():
            o1 = Object(k=10)

        with db.view() as v:
            self.assertEqual(Object.lookupAll(k=10), (o1,))
            self.assertEqual(Object.lookupAll(k2=20), (o1,))
            self.assertEqual(Object.lookupAll(k=20), ())
            self.assertEqual(o1.k2(), o1.k * 2)

            self.assertEqual(Object.lookupAll(pair_index=(10,10)), (o1,))
            self.assertEqual(Object.lookupAll(pair_index=(10,11)), ())

            with self.assertRaises(Exception):
                self.assertEqual(Object.lookupAll(pair_index=(10,"hi")), (o1,))

    def test_index_functions_None_semantics(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            k=Indexed(int)

            @Indexed
            def index(self):
                return True if self.k > 10 else None

        db.subscribeToSchema(schema)

        with db.transaction() as v:
            self.assertEqual(Object.lookupAll(index=True), ())
            o1 = Object(k=10)
            self.assertEqual(Object.lookupAll(index=True), ())
            o1.k = 20
            self.assertEqual(Object.lookupAll(index=True), (o1,))
            o1.k = 10
            self.assertEqual(Object.lookupAll(index=True), ())
            o1.k = 20
            self.assertEqual(Object.lookupAll(index=True), (o1,))
            o1.delete()
            self.assertEqual(Object.lookupAll(index=True), ())

    def test_indices_update_during_transactions(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            k=Indexed(int)

        db.subscribeToSchema(schema)
        
        with db.transaction() as v:
            self.assertEqual(Object.lookupAll(k=10), ())
            o1 = Object(k=10)

            self.assertEqual(Object.lookupAll(k=10), (o1,))
            
            o1.k = 20

            self.assertEqual(Object.lookupAll(k=10), ())
            self.assertEqual(Object.lookupAll(k=20), (o1,))

            o1.delete()

            self.assertEqual(Object.lookupAll(k=10), ())
            self.assertEqual(Object.lookupAll(k=20), ())

    def test_index_transaction_conflicts(self):
        db = self.createNewDb()

        schema = Schema("test_schema")

        @schema.define
        class Object:
            k=Indexed(int)

        db.subscribeToSchema(schema)

        with db.transaction():
            o1 = Object(k=10)
            o2 = Object(k=20)
            o3 = Object(k=30)

        t1 = db.transaction().consistency(full=True)
        t2 = db.transaction().consistency(full=True)

        with t1.nocommit():
            o2.k=len(Object.lookupAll(k=10))

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
            c0 = Counter(k = 0)
            c1 = Counter(k = 1)

            c0.x = 20
            c1.x = 30

        db1.subscribeToIndex(Counter,k=0)
        db2.subscribeToIndex(Counter,k=1)

        with db1.view():
            self.assertTrue(c0.exists())
            self.assertEqual(c0.x, 20)
            self.assertFalse(c1.exists())

        with db2.view():
            self.assertTrue(c1.exists())
            self.assertEqual(c1.x, 30)
            self.assertFalse(c0.exists())

        #create a new value in the view and verify it shows up
        with db_all.transaction():
            c2_0 = Counter(k=0)
            c2_1 = Counter(k=1)

        db1.waitForCondition(lambda: c2_0.exists(), 2)
        db2.waitForCondition(lambda: c2_1.exists(), 2)

        with db2.view():
            self.assertFalse(c2_0.exists())
        with db1.view():
            self.assertFalse(c2_1.exists())

        #now move c2_0 from '0' to '1'. It should show up in db2 and still in db1
        with db_all.transaction():
            c2_0.k = 1

        db1.waitForCondition(lambda: c2_0.exists(), 2)
        db2.waitForCondition(lambda: c2_0.exists(), 2)
        
        #now, we should see it get subscribed to in both
        with db_all.transaction():
            c2_0.x = 40

        db1.waitForCondition(lambda: c2_0.x == 40, 2)
        db2.waitForCondition(lambda: c2_0.x == 40, 2)
        
        #but if we make a new database connection and subscribe, we won't see it
        db3 = self.createNewDb()
        db3.subscribeToIndex(Counter,k=0)
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

    def test_adding_while_subscribing(self, shouldSubscribeToIndex=False):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)

        with db1.transaction():
            c1 = Counter(k = 123)
            c1.x = 1

        blocker = BlockingCallback()

        self.server._subscriptionBackgroundThreadCallback = blocker.callback

        if shouldSubscribeToIndex:
            subscriptionEvents = db2.subscribeToIndex(Counter, k=123, block=False)
        else:
            subscriptionEvents = db2.subscribeToType(Counter, block=False)

        self.assertEqual(blocker.waitForCallback(1.0), 0)

        #make a transaction
        with db1.transaction():
            c1.x = 2
            c2 = Counter(k = 123)

        blocker.releaseCallback()
        self.assertEqual(blocker.waitForCallback(1.0), "DONE")
        blocker.releaseCallback()

        for e in subscriptionEvents:
            assert e.wait(timeout=1.0)

        with db2.transaction():
            #verify we see the write on c1
            self.assertTrue(c1.exists())
            self.assertTrue(c1.x == 2)

            #check we see the creation of c2
            self.assertTrue(c2.exists())

    def test_adding_while_subscribing_and_moving_into_index(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)

        with db1.transaction():
            c1 = Counter(k = 0)

        blocker = BlockingCallback()

        self.server._subscriptionBackgroundThreadCallback = blocker.callback

        subscriptionEvents = db2.subscribeToIndex(Counter, k=123, block=False)
        
        self.assertEqual(blocker.waitForCallback(1.0), 0)

        #make a transaction
        with db1.transaction():
            c1.k = 123

        blocker.releaseCallback()
        self.assertEqual(blocker.waitForCallback(1.0), "DONE")
        blocker.releaseCallback()

        for e in subscriptionEvents:
            assert e.wait(timeout=1.0)

        with db2.transaction():
            #verify we see the write on c1
            self.assertTrue(c1.exists())

    def test_moving_into_index(self):
        db1 = self.createNewDb()
        db2 = self.createNewDb()

        db1.subscribeToSchema(schema)
        db2.subscribeToIndex(Counter, k = 123)

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

    def test_subscription_matching_is_linear(self):
        schemas = []
        dbs = []
        
        db = self.createNewDb()

        while len(schemas) < 20:
            #make a new schema
            s = Schema("schema_" + str(len(schemas)))
            
            @s.define
            class Thing:
                x = int

            schemas.append(s)

            #create a new database for this new schema and subscribe in both this one and
            #the main connection
            dbs.append(self.createNewDb())
            dbs[-1].subscribeToSchema(s)
            db.subscribeToSchema(s)

            #create a new object in the schema
            things = []
            for i in range(len(schemas)):
                with dbs[i].transaction():
                    things.append(schemas[i].Thing(x=10))

            #make sure that the main db sees it
            for thing in things:
                db.waitForCondition(lambda: thing.exists(), 10)

            #verify the main db sees something quadratic in the number of transactions plus a constant
            self.assertLess(db._messages_received, (len(schemas) + 1) * (len(schemas) + 2) + 8)
            
            #each database sees two transactions each pass
            for i in range(len(dbs)):
                self.assertTrue(dbs[i]._messages_received < (len(schemas) - i) * 2 + 10)

class ObjectDatabaseOverChannelTestsWithRedis(unittest.TestCase, ObjectDatabaseTests):
    def setUp(self):
        self.tempDir = tempfile.TemporaryDirectory()
        self.tempDirName = self.tempDir.__enter__()

        if hasattr(self, 'redisProcess') and self.redisProcess:
            self.redisProcess.terminate()
            self.redisProcess.wait()

        self.redisProcess = subprocess.Popen(
            ["/usr/bin/redis-server",'--port', '1115', '--logfile', os.path.join(self.tempDirName, "log.txt"), 
                "--dbfilename", "db.rdb", "--dir", os.path.join(self.tempDirName)]
            )
        time.sleep(.5)
        assert self.redisProcess.poll() is None

        redis.StrictRedis(db=0, decode_responses=True, port=1115).echo("hi")
        self.mem_store = RedisPersistence(port=1115)
        self.server = InMemServer(self.mem_store)
        self.server.start()

    def createNewDb(self):
        db = DatabaseConnection(self.server.getChannel())
        db.initialized.wait()
        return db

    def tearDown(self):
        self.server.stop()
        self.redisProcess.terminate()
        self.redisProcess.wait()
        self.redisProcess = None
        self.tempDir.__exit__(None, None, None)

    def test_throughput(self):
        pass

    def test_object_versions_robust(self):
        pass

    def test_flush_db_works(self):
        pass



class ObjectDatabaseOverChannelTests(unittest.TestCase, ObjectDatabaseTests):
    def setUp(self):
        self.mem_store = InMemoryPersistence()
        self.server = InMemServer(self.mem_store)
        self.server.start()

    def createNewDb(self):
        db = DatabaseConnection(self.server.getChannel())
        db.initialized.wait()
        return db

    def tearDown(self):
        self.server.stop()

    def test_heartbeats(self):
        old_interval = messages.getHeartbeatInterval()
        messages.setHeartbeatInterval(.25)

        try:
            db1 = self.createNewDb()
            db2 = self.createNewDb()

            db1.subscribeToSchema(core_schema)
            db2.subscribeToSchema(core_schema)

            with db1.view():
                self.assertTrue(len(core_schema.Connection.lookupAll()), 2)

            with db2.view():
                self.assertTrue(len(core_schema.Connection.lookupAll()), 2)

            db1._stopHeartbeating()

            db2.waitForCondition(lambda: len(core_schema.Connection.lookupAll()) == 1, 5.0)

            with db2.view():
                self.assertEqual(len(core_schema.Connection.lookupAll()), 1)

            with self.assertRaises(DisconnectedException):
                with db1.view():
                    pass
        finally:
            messages.setHeartbeatInterval(old_interval)

class ObjectDatabaseOverSocketTests(unittest.TestCase, ObjectDatabaseTests):
    def setUp(self):
        self.mem_store = InMemoryPersistence()
        self.server = TcpServer(host="localhost", port=8888, mem_store=self.mem_store)
        self.server.start()

    def createNewDb(self, useSecondaryLoop=False):
        db = self.server.connect(useSecondaryLoop=useSecondaryLoop)

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

            for ix in range(1,3):
                with db1.transaction():
                    for i in range(5000):
                        Counter(k=ix,x=i)

            #now there's a lot of stuff in the database

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

            #verify the properties of the subscription. we shouldn't be disconnected!
            with db2.view():
                self.assertEqual(len(Counter.lookupAll(k=1)), 5000)
                self.assertEqual(len(Counter.lookupAll(k=2)), 5000)
                self.assertEqual(
                    sorted(set([c.x for c in Counter.lookupAll(k=1)])),
                    sorted(range(5000))
                    )

            #we should never have had a really long latency
            self.assertTrue(maxLatency[0] < subscriptionTime / 10.0, (maxLatency[0], subscriptionTime))

        finally:
            messages.setHeartbeatInterval(old_interval)





