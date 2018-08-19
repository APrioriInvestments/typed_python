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

from object_database.database import Database, RevisionConflictException, Indexed, Index, Schema
import object_database.database
import object_database.InMemoryJsonStore as InMemoryJsonStore

import unittest
import random
import time

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

schema = Schema()

@schema.define
class Root:
    obj=OneOf(None, schema.Object)

@schema.define
class Object:
    k=Indexed(expr)
    other=OneOf(None, schema.Object)

@schema.define
class Counter:
    k = Indexed(int)
    x = int

    def f(self):
        return self.k + 1
    
    def __str__(self):
        return "Counter(k=%s)" % self.k

class ObjectDatabaseTests:
    def test_methods(self):
        db = self.createNewDb()

        with db.transaction():
            counter = Counter()
            counter.k = 2
            self.assertEqual(counter.f(), 3)
            self.assertEqual(str(counter), "Counter(k=2)")

    def test_identity_transfer(self):
        db = self.createNewDb()

        with db.transaction():
            root = Root()
            root2 = Root.fromIdentity(root._identity)

            root.obj = Object(k=expr.Constant(value=23))
            self.assertEqual(root2.obj.k.value, 23)

    def test_basic(self):
        db = self.createNewDb()

        with db.transaction():
            root = Root()

            self.assertTrue(root.obj is None)

            root.obj = Object(k=expr.Constant(value=23))

        db2 = self.createNewDb()

        with db2.view():
            self.assertEqual(root.obj.k.value, 23)

    def test_throughput(self):
        db = self.createNewDb()

        with db.transaction():
            root = Root()
            root.obj = Object(k=expr.Constant(value=0))

        t0 = time.time()
        while time.time() < t0 + 1.0:
            with db.transaction() as t:
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)
        
        with db.view():
            self.assertTrue(root.obj.k.value > 500, root.obj.k.value)
                
    def test_exists(self):
        db = self.createNewDb()

        with db.transaction():
            root = Root()

            self.assertTrue(root.exists())

            root.delete()

            self.assertFalse(root.exists())

        with db.view():
            self.assertFalse(root.exists())

        db = self.createNewDb()

        with db.view():
            self.assertFalse(root.exists())

    def test_read_performance(self):
        db = self.createNewDb()

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
                for subix in range(int(random.random() * 5 + 1)):
                    counter = counters[int(random.random() * len(counters))]

                    if counter.exists():
                        if random.random() < .001:
                            counter.delete()
                        else:
                            counter.k = int(random.random() * 100)
                        total_writes += 1

                counter_vals_by_tn[db._cur_transaction_num + 1] = {c: c.k for c in counters if c.exists()}

            views_by_tn[db._cur_transaction_num] = db.view()

            while views_by_tn and random.random() < .5 or len(views_by_tn) > 10:
                #pick a random view and check that it's consistent
                all_tids = list(views_by_tn)
                tid = all_tids[int(random.random() * len(all_tids))]

                with views_by_tn[tid]:
                    for c in counters:
                        if not c.exists():
                            assert c not in counter_vals_by_tn[tid]
                        else:
                            self.assertEqual(c.k, counter_vals_by_tn[tid][c])

                del views_by_tn[tid]

            if random.random() < .05 and views_by_tn:
                with db.view():
                    max_counter_vals = {}
                    for c in counters:
                        if c.exists():
                            max_counter_vals[c] = c.k

                #reset the database
                db = self.createNewDb()

                new_counters = list(counters)

                views_by_tn = {db._cur_transaction_num: db.view()}

                counter_vals_by_tn = {db._cur_transaction_num: 
                    {new_counters[ix]: max_counter_vals[counters[ix]] for ix in 
                        range(len(counters)) if counters[ix] in max_counter_vals}
                    }

                counters = new_counters

        self.assertLess(self.mem_store.storedStringCount(), 100)
        self.assertTrue(total_writes > 500)

    def test_flush_db_works(self):
        db = self.createNewDb()

        with db.transaction():
            c = Counter()
            c.k = 1

        self.assertTrue(self.mem_store.values)

        view = db.view()

        with db.transaction():
            c.delete()

        #database doesn't have this
        self.assertFalse(self.mem_store.storedStringCount())

        #but the view does!
        with view:
            self.assertTrue(c.exists())

        self.assertFalse(self.mem_store.storedStringCount())

    def test_read_write_conflict(self):
        db = self.createNewDb()

        schema = Schema()

        @schema.define
        class Counter:
            k = int

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

        with db.transaction():
            o = Counter(k=1)
            o.x = 10

        db = self.createNewDb()

        with db.transaction() as v:
            o = Counter.lookupOne(k=1)
            self.assertEqual(o.x, 10)
            o.k = 2
            o.x = 11

        db = self.createNewDb()

        with db.transaction() as v:
            o = Counter.lookupOne(k=2)
            o.k = 3
            self.assertEqual(o.x, 11)
            
        db = self.createNewDb()

        with db.transaction() as v:
            self.assertFalse(Counter.lookupAny(k=2))
            
            o = Counter.lookupOne(k=3)
            o.k = 3
            self.assertEqual(o.x, 11)
            
    def test_index_consistency(self):
        db = self.createNewDb()

        schema = Schema()

        @schema.define
        class Object:
            x = int
            y = int

            @Indexed
            def pair(self):
                return (self.x, self.y)

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

        with db.transaction():
            o1 = Object(k=expr.Constant(value=123))

        with db.view() as v:
            self.assertEqual(Object.lookupAll(k=expr.Constant(value=123)), (o1,))

    def test_index_functions(self):
        db = self.createNewDb()

        schema = Schema()
        @schema.define
        class Object:
            k=Indexed(int)

            @Indexed
            def k2(self) -> int:
                return self.k * 2

            pair_index = Index('k', 'k')

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

        schema = Schema()
        @schema.define
        class Object:
            k=Indexed(int)

            @Indexed
            def index(self):
                return True if self.k > 10 else None

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

        schema = Schema()

        @schema.define
        class Object:
            k=Indexed(int)
        
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

        schema = Schema()

        @schema.define
        class Object:
            k=Indexed(int)

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

        schema = Schema()

        @schema.define
        class Object:
            x = TupleOf(int)

        with db.transaction():
            n = Object()
            self.assertEqual(len(n.x), 0)

class ObjectDatabaseInMemTests(unittest.TestCase, ObjectDatabaseTests):
    def setUp(self):
        self.mem_store = InMemoryJsonStore.InMemoryJsonStore()

    def createNewDb(self):
        db = Database(self.mem_store)
        return db

class ObjectDatabaseOverChannelTests(unittest.TestCase, ObjectDatabaseTests):
    def setUp(self):
        self.mem_store = InMemoryJsonStore.InMemoryJsonStore()
        self.core_db = Database(self.mem_store)
        self.channels = []

    def createNewDb(self):
        channel = object_database.database.InMemoryChannel()

        self.core_db.addConnection(channel)

        db = object_database.database.DatabaseConnection(channel)

        channel.start()

        self.channels.append(channel)

        db.initialized.wait()

        return db

    def tearDown(self):
        for c in self.channels:
            c.stop()


class ObjectDatabaseOverSocketTests(unittest.TestCase, ObjectDatabaseTests):
    def setUp(self):
        self.mem_store = InMemoryJsonStore.InMemoryJsonStore()
        self.core_db = Database(self.mem_store)
        self.databaseServer = object_database.database.DatabaseServer(self.core_db, host="localhost", port=8888)
        self.databaseServer.start()
        self.channels = []

    def createNewDb(self, ):
        db = object_database.database.connect("localhost", 8888)

        self.channels.append(db)
        
        db.initialized.wait()

        return db

    def tearDown(self):
        self.databaseServer.stop()


