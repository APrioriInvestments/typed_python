from typed_python import Alternative, TupleOf

from object_database.database import Database, RevisionConflictException, Indexed, Index

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

def initialize_types(db):
    @db.define
    class Root:
        obj=db.Object

    @db.define
    class Object:
        k=Indexed(expr)
        other=db.Object

    @db.define
    class Counter:
        k = Indexed(int)
        x = int

        def f(self):
            return self.k + 1
        
        def __str__(self):
            return "Counter(k=%s)" % self.k

class ObjectDatabaseTests(unittest.TestCase):
    def test_methods(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)

        with db.transaction():
            counter = db.Counter.New()
            counter.k = 2
            self.assertEqual(counter.f(), 3)
            self.assertEqual(str(counter), "Counter(k=2)")

    def test_basic(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)

        with db.transaction():
            root = db.Root.New()

            self.assertTrue(root.obj is db.Object.Null)

            root.obj = db.Object.New(k=expr.Constant(value=23))

        db2 = Database(mem_store)
        initialize_types(db2)

        with db2.view():
            root = db2.Root(root._identity)
            self.assertEqual(root.obj.k.value, 23)

    def test_throughput(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)

        with db.transaction():
            root = db.Root.New()
            root.obj = db.Object.New(k=expr.Constant(value=0))

        t0 = time.time()
        while time.time() < t0 + 1.0:
            with db.transaction() as t:
                root.obj.k = expr.Constant(value=root.obj.k.value + 1)
        
        with db.view():
            self.assertTrue(root.obj.k.value > 1000, root.obj.k.value)

    def test_read_performance(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)

        objects = {}
        with db.transaction():
            for i in range(100):
                root = db.Root.New()

                e = expr.Constant(value=i)
                e = expr.Add(l=e,r=e)
                e = expr.Add(l=e,r=e)
                e = expr.Add(l=e,r=e)

                root.obj = db.Object.New(k=e)

                objects[i] = root


        db = Database(mem_store)
        initialize_types(db)

        objects = {k: db.Root(v._identity) for k,v in objects.items()}

        t0 = time.time()
        count = 0
        steps = 0
        while time.time() < t0 + 1.0:
            with db.transaction() as t:
                for i in range(100):
                    count += objects[i].obj.k.l.r.l.value
                    steps += 1
        
        

    def test_transactions(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)

        with db.transaction():
            root = db.Root.New()

        views = [db.view()]

        for i in [1,2,3]:
            with db.transaction():
                root.obj = db.Object.New(k=expr.Constant(value=i))
            views.append(db.view())

        vals = []
        for v in views:
            with v:
                if root.obj is db.Object.Null:
                    vals.append(None)
                else:
                    vals.append(root.obj.k.value)

        self.assertEqual(vals, [None, 1,2,3])

    def test_conflicts(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)


        with db.transaction():
            root = db.Root.New()
            root.obj = db.Object.New(k=expr.Constant(value=0))

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
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)

        counters = []
        counter_vals_by_tn = {}
        views_by_tn = {}

        random.seed(123)

        #expect nothing initially
        views_by_tn[db._cur_transaction_num] = db.view()
        counter_vals_by_tn[db._cur_transaction_num] = {}

        #seed the initial state
        with db.transaction():
            for i in range(20):
                counter = db.Counter.New(_identity="C_%s" % i)
                counter.k = int(random.random() * 100)
                counters.append(counter)

            counter_vals_by_tn[db._cur_transaction_num + 1] = {c: c.k for c in counters}

        total_writes = 0

        for passIx in range(1000):
            #keyname = "Counter-val:C_19:k"
            #print "C19: ", db._tail_values.get(keyname), [(tid, db._key_and_version_to_object.get((keyname, tid)))
            #    for tid in sorted(db._key_version_numbers.get(keyname,()))], mem_store.get(keyname)
            
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
                    #print "checking consistency of ", tid
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
                db = Database(mem_store)
                initialize_types(db)

                new_counters = [db.Counter(c._identity) for x in counters]

                views_by_tn = {0: db.view()}
                counter_vals_by_tn = {0: 
                    {new_counters[ix]: max_counter_vals[counters[ix]] for ix in 
                        range(len(counters)) if counters[ix] in max_counter_vals}
                    }

                counters = new_counters

        self.assertLess(mem_store.storedStringCount(), 100)
        self.assertTrue(total_writes > 500)

    def test_flush_db_works(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)

        with db.transaction():
            c = db.Counter.New()
            c.k = 1

        self.assertTrue(mem_store.values)

        view = db.view()

        with db.transaction():
            c.delete()

        #database doesn't have this
        self.assertFalse(mem_store.storedStringCount())

        #but the view does!
        with view:
            self.assertTrue(c.exists())

        self.assertFalse(mem_store.storedStringCount())

    def test_read_write_conflict(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)

        with db.transaction():
            o1 = db.Counter.New()
            o2 = db.Counter.New()

        t1 = db.transaction()
        t2 = db.transaction()

        with t1.nocommit():
            o1.k = o2.k + 1
        
        with t2.nocommit():
            o2.k = o1.k + 1

        t1.commit()

        with self.assertRaises(RevisionConflictException):
            t2.commit()
        
    def test_indices(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)
        
        with db.view() as v:
            self.assertEqual(v.indexLookup(db.Counter,k=20), ())
            self.assertEqual(v.indexLookup(db.Counter,k=30), ())

        with db.transaction():
            o1 = db.Counter.New(k = 20)

        with db.view() as v:
            self.assertEqual(v.indexLookup(db.Counter,k=20), (o1,))
            self.assertEqual(v.indexLookup(db.Counter,k=30), ())

        with db.transaction():
            o1.k = 30

        with db.view() as v:
            self.assertEqual(v.indexLookup(db.Counter,k=20), ())
            self.assertEqual(v.indexLookup(db.Counter,k=30), (o1,))

        with db.transaction():
            o1.delete()

        with db.view() as v:
            self.assertEqual(v.indexLookup(db.Counter,k=20), ())
            self.assertEqual(v.indexLookup(db.Counter,k=30), ())

    def test_indices_multiple_values(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)
        
        with db.transaction() as v:
            k1 = db.Counter.New(k=20)
            k2 = db.Counter.New(k=20)

            self.assertEqual(len(db.Counter.lookupAll(k=20)), 2)

            k1.k = 30

            self.assertEqual(len(db.Counter.lookupAll(k=20)), 1)

            k1.k = 20

            self.assertEqual(len(db.Counter.lookupAll(k=20)), 2)

        with db.transaction() as v:
            self.assertEqual(len(db.Counter.lookupAll(k=20)), 2)

            k1.k = 30

            self.assertEqual(len(db.Counter.lookupAll(k=20)), 1)

            k1.k = 20

            self.assertEqual(len(db.Counter.lookupAll(k=20)), 2)

    def test_indices_across_invocations(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)
        
        with db.transaction():
            o = db.Counter.New(k=1)
            o.x = 10

        db = Database(mem_store)
        initialize_types(db)
    
        with db.transaction() as v:
            o = db.Counter.lookupOne(k=1)
            self.assertEqual(o.x, 10)
            o.k = 2
            o.x = 11

        db = Database(mem_store)
        initialize_types(db)
    
        with db.transaction() as v:
            o = db.Counter.lookupOne(k=2)
            o.k = 3
            self.assertEqual(o.x, 11)
            
        db = Database(mem_store)
        initialize_types(db)
    
        with db.transaction() as v:
            self.assertFalse(db.Counter.lookupAny(k=2))
            
            o = db.Counter.lookupOne(k=3)
            o.k = 3
            self.assertEqual(o.x, 11)
            

    def test_indices_of_algebraics(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        initialize_types(db)
        
        with db.transaction():
            o1 = db.Object.New(k=expr.Constant(value=123))

        with db.view() as v:
            self.assertEqual(v.indexLookup(db.Object,k=expr.Constant(value=123)), (o1,))

    def test_index_functions(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        
        @db.define
        class Object:
            k=Indexed(int)

            @Indexed
            def k2(self) -> int:
                return self.k * 2

            pair_index = Index('k', 'k')

        with db.transaction():
            o1 = db.Object.New(k=10)

        with db.view() as v:
            self.assertEqual(v.indexLookup(db.Object,k=10), (o1,))
            self.assertEqual(v.indexLookup(db.Object,k2=20), (o1,))
            self.assertEqual(v.indexLookup(db.Object,k=20), ())
            self.assertEqual(o1.k2(), o1.k * 2)

            self.assertEqual(v.indexLookup(db.Object,pair_index=(10,10)), (o1,))
            self.assertEqual(v.indexLookup(db.Object,pair_index=(10,11)), ())

            with self.assertRaises(Exception):
                self.assertEqual(v.indexLookup(db.Object,pair_index=(10,"hi")), (o1,))

    def test_index_functions_None_semantics(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        db.Object.define(k=int)
        db.addIndex(db.Object, 'index', lambda o: True if o.k > 10 else None)
        
        with db.transaction() as v:
            self.assertEqual(v.indexLookup(db.Object,index=True), ())
            o1 = db.Object.New(k=10)
            self.assertEqual(v.indexLookup(db.Object,index=True), ())
            o1.k = 20
            self.assertEqual(v.indexLookup(db.Object,index=True), (o1,))
            o1.k = 10
            self.assertEqual(v.indexLookup(db.Object,index=True), ())
            o1.k = 20
            self.assertEqual(v.indexLookup(db.Object,index=True), (o1,))
            o1.delete()
            self.assertEqual(v.indexLookup(db.Object,index=True), ())

    def test_indices_update_during_transactions(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        db.Object.define(k=int)
        db.addIndex(db.Object, 'k')
        
        with db.transaction() as v:
            self.assertEqual(v.indexLookup(db.Object,k=10), ())
            o1 = db.Object.New(k=10)

            self.assertEqual(v.indexLookup(db.Object,k=10), (o1,))
            
            o1.k = 20

            self.assertEqual(v.indexLookup(db.Object,k=10), ())
            self.assertEqual(v.indexLookup(db.Object,k=20), (o1,))

            o1.delete()

            self.assertEqual(v.indexLookup(db.Object,k=10), ())
            self.assertEqual(v.indexLookup(db.Object,k=20), ())

    def test_index_transaction_conflicts(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        db.Object.define(k=int)
        db.Other.define(k=int)
        db.addIndex(db.Object, 'k')
        
        with db.transaction():
            o1 = db.Object.New(k=10)
            o2 = db.Object.New(k=20)
            o3 = db.Object.New(k=30)

        t1 = db.transaction()
        t2 = db.transaction()

        with t1.nocommit():
            o2.k=len(t1.indexLookup(db.Object,k=10))

        with t2.nocommit():
            o1.k = 20

        t2.commit()

        with self.assertRaises(RevisionConflictException):
            t1.commit()

    def test_default_constructor_for_list(self):
        mem_store = InMemoryJsonStore.InMemoryJsonStore()

        db = Database(mem_store)
        db.Object.define(x = TupleOf(int))

        with db.transaction():
            n = db.Object.New()
            self.assertEqual(len(n.x), 0)
