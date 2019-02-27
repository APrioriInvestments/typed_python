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

from typed_python import OneOf

from object_database.schema import Indexed, Schema
from object_database.inmem_server import InMemServer
from object_database.persistence import InMemoryPersistence
from object_database.util import genToken

import unittest
import numpy

from object_database.util import configureLogging

schema = Schema("test_schema")


@schema.define
class Ring:
    left = Indexed(OneOf(None, schema.Ring))
    right = Indexed(OneOf(None, schema.Ring))
    k = Indexed(int)

    @staticmethod
    def New():
        r = Ring()
        r.left = r.right = r
        return r

    def insert(self, k):
        r = Ring(k=k)

        r.left = self
        r.right = self.right
        self.right = r
        r.right.left = r

        kRight = k // 2
        kLeft = k - kRight

        r.left.k -= kLeft
        r.right.k -= kRight

    def check(self):
        count = 1
        sum = self.k

        r = self.right
        while r != self:
            assert r.left.right == r
            assert r.right.left == r

            count += 1
            sum += r.k
            r = r.right

        return count, sum


class RingInvariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configureLogging('database_ring_invariant_test')

    def setUp(self):
        self.token = genToken()
        self.mem_store = InMemoryPersistence()
        self.server = InMemServer(self.mem_store, self.token)
        self.server.start()

    def createNewDb(self):
        return self.server.connect(self.token)

    def tearDown(self):
        self.server.stop()

    def test_ring_invariants_basic(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)
        with db.transaction():
            # create the empty ring
            r = Ring.New()
            for i in range(10):
                r.insert(i)
                self.assertEqual(r.check(), (i+2, 0))

    def test_ring_invariants_reader_writer(self):
        db = self.createNewDb()
        db.subscribeToSchema(schema)

        with db.transaction():
            # create the empty ring
            Ring.New()

        def writeSome():
            with db.transaction():
                rings = Ring.lookupAll()
                ring = rings[numpy.random.choice(len(rings))]
                ring.insert(numpy.random.choice(10))

        def checkSome(lazy, k=None):
            db2 = self.createNewDb()

            if k is not None:
                assert lazy
                db2.subscribeToType(schema.Ring, lazySubscription=True)
                db2.subscribeToIndex(schema.Ring, k=k)
            else:
                db2.subscribeToSchema(schema)

            with db2.transaction():
                rings = Ring.lookupAll()
                return rings[numpy.random.choice(len(rings))].check()

        for i in range(100):
            writeSome()

            isLazy = (i%2) == 0

            k = None if i % 5 != 3 or not isLazy else numpy.random.choice(10)

            print("Pass ", i, 'isLazy=', isLazy, 'k=', k)

            count, sum = checkSome(isLazy, k)
            self.assertEqual(count, i+2)
            self.assertEqual(sum, 0)
