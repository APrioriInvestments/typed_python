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

from object_database.web.cells import Cells, Sequence, Container, Subscribed, Span, SubscribedSequence, ensureSubscribedType
from object_database import InMemServer, Schema, Indexed
from object_database.util import genToken

import unittest
import threading

test_schema = Schema("core.web.test")

@test_schema.define
class Thing:
    k = Indexed(int)
    x = int


class CellsTests(unittest.TestCase):
    def setUp(self):
        self.token = genToken()
        self.server = InMemServer(auth_token=self.token)
        self.server.start()

        self.db = self.server.connect(self.token)
        self.db.subscribeToSchema(test_schema)
        self.cells = Cells(self.db)

    def tearDown(self):
        self.server.stop()

    def test_messages(self):
        pair = [
            Container("HI"),
            Container("HI2")
            ]
        pairCell = Sequence(pair)
        self.cells.root.setChild(pairCell)

        self.cells.recalculate()

        expectedCells = [self.cells.root, pairCell, pair[0], pair[1]]

        self.assertTrue(self.cells.root.identity in self.cells.cells)
        self.assertTrue(pairCell.identity in self.cells.cells)
        self.assertTrue(pair[0].identity in self.cells.cells)
        self.assertTrue(pair[1].identity in self.cells.cells)

        messages = {}
        for m in self.cells.renderMessages():
            assert m['id'] not in messages

            messages[m['id']] = m

        for c in expectedCells:
            self.assertTrue(c.identity in messages)

        self.assertEqual(
            set(messages[pairCell.identity]['replacements'].values()),
            set([pair[0].identity, pair[1].identity])
            )

        self.assertEqual(
            set(messages[self.cells.root.identity]['replacements'].values()),
            set([pairCell.identity])
            )

    def test_recalculation(self):
        pair = [
            Container("HI"),
            Container("HI2")
            ]

        self.cells.root.setChild(
            Sequence(pair)
            )

        self.cells.recalculate()
        self.cells.renderMessages()

        pair[0].setChild("HIHI")

        self.cells.recalculate()

        #a new message for the child, and also for 'pair[0]'
        self.assertEqual(len(self.cells.renderMessages()), 3)

    def test_subscriptions(self):
        self.cells.root.setChild(
            Subscribed(lambda:
                Sequence([
                    Span("Thing(k=%s).x = %s" % (thing.k, thing.x)) for thing in Thing.lookupAll()
                    ])
                )
            )

        self.cells.recalculate()
        self.cells.renderMessages()

        with self.db.transaction():
            Thing(x=1,k=1)
            Thing(x=2,k=2)

        self.cells.recalculate()

        with self.db.transaction():
            Thing(x=3,k=3)

        self.cells.recalculate()

        #three 'Span', three 'Text', the Sequence, the Subscribed, and a delete
        self.assertEqual(len(self.cells.renderMessages()), 9)

    def test_ensure_subscribed(self):
        schema = Schema("core.web.test2")

        @schema.define
        class Thing2:
            k = Indexed(int)
            x = int

        computed = threading.Event()

        def checkThing2s():
            ensureSubscribedType(Thing2)

            res = Sequence([
                    Span("Thing(k=%s).x = %s" % (thing.k, thing.x)) for thing in Thing2.lookupAll()
                    ])

            computed.set()

            return res

        self.cells.root.setChild(Subscribed(checkThing2s))

        self.cells.recalculate()
        self.cells.renderMessages()

        self.assertTrue(computed.wait(timeout=5.0))


    def test_garbage_collection(self):
        #create a cell that subscribes to a specific 'thing', but that
        #creates new cells each time, and verify that we reduce our
        #cell count, and that we send deletion messages

        #subscribes to the set of cells with k=0 and displays something
        self.cells.root.setChild(
            SubscribedSequence(
                lambda: Thing.lookupAll(k=0),
                lambda thing: Subscribed(
                    lambda: Span("Thing(k=%s).x = %s" % (thing.k, thing.x))
                    )
                )
            )

        with self.db.transaction():
            thing = Thing(x=1, k=0)

        for i in range(100):
            with self.db.transaction():
                thing.k = 1
                thing = Thing(x=i,k=0)

                for anything in Thing.lookupAll():
                    anything.x = anything.x + 1

            self.cells.recalculate()
            messages = self.cells.renderMessages()

            self.assertTrue(len(self.cells.cells) < 20, "Have %s cells at pass %s" % (len(self.cells.cells), i))
            self.assertTrue(len(messages) < 20, "Got %s messages at pass %s" % (len(messages), i))
