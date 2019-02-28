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

from object_database.web.cells import (
    Cells, Sequence, Container, Subscribed, Span, SubscribedSequence,
    Card, Text, Slot, ensureSubscribedType
)
from object_database import InMemServer, Schema, Indexed, connect
from object_database.util import genToken, configureLogging
from object_database.test_util import (
    currentMemUsageMb,
    autoconfigure_and_start_service_manager,
    log_cells_stats
)

import logging
import unittest
import threading

test_schema = Schema("core.web.test")


@test_schema.define
class Thing:
    k = Indexed(int)
    x = int


class CellsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configureLogging(
            preamble="cells_test",
            level=logging.INFO
        )
        cls._logger = logging.getLogger(__name__)

    def setUp(self):
        self.token = genToken()
        self.server = InMemServer(auth_token=self.token)
        self.server.start()

        self.db = self.server.connect(self.token)
        self.db.subscribeToSchema(test_schema)
        self.cells = Cells(self.db)

    def tearDown(self):
        self.server.stop()

    def test_cells_messages(self):
        pair = [
            Container("HI"),
            Container("HI2")
        ]
        pairCell = Sequence(pair)
        self.cells.root.setChild(pairCell)

        msgs = self.cells.renderMessages()

        expectedCells = [self.cells.root, pairCell, pair[0], pair[1]]

        self.assertTrue(self.cells.root in self.cells)
        self.assertTrue(pairCell in self.cells)
        self.assertTrue(pair[0] in self.cells)
        self.assertTrue(pair[1] in self.cells)

        messages = {}
        for m in msgs:
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

    def test_cells_recalculation(self):
        pair = [
            Container("HI"),
            Container("HI2")
        ]

        self.cells.root.setChild(
            Sequence(pair)
        )

        self.cells.renderMessages()

        pair[0].setChild("HIHI")

        # a new message for the child, and also for 'pair[0]'
        self.assertEqual(len(self.cells.renderMessages()), 3)

    def test_cells_reusable(self):
        c1 = Card(Text("HI"))
        c2 = Card(Text("HI2"))
        slot = Slot(0)

        self.cells.root.setChild(
            Subscribed(lambda: c1 if slot.get() else c2)
        )

        self.cells.renderMessages()
        slot.set(1)
        self.cells.renderMessages()
        slot.set(0)
        self.cells.renderMessages()

        self.assertFalse(self.cells.root.childrenWithExceptions())

    def test_cells_subscriptions(self):
        self.cells.root.setChild(
            Subscribed(
                lambda: Sequence([
                    Span("Thing(k=%s).x = %s" % (thing.k, thing.x))
                    for thing in Thing.lookupAll()
                ])
            )
        )

        self.cells.renderMessages()

        with self.db.transaction():
            Thing(x=1, k=1)
            Thing(x=2, k=2)

        self.cells._recalculateCells()

        with self.db.transaction():
            Thing(x=3, k=3)

        # three 'Span', three 'Text', the Sequence, the Subscribed, and a delete
        self.assertEqual(len(self.cells.renderMessages()), 9)

    def test_cells_ensure_subscribed(self):
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

        self.cells.renderMessages()

        self.assertTrue(computed.wait(timeout=5.0))

    def test_cells_garbage_collection(self):
        # create a cell that subscribes to a specific 'thing', but that
        # creates new cells each time, and verify that we reduce our
        # cell count, and that we send deletion messages

        # subscribes to the set of cells with k=0 and displays something
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
                thing = Thing(x=i, k=0)

                for anything in Thing.lookupAll():
                    anything.x = anything.x + 1

            messages = self.cells.renderMessages()

            self.assertTrue(len(self.cells) < 20, "Have %s cells at pass %s" % (len(self.cells), i))
            self.assertTrue(len(messages) < 20, "Got %s messages at pass %s" % (len(messages), i))

    def helper_memory_leak(self, cell, initFn, workFn, thresholdMB):
        port = 8021
        server, cleanupFn = autoconfigure_and_start_service_manager(
            port=port,
            auth_token=self.token,
        )
        try:
            db = connect("localhost", port, self.token, retry=True)
            db.subscribeToSchema(test_schema)
            cells = Cells(db)

            cells.root.setChild(cell)

            initFn(db, cells)

            rss0 = currentMemUsageMb()
            log_cells_stats(cells, self._logger.info, indentation=4)

            workFn(db, cells)
            log_cells_stats(cells, self._logger.info, indentation=4)

            rss = currentMemUsageMb()
            self._logger.info("Initial Memory Usage: {} MB".format(rss0))
            self._logger.info("Final   Memory Usage: {} MB".format(rss))
            self.assertTrue(
                rss - rss0 < thresholdMB,
                "Memory Usage Increased by {} MB.".format(rss - rss0)
            )
        finally:
            cleanupFn()

    def test_cells_memory_leak1(self):
        cell = Subscribed(
            lambda: Sequence([
                Span("Thing(k=%s).x = %s" % (thing.k, thing.x))
                for thing in Thing.lookupAll(k=0)
            ])
        )

        def workFn(db, cells, iterations=5000):
            with db.view():
                thing = Thing.lookupAny(k=0)

            for counter in range(iterations):
                with db.transaction():
                    thing.delete()
                    thing = Thing(k=0, x=counter)

                cells.renderMessages()

        def initFn(db, cells):
            with db.transaction():
                Thing(k=0, x=0)

            cells.renderMessages()

            workFn(db, cells, iterations=500)

        self.helper_memory_leak(cell, initFn, workFn, 1)

    def test_cells_memory_leak2(self):
        cell = (
            SubscribedSequence(
                lambda: Thing.lookupAll(k=0),
                lambda thing: Subscribed(
                    lambda: Span("Thing(k=%s).x = %s" % (thing.k, thing.x))
                )
            ) +
            SubscribedSequence(
                lambda: Thing.lookupAll(k=1),
                lambda thing: Subscribed(
                    lambda: Span("Thing(k=%s).x = %s" % (thing.k, thing.x))
                )
            )
        )

        def workFn(db, cells, iterations=5000):
            with db.view():
                thing = Thing.lookupAny(k=0)

            for counter in range(iterations):
                with db.transaction():
                    if counter % 3 == 0:
                        thing.k = 1 - thing.k
                        thing.delete()
                        thing = Thing(x=counter, k=0)

                    all_things = Thing.lookupAll()
                    self.assertEqual(len(all_things), 1)
                    for anything in all_things:
                        anything.x = anything.x + 1

                cells.renderMessages()

        def initFn(db, cells):
            with db.transaction():
                Thing(x=1, k=0)

            cells.renderMessages()

            workFn(db, cells, iterations=500)

        self.helper_memory_leak(cell, initFn, workFn, 1)
