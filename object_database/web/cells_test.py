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

from object_database.web.cells import Cells, Sequence, Container, Subscribed, Span
from object_database import InMemServer, Schema, Indexed

import unittest


test_schema = Schema("core.web.test")

@test_schema.define
class Thing:
    k = Indexed(int)
    x = int


class CellsTests(unittest.TestCase):
    def setUp(self):
        self.server = InMemServer()
        self.server.start()

        self.db = self.server.connect()
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
        self.assertEqual(len(self.cells.renderMessages()), 2)

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

        #three 'Span', three 'Text', the Sequence, and the Subscribed
        self.assertEqual(len(self.cells.renderMessages()), 8)
            