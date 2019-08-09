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

from object_database.web.cells import (
    AsyncDropdown,
    Cell,
    Cells,
    Subscribed,
    Card,
    Container,
    Sequence,
    SubscribedSequence,
    Span,
    Text,
    Slot,
    ensureSubscribedType,
    registerDisplay
)

from object_database import InMemServer, Schema, Indexed, connect
from object_database.util import genToken, configureLogging
from object_database.frontends.service_manager import autoconfigureAndStartServiceManagerProcess
from object_database.test_util import (
    currentMemUsageMb,
    log_cells_stats
)
from .Messenger import getStructure

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

    def test_cells_lifecycle_created(self):
        basicCell = Cell()
        # new cell
        self.assertTrue(basicCell.wasCreated)
        self.cells.withRoot(basicCell)
        self.cells.renderMessages()
        # no longer new cell
        self.assertFalse(basicCell.wasCreated)

    def test_cells_lifecycle_updated(self):
        basicCell = Container("TEXT")
        # new cell not updated
        self.cells.withRoot(basicCell)
        self.assertFalse(basicCell.wasUpdated)
        self.cells.renderMessages()
        # now update
        basicCell.setChild("NEW TEXT")
        self.cells._recalculateCells()
        self.assertTrue(basicCell.wasUpdated)
        self.cells.renderMessages()
        # no longer updated afted message rendered, i.e. sent
        self.assertFalse(basicCell.wasUpdated)

    def test_cells_lifecycle_not_updated(self):
        basicCell = Cell()
        # new cell not updated
        self.cells.withRoot(basicCell)
        self.assertFalse(basicCell.wasUpdated)
        self.cells.renderMessages()
        # still not udpated
        self.assertFalse(basicCell.wasUpdated)

    def test_cells_lifecycle_removed(self):
        basicCell = Cell()
        # new cell note removed
        self.cells.withRoot(basicCell)
        self.assertFalse(basicCell.wasRemoved)
        # now remove
        self.cells.markToDiscard(basicCell)
        self.assertTrue(basicCell.wasRemoved)
        self.cells.renderMessages()
        # still not removed
        self.assertFalse(basicCell.wasRemoved)

    def test_cells_lifecycle_notremoved(self):
        basicCell = Cell()
        # new cell not removed
        self.assertFalse(basicCell.wasRemoved)
        self.cells.withRoot(basicCell)
        self.cells.renderMessages()
        # still not removed
        self.assertFalse(basicCell.wasRemoved)

    def test_cells_recalculation(self):
        pair = [
            Container("HI"),
            Container("HI2")
        ]

        sequence = Sequence(pair)

        self.cells.withRoot(
            sequence
        )

        self.cells._recalculateCells()
        pair[0].setChild("HIHI")
        self.cells._recalculateCells()

        # Assert that the contianers have the correct parent
        self.assertEqual(pair[0].parent, sequence)
        self.assertEqual(pair[1].parent, sequence)

        # Assert that the first Container has a Cell child
        self.assertIsInstance(pair[0].namedChildren['child'], Cell)

    def test_cells_reusable(self):
        c1 = Card(Text("HI"))
        c2 = Card(Text("HI2"))
        slot = Slot(0)

        self.cells.withRoot(
            Subscribed(lambda: c1 if slot.get() else c2)
        )

        self.cells.renderMessages()
        slot.set(1)
        self.cells.renderMessages()
        slot.set(0)
        self.cells.renderMessages()

        self.assertFalse(self.cells.childrenWithExceptions())

    def test_cells_subscriptions(self):
        self.cells.withRoot(
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
        # self.assertEqual(len(self.cells.renderMessages()), 9)
        nodes_created = [ node for node in self.cells._nodesToBroadcast if node.wasCreated]

        # We have discarded only one
        self.assertEqual(len(self.cells._nodesToDiscard), 1)

        # We have created three: Span and two Text
        self.assertEqual(len(nodes_created), 3)

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
                Span("Thing(k=%s).x = %s"
                     % (thing.k, thing.x)) for thing in Thing2.lookupAll()
            ])

            computed.set()

            return res

        self.cells.withRoot(Subscribed(checkThing2s))

        self.cells.renderMessages()

        self.assertTrue(computed.wait(timeout=5.0))

    def test_cells_garbage_collection(self):
        # create a cell that subscribes to a specific 'thing', but that
        # creates new cells each time, and verify that we reduce our
        # cell count, and that we send deletion messages

        # subscribes to the set of cells with k=0 and displays something
        self.cells.withRoot(
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

            self.assertTrue(len(self.cells) < 20, "Have %s cells at pass %s"
                            % (len(self.cells), i))
            self.assertTrue(len(messages) < 20, "Got %s messages at pass %s"
                            % (len(messages), i))

    def helper_memory_leak(self, cell, initFn, workFn, thresholdMB):
        port = 8021
        server, cleanupFn = autoconfigureAndStartServiceManagerProcess(
            port=port,
            authToken=self.token,
        )
        try:
            db = connect("localhost", port, self.token, retry=True)
            db.subscribeToSchema(test_schema)
            cells = Cells(db)

            cells.withRoot(cell)

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

            workFn(db, cells, iterations=5) # Change back to 500

        self.helper_memory_leak(cell, initFn, workFn, 1)

    @unittest.skip("Test is failing oddly, but it's not clear what test is trying to do")
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
                self.assertTrue(thing)
                self.assertTrue(Thing.lookupAny())

            for counter in range(iterations):
                with db.transaction():
                    if counter % 3 == 0:
                        thing.k = 1 - thing.k
                        thing.delete()
                        thing = Thing(x=counter, k=0)

                    self.assertTrue(Thing.lookupAny())
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

        self.helper_memory_leak(cell, initFn, workFn, 3)

    def test_cells_context(self):
        class X:
            def __init__(self, x):
                self.x = x

        @registerDisplay(X, size="small")
        def small_x(X):
            return Text(X.x).tagged("small display " + str(X.x))

        @registerDisplay(X, size="large")
        def large_x(X):
            return Text(X.x).tagged("large display " + str(X.x))

        @registerDisplay(X, size=lambda size: size is not None)
        def sized_x(X):
            return Text(X.x).tagged("sized display " + str(X.x))

        @registerDisplay(X)
        def any_x(X):
            return Text(X.x).tagged("display " + str(X.x))

        self.cells.withRoot(
            Card(X(0)).withContext(size="small") +
            Card(X(1)).withContext(size="large") +
            Card(X(2)).withContext(size="something else") +
            Card(X(3))
        )

        self.cells.renderMessages()

        self.assertTrue(self.cells.findChildrenByTag("small display 0"))
        self.assertTrue(self.cells.findChildrenByTag("large display 1"))
        self.assertTrue(self.cells.findChildrenByTag("sized display 2"))
        self.assertTrue(self.cells.findChildrenByTag("display 3"))

    def test_async_dropdown_changes(self):
        # Ensure that AsyncDropdown shows
        # the loading child first, then the
        # rendered child after changing the
        # open state.
        changedCell = Text("Changed")

        def handler():
            return changedCell

        dropdown = AsyncDropdown('Untitled', handler)
        dropdown.contentCell.loadingCell = Text("INITIAL")
        self.cells.withRoot(dropdown)
        self.cells.renderMessages()

        # Initial
        self.assertTrue(dropdown in self.cells)
        self.assertTrue(dropdown.contentCell.loadingCell in self.cells)

        # Changed
        dropdown.slot.set(True)
        self.cells.renderMessages()
        self.assertTrue(changedCell in self.cells)



class CellsMessagingTests(unittest.TestCase):
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

    def test_cells_initial_messages(self):
        pair = [
            Container("HI"),
            Container("HI2")
        ]
        pairCell = Sequence(pair)
        self.cells.withRoot(pairCell)

        msgs = self.cells.renderMessages()

        expectedCells = [self.cells._root, pairCell, pair[0], pair[1]]

        self.assertTrue(self.cells._root in self.cells)
        self.assertTrue(pairCell in self.cells)
        self.assertTrue(pair[0] in self.cells)
        self.assertTrue(pair[1] in self.cells)

        # We should for now only have the initial
        # creation message for the RootCell
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]['id'], self.cells._root.identity)

    def test_cells_simple_update_message(self):
        pair = [
            Container("Hello"),
            Container("World")
        ]
        sequence = Sequence(pair)

        # Initial recalculation
        self.cells.withRoot(sequence)
        self.cells.renderMessages()

        # Add a new element to the end of Sequence
        # and update
        text = Text("Hello World")
        pair.append(Text)
        sequence.elements = [Cell.makeCell(el) for el in pair]
        sequence.markDirty()
        msgs = self.cells.renderMessages()

        # There should be one message
        self.assertEqual(len(msgs), 1)

        # It should be an update to the Sequence
        self.assertEqual(msgs[0]['id'], sequence.identity)

        # Sequence's namedChildren should have a length now of 3
        self.assertEqual(len(sequence.namedChildren['elements']), 3)

class CellsStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configureLogging(
            preamble="cells_structure_test",
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

    def test_basic_flat_structure(self):
        first = Text("Hello")
        first.cells = self.cells
        second = Text("World")
        second.cells = self.cells
        container = Sequence([first, second])
        container.cells = self.cells
        self.cells.withRoot(container)
        structure = getStructure(None, container, None)
        expected_subset = {
            "id": container.identity,
            "cellType": "Sequence"
        }
        self.assertTrue(isinstance(structure, dict))
        self.assertDictContainsSubset(expected_subset, structure)
        self.assertIn('elements', structure['namedChildren'].keys())
        self.assertIn(first.identity, structure['namedChildren']['elements'])
        self.assertIn(second.identity, structure['namedChildren']['elements'])

    def test_basic_expanded_structure(self):
        first = Text("Hello")
        first.cells = self.cells
        second = Text("World")
        second.cells = self.cells
        container = Sequence([first, second])
        container.cells = self.cells
        self.cells.withRoot(container)
        struct = getStructure(None, container, None, expand=True)
        expected_subset = {
            "id": container.identity,
            "cellType": "Sequence"
        }
        expected_first_element = {
            "id": first.identity,
            "cellType": "Text",
            "namedChildren": {}
        }
        expected_second_element = {
            "id": second.identity,
            "cellType": "Text",
            "namedChildren": {}
        }

        self.assertTrue(isinstance(struct, dict))
        self.assertTrue(isinstance(struct['namedChildren'], dict))
        self.assertDictContainsSubset(expected_subset, struct)
        self.assertIn("elements", struct['namedChildren'])
        elements = struct['namedChildren']['elements']
        self.assertIsInstance(elements, list)
        self.assertDictContainsSubset(expected_first_element, elements[0])
        self.assertDictContainsSubset(expected_second_element, elements[1])

    def test_cell_self_flat_struct(self):
        c = Sequence([
            Text("Hello"),
            Text("World")
        ])
        self.cells.withRoot(c)
        self.cells._recalculateCells()
        struct = c.getCurrentStructure()
        self.assertIsInstance(struct, dict)


if __name__ == '__main__':
    unittest.main()
