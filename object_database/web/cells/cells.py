#   Copyright 2017-2019 Nativepython Authors
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

import json
import queue
import os
import cgi
import time
import traceback
import logging
import gevent
import gevent.fileobject
import threading
import numpy

from inspect import signature

from object_database.view import RevisionConflictException
from object_database.view import current_transaction
from object_database.util import Timer
from object_database.web.html.html_gen import HTMLElement, HTMLTextContent
from typed_python.Codebase import Codebase as TypedPythonCodebase

MAX_TIMEOUT = 1.0
MAX_TRIES = 10

_cur_cell = threading.local()


def registerDisplay(type, **context):
    """Register a display function for any instances of a given type. For instance

    @registerDisplay(MyType, size="small")
    def display(value):
        return cells.Text("For small values")

    @registerDisplay(MyType)
    def display(value):
        return cells.Text("For any other kinds of values")

    Arguments:
        type - the type object to display. Instances of _exactly_ this type
            will match this if we don't have a display for the object already.
        context - a dict from str->value. we'll only use this display if this context
            is exactly matched in the parent cell. We'll check contexts in the
            order in which they were registered.
    """
    def registrar(displayFunc):
        ContextualDisplay._typeToDisplay.setdefault(type, []).append(
            (ContextualDisplay.ContextMatcher(context), displayFunc)
        )

        return displayFunc

    return registrar


def context(contextKey):
    """During cell evaluation, lookup context from our parent cell by name."""
    if not hasattr(_cur_cell, 'cell'):
        raise Exception("Please call 'context' from within a message or cell update function.")

    return _cur_cell.cell.getContext(contextKey)


def quoteForJs(string, quoteType):
    if quoteType == "'":
        return string.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    else:
        return string.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def multiReplace(msg, replacements):
    for k, v in replacements.items():
        assert k[:4] == "____", k

    chunks = msg.split("____")
    outChunks = []
    for chunk in chunks:
        subchunk = chunk.split("__", 1)
        if len(subchunk) == 2:
            lookup = "____" + subchunk[0] + "__"
            if lookup in replacements:
                outChunks.append(replacements.pop(lookup))
                outChunks.append(subchunk[1])
            else:
                outChunks.append("____" + chunk)
        else:
            outChunks.append("____" + chunk)

    assert not replacements, "Didn't use up replacement %s in %s" % (
        replacements.keys(), msg)

    return "".join(outChunks)


def wrapCallback(callback):
    """Make a version of callback that will run on the main cells ui thread when invoked.

    This must be called from within a 'cell' or message update.
    """
    cells = _cur_cell.cell.cells

    def realCallback(*args, **kwargs):
        cells.scheduleCallback(lambda: callback(*args, **kwargs))

    realCallback.__name__ = callback.__name__

    return realCallback


def augmentToBeUnique(listOfItems):
    """Returns a list of [(x,index)] for each 'x' in listOfItems, where index is the number of times
    we've seen 'x' before.
    """
    counts = {}
    output = []
    for x in listOfItems:
        counts[x] = counts.setdefault(x, 0) + 1
        output.append((x, counts[x]-1))

    return output


class GeventPipe:
    """A simple mechanism for triggering the gevent webserver from a thread other than
    the webserver thread. Gevent itself expects everything to happen on greenlets. The
    database connection in the background is not based on gevent, so we cannot use any
    standard gevent-based event or queue objects from the db-trigger thread.
    """

    def __init__(self):
        self.read_fd, self.write_fd = os.pipe()
        self.fileobj = gevent.fileobject.FileObjectPosix(
            self.read_fd, bufsize=2)
        self.netChange = 0

    def wait(self):
        self.fileobj.read(1)
        self.netChange -= 1

    def trigger(self):
        # it's OK that we don't check if the bytes are written because we're just
        # trying to wake up the other side. If the operating system's buffer is full,
        # then that means the other side hasn't been clearing the bytes anyways,
        # and that it will come back around and read our data.
        if self.netChange > 2:
            return

        self.netChange += 1
        os.write(self.write_fd, b"\n")


class Cells:
    def __init__(self, db):
        self.db = db

        self._gEventHasTransactions = GeventPipe()

        self.db.registerOnTransactionHandler(self._onTransaction)

        # map: Cell.identity ->  Cell
        self._cells = {}

        # map: Cell.identity -> set(Cell)
        self._cellsKnownChildren = {}

        # set(Cell)
        self._dirtyNodes = set()

        # set(Cell)
        self._nodesToBroadcast = set()

        # set(Cell)
        self._nodesToDiscard = set()

        self._transactionQueue = queue.Queue()

        # set(Slot) containing slots that have been dirtied but whose
        # values have not been updated yet.
        self._dirtySlots = set()

        # map: db.key -> set(Cell)
        self._subscribedCells = {}

        self._pendingPostscripts = []

        # used by _newID to generate unique identifiers
        self._id = 0

        # a list of pending callbacks that want to run on the main thread
        self._callbacks = queue.Queue()

        self._logger = logging.getLogger(__name__)

        self._root = RootCell()

        self._shouldStopProcessingTasks = threading.Event()

        self._addCell(self._root, parent=None)

    def _processCallbacks(self):
        """Execute any callbacks that have been scheduled to run on the main UI thread."""
        try:
            while True:
                callback = self._callbacks.get(block=False)

                try:
                    callback()
                except Exception:
                    self._logger.error(
                        "Callback %s threw an unexpected exception:\n%s",
                        callback,
                        traceback.format_exc()
                    )
        except queue.Empty:
            return

    def scheduleCallback(self, callback):
        """Schedule a callback that will execute on the main cells thread as soon as possible.

        Code in other threads shouldn't modify cells or slots. Cells that want to trigger
        asynchronous work can do so and then push content back into Slot objects using these callbacks.
        """
        self._callbacks.put(callback)
        self._gEventHasTransactions.trigger()

    def withRoot(self, root_cell, serialization_context=None, session_state=None):
        self._root.setChild(
            root_cell
        )
        self._root.setContext(
            SessionState,
            session_state
            or self._root.context.get(SessionState)
            or SessionState()._reset(self)
        )
        self._root.withSerializationContext(
            serialization_context
            or self._root.serializationContext
            or self.db.serializationContext
        )
        return self

    def __contains__(self, cell_or_id):
        if isinstance(cell_or_id, Cell):
            return cell_or_id.identity in self._cells
        else:
            return cell_or_id in self._cells

    def __len__(self):
        return len(self._cells)

    def __getitem__(self, ix):
        return self._cells.get(ix)

    def _newID(self):
        self._id += 1
        return str(self._id)

    def triggerIfHasDirty(self):
        if self._dirtyNodes:
            self._gEventHasTransactions.trigger()

    def wait(self):
        self._gEventHasTransactions.wait()

    def _onTransaction(self, *trans):
        self._transactionQueue.put(trans)
        self._gEventHasTransactions.trigger()

    def _handleTransaction(self, key_value, set_adds, set_removes, transactionId):
        """ Given the updates coming from a transaction, update self._subscribedCells. """
        for k in list(key_value) + list(set_adds) + list(set_removes):
            if k in self._subscribedCells:

                self._subscribedCells[k] = set(
                    cell for cell in self._subscribedCells[k] if not cell.garbageCollected
                )

                for cell in self._subscribedCells[k]:
                    cell.markDirty()

                if not self._subscribedCells[k]:
                    del self._subscribedCells[k]

    def _addCell(self, cell, parent):
        assert isinstance(cell, Cell), type(cell)
        assert cell.cells is None, cell

        cell.cells = self
        cell.parent = parent
        cell.level = parent.level + 1 if parent else 0

        assert cell.identity not in self._cellsKnownChildren
        self._cellsKnownChildren[cell.identity] = set()

        assert cell.identity not in self._cells
        self._cells[cell.identity] = cell

        self.markDirty(cell)

    def _cellOutOfScope(self, cell):
        for c in cell.children.values():
            self._cellOutOfScope(c)

        self.markToDiscard(cell)

        if cell.cells is not None:
            assert cell.cells == self
            del self._cells[cell.identity]
            del self._cellsKnownChildren[cell.identity]
            for sub in cell.subscriptions:
                self.unsubscribeCell(cell, sub)

        cell.garbageCollected = True

    def subscribeCell(self, cell, subscription):
        self._subscribedCells.setdefault(subscription, set()).add(cell)

    def unsubscribeCell(self, cell, subscription):
        if subscription in self._subscribedCells:
            self._subscribedCells[subscription].discard(cell)
            if not self._subscribedCells[subscription]:
                del self._subscribedCells[subscription]

    def markDirty(self, cell):
        assert not cell.garbageCollected, (cell, cell.text if isinstance(
            cell, Text) else "")

        self._dirtyNodes.add(cell)

    def markToDiscard(self, cell):
        assert not cell.garbageCollected, (cell, cell.text if isinstance(
            cell, Text) else "")

        self._nodesToDiscard.add(cell)

    def markToBroadcast(self, node):
        assert node.cells is self

        self._nodesToBroadcast.add(node)

    def getChildrenJSON(self, cellId, cellObject):
        childrenInfo = []
        for replacementKey, childCell in cellObject.children.items():
            childrenInfo.append({
                'id': childCell.identity,
                'cellType': childCell.__class__.__name__,
                'children': self.getChildrenJSON(childCell.identity, childCell),
                'replacementKey': replacementKey,
                'properties': childCell.exportData,
                'parentId': cellId
            })
        return childrenInfo

    def currentStructureFromCell(self, aCell, replacement_key=None):
        """Responds with a JSON representation of
        the current cell tree with the passed cell as
        the root"""
        result = {
            'id': aCell.identity,
            'cellType': aCell.__class__.__name__,
            'children': self.getChildrenJSON(aCell.identity, aCell),
            'replacementKey': replacement_key,
            'properties': aCell.exportData
        }
        return result

    def renderMessages(self):
        self._processCallbacks()
        self._recalculateCells()

        res = []

        # map<level:int -> cells:set<Cells> >
        cellsByLevel = {}

        for n in self._nodesToBroadcast:
            if n not in self._nodesToDiscard:
                cellsByLevel.setdefault(n.level, set()).add(n)

        for level, cells in reversed(sorted(cellsByLevel.items())):
            for n in cells:
                res.append(self.updateMessageFor(n))

        for n in self._nodesToDiscard:
            if n.cells is not None:
                assert n.cells == self
                res.append({'id': n.identity, 'discard': True})

        # the client reverses the order of postscripts because it wants
        # to do parent nodes before child nodes. We want our postscripts
        # here to happen in order, because they're triggered by messages,
        # so we have to reverse the order in which we append them, and
        # put them on the front.
        res = [{'postscript': js}
               for js in reversed(self._pendingPostscripts)] + res

        self._pendingPostscripts.clear()

        self._nodesToBroadcast = set()
        self._nodesToDiscard = set()

        return res

    def _recalculateCells(self):
        # handle all the transactions so far
        old_queue = self._transactionQueue
        self._transactionQueue = queue.Queue()

        try:
            while True:
                self._handleTransaction(*old_queue.get_nowait())
        except queue.Empty:
            pass

        while self._dirtyNodes:
            n = self._dirtyNodes.pop()

            if not n.garbageCollected:
                self.markToBroadcast(n)

                origChildren = self._cellsKnownChildren[n.identity]

                try:
                    _cur_cell.cell = n
                    _cur_cell.isProcessingMessage = False
                    _cur_cell.isProcessingCell = True
                    while True:
                        try:
                            n.prepare()
                            n.recalculate()
                            break
                        except SubscribeAndRetry as e:
                            e.callback(self.db)
                    # GLORP
                    for childname, child_cell in n.children.items():
                        if not isinstance(child_cell, Cell):
                            raise Exception("Cell of type %s had a non-cell child %s of type %s != Cell." % (
                                type(n),
                                childname,
                                type(child_cell)
                            ))
                        if child_cell.cells:
                            child_cell.prepareForReuse()

                except Exception:
                    self._logger.error(
                        "Node %s had exception during recalculation:\n%s", n, traceback.format_exc())
                    self._logger.error(
                        "Subscribed cell threw an exception:\n%s", traceback.format_exc())
                    n.children = {'____contents__': Traceback(
                        traceback.format_exc())}
                    n.contents = "____contents__"
                finally:
                    _cur_cell.cell = None
                    _cur_cell.isProcessingMessage = False
                    _cur_cell.isProcessingCell = False

                newChildren = set(n.children.values())

                for c in newChildren.difference(origChildren):
                    self._addCell(c, n)

                for c in origChildren.difference(newChildren):
                    self._cellOutOfScope(c)

                self._cellsKnownChildren[n.identity] = newChildren

    def updateMessageFor(self, cell):
        contents = cell.contents
        assert isinstance(
            contents, str), "Cell %s produced %s for its contents which is not a string" % (cell, contents)

        formatArgs = {}

        replaceDict = {}

        for childName, childNode in cell.children.items():
            formatArgs[childName] = "<div id='%s'></div>" % (
                cell.identity + "_" + childName)
            replaceDict[cell.identity + "_" + childName] = childNode.identity

        try:
            contents = multiReplace(contents, formatArgs)
        except Exception:
            raise Exception(
                "Failed to format these contents with args %s:\n\n%s", formatArgs, contents)

        res = {
            'id': cell.identity,
            'contents': contents,
            'replacements': replaceDict,
            'component_name': cell.__class__.__name__,  # TODO: TEMP replace when ready
            'component_contents': cell.contents,  # TODO: TEMP replace when ready
            'replacement_keys': [k for k in cell.children.keys()],
            'extra_data': cell.exportData
        }
        # print()
        # print(str(res))
        # print()

        if cell.postscript:
            res['postscript'] = cell.postscript
        return res

    def childrenWithExceptions(self):
        return self._root.findChildrenMatching(lambda cell: isinstance(cell, Traceback))

    def findChildrenByTag(self, tag, stopSearchingAtFoundTag=True):
        return self._root.findChildrenByTag(
            tag,
            stopSearchingAtFoundTag=stopSearchingAtFoundTag
        )

    def findChildrenMatching(self, filtr):
        return self._root.findChildrenMatching(filtr)


class Slot:
    """Represents a piece of session-specific interface state. Any cells
    that call 'get' will be recalculated if the value changes. UX is allowed
    to change the state (say, because of a button call), thereby causing any
    cells that depend on the Slot to recalculate.

    For the most part, slots are specific to a particular part of a UX tree,
    so they don't have memory. Eventually, it would be good to give them a
    specific name based on where they are in the UX, so that we don't lose
    UX state when we navigate away. We could also keep this in ODB so that
    the state is preserved when we bounce the page.
    """

    def __init__(self, value=None):
        self._value = value
        self._subscribedCells = set()
        self._lock = threading.Lock()

        # mark the cells object present when we were created so we can trigger
        # it when we get updated.
        if hasattr(_cur_cell, 'cell') and _cur_cell.cell is not None:
            self._cells = _cur_cell.cell.cells
        else:
            self._cells = None

    def setter(self, val):
        return lambda: self.set(val)

    def getWithoutRegisteringDependency(self):
        return self._value

    def get(self):
        """Get the value of the Slot, and register a dependency on the calling cell."""
        with self._lock:
            # we can only create a dependency if we're being read
            # as part of a cell's state recalculation.
            if getattr(_cur_cell, 'cell', None) is not None and getattr(_cur_cell, 'isProcessingCell', False):
                self._subscribedCells.add(_cur_cell.cell)

            return self._value

    def set(self, val):
        """Write to a slot.

        If the outside context is a Task, this gets placed on a 'pendingValue' and
        the primary value gets updated between Task cycles. Otherwise, the write
        is synchronous.
        """
        with self._lock:
            if val == self._value:
                return

            self._value = val

            for c in self._subscribedCells:
                logging.debug("Setting value %s triggering cell %s", val, c)
                c.markDirty()

            self._subscribedCells = set()


class SessionState(object):
    """Represents a piece of session-specific interface state. You may access state
    using attributes, which will register a dependency
    """

    def __init__(self):
        self._slots = {}

    def _reset(self, cells):
        self._slots = {
            k: Slot(v.getWithoutRegisteringDependency())
            for k, v in self._slots.items()
        }

        for s in self._slots.values():
            s._cells = cells
            if isinstance(s._value, Cell):
                try:
                    s._value.prepareForReuse()
                except Exception:
                    logging.warn(
                        f"Reusing a Cell slot could create a problem: {s._value}")
        return self

    def _slotFor(self, name):
        if name not in self._slots:
            self._slots[name] = Slot()
        return self._slots[name]

    def __getattr__(self, attr):
        if attr[:1] == "_":
            raise AttributeError(attr)

        return self._slotFor(attr).get()

    def __setattr__(self, attr, val):
        if attr[:1] == "_":
            self.__dict__[attr] = val
            return

        return self._slotFor(attr).set(val)

    def setdefault(self, attr, value):
        if attr not in self._slots:
            self._slots[attr] = Slot(value)

    def set(self, attr, value):
        self.__setattr__(attr, value)

    def toggle(self, attr):
        self.set(attr, not self.__getattr__(attr))

    def get(self, attr):
        return self.__getattr__(attr)


def sessionState():
    return context(SessionState)


class Cell:
    def __init__(self):
        self.cells = None  # will get set when its added to a 'Cells' object
        self.parent = None
        self.level = None
        self.children = {}  # local node def to global node def
        self.namedChildren = {} # The explicitly named version for front-end (refactoring)
        self.contents = ""  # some contents containing a local node def
        self._identity = None
        self._tag = None
        self._nowrap = None
        self._background_color = None
        self._height = None
        self._width = None
        self._overflow = None
        self._color = None
        self.postscript = None
        self.garbageCollected = False
        self.subscriptions = set()
        self._style = {}
        self.serializationContext = TypedPythonCodebase.coreSerializationContext()
        self.context = {}

        self._logger = logging.getLogger(__name__)

        # This is for interim JS refactoring.
        # Cells provide extra data that JS
        # components will need to know about
        # when composing DOM.
        self.exportData = {}

    def evaluateWithDependencies(self, fun):
        """Evaluate function within a view and add dependencies for whatever
        we read."""
        with self.view() as v:
            result = fun()

            self._resetSubscriptionsToViewReads(v)

            return result

    def triggerPostscript(self, javascript):
        """Queue a postscript (piece of javascript to execute) to be run at the end of message processing."""
        self.cells._pendingPostscripts.append(javascript)

    def tagged(self, tag):
        """Give a tag to the cell, which can help us find interesting cells during test."""
        self._tag = tag
        return self

    def findChildrenByTag(self, tag, stopSearchingAtFoundTag=False):
        """Search the cell and its children for all cells with the given tag.

        If `stopSearchingAtFoundTag`, we don't search recursively the children of a
        cell that matched our search.
        """
        cells = []

        if self._tag == tag:
            cells.append(self)
            if stopSearchingAtFoundTag:
                return cells

        for child in self.children:
            cells.extend(
                self.children[child].findChildrenByTag(
                    tag, stopSearchingAtFoundTag)
            )

        return cells

    def visitAllChildren(self, visitor):
        visitor(self)
        for child in self.children.values():
            child.visitAllChildren(visitor)

    def findChildrenMatching(self, filtr):
        res = []

        def visitor(cell):
            if filtr(cell):
                res.append(cell)

        self.visitAllChildren(visitor)

        return res

    def childByIndex(self, ix):
        return self.children[sorted(self.children)[ix]]

    def childrenWithExceptions(self):
        return self.findChildrenMatching(lambda cell: isinstance(cell, Traceback))

    def onMessageWithTransaction(self, *args):
        """Call our inner 'onMessage' function with a transaction and a revision conflict retry loop."""
        tries = 0
        t0 = time.time()
        while True:
            try:
                _cur_cell.cell = self
                _cur_cell.isProcessingMessage = True
                _cur_cell.isProcessingCell = False

                with self.transaction():
                    self.onMessage(*args)
                    return
            except RevisionConflictException:
                tries += 1
                if tries > MAX_TRIES or time.time() - t0 > MAX_TIMEOUT:
                    self._logger.error(
                        "OnMessage timed out. This should really fail.")
                    return
            except Exception:
                self._logger.error(
                    "Exception in dropdown logic:\n%s", traceback.format_exc())
                return
            finally:
                _cur_cell.cell = None
                _cur_cell.isProcessingMessage = False
                _cur_cell.isProcessingCell = False

    def withSerializationContext(self, context):
        self.serializationContext = context
        return self

    def _clearSubscriptions(self):
        if self.cells:
            for sub in self.subscriptions:
                self.cells.unsubscribeCell(self, sub)

        self.subscriptions = set()

    def _resetSubscriptionsToViewReads(self, view):
        new_subscriptions = set(view.getFieldReads()).union(set(view.getIndexReads()))

        for k in new_subscriptions.difference(self.subscriptions):
            self.cells.subscribeCell(self, k)

        for k in self.subscriptions.difference(new_subscriptions):
            self.cells.unsubscribeCell(self, k)

        self.subscriptions = new_subscriptions

    def view(self):
        return self.cells.db.view().setSerializationContext(self.serializationContext)

    def transaction(self):
        return self.cells.db.transaction().setSerializationContext(self.serializationContext)

    def prepare(self):
        if self.serializationContext is TypedPythonCodebase.coreSerializationContext() and self.parent is not None:
            if self.parent.serializationContext is TypedPythonCodebase.coreSerializationContext():
                self.parent.prepare()
            self.serializationContext = self.parent.serializationContext

    def sortsAs(self):
        return None

    def _divStyle(self, existing=None):
        if existing:
            res = [existing]
        else:
            res = []

        if self._nowrap:
            res.append("display:inline-block")

        if self._width is not None:
            if isinstance(self._width, int) or self._width.isdigit():
                res.append("width:%spx" % self._width)
            else:
                res.append("width:%s" % self._width)

        if self._height is not None:
            if isinstance(self._height, int) or self._height.isdigit():
                res.append("height:%spx" % self._height)
            else:
                res.append("height:%s" % self._height)

        if self._color is not None:
            res.append("color:%s" % self._color)

        if self._background_color is not None:
            res.append("background-color:%s" % self._background_color)

        if self._overflow is not None:
            res.append("overflow:%s" % self._overflow)

        if not res:
            return ""
        else:
            return ";".join(res)

    def nowrap(self):
        self._nowrap = True
        return self

    def width(self, width):
        self._width = width
        return self

    def overflow(self, overflow):
        self._overflow = overflow
        return self

    def height(self, height):
        self._height = height
        return self

    def color(self, color):
        self._color = color
        return self

    def background_color(self, color):
        self._background_color = color
        return self

    def isActive(self):
        """Is this cell installed in the tree and active?"""
        return self.cells and not self.garbageCollected

    def prepareForReuse(self):
        if not self.garbageCollected:
            return False

        self.cells = None
        self.postscript = None
        self.garbageCollected = False
        self._identity = None
        self.parent = None

        for c in self.children.values():
            c.prepareForReuse()

        return True

    @property
    def identity(self):
        if self._identity is None:
            assert self.cells is not None, "Can't ask for identity for %s as it's not part of a cells package" % self
            self._identity = self.cells._newID()
        return self._identity

    def markDirty(self):
        if not self.garbageCollected and self.cells is not None:
            self.cells.markDirty(self)

    def recalculate(self):
        pass

    @staticmethod
    def makeCell(x):
        if isinstance(x, (str, float, int, bool)):
            return Text(str(x), x)
        if x is None:
            return Span("")
        if isinstance(x, Cell):
            return x

        return ContextualDisplay(x)

    def __add__(self, other):
        return Sequence([self, Cell.makeCell(other)])

    def withContext(self, **kwargs):
        """Modify our context, and then return self."""
        self.context.update(kwargs)
        return self

    def setContext(self, key, val):
        self.context[key] = val
        return self

    def getContext(self, contextKey):
        if contextKey in self.context:
            return self.context[contextKey]

        if self.parent:
            return self.parent.getContext(contextKey)

        return None


class Card(Cell):
    def __init__(self, body, header=None, padding=None):
        super().__init__()

        self.padding = padding
        self.body = body
        self.header = header

    def recalculate(self):
        bodyCell = Cell.makeCell(self.body)
        self.children = {"____contents__": bodyCell}
        self.namedChildren['body'] = bodyCell

        other = ""
        if self.padding:
            other += " p-" + str(self.padding)

        body = HTMLElement.div().add_class(
            "card-body").add_class(
                other).add_child(
                    HTMLTextContent(" ____contents__"))
        card = HTMLElement.div().add_class("card")

        if self.header is not None:
            header = HTMLElement.div().add_class(
                "card-header").add_child(
                    HTMLTextContent("____header__")
            )
            headerCell = Cell.makeCell(self.header)
            self.children['____header__'] = headerCell
            self.namedChildren['header'] = headerCell
            card.add_child(header)

        card.add_child(body)
        card.attributes["style"] = self._divStyle()

        self.contents = str(card)

        # temporary js WS refactoring data
        self.exportData['divStyle'] = self._divStyle()
        self.exportData['padding'] = self.padding

    def sortsAs(self):
        return self.contents.sortsAs()


class CardTitle(Cell):
    def __init__(self, inner):
        super().__init__()
        innerCell = Cell.makeCell(inner)
        self.children = {"____contents__": innerCell}
        self.namedChildren['inner'] = innerCell
        self.contents = str(
            HTMLElement.div()
            .add_child(HTMLTextContent('____contents__'))
        )

    def sortsAs(self):
        return self.inner.sortsAs()


class Modal(Cell):
    def __init__(self, title, message, **buttonActions):
        """Initialize a modal dialog.

        title - string for the title
        message - string for the message body
        buttonActions - a dict from string to a button action function.
        """
        super().__init__()
        self.title = Cell.makeCell(title).tagged("title")
        self.message = Cell.makeCell(message).tagged("message")
        self.buttons = {
            f"____button_{k}__": Button(k, v).tagged(k)
            for k, v in buttonActions.items()
        }

    def recalculate(self):
        mainStyle = 'display: block; padding-right: 15px;'
        self.contents = str(
            HTMLElement.div()
            .add_classes(['modal', 'fade', 'show'])
            .set_attribute('role', 'dialog')
            .set_attribute('style', mainStyle)
            .add_child(
                HTMLElement.div()
                .add_class('modal-dialog')
                .set_attribute('role', 'document')
                .add_child(
                    HTMLElement.div()
                    .add_class('modal-content')
                    .with_children(
                        HTMLElement.div()
                        .add_class('modal-header')
                        .add_child(
                            HTMLElement.h5()
                            .add_class('modal-title')
                            .add_child(HTMLTextContent('____title__'))
                        ),
                        HTMLElement.div()
                        .add_class('modal-body')
                        .add_child(HTMLTextContent('____message__')),
                        HTMLElement.div()
                        .add_class('modal-footer')
                        .add_child(HTMLTextContent(" ".join(self.buttons)))

                    )
                )
            )
        )
        self.children = dict(self.buttons)
        self.namedChildren['buttons'] = list(self.buttons.values())
        self.children["____title__"] = self.title
        self.namedChildren['title'] = self.title
        self.children["____message__"] = self.message
        self.namedChildren['message'] = self.message


class Octicon(Cell):
    def __init__(self, which):
        super().__init__()
        self.whichOcticon = which

    def sortsAs(self):
        return self.whichOcticon

    def recalculate(self):
        octiconClasses = ['octicon', ('octicon-%s' % self.whichOcticon)]
        self.exportData['octiconClasses'] = octiconClasses
        self.exportData['divStyle'] = self._divStyle()
        self.contents = str(
            HTMLElement.span()
            .add_classes(octiconClasses)
            .set_attribute('aria-hidden', 'true')
            .set_attribute('style', self._divStyle())
        )


class Badge(Cell):
    def __init__(self, inner, style='primary'):
        super().__init__()
        self.inner = self.makeCell(inner)
        self.style = style
        self.exportData['badgeStyle'] = self.style

    def sortsAs(self):
        return self.inner.sortsAs()

    def recalculate(self):
        self.contents = str(
            HTMLElement.span()
            .add_class('badge')
            .add_class('badge-{}'.format(self.style))
            .add_child(HTMLTextContent('____child__'))
        )
        self.children = {'____child__': self.inner}
        self.namedChildren['inner'] = self.inner


class CollapsiblePanel(Cell):
    def __init__(self, panel, content, isExpanded):
        super().__init__()
        self.panel = panel
        self.content = content
        self.isExpanded = isExpanded

    def sortsAs(self):
        return self.content.sortsAs()

    def recalculate(self):
        expanded = self.evaluateWithDependencies(self.isExpanded)
        self.exportData['isExpanded'] = expanded
        self.exportData['divStyle'] = self._divStyle()
        if expanded:
            self.contents = str(
                HTMLElement.div()
                .add_class('container-fluid')
                .set_attribute('style', self._divStyle())
                .add_child(
                    HTMLElement.div()
                    .add_classes(['row', 'flex-nowrap', 'no-gutters'])
                    .with_children(
                        HTMLElement.div()
                        .add_class('col-md-auto')
                        .add_child(HTMLTextContent('____panel__')),
                        HTMLElement.div()
                        .add_class('col-sm')
                        .add_child(HTMLTextContent('____content__'))
                    )
                )
            )
        else:
            self.contents = str(
                HTMLElement.div()
                .add_child(HTMLTextContent('____content__'))
                .set_attribute('style', self._divStyle())
            )

        self.children = {
            '____content__': self.content
        }
        self.namedChildren['content'] = self.content

        if expanded:
            self.children['____panel__'] = self.panel
            self.namedChildren['panel'] = self.panel


class Text(Cell):
    def __init__(self, text, sortAs=None):
        super().__init__()
        self.text = text
        self._sortAs = sortAs if sortAs is not None else text

    def sortsAs(self):
        return self._sortAs

    def recalculate(self):
        escapedText = cgi.escape(str(self.text)) if self.text else " "
        self.exportData['escapedText'] = escapedText
        self.exportData['rawText'] = self.text
        self.exportData['divStyle'] = self._divStyle()
        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', self._divStyle())
            .add_child(HTMLTextContent(escapedText))
        )


class Padding(Cell):
    def __init__(self):
        super().__init__()
        self.contents = str(
            HTMLElement.span()
            .add_class('px-2')
            .add_child(HTMLTextContent('&nbsp'))
        )

    def sortsAs(self):
        return " "


class Span(Cell):
    def __init__(self, text):
        super().__init__()
        self.exportData['text'] = text
        self.contents = str(
            HTMLElement.span()
            .add_child(HTMLTextContent(cgi.escape(str(text))))
        )

    def sortsAs(self):
        return self.contents


class Sequence(Cell):
    def __init__(self, elements):
        super().__init__()
        elements = [Cell.makeCell(x) for x in elements]

        self.elements = elements
        self.namedChildren['elements'] = elements
        self.children = {"____c_%s__" %
                         i: elements[i] for i in range(len(elements))}

    def __add__(self, other):
        other = Cell.makeCell(other)
        if isinstance(other, Sequence):
            return Sequence(self.elements + other.elements)
        else:
            return Sequence(self.elements + [other])

    def recalculate(self):
        sequenceChildrenStr = '\n'.join("____c_%s__" %
                                        i for i in range(len(self.elements)))
        self.namedChildren['elements'] = self.elements
        self.exportData['divStyle'] = self._divStyle()
        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', self._divStyle())
            .add_child(HTMLTextContent(sequenceChildrenStr))
        )

    def sortsAs(self):
        if self.elements:
            return self.elements[0].sortsAs()
        return None


class Columns(Cell):
    def __init__(self, *elements):
        super().__init__()
        elements = [Cell.makeCell(x) for x in elements]
        self.exportData['divStyle'] = self._divStyle()

        self.elements = elements
        self.children = {"____c_%s__" %
                         i: elements[i] for i in range(len(elements))}
        self.namedChildren['elements'] = self.elements

        innerChildren = [
            HTMLElement.div()
            .add_class('col-sm')
            .add_child(HTMLTextContent('____c_%s__' % i))
            for i in range(len(elements))
        ]
        self.contents = str(
            HTMLElement.div()
            .add_class('container-fluid')
            .set_attribute('style', self._divStyle())
            .add_child(
                HTMLElement.div()
                .add_classes(['row', 'flex-nowrap'])
                .add_children(innerChildren)
            )
        )

    def __add__(self, other):
        other = Cell.makeCell(other)
        if isinstance(other, Columns):
            return Columns(*(self.elements + other.elements))
        else:
            return super().__add__(other)

    def sortsAs(self):
        if self.elements:
            return self.elements[0].sortsAs()
        return None


class LargePendingDownloadDisplay(Cell):
    def __init__(self):
        super().__init__()

        self.contents = str(
            HTMLElement.span()
            .set_attribute('id', 'object_database_large_pending_download_text')
        )


class HeaderBar(Cell):
    def __init__(self, leftItems, centerItems=(), rightItems=()):
        super().__init__()
        self.leftItems = leftItems
        self.centerItems = centerItems
        self.rightItems = rightItems

        self.namedChildren = {
            'leftItems': self.leftItems,
            'centerItems': self.centerItems,
            'rightItems': self.rightItems
        }

        self.leftElements = [
            HTMLElement.span()
            .add_classes(['flex-item', 'px-3'])
            .add_child(HTMLTextContent('____left_%s__' % i))
            for i in range(len(self.leftItems))
        ]
        self.centerElements = [
            HTMLElement.span()
            .add_classes(['flex-item', 'px-3'])
            .add_child(HTMLTextContent('____center_%s__' % i))
            for i in range(len(self.centerItems))
        ]
        self.rightElements = [
            HTMLElement.span()
            .add_classes(['flex-item', 'px-3'])
            .add_child(HTMLTextContent('____right_%s__' % i))
            for i in range(len(self.rightItems))
        ]

        self.contents = str(
            HTMLElement.div()
            .add_classes(['p-2', 'bg-light', 'flex-container'])
            .set_attribute('style', 'display:flex;align-items:baseline;')
            .with_children(
                # Left Elements
                HTMLElement.div()
                .add_class('flex-item')
                .set_attribute('style', 'flex-grow:0;')
                .add_child(
                    HTMLElement.div()
                    .add_class('flex-container')
                    .set_attribute('style',
                                   'display:flex;justify-content:center;align-items:baseline;')
                    .add_children(self.leftElements)
                ),
                # Center Elements
                HTMLElement.div()
                .add_class('flex-item')
                .set_attribute('style', 'flex-grow:1;')
                .add_child(
                    HTMLElement.div()
                    .add_class('flex-container')
                    .set_attribute('style',
                                   'display:flex;justify-content:center;align-items:baseline;')
                    .add_children(self.centerElements)
                ),
                # Right Elements
                HTMLElement.div()
                .add_class('flex-item')
                .set_attribute('style', 'flex-grow:0;')
                .add_child(
                    HTMLElement.div()
                    .add_class('flex-container')
                    .set_attribute('style',
                                   'display:flex;justify-content:center;align-items:baseline;')
                    .add_children(self.rightElements)
                )
            )
        )

        self.children = {'____left_%s__' %
                         i: self.leftItems[i] for i in range(len(self.leftItems))}
        self.children.update(
            {'____center_%s__' % i: self.centerItems[i] for i in range(len(self.centerItems))})
        self.children.update(
            {'____right_%s__' % i: self.rightItems[i] for i in range(len(self.rightItems))})


class Main(Cell):
    def __init__(self, child):
        super().__init__()

        self.contents = str(
            HTMLElement.main()
            .add_class('py-md-2')
            .add_child(
                HTMLElement.div()
                .add_class('container-fluid')
                .add_child(HTMLTextContent('____child__'))
            )
        )
        self.children = {'____child__': child}
        self.namedChildren['child'] = child


class _NavTab(Cell):
    def __init__(self, slot, index, target, child):
        super().__init__()

        self.slot = slot
        self.index = index
        self.target = target
        self.child = child

    def recalculate(self):
        inlineScript = """
        cellSocket.sendString(JSON.stringify({'event': 'click', 'ix': __ix__, 'target_cell': '__identity__'}))
        """.replace('__identity__', self.target).replace('__ix__', str(self.index))
        self.exportData['clickData'] = {
            'event': 'click',
            'ix': str(self.index),
            'target_cell': self.target
        }
        navLinkClasses = ['nav-link']
        if self.index == self.slot.get():
            navLinkClasses.append('active')
            self.exportData['isActive'] = True
        else:
            self.exportData['isActive'] = False
        self.contents = str(
            HTMLElement.li()
            .add_class('nav-item')
            .add_child(
                HTMLElement.a()
                .add_classes(navLinkClasses)
                .set_attribute('role', 'tab')
                .set_attribute('onclick', inlineScript)
                .add_child(HTMLTextContent('____child__'))
            )
        )

        childCell = Cell.makeCell(self.child)
        self.children['____child__'] = childCell
        self.namedChildren['child'] = childCell


class Tabs(Cell):
    def __init__(self, headersAndChildren=(), **headersAndChildrenKwargs):
        super().__init__()

        self.whichSlot = Slot(0)
        self.headersAndChildren = list(headersAndChildren)
        self.headersAndChildren.extend(headersAndChildrenKwargs.items())

    def sortsAs(self):
        return None

    def setSlot(self, index):
        self.whichSlot.set(index)

    def recalculate(self):
        displayCell = Subscribed(
            lambda: self.headersAndChildren[self.whichSlot.get()][1])
        self.children['____display__'] = displayCell
        self.namedChildren['display'] = displayCell
        self.namedChildren['headers'] = []

        for i in range(len(self.headersAndChildren)):
            headerCell = _NavTab(
                self.whichSlot, i, self._identity, self.headersAndChildren[i][0])
            self.children['____header_{ix}__'.format(ix=i)] = headerCell
            self.namedChildren['headers'].append(headerCell)

        headerItems = "".join(
            """ ____header___ix____ """.replace('__ix__', str(i))
            for i in range(len(self.headersAndChildren)))
        self.contents = str(
            HTMLElement.div()
            .add_classes(['container-fluid', 'mb-3'])
            .with_children(
                HTMLElement.ul()
                .add_classes(['nav', 'nav-tabs'])
                .set_attribute('role', 'tablist')
                .add_child(HTMLTextContent(headerItems)),

                HTMLElement.div()
                .add_class("tab-content")
                .add_child(
                    HTMLElement.div()
                    .add_classes(['tab-pane', 'fade', 'show', 'active'])
                    .set_attribute('role', 'tabpanel')
                    .add_child(HTMLTextContent('____display__'))
                )
            )
        )

    def onMessage(self, msgFrame):
        self.whichSlot.set(int(msgFrame['ix']))


class Dropdown(Cell):
    def __init__(self, title, headersAndLambdas, singleLambda=None, rightSide=False):
        """
        Initialize a Dropdown menu.

            title - a cell containing the current value.
            headersAndLambdas - a list of pairs containing (cell, callback) for each menu item.

        OR

            title - a cell containing the current value.
            headersAndLambdas - a list of pairs containing cells for each item
            callback - a primary callback to call with the selected cell
        """
        super().__init__()

        if singleLambda is not None:
            def makeCallback(cell):
                def callback():
                    singleLambda(cell)
                return callback

            self.headersAndLambdas = [(header, makeCallback(header))
                                      for header in headersAndLambdas]
        else:
            self.headersAndLambdas = headersAndLambdas

        self.title = Cell.makeCell(title)

    def sortsAs(self):
        return self.title.sortsAs()

    def recalculate(self):
        items = []

        self.children['____title__'] = self.title
        self.namedChildren['title'] = self.title
        self.namedChildren['dropdownItems'] = []

        # Because the items here are not separate cells,
        # we have to perform an extra hack of a dict
        # to get the proper data to the temporary
        # JS Component
        self.exportData['targetIdentity'] = self.identity
        self.exportData['dropdownItemInfo'] = {}

        for i in range(len(self.headersAndLambdas)):
            header, onDropdown = self.headersAndLambdas[i]
            childCell = Cell.makeCell(header)
            self.children["____child_%s__" % i] = childCell
            self.namedChildren['dropdownItems'].append(childCell)
            if not isinstance(onDropdown, str):
                inlineScript = """
                cellSocket.sendString(JSON.stringify({'event':'menu', 'ix': __ix__, 'target_cell': '__identity__'}))
                """.replace('__ix__', str(i)).replace('__identity__', self.identity)
                self.exportData['dropdownItemInfo'][i] = 'callback'
            else:
                inlineScript = quoteForJs("window.location.href = '%s'"
                                          % (quoteForJs(onDropdown, "'")), '"')
                self.exportData['dropdownItemInfo'][i] = onDropdown
            childSubstitute = '____child_%s__' % str(i)
            items.append(
                HTMLElement.a()
                .add_class('dropdown-item')
                .set_attribute('onclick', inlineScript)
                .add_child(HTMLTextContent(childSubstitute))
            )
        self.contents = str(
            HTMLElement.div()
            .add_class('btn-group')
            .with_children(
                HTMLElement.a()
                .add_classes(['btn', 'btn-xs', 'btn-outline-secondary'])
                .add_child(HTMLTextContent('____title__')),

                HTMLElement.button()
                .add_classes(['btn', 'btn-xs',
                              'btn-outline-secondary',
                              'dropdown-toggle',
                              'dropdown-toggle-split'])
                .set_attribute('type', 'button')
                .set_attribute('id', '%s-dropdownMenuButton' % self.identity)
                .set_attribute('data-toggle', 'dropdown'),

                HTMLElement.div()
                .add_class('dropdown-menu')
                .add_children(items)
            )
        )

    def onMessage(self, msgFrame):
        self._logger.info(msgFrame)
        fun = self.headersAndLambdas[msgFrame['ix']][1]
        fun()


class CircleLoader(Cell):
    """A simple circular loading indicator
    """
    def __init__(self):
        super().__init__()

    def recalculate(self):
        self.contents = str(
            HTMLElement.div()
            .add_class('spinner-grow')
            .set_attribute('role', 'status')
        )


class AsyncDropdown(Cell):
    """A Bootstrap-styled Dropdown Cell

    whose dropdown-menu contents can be loaded
    asynchronously each time the dropdown is opened.

    Example
    -------
    The following dropdown will display a
    Text cell that displays "LOADING" for
    a second before switching to a different
    Text cell that says "NOW CONTENT HAS LOADED"::
        def someDisplayMethod():
            def delayAndDisplay():
                time.sleep(1)
                return Text('NOW CONTENT HAS LOADED')

            return Card(
                AsyncDropdown(delayAndDisplay)
            )

    """
    def __init__(self, labelText, contentCellFunc, loadingIndicatorCell=None):
        """
        Parameters
        ----------
        labelText: String
            A label for the dropdown
        contentCellFunc: Function or Lambda
            A lambda or function that will
            return a Cell to display asynchronously.
            Usually some computation that takes time
            is performed first, then the Cell gets
            returned
        loadingIndicatorCell: Cell
            A cell that will be displayed while
            the content of the contentCellFunc is
            loading. Defaults to CircleLoader.
        """
        super().__init__()
        self.slot = Slot(False)
        self.labelText = labelText
        self.exportData['labelText'] = self.labelText
        if not loadingIndicatorCell:
            loadingIndicatorCell = CircleLoader()
        self.contentCell = AsyncDropdownContent(self.slot, contentCellFunc, loadingIndicatorCell)
        self.children = {'____contents__': self.contentCell}
        self.namedChildren['content'] = self.contentCell
        self.namedChildren['loadingIndicator'] = loadingIndicatorCell

    def recalculate(self):
        inlinescript = self.getInlineScript()
        self.contents = str(
            HTMLElement.div()
            .add_class('btn-group')
            .with_children(
                HTMLElement.a()
                .add_classes(['btn', 'btn-xs', 'btn-outline-secondary'])
                .add_child(HTMLTextContent(self.labelText)),

                HTMLElement.button()
                .add_classes(['btn', 'btn-xs',
                              'btn-outline-secondary',
                              'dropdown-toggle',
                              'dropdown-toggle-split'])
                .set_attribute('type', 'button')
                .set_attribute('id', '%s-dropdownMenuButton' % self.identity)
                .set_attribute('data-toggle', 'dropdown')
                .set_attribute('onclick', inlinescript)
                .set_attribute('data-firstclick', 'true'),

                HTMLElement.div()
                .set_attribute('id', '%s-dropdownContentWrapper' % self.identity)
                .add_classes(['dropdown-menu'])
                .add_child(HTMLTextContent('____contents__'))
            )
        )

    def getInlineScript(self):
        """Binds a specific call to the JS
        side `cellHandler` who now has a
        special method for dealing with initial
        dropdown events. We dynamically pass the
        Cell identity as the arg and add this as
        an `onclick` inline method to the
        Dropdown's `button` element
        """
        template = """
        cellHandler.dropdownInitialBindFor('__target__')
        """.replace('__target__', self.identity)
        return template

    def onMessage(self, messageFrame):
        """On `dropdown` events sent to this
        Cell over the socket, we will be told
        whether the dropdown menu is open or not
        """
        if messageFrame['event'] == 'dropdown':
            self.slot.set(not messageFrame['isOpen'])


class AsyncDropdownContent(Cell):
    """A dynamic content cell designed for use

    inside of a parent `AsyncDropdown` Cell.

    Notes
    -----
    This Cell should only be used by `AsyncDropdown`.

    Because of the nature of slots and rendering,
    we needed to decompose the actual Cell that
    is dynamically updated using `Subscribed` into
    a separate unit from `AsyncDropdown`.

    Without this separate decomposition,
    the entire Cell would be replaced on
    the front-end, meaning the drawer would never open
    or close since Dropdowns render closed initially.
    """
    def __init__(self, slot, contentFunc, loadingIndicatorCell):
        """
        Parameters
        ----------
        slot: Slot
            A slot that contains a Boolean value
            that tells this cell whether it is in
            the open or closed state on the front
            end. Changes are used to update the
            loading of dynamic Cells to display
            on open.
        contentFunc: Function or Lambda
            A function or lambda that will return
            a Cell to display. Will be called whenever
            the Dropdown is opened. This gets passed
            from the parent `AsyncDropdown`
        loadingIndicatorCell: Cell
            A Cell that will be displayed while
            the content from contentFunc is loading
        """
        super().__init__()
        self.slot = slot
        self.contentFunc = contentFunc
        self.loadingCell = loadingIndicatorCell
        self.contentCell = Subscribed(self.changeHandler)
        self.children = {
            '____contents__': self.contentCell
        }
        self.namedChildren = {
            'content': self.contentCell,
            'loadingIndicator': self.loadingCell
        }

    def changeHandler(self):
        """If the slot is true, the
        dropdown is open and we call the
        `contentFunc` to get something to
        display. Until then, we show the
        Loading message.
        """
        slotState = self.slot.get()
        if slotState:
            return self.contentFunc()
        else:
            return self.loadingCell

    def recalculate(self):
        elementId = ('dropdownContent-%s' % self.identity)
        self.contents = str(
            HTMLElement.div()
            .set_attribute('id', elementId)
            .add_child(HTMLTextContent('____contents__')))


class Container(Cell):
    def __init__(self, child=None):
        super().__init__()
        if child is None:
            self.contents = ""
            self.children = {}
            self.namedChildren['child'] = None
        else:
            self.contents = str(
                HTMLElement.div()
                .add_child(HTMLTextContent('____child__'))
            )
            childCell = Cell.makeCell(child)
            self.children = {"____child__": childCell}
            self.namedChildren['child'] = childCell

    def setChild(self, child):
        childElement = (
            HTMLElement.div()
            .add_child(HTMLTextContent('____child__'))
        )
        self.setContents(str(childElement),
                         {"____child__": Cell.makeCell(child)})

    def setContents(self, newContents, newChildren):
        self.contents = newContents
        self.children = newChildren
        self.namedChildren['child'] = list(newChildren.values())[0] # Hacky!
        self.markDirty()


class Scrollable(Container):
    def __init__(self, child=None):
        super().__init__(child)
        self.overflow('auto')


class RootCell(Container):
    @property
    def identity(self):
        return "page_root"

    def setChild(self, child):
        childElement = (
            HTMLElement.div()
            .add_child(HTMLTextContent('____c__'))
        )
        self.setContents(str(childElement), {"____c__": child})


class Traceback(Cell):
    # TODO: It seems like the passed-in traceback
    # value might not need to be its own Cell, but
    # rather just some data that is passed to this
    # cell.
    def __init__(self, traceback):
        super().__init__()
        self.contents = str(
            HTMLElement.div()
            .add_classes(['alert', 'alert-primary'])
            .add_child(
                HTMLElement.pre()
                .add_child(HTMLTextContent('____child__'))
            )
        )
        self.traceback = traceback
        tracebackCell = Cell.makeCell(traceback)
        self.children = {"____child__": tracebackCell}
        self.namedChildren['traceback'] = tracebackCell

    def sortsAs(self):
        return self.traceback


class Code(Cell):
    # TODO: It looks like codeContents might not
    # need to be an actual Cell, but instead just
    # some data passed to this Cell.
    def __init__(self, codeContents):
        super().__init__()
        self.contents = str(
            HTMLElement.pre()
            .add_child(
                HTMLElement.code()
                .add_child(HTMLTextContent('____child__'))
            )
        )
        self.codeContents = codeContents
        codeContentsCell = Cell.makeCell(codeContents)
        self.children = {"____child__": codeContentsCell}
        self.namedChildren['code'] = codeContentsCell

    def sortsAs(self):
        return self.codeContents


class ContextualDisplay(Cell):
    """Display an arbitrary python object by checking registered display handlers"""

    # map from type -> [(ContextMatcher, displayFun)]
    _typeToDisplay = {}

    class ContextMatcher:
        """Checks if a cell matches a context dict."""

        def __init__(self, contextDict):
            """Initialize a context matcher."""
            self.contextDict = contextDict

        def matchesCell(self, cell):
            for key, value in self.contextDict.items():
                ctx = cell.getContext(key)
                if callable(value):
                    if not value(ctx):
                        return False
                else:
                    if ctx != value:
                        return False
            return True

    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self.contents = str(
            HTMLElement.div()
            .add_child(HTMLTextContent('____child__'))
        )

    def getChild(self):
        if type(self.obj) in ContextualDisplay._typeToDisplay:
            for context, dispFun in ContextualDisplay._typeToDisplay[type(self.obj)]:
                if context.matchesCell(self):
                    return dispFun(self.obj)

        if hasattr(self.obj, "cellDisplay"):
            return self.obj.cellDisplay()

        return Traceback(f"Invalid object of type {type(self.obj)}")

    def recalculate(self):
        with self.view():
            childCell = self.getChild()
            self.children = {"____child__": childCell}
            self.namedChildren['child'] = childCell


class Subscribed(Cell):
    def __init__(self, f):
        super().__init__()

        self.f = f # What is f? A lambda?

    def prepareForReuse(self):
        if not self.garbageCollected:
            return False
        self._clearSubscriptions()
        return super().prepareForReuse()

    def __repr__(self):
        return "Subscribed(%s)" % self.f

    def sortsAs(self):
        for c in self.children.values():
            return c.sortsAs()

        return Cell.makeCell(self.f()).sortsAs()

    def recalculate(self):
        with self.view() as v:
            self.contents = str(
                HTMLElement.div()
                .set_attribute('style', self._divStyle())
                .add_child(HTMLTextContent('____contents__'))
            )
            try:
                c = Cell.makeCell(self.f())
                if c.cells is not None:
                    c.prepareForReuse()
                self.children = {'____contents__': c}
                self.namedChildren['content'] = c
            except SubscribeAndRetry:
                raise
            except Exception:
                tracebackCell = Traceback(
                    traceback.format_exc())
                self.children = {'____contents__': tracebackCell}
                self.namedChildren['content'] = tracebackCell
                self._logger.error(
                    "Subscribed inner function threw exception:\n%s", traceback.format_exc())

            self._resetSubscriptionsToViewReads(v)

            # temporary js WS refactoring data
            self.exportData['divStyle'] = self._divStyle()


class SubscribedSequence(Cell):
    # TODO: Get a better idea of what is actually happening
    # in this cell. For example, what is with all the existing
    # items funging, and what is actually needed in terms of
    # information to display this correctly in the new
    # JS-component based setup?
    def __init__(self, itemsFun, rendererFun, asColumns=False):
        super().__init__()

        self.itemsFun = itemsFun
        self.rendererFun = rendererFun
        self.existingItems = {}
        self.spine = []
        self.asColumns = asColumns

    def prepareForReuse(self):
        if not self.garbageCollected:
            return False
        self._clearSubscriptions()
        self.existingItems = {}
        self.spine = []
        return super().prepareForReuse()

    def sortsAs(self):
        if '____child_0__' in self.children:
            return self.children['____child_0__'].sortsAs()

    def makeCell(self, item):
        return Subscribed(lambda: Cell.makeCell(self.rendererFun(item)))

    def recalculate(self):
        with self.view() as v:
            try:
                self.spine = augmentToBeUnique(self.itemsFun())
            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error(
                    "Spine calc threw an exception:\n%s", traceback.format_exc())
                self.spine = []

            self._resetSubscriptionsToViewReads(v)

            new_children = {}
            new_named_children = {
                'children': []
            }
            for ix, rowKey in enumerate(self.spine):
                if rowKey in self.existingItems:
                    new_children["____child_%s__" %
                                 ix] = self.existingItems[rowKey]
                    new_named_children['children'].append(self.existingItems[rowKey])
                else:
                    try:
                        childCell = self.makeCell(rowKey[0])
                        self.existingItems[rowKey] = new_children["____child_%s__" % ix] = childCell
                        new_named_children[ix] = childCell
                    except SubscribeAndRetry:
                        raise
                    except Exception:
                        tracebackCell = Traceback(traceback.format_exc())
                        self.existingItems[rowKey] = new_children["____child_%s__" % ix] = tracebackCell
                        new_named_children[ix] = tracebackCell

        self.children = new_children
        self.namedChildren = new_named_children

        spineAsSet = set(self.spine)
        for i in list(self.existingItems):
            if i not in spineAsSet:
                del self.existingItems[i]

        if self.asColumns:
            spineChildren = []
            for i in range(len(self.spine)):
                spineChildren.append(
                    HTMLElement.div()
                    .add_class('col-sm')
                    .add_child(HTMLTextContent('____child_%s__' % str(i)))
                )
            self.contents = str(
                HTMLElement.div()
                .add_class('container-fluid')
                .set_attribute('style', self._divStyle())
                .add_child(
                    HTMLElement.div()
                    .add_classes(['row', 'flex-nowrap'])
                    .add_children(spineChildren)
                )
            )
        else:
            spineChildren = []
            for i in range(len(self.spine)):
                spineChildren.append('____child_%s__' % i)
            self.contents = str(
                HTMLElement.div()
                .set_attribute('style', self._divStyle())
                .add_child(HTMLTextContent("\n".join(spineChildren)))
            )

        # temporary js WS refactoring data
        self.exportData['divStyle'] = self._divStyle()
        self.exportData['asColumns'] = self.asColumns
        self.exportData['numSpineChildren'] = len(self.spine)


class Popover(Cell):
    # TODO: Does title actually need to be a cell here? What about detail?
    # What is the purpose of the sortAs method here and why are we using
    # it on the title cell?
    def __init__(self, contents, title, detail, width=400):
        super().__init__()

        self.width = width
        contentCell = Cell.makeCell(contents)
        detailCell = Cell.makeCell(detail)
        titleCell = Cell.makeCell(title)
        self.children = {
            '____contents__': contentCell,
            '____detail__': detailCell,
            '____title__': titleCell
        }
        self.namedChildren = {
            'content': contentCell,
            'detail': detailCell,
            'title': titleCell
        }

    def recalculate(self):
        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', self._divStyle())
            .with_children(
                HTMLElement.a()
                .set_attribute('href', '#popmain_%s' % self.identity)
                .set_attribute('data-toggle', 'popover')
                .set_attribute('data-trigger', 'focus')
                .set_attribute('data-bind', '#pop_%s' % self.identity)
                .set_attribute('data-placement', 'bottom')
                .set_attribute('role', 'button')
                .add_classes(['btn', 'btn-xs'])
                .add_child(HTMLTextContent('____contents__')),

                HTMLElement.div()
                .set_attribute('style', 'display:none;')
                .add_child(
                    HTMLElement.div()
                    .set_attribute('id', 'pop_%s' % self.identity)
                    .with_children(

                        HTMLElement.div()
                        .add_class('data-title')
                        .add_child(HTMLTextContent('____title__')),

                        HTMLElement.div()
                        .add_class('data-content')
                        .add_child(
                            HTMLElement.div()
                            .set_attribute('style', 'width: %spx' % str(self.width))
                            .add_child(HTMLTextContent('____detail__'))
                        )
                    )
                )
            )
        )

        # temporary js WS refactoring data
        self.exportData['divStyle'] = self._divStyle()
        self.exportData['width'] = self.width

    def sortsAs(self):
        if '____title__' in self.children:
            return self.children['____title__'].sortsAs()


class Grid(Cell):
    # TODO: Do the individual data cells (in grid terms) need to be actual Cell objects?
    # Is there a way to let the Components on the front end handle the updating of the
    # data that gets presented, without having to wrap each datum in a Cell object?
    def __init__(self, colFun, rowFun, headerFun, rowLabelFun, rendererFun):
        super().__init__()
        self.colFun = colFun
        self.rowFun = rowFun
        self.headerFun = headerFun
        self.rowLabelFun = rowLabelFun
        self.rendererFun = rendererFun

        self.existingItems = {}
        self.rows = []
        self.cols = []

    def prepareForReuse(self):
        if not self.garbageCollected:
            return False
        self._clearSubscriptions()
        self.existingItems = {}
        self.rows = []
        self.cols = []
        super().prepareForReuse()

    def recalculate(self):
        with self.view() as v:
            try:
                self.rows = augmentToBeUnique(self.rowFun())
            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error(
                    "Row fun calc threw an exception:\n%s", traceback.format_exc())
                self.rows = []
            try:
                self.cols = augmentToBeUnique(self.colFun())
            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error(
                    "Col fun calc threw an exception:\n%s", traceback.format_exc())
                self.cols = []

            self._resetSubscriptionsToViewReads(v)

        new_children = {}
        new_named_children = {
            'headers': [],
            'rowLabels': [],
            'dataCells': []
        }
        seen = set()

        for col_ix, col in enumerate(self.cols):
            seen.add((None, col))
            if (None, col) in self.existingItems:
                new_children["____header_%s__" %
                             (col_ix)] = self.existingItems[(None, col)]
                new_named_children['headers'].append(self.existingItems[(None, col)])
            else:
                try:
                    headerCell = Cell.makeCell(self.headerFun(col[0]))
                    self.existingItems[(None, col)] = \
                        new_children["____header_%s__" % col_ix] = \
                        headerCell
                    new_named_children['headers'].append(headerCell)
                except SubscribeAndRetry:
                    raise
                except Exception:
                    tracebackCell = Traceback(traceback.format_exc)()
                    self.existingItems[(None, col)] = \
                        new_children["____header_%s__" % col_ix] = \
                        tracebackCell
                    new_named_children['headers'].append(tracebackCell)

        if self.rowLabelFun is not None:
            for row_ix, row in enumerate(self.rows):
                seen.add((None, row))
                if (row, None) in self.existingItems:
                    rowLabelCell = self.existingItems[(row, None)]
                    new_children["____rowlabel_%s__" %
                                 (row_ix)] = rowLabelCell
                    new_named_children['rowLabels'].append(rowLabelCell)
                else:
                    try:
                        rowLabelCell = Cell.makeCell(self.rowLabelFun(row[0]))
                        self.existingItems[(row, None)] = \
                            new_children["____rowlabel_%s__" % row_ix] = \
                            rowLabelCell
                        new_named_children['rowLabels'].append(rowLabelCell)
                    except SubscribeAndRetry:
                        raise
                    except Exception:
                        tracebackCell = Traceback(traceback.format_exc())
                        self.existingItems[(row, None)] = \
                            new_children["____rowlabel_%s__" % row_ix] = \
                            tracebackCell
                        new_named_children['rowLabels'].append(tracebackCell)

        seen = set()
        for row_ix, row in enumerate(self.rows):
            new_named_children_column = []
            new_named_children['dataCells'].append(new_named_children_column)
            for col_ix, col in enumerate(self.cols):
                seen.add((row, col))
                if (row, col) in self.existingItems:
                    new_children["____child_%s_%s__" % (row_ix, col_ix)] = \
                        self.existingItems[(row, col)]
                    new_named_children_column.append(self.existingItems[(row, col)])
                else:
                    try:
                        dataCell = Cell.makeCell(self.rendererFun(row[0], col[0]))
                        self.existingItems[(row, col)] = \
                            new_children["____child_%s_%s__" % (row_ix, col_ix)] = \
                            dataCell
                        new_named_children_column.append(dataCell)
                    except SubscribeAndRetry:
                        raise
                    except Exception:
                        tracebackCell = Traceback(traceback.format_exc())
                        self.existingItems[(row, col)] = \
                            new_children["____child_%s_%s__" % (row_ix, col_ix)] = \
                            tracebackCell
                        new_named_children_column.append(tracebackCell)

        self.children = new_children
        self.namedChildren = new_named_children

        for i in list(self.existingItems):
            if i not in seen:
                del self.existingItems[i]

        tableHeaders = []
        for colIndex in range(len(self.cols)):
            tableHeaders.append(
                HTMLElement.th()
                .add_child(HTMLTextContent('____header_%s__' % colIndex))
            )
        tableRows = []
        for rowIndex in range(len(self.rows)):
            rowLabel = HTMLTextContent("")
            if self.rowLabelFun:
                rowLabel = (
                    HTMLElement.td()
                    .add_child(HTMLTextContent('____rowlabel_%s__' % rowIndex))
                )
            tableColumns = []
            for colIndex in range(len(self.cols)):
                tableColumns.append(
                    HTMLElement.td()
                    .add_child(HTMLTextContent('____child_%s_%s__' % (rowIndex, colIndex)))
                )
            tableRows.append(
                HTMLElement.tr()
                .add_child(rowLabel)
                .add_children(tableColumns)
            )
        topTableHeader = HTMLTextContent("")
        if self.rowLabelFun is not None:
            topTableHeader = HTMLElement.th()
        self.contents = str(
            HTMLElement.table()
            .add_classes(['table-hscroll', 'table-sm', 'table-striped'])
            .with_children(
                HTMLElement.thead()
                .add_child(
                    HTMLElement.tr()
                    .add_child(topTableHeader)
                    .add_children(tableHeaders)
                ),
                HTMLElement.tbody()
                .add_children(tableRows)
            )
        )

        self.exportData['rowNum'] = len(self.rows)
        self.exportData['colNum'] = len(self.cols)
        self.exportData['hasTopHeader'] = (self.rowLabelFun is not None)


class SortWrapper:
    def __init__(self, x):
        self.x = x

    def __lt__(self, other):
        try:
            if type(self.x) is type(other.x):  # noqa: E721
                return self.x < other.x
            else:
                return str(type(self.x)) < str(type(other.x))
        except Exception:
            try:
                return str(self.x) < str(self.other)
            except Exception:
                return False

    def __eq__(self, other):
        try:
            if type(self.x) is type(other.x):  # noqa: E721
                return self.x == other.x
            else:
                return str(type(self.x)) == str(type(other.x))
        except Exception:
            try:
                return str(self.x) == str(self.other)
            except Exception:
                return True


class SingleLineTextBox(Cell):
    def __init__(self, slot, pattern=None):
        super().__init__()
        self.children = {}
        self.pattern = None
        self.slot = slot

    def recalculate(self):
        inlineScript = """
        cellSocket.sendString(
            JSON.stringify({
                'event':'click',
                'target_cell': '__identity__',
                'text': this.value
            })
        )
        """
        inlineScript = inlineScript.replace('__identity__', self.identity)
        inputValue = quoteForJs(self.slot.get(), '"')
        element = (
            HTMLElement.input()
            .set_attribute('type', 'text')
            .set_attribute('id', 'text_%s' % self.identity)
            .set_attribute('onchange', inlineScript)
            .set_attribute('value', inputValue)
        )
        if self.pattern:
            element.set_attribute('pattern', self.pattern)
        self.contents = str(element)

        # temporary js WS refactoring data
        if self.pattern:
            self.exportData['pattern'] = self.pattern

    def onMessage(self, msgFrame):
        self.slot.set(msgFrame['text'])


class Table(Cell):
    """An active table with paging, filtering, sortable columns."""

    def __init__(self, colFun, rowFun, headerFun, rendererFun, maxRowsPerPage=20):
        super().__init__()
        self.colFun = colFun
        self.rowFun = rowFun
        self.headerFun = headerFun
        self.rendererFun = rendererFun

        self.existingItems = {}
        self.rows = []
        self.cols = []

        self.maxRowsPerPage = maxRowsPerPage

        self.curPage = Slot("1")
        self.sortColumn = Slot(None)
        self.sortColumnAscending = Slot(True)
        self.columnFilters = {}

    def prepareForReuse(self):
        if not self.garbageCollected:
            return False
        self._clearSubscriptions()
        self.existingItems = {}
        self.rows = []
        self.cols = []
        super().prepareForReuse()

    def cachedRenderFun(self, row, col):
        if (row, col) in self.existingItems:
            return self.existingItems[row, col]
        else:
            return self.rendererFun(row, col)

    def filterRows(self, rows):
        for col in self.cols:
            if col not in self.columnFilters:
                self.columnFilters[col] = Slot(None)

            filterString = self.columnFilters.get(col).get()

            if filterString:
                new_rows = []
                for row in rows:
                    filterAs = self.cachedRenderFun(row, col).sortsAs()

                    if filterAs is None:
                        filterAs = ""
                    else:
                        filterAs = str(filterAs)

                    if filterString in filterAs:
                        new_rows.append(row)
                rows = new_rows

        return rows

    def sortRows(self, rows):
        sc = self.sortColumn.get()

        if sc is not None and sc < len(self.cols):
            col = self.cols[sc]

            keymemo = {}

            def key(row):
                if row not in keymemo:
                    try:
                        r = self.cachedRenderFun(row, col)
                        keymemo[row] = SortWrapper(r.sortsAs())
                    except Exception:
                        self._logger.error(traceback.format_exc())
                        keymemo[row] = SortWrapper(None)

                return keymemo[row]

            rows = sorted(rows, key=key)

            if not self.sortColumnAscending.get():
                rows = list(reversed(rows))

        page = 0
        try:
            page = max(0, int(self.curPage.get())-1)
            page = min(page, (len(rows) - 1) // self.maxRowsPerPage)
        except Exception:
            self._logger.error(
                "Failed to parse current page: %s", traceback.format_exc())

        return rows[page * self.maxRowsPerPage:(page+1) * self.maxRowsPerPage]

    def makeHeaderCell(self, col_ix):
        col = self.cols[col_ix]

        if col not in self.columnFilters:
            self.columnFilters[col] = Slot(None)

        def icon():
            if self.sortColumn.get() != col_ix:
                return ""
            return Octicon("arrow-up" if not self.sortColumnAscending.get() else "arrow-down")

        cell = Cell.makeCell(self.headerFun(col)).nowrap() + \
            Padding() + Subscribed(icon).nowrap()

        def onClick():
            if self.sortColumn.get() == col_ix:
                self.sortColumnAscending.set(
                    not self.sortColumnAscending.get())
            else:
                self.sortColumn.set(col_ix)
                self.sortColumnAscending.set(False)

        res = Clickable(cell, onClick, makeBold=True)

        if self.columnFilters[col].get() is None:
            res = res.nowrap() + Clickable(Octicon("search"),
                                           lambda: self.columnFilters[col].set("")).nowrap()
        else:
            res = res + SingleLineTextBox(self.columnFilters[col]).nowrap() + \
                Button(Octicon("x"), lambda: self.columnFilters[col].set(
                    None), small=True)

        return Card(res, padding=1)

    def recalculate(self):
        with self.view() as v:
            try:
                self.cols = list(self.colFun())
            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error(
                    "Col fun calc threw an exception:\n%s", traceback.format_exc())
                self.cols = []

            try:
                self.unfilteredRows = list(self.rowFun())
                self.filteredRows = self.filterRows(self.unfilteredRows)
                self.rows = self.sortRows(self.filteredRows)

            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error(
                    "Row fun calc threw an exception:\n%s", traceback.format_exc())
                self.rows = []

            self._resetSubscriptionsToViewReads(v)

        new_children = {}
        new_named_children = {
            'headers': [],
            'dataCells': [],
            'page': None,
            'right': None,
            'left': None
        }
        seen = set()

        for col_ix, col in enumerate(self.cols):
            seen.add((None, col))
            if (None, col) in self.existingItems:
                new_children["____header_%s__" %
                             (col_ix)] = self.existingItems[(None, col)]
                new_named_children['headers'].append(self.existingItems[(None, col)])
            else:
                try:
                    headerCell = self.makeHeaderCell(col_ix)
                    self.existingItems[(None, col)] = \
                        new_children["____header_%s__" % col_ix] = \
                        headerCell
                    new_named_children['headers'].append(headerCell)
                except SubscribeAndRetry:
                    raise
                except Exception:
                    tracebackCell = Traceback(traceback.format_exc())
                    self.existingItems[(None, col)] = \
                        new_children["____header_%s__" % col_ix] = \
                        tracebackCell
                    new_named_children['headers'].append(tracebackCell)

        seen = set()
        for row_ix, row in enumerate(self.rows):
            new_named_children_columns = []
            new_named_children['dataCells'].append(new_named_children_columns)
            for col_ix, col in enumerate(self.cols):
                seen.add((row, col))
                if (row, col) in self.existingItems:
                    new_children["____child_%s_%s__" % (row_ix, col_ix)] = \
                        self.existingItems[(row, col)]
                    new_named_children_columns.append(self.existingItems[(row, col)])
                else:
                    try:
                        dataCell = Cell.makeCell(self.rendererFun(row, col))
                        self.existingItems[(row, col)] = \
                            new_children["____child_%s_%s__" % (row_ix, col_ix)] = \
                            dataCell
                        new_named_children_columns.append(dataCell)
                    except SubscribeAndRetry:
                        raise
                    except Exception:
                        tracebackCell = Traceback(traceback.format_exc())
                        self.existingItems[(row, col)] = \
                            new_children["____child_%s_%s__" % (row_ix, col_ix)] = \
                            tracebackCell
                        new_named_children_columns.append(tracebackCell)

        self.children = new_children
        self.namedChildren = new_named_children

        for i in list(self.existingItems):
            if i not in seen:
                del self.existingItems[i]

        totalPages = ((len(self.filteredRows) - 1) // self.maxRowsPerPage + 1)

        rowDisplay = "____left__ ____right__ Page ____page__ of " + \
            str(totalPages)
        if totalPages <= 1:
            pageCell = Cell.makeCell(totalPages).nowrap()
            self.children['____page__'] = pageCell
            self.namedChildren['page'] = pageCell
        else:
            pageCell = (
                SingleLineTextBox(self.curPage, pattern="[0-9]+")
                .width(10 * len(str(totalPages)) + 6)
                .height(20)
                .nowrap()
            )
            self.children['____page__'] = pageCell
            self.namedChildren['page'] = pageCell
        if self.curPage.get() == "1":
            leftCell = Octicon(
                "triangle-left").nowrap().color("lightgray")
            self.children['____left__'] = leftCell
            self.namedChildren['left'] = leftCell
        else:
            leftCell = (
                Clickable(
                    Octicon("triangle-left"),
                    lambda: self.curPage.set(str(int(self.curPage.get())-1))
                ).nowrap()
            )
            self.children['____left__'] = leftCell
            self.namedChildren['left'] = leftCell
        if self.curPage.get() == str(totalPages):
            rightCell = Octicon(
                "triangle-right").nowrap().color("lightgray")
            self.children['____right__'] = rightCell
            self.namedChildren['right'] = rightCell
        else:
            rightCell = (
                Clickable(
                    Octicon("triangle-right"),
                    lambda: self.curPage.set(str(int(self.curPage.get())+1))
                ).nowrap()
            )
            self.children['____right__'] = rightCell
            self.namedChildren['right'] = rightCell
        # NOTE:
        # It is unclear where the following children are
        # rendered here:
        # ____right__
        # ____left__
        # ____page__

        headerElements = []
        for colIndex in range(len(self.cols)):
            headerElements.append(
                HTMLElement.th()
                .set_attribute('style', 'vertical-align:top;')
                .add_child(HTMLTextContent('____header_%s__' % str(colIndex)))
            )

        rowElements = []
        for rowIndex in range(len(self.rows)):
            colElements = []
            colElements.append(
                HTMLElement.td()
                .add_child(HTMLTextContent(str(rowIndex+1)))
            )
            for colIndex in range(len(self.cols)):
                colElements.append(
                    HTMLElement.td()
                    .add_child(HTMLTextContent('____child_%s_%s__' % (rowIndex, colIndex)))
                )
            rowElements.append(
                HTMLElement.tr()
                .add_children(colElements)
            )

        firstRowElement = (
            HTMLElement.tr()
            .add_child(
                HTMLElement.th()
                .set_attribute('style', 'vertical-align:top;')
                .add_child(
                    HTMLElement.div()
                    .add_class('card')
                    .add_child(
                        HTMLElement.div()
                        .add_classes(['card-body', 'p-1'])
                        .add_child(HTMLTextContent(rowDisplay))
                    )
                )
            )
        )
        firstRowElement.add_children(headerElements)

        self.contents = str(
            HTMLElement.table()
            .add_classes(['table-hscroll', 'table-sm', 'table-striped'])
            .with_children(
                HTMLElement.thead()
                .set_attribute('style', 'border-bottom: black;border-bottom-style:solid;border-bottom-width:thin;')
                .add_child(firstRowElement),

                HTMLElement.tbody()
                .add_children(rowElements)
            )
        )

        # print("THIS IS INSANE TABLE DATA")
        # print("first row")
        # print(str(firstRowElement))
        # print("rows")
        # [print(r) for r in rowElements]
        # print()

        # temporary js WS refactoring data
        self.exportData['totalPages'] = totalPages
        self.exportData['numColumns'] = len(self.cols)
        self.exportData['numRows'] = len(self.rows)


class Clickable(Cell):
    def __init__(self, content, f, makeBold=False, makeUnderling=False):
        super().__init__()
        self.f = f # What is this?
        self.content = Cell.makeCell(content)
        self.bold = makeBold

    def calculatedOnClick(self):
        if isinstance(self.f, str):
            return quoteForJs("window.location.href = '__url__'".replace("__url__", quoteForJs(self.f, "'")), '"')
        else:
            return (
                "cellSocket.sendString(JSON.stringify({'event':'click', 'target_cell': '__identity__'}))"
                .replace("__identity__", self.identity)
            )

    def recalculate(self):
        self.children = {'____contents__': self.content}
        self.namedChildren = {'content': self.content}
        style = self._divStyle("cursor:pointer;*cursor: hand" +
                               (";font-weight:bold" if self.bold else ""))

        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', style)
            .set_attribute('onclick', self.calculatedOnClick())
            .add_child(HTMLTextContent('____contents__'))
        )

        # temporary js WS refactoring data
        self.exportData['divStyle'] = style
        # TODO: this event handling situation must be refactored
        self.exportData['events'] = {"onclick": self.calculatedOnClick()}

    def sortsAs(self):
        return self.content.sortsAs()

    def onMessage(self, msgFrame):
        val = self.f()
        if isinstance(val, str):
            self.triggerPostscript(quoteForJs("window.location.href = '__url__'".replace(
                "__url__", quoteForJs(val, "'")), '"'))


class Button(Clickable):
    def __init__(self, *args, small=False, active=True, style="primary", **kwargs):
        Clickable.__init__(self, *args, **kwargs)
        self.small = small
        self.active = active
        self.style = style

    def recalculate(self):
        self.children = {'____contents__': self.content}
        self.namedChildren = {'content': self.content}

        classList = ['btn']
        buttonStateClass = f"""btn{'-outline' if not self.active else ''}-{self.style}"""
        classList.append(buttonStateClass)
        buttonSize = ("" if not self.small else "btn-xs")
        classList.append(buttonSize)
        self.contents = str(
            HTMLElement.button()
            .add_classes(classList)
            .set_attribute('onclick', self.calculatedOnClick())
            .add_child(HTMLTextContent('____contents__'))
        )

        # temporary js WS refactoring data
        self.exportData['classes'] = classList
        # TODO: this event handling situation must be refactored
        self.exportData['events'] = {"onclick": self.calculatedOnClick()}


class ButtonGroup(Cell):
    def __init__(self, buttons):
        super().__init__()
        self.buttons = buttons

    def recalculate(self):
        self.children = {
            f'____button_{i}__': self.buttons[i] for i in range(len(self.buttons))}
        self.namedChildren['buttons'] = self.buttons
        innerButtonsText = " ".join(f"____button_{i}__" for i in range(len(self.buttons)))
        self.contents = str(
            HTMLElement.div()
            .add_class('btn-group')
            .set_attribute('role', 'group')
            .add_child(HTMLTextContent(innerButtonsText))
        )

        # temporary js WS refactoring data
        self.exportData['innerButtonsText'] = innerButtonsText


class LoadContentsFromUrl(Cell):
    def __init__(self, targetUrl):
        Cell.__init__(self)
        self.targetUrl = targetUrl

    def recalculate(self):
        self.children = {}

        self.contents = str(
            HTMLElement.div()
            .add_child(
                HTMLElement.div()
                .set_attribute('id', 'loadtarget%s' % self._identity)
            )
        )

        # temporary js WS refactoring data
        self.exportData['loadTargetId'] = 'loadtarget%s' % self._identity

        self.postscript = (
            "$('#loadtarget__identity__').load('__url__')"
            .replace("__identity__", self._identity)
            .replace("__url__", quoteForJs(self.targetUrl, "'"))
        )


class SubscribeAndRetry(Exception):
    def __init__(self, callback):
        super().__init__("SubscribeAndRetry")
        self.callback = callback


def ensureSubscribedType(t, lazy=False):
    if not current_transaction().db().isSubscribedToType(t):
        raise SubscribeAndRetry(
            Timer("Subscribing to type %s%s", t, " lazily" if lazy else "")(
                lambda db: db.subscribeToType(t, lazySubscription=lazy)
            )
        )


def ensureSubscribedSchema(t, lazy=False):
    if not current_transaction().db().isSubscribedToSchema(t):
        raise SubscribeAndRetry(
            Timer("Subscribing to schema %s%s", t, " lazily" if lazy else "")(
                lambda db: db.subscribeToSchema(t, lazySubscription=lazy)
            )
        )


class Expands(Cell):
    # TODO: Do the icons really need to be their own Cell objects? In fact, does Octicon need to be its own Cell class/object at all, considering it is a styling/visual issue that can
    # more easily be handled by passing names to the front end?
    def __init__(self, closed, open, closedIcon=None, openedIcon=None, initialState=False):
        super().__init__()
        self.isExpanded = initialState
        self.closed = closed
        self.open = open
        self.openedIcon = openedIcon or Octicon("diff-removed")
        self.closedIcon = closedIcon or Octicon("diff-added")

    def sortsAs(self):
        if self.isExpanded:
            return self.open.sortsAs()
        return self.closed.sortsAs()

    def recalculate(self):
        inlineScript = "cellSocket.sendString(JSON.stringify({'event':'click', 'target_cell': '%s'}))" % self.identity
        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', self._divStyle())
            .with_children(
                HTMLElement.div()
                .set_attribute('style', 'display:inline-block;vertical-align:top')
                .set_attribute('onclick', inlineScript)
                .add_child(HTMLTextContent('____icon__')),

                HTMLElement.div()
                .set_attribute('style', 'display:inline-block')
                .add_child(HTMLTextContent('____child__'))

            )
        )

        self.children = {
            '____child__': self.open if self.isExpanded else self.closed,
            '____icon__': self.openedIcon if self.isExpanded else self.closedIcon
        }
        self.namedChildren = {
            'content': self.open if self.isExpanded else self.closed,
            'icon': self.openedIcon if self.isExpanded else self.closedIcon
        }

        # temporary js WS refactoring data
        self.exportData['divStyle'] = self._divStyle()
        self.exportData['events'] = {"onclick": inlineScript}

        for c in self.children.values():
            if c.cells is not None:
                c.prepareForReuse()

    def onMessage(self, msgFrame):
        self.isExpanded = not self.isExpanded

        self.markDirty()


class CodeEditor(Cell):
    """Produce a code editor."""

    def __init__(self,
                 keybindings=None,
                 noScroll=False,
                 minLines=None,
                 fontSize=None,
                 autocomplete=True,
                 onTextChange=None):
        """Create a code editor

        keybindings - map from keycode to a lambda function that will receive
            the current buffer and the current selection range when the user
            types ctrl-X and 'X' is a valid keycode. Common values here are also
            'Enter' and 'space'

        You may call 'setContents' to override the current contents of the editor.
        This version is not robust to mutiple users editing at the same time.

        onTextChange - called when the text buffer changes with the new buffer
            and a json selection.
        """
        super().__init__()
        # contains (current_iteration_number: int, text: str)
        self._slot = Slot((0, ""))
        self.keybindings = keybindings or {}
        self.noScroll = noScroll
        self.fontSize = fontSize
        self.minLines = minLines
        self.autocomplete = autocomplete
        self.onTextChange = onTextChange
        self.initialText = ""

    def getContents(self):
        return self._slot.get()[1]

    def setContents(self, contents):
        newSlotState = (self._slot.getWithoutRegisteringDependency()[
                        0]+1000000, contents)
        self._slot.set(newSlotState)
        self.sendCurrentStateToBrowser(newSlotState)

    def onMessage(self, msgFrame):
        if msgFrame['event'] == 'keybinding':
            self.keybindings[msgFrame['key']](
                msgFrame['buffer'], msgFrame['selection'])
        elif msgFrame['event'] == 'editing':
            if self.onTextChange:
                self._slot.set((self._slot.getWithoutRegisteringDependency()[
                               0] + 1, msgFrame['buffer']))
                self.onTextChange(msgFrame['buffer'], msgFrame['selection'])

    def recalculate(self):
        self.children = {} # Is there ever any children for this Cell type?

        editorStyle = 'width:100%;height:100%;margin:auto;border:1px solid lightgray;'
        self.contents = str(
            HTMLElement.div()
            .set_attribute('style', self._divStyle())
            .add_child(
                HTMLElement.div()
                .set_attribute('id', 'editor%s' % self.identity)
                .set_attribute('style', editorStyle)
            )
        )

        # temporary js WS refactoring data
        self.exportData['divStyle'] = self._divStyle()
        self.exportData['initialText'] = self.initialText
        self.exportData['autocomplete'] = self.autocomplete
        self.exportData['noScroll'] = self.noScroll
        if self.fontSize is not None:
            self.exportData['fontSize'] = self.fontSize
        if self.minLines is not None:
            self.exportData['minLines'] = self.minLines
        self.exportData['keybindings'] = [k for k in self.keybindings.keys()]

        print()
        print("CODEEDITOR")
        print()
        print(self.postscript)
        print()

    def sendCurrentStateToBrowser(self, newSlotState):
        if self.cells is not None:
            # if self.identity is None, then we have not been installed in the tree yet
            # so sending ourselves a message makes no sense.
            self.triggerPostscript(
                """
                var editor = aceEditors["editor__identity__"]

                editor.last_edit_millis = Date.now()
                editor.current_iteration = __iteration__;

                curRange = editor.selection.getRange()
                var Range = require('ace/range').Range
                var range = new Range(curRange.start.row,curRange.start.column,curRange.end.row,curRange.end.column)

                newText = "__text__";

                console.log("Resetting editor text to " + newText.length + " because it changed on the server" +
                    " Cur iteration is __iteration__.")

                editor.setValue(newText, 1)
                editor.selection.setRange(range)

                """
                .replace("__identity__", self.identity)
                .replace("__text__", quoteForJs(newSlotState[1], '"'))
                .replace("__iteration__", str(newSlotState[0]))
            )
        else:
            self.initialText = newSlotState[1]


class Sheet(Cell):
    """Make a nice spreadsheet viewer. The dataset needs to be static in this implementation."""

    def __init__(self, columnNames, rowCount, rowFun,
                 colWidth=200,
                 onCellDblClick=None):
        """
        columnNames:
            names to go in column Header
        rowCount:
            number of rows in table
        rowFun:
            function taking integer row as argument that returns list of values
            to populate that row of the table
        colWidth:
            width of columns
        onCellDblClick:
            function to run after user double clicks a cell. It takes as keyword
            arguments row, col, and sheet where row and col represent the row and
            column clicked and sheet is the Sheet object. Clicks on row(col)
            headers will return row(col) values of -1
        """
        super().__init__()

        self.columnNames = columnNames
        self.rowCount = rowCount
        # for a row, the value of all the columns in a list.
        self.rowFun = rowFun
        self.colWidth = colWidth
        self.error = Slot(None)
        self._overflow = "auto"
        self.rowsSent = set()

        self._hookfns = {}
        if onCellDblClick is not None:
            def _makeOnCellDblClick(func):
                def _onMessage(sheet, msgFrame):
                    return onCellDblClick(sheet=sheet,
                                          row=msgFrame["row"],
                                          col=msgFrame["col"])
                return _onMessage

            self._hookfns["onCellDblClick"] = _makeOnCellDblClick(
                onCellDblClick)

    def _addHandsontableOnCellDblClick(self):
        pass

    def recalculate(self):
        self.contents = str(
            HTMLElement.div()
            .add_child(
                HTMLElement.div()
                .set_attribute('id', 'sheet%s' % self.identity)
                .set_attribute('style', self._divStyle())
                .add_class('handsontable')
                .add_child(HTMLTextContent('____error__'))
            )
        )
        errorCell = Subscribed(lambda: Traceback(self.error.get()) if self.error.get() is not None else Text(""))
        self.children = {
            '____error__': errorCell
        }
        self.namedChildren['error'] = errorCell


        # Deleted the postscript that was here.
        # Should now be implemented completely
        # in the JS side component.

        self.exportData['divStyle'] = self._divStyle()
        self.exportData['columnNames'] = [x for x in self.columnNames]
        self.exportData['rowCount'] = self.rowCount
        self.exportData['columnWidth'] = self.colWidth
        self.exportData['handlesDoubleClick'] = ("onCellDblClick" in self._hookfns)

    def onMessage(self, msgFrame):
        """TODO: We will need to update the Cell lifecycle
        and data handling before we can move this
        to the JS side"""

        if msgFrame["event"] == 'sheet_needs_data':
            row = msgFrame['data']

            if row in self.rowsSent:
                return

            rows = []
            for rowToRender in range(max(0, row-100), min(row+100, self.rowCount)):
                if rowToRender not in self.rowsSent:
                    self.rowsSent.add(rowToRender)
                    rows.append(rowToRender)

                    rowData = self.rowFun(rowToRender)

                    self.triggerPostscript(
                        """
                        var hot = handsOnTables["__identity__"].table

                        hot.getSettings().data.cache[__row__] = __data__
                        """
                        .replace("__row__", str(rowToRender))
                        .replace("__identity__", self._identity)
                        .replace("__data__", json.dumps(rowData))
                    )

            if rows:
                self.triggerPostscript(
                    """handsOnTables["__identity__"].table.render()"""
                    .replace("__identity__", self._identity)
                )
        else:
            return self._hookfns[msgFrame["event"]](self, msgFrame)


class Plot(Cell):
    """Produce some reactive line plots."""

    def __init__(self, namedDataSubscriptions, xySlot=None):
        """Initialize a line plot.

        namedDataSubscriptions: a map from plot name to a lambda function
            producing either an array, or {x: array, y: array}
        """
        super().__init__()

        self.namedDataSubscriptions = namedDataSubscriptions
        self.curXYRanges = xySlot or Slot(None)
        self.error = Slot(None)

    def recalculate(self):
        self.contents = str(
            HTMLElement.div()
            .add_child(
                HTMLElement.div()
                .set_attribute('id', 'plot%s' % self.identity)
                .set_attribute('style', self._divStyle())
            )
            .add_children([
                HTMLTextContent('____chart_updater__'),
                HTMLTextContent('____error__')
            ])
        )
        chartUpdaterCell = Subscribed(lambda: _PlotUpdater(self))
        errorCell = Subscribed(lambda: Traceback(self.error.get()) if self.error.get() is not None else Text(""))
        self.children = {
            '____chart_updater__': chartUpdaterCell,
            '____error__': errorCell
        }
        self.namedChildren = {
            'chartUpdater': chartUpdaterCell,
            'error': errorCell
        }
        self.postscript = ""

        # sdtemporary js WS refactoring data
        self.exportData['divStyle'] = self._divStyle()

    def onMessage(self, msgFrame):
        d = msgFrame['data']
        curVal = self.curXYRanges.get() or ((None, None), (None, None))

        self.curXYRanges.set(
            ((d.get('xaxis.range[0]', curVal[0][0]), d.get('xaxis.range[1]', curVal[0][1])),
             (d.get('yaxis.range[0]', curVal[1][0]), d.get('yaxis.range[1]', curVal[1][1])))
        )

        self.cells._logger.info(
            "User navigated plot to %s", self.curXYRanges.get())

    def setXRange(self, low, high):
        curXY = self.curXYRanges.getWithoutRegisteringDependency()
        self.curXYRanges.set(
            ((low, high), curXY[1] if curXY else (None, None))
        )

        self.triggerPostscript(f"""
            plotDiv = document.getElementById('plot__identity__');
            newLayout = plotDiv.layout

            if (typeof(newLayout.xaxis.range[0]) === 'string') {{
                formatDate = function(d) {{
                    return (d.getYear() + 1900) + "-" + ("00" + (d.getMonth() + 1)).substr(-2) + "-" +
                            ("00" + d.getDate()).substr(-2) + " " + ("00" + d.getHours()).substr(-2) + ":" +
                            ("00" + d.getMinutes()).substr(-2) + ":" + ("00" + d.getSeconds()).substr(-2) + "." +
                            ("000000" + d.getMilliseconds()).substr(-3)
                    }};

                newLayout.xaxis.range[0] = formatDate(new Date({low*1000}));
                newLayout.xaxis.range[1] = formatDate(new Date({high*1000}));
                newLayout.xaxis.autorange = false;
            }} else {{
                newLayout.xaxis.range[0] = {low};
                newLayout.xaxis.range[1] = {high};
                newLayout.xaxis.autorange = false;
            }}


            plotDiv.is_server_defined_move = true;
            Plotly.react(plotDiv, plotDiv.data, newLayout);
            plotDiv.is_server_defined_move = false;

            console.log("cells.Plot: range for 'plot__identity__' is now " +
                plotDiv.layout.xaxis.range[0] + " to " + plotDiv.layout.xaxis.range[1])

            """.replace("__identity__", self._identity))


class _PlotUpdater(Cell):
    """Helper utility to push data into an existing line plot."""

    def __init__(self, linePlot):
        super().__init__()

        self.linePlot = linePlot
        self.namedDataSubscriptions = linePlot.namedDataSubscriptions
        self.chartId = linePlot._identity

    def calculatedDataJson(self):
        series = self.callFun(self.namedDataSubscriptions)

        assert isinstance(series, (dict, list))

        if isinstance(series, dict):
            return [self.processSeries(callableOrData, name)
                    for name, callableOrData in series.items()]
        else:
            return [self.processSeries(callableOrData, None) for callableOrData in series]

    def callFun(self, fun):
        if not callable(fun):
            return fun

        sig = signature(fun)
        if len(sig.parameters) == 0:
            return fun()
        if len(sig.parameters) == 1:
            return fun(self.linePlot)
        assert False, "%s expects more than 1 argument" % fun

    def processSeries(self, callableOrData, name):
        data = self.callFun(callableOrData)

        if isinstance(data, list):
            res = {'x': [float(x) for x in range(len(data))],
                   'y': [float(d) for d in data]}
        else:
            assert isinstance(data, dict)
            res = dict(data)

            for k, v in res.items():
                if isinstance(v, numpy.ndarray):
                    res[k] = v.astype('float64').tostring().hex()

        if name is not None:
            res['name'] = name

        return res

    def recalculate(self):
        with self.view() as v:
            # we only exist to run our postscript
            self.contents = """<div style="display:none">"""
            self.children = {} # Does this Cell type ever have children?
            self.postscript = ""
            self.linePlot.error.set(None)

            # temporary js WS refactoring data
            self.exportData['plotId'] = self.chartId
            self.exportData['exceptionOccured'] = False

            try:
                jsonDataToDraw = self.calculatedDataJson()
                self.exportData['plotData'] = jsonDataToDraw
            except SubscribeAndRetry:
                raise
            except Exception:
                # temporary js WS refactoring data
                self.exportData['exceptionOccured'] = True

                self._logger.error(traceback.format_exc())
                self.linePlot.error.set(traceback.format_exc())

            self._resetSubscriptionsToViewReads(v)
