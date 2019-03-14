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

MAX_TIMEOUT = 1.0
MAX_TRIES = 10
MAX_FPS = 10

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

    assert not replacements, "Didn't use up replacement %s in %s" % (replacements.keys(), msg)

    return "".join(outChunks)


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
        self.fileobj = gevent.fileobject.FileObjectPosix(self.read_fd, bufsize=2)
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

        # a list of pending task objects
        self._tasks = queue.Queue()

        self._logger = logging.getLogger(__name__)

        self._root = RootCell()

        self._shouldStopProcessingTasks = threading.Event()

        self._addCell(self._root, parent=None)

    def createTask(self, owningCell, taskFun):
        """Create a long-running task (such as a task that acquires data) that can push content into slots.

        Tasks are run in the order received by a set of worker threads. They can interact
        with object_db, and write into slots.

        Tasks are callables that are invoked as

            task(owningCell, shouldStop)

        where 'owningCell' is the cell that created the task, 'shouldStop' is a threading.Event that
        we set if we know the task should stop (because we've navigated away from the cell
        that produced it.)

        Tasks should use 'owningCell' to create transactions and views because it will have the
        correct serialization context.
        """
        self._tasks.put((owningCell, taskFun))

    def markStopProcessingTasks(self):
        """Our calling context is leaving, so we should stop processing."""
        self._shouldStopProcessingTasks.set()

    def processTasks(self):
        while not self._shouldStopProcessingTasks.is_set():
            self.processOneTask()

    def processOneTask(self):
        try:
            creatingCell, task = self._tasks.get(timeout=.25)
            if creatingCell.isActive():
                try:
                    self._logger.info("Starting task %s with %s remaining", task, self._tasks.qsize())
                    t0 = time.time()
                    task(creatingCell, self._shouldStopProcessingTasks)
                    self._logger.info("Task %s took %s", task, time.time() - t0)
                except Exception:
                    self._logger.error("Unexpected Exception in cells.processTasks:\n%s", traceback.format_exc())
            return True
        except queue.Empty:
            return False


    @property
    def root(self):
        return self._root

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

    def markSlotDirtyForNextRecompute(self, slot):
        self._dirtySlots.add(slot)
        self._gEventHasTransactions.trigger()


    def _onTransaction(self, *trans):
        self._transactionQueue.put(trans)
        self._gEventHasTransactions.trigger()

    def _handleTransaction(self, key_value, priors, set_adds, set_removes, transactionId):
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
        assert not cell.garbageCollected, (cell, cell.text if isinstance(cell, Text) else "")

        self._dirtyNodes.add(cell)

    def markToDiscard(self, cell):
        assert not cell.garbageCollected, (cell, cell.text if isinstance(cell, Text) else "")

        self._nodesToDiscard.add(cell)

    def markToBroadcast(self, node):
        assert node.cells is self

        self._nodesToBroadcast.add(node)

    def renderMessages(self):
        self._processDirtySlots()
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
        res = [{'postscript': js} for js in reversed(self._pendingPostscripts)] + res

        self._pendingPostscripts.clear()

        self._nodesToBroadcast = set()
        self._nodesToDiscard = set()

        return res

    def _processDirtySlots(self):
        slots = list(self._dirtySlots)
        self._dirtySlots.clear()

        for slot in slots:
            slot.absorbPendingWrite()

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
                    while True:
                        try:
                            n.prepare()
                            n.recalculate()
                            break
                        except SubscribeAndRetry as e:
                            e.callback(self.db)

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
                    self._logger.error("Node %s had exception during recalculation:\n%s", n, traceback.format_exc())
                    self._logger.error("Subscribed cell threw an exception:\n%s", traceback.format_exc())
                    n.children = {'____contents__': Traceback(traceback.format_exc())}
                    n.contents = "____contents__"
                    return
                finally:
                    _cur_cell.cell = None
                    _cur_cell.isProcessingMessage = False

                newChildren = set(n.children.values())

                for c in newChildren.difference(origChildren):
                    self._addCell(c, n)

                for c in origChildren.difference(newChildren):
                    self._cellOutOfScope(c)

                self._cellsKnownChildren[n.identity] = newChildren

    def updateMessageFor(self, cell):
        contents = cell.contents
        assert isinstance(contents, str), "Cell %s produced %s for its contents which is not a string" % (cell, contents)

        formatArgs = {}

        replaceDict = {}

        for childName, childNode in cell.children.items():
            formatArgs[childName] = "<div id='%s'></div>" % (cell.identity + "_" + childName)
            replaceDict[cell.identity + "_" + childName] = childNode.identity

        try:
            contents = multiReplace(contents, formatArgs)
        except Exception:
            raise Exception("Failed to format these contents with args %s:\n\n%s", formatArgs, contents)

        res = {
            'id': cell.identity,
            'contents': contents,
            'replacements': replaceDict
        }

        if cell.postscript:
            res['postscript'] = cell.postscript
        return res


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
        self._pendingValue = value
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
            if _cur_cell.cell and not getattr(_cur_cell, 'isProcessingMessage', False):
                self._subscribedCells.add(_cur_cell.cell)

            return self._value

    def set(self, val):
        """Schedule a write to a slot. This does _NOT_ happen synchronously, because
        we need to buffer writes to Slots that can happen on background threads.
        """
        with self._lock:
            if val == self._pendingValue:
                return

            self._pendingValue = val

        # delay triggering this slot as dirty
        if self._cells:
            self._cells.markSlotDirtyForNextRecompute(self)

    def absorbPendingWrite(self):
        """Update our value and dirty any listening cells.

        If our pending value hasn't actually changed, do nothing.
        """
        if self._value == self._pendingValue:
            return

        self._value = self._pendingValue

        for c in self._subscribedCells:
            c.markDirty()
        self._subscribedCells = set()

class SessionState(object):
    """Represents a piece of session-specific interface state. You may access state
    using attributes, which will register a dependency
    """
    def __init__(self):
        self._slots = {}

    def _reset(self, cells):
        self._slots = {k:Slot(v.getWithoutRegisteringDependency()) for k,v in self._slots.items()}
        for s in self._slots.values():
            s._cells = cells
            if isinstance(s._value, Cell):
                try:
                    s._value.prepareForReuse()
                except Exception:
                    logging.warn("Reusing a Cell slot could create a problem: %s", s._value)

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

    def ensure(self, attr, value):
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
        self.contents = ""  # some contents containing a local node def
        self._identity = None
        self._tag = None
        self.postscript = None
        self.garbageCollected = False
        self.subscriptions = set()
        self._nowrap = False
        self._background_color = None
        self._width = None
        self._overflow = None
        self._color = None
        self._height = None
        self.serializationContext = None
        self.context = {}

        self._logger = logging.getLogger(__name__)

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

    def findChildrenByTag(self, tag, stopSearchingAtTag=True, isRoot=True):
        """Search the cell and its children for all cells with the given tag.

        If `stopSearchingAtTag`, then if we encounter a non-None tag that doesn't
        match, stop searching immediately.
        """
        cells = []

        if self._tag == tag:
            cells.append(self)

        if self._tag is not None and stopSearchingAtTag and not isRoot:
            return cells

        for child in self.children:
            cells.extend(self.children[child].findChildrenByTag(tag, stopSearchingAtTag, False))

        return cells

    def visitAllChildren(self, visitor):
        visitor(self)
        for child in self.children.values():
            child.visitAllChildren(visitor)

    def findChildrenMatching(self, filter):
        res = []

        def visitor(cell):
            if filter(cell):
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

                with self.transaction():
                    self.onMessage(*args)
                    return
            except RevisionConflictException:
                tries += 1
                if tries > MAX_TRIES or time.time() - t0 > MAX_TIMEOUT:
                    self._logger.error("OnMessage timed out. This should really fail.")
                    return
            except Exception:
                self._logger.error("Exception in dropdown logic:\n%s", traceback.format_exc())
                return
            finally:
                _cur_cell.cell = None
                _cur_cell.isProcessingMessage = False

    def withSerializationContext(self, context):
        self.serializationContext = context
        return self

    def _clearSubscriptions(self):
        if self.cells:
            for sub in self.subscriptions:
                self.cells.unsubscribeCell(self, sub)

        self.subscriptions = set()

    def _resetSubscriptionsToViewReads(self, view):
        new_subscriptions = set(view._reads).union(set(view._indexReads))

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
        if self.serializationContext is None and self.parent is not None:
            if self.parent.serializationContext is None:
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
            return "style='%s'" % ";".join(res)

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

    def getContext(self, contextKey):
        if contextKey in self.context:
            return self.context[contextKey]

        if self.parent:
            return self.parent.getContext(contextKey)

        return None


class Card(Cell):
    def __init__(self, inner, padding=None):
        super().__init__()

        self.children = {"____contents__": Cell.makeCell(inner)}
        self.padding = padding

    def recalculate(self):
        other = ""
        if self.padding:
            other += " p-" + str(self.padding)

        self.contents = """
            <div class="card" __style__>
              <div class="card-body __other__">
                ____contents__
              </div>
            </div>
            """.replace('__other__', other).replace('__style__', self._divStyle())

    def sortsAs(self):
        return self.inner.sortsAs()


class CardTitle(Cell):
    def __init__(self, inner):
        super().__init__()

        self.children = {"____contents__": Cell.makeCell(inner)}
        self.contents = """
        <div class='card-title'>
          ____contents__
        </div>
        """

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
        self.title = title
        self.message = message
        self.buttons = {f"____button_{k}__": Button(k, v) for k,v in buttonActions.items()}

    def recalculate(self):
        self.contents = (
            f"""
            <div class="modal fade show" tabindex="-1" role="dialog" style="display: block; padding-right: 15px;">
              <div class="modal-dialog" role="document">
                <div class="modal-content">
                  <div class="modal-header">
                    <h5 class="modal-title">____title__</h5>
                  </div>
                  <div class="modal-body">
                    ____message__
                  </div>
                  <div class="modal-footer">
                    __buttons__
                  </div>
                </div>
              </div>
            </div>
            """
            .replace("__buttons__",
                " ".join(self.buttons)
                )
            )
        self.children = dict(self.buttons)
        self.children["____title__"] = Cell.makeCell(self.title)
        self.children["____message__"] = Cell.makeCell(self.message)

class Octicon(Cell):
    def __init__(self, which):
        super().__init__()
        self.whichOcticon = which

    def sortsAs(self):
        return self.whichOcticon

    def recalculate(self):
        self.contents = (
            '<span class="octicon octicon-%s" aria-hidden="true" __style__></span>' % self.whichOcticon
        ).replace('__style__', self._divStyle())


class Badge(Cell):
    def __init__(self, inner, style='primary'):
        super().__init__()
        self.inner = self.makeCell(inner)
        self.style = style

    def sortsAs(self):
        return self.inner.sortsAs()

    def recalculate(self):
        self.contents = """<span class="badge badge-__style__">____child__</span>""".replace(
            "__style__", self.style
        )
        self.children = {'____child__': self.inner}

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

        self.contents = """
            <div class="container-fluid" __style__>
                <div class="row flex-nowrap no-gutters">
                    <div class="col-md-auto">
                        ____panel__
                    </div>
                    <div class="col-sm">
                        ____content__
                    </div>
                </div>
            </div>
            """ if expanded else """
            <div __style__>
                ____content__
            </div>
            """

        self.contents = (
            self.contents.replace("__identity__", self.identity)
                         .replace("__style__", self._divStyle())
            )

        self.children = {
            '____content__': self.content
            }

        if expanded:
            self.children['____panel__'] = self.panel


class Text(Cell):
    def __init__(self, text, sortAs=None):
        super().__init__()
        self.text = text
        self._sortAs = sortAs if sortAs is not None else text

    def sortsAs(self):
        return self._sortAs

    def recalculate(self):
        self.contents = "<div %s>%s</div>" % (
            self._divStyle(),
            cgi.escape(str(self.text)) if self.text else "&nbsp;"
        )


class Padding(Cell):
    def __init__(self):
        super().__init__()
        self.contents = "<span class='px-2'>&nbsp</span>"

    def sortsAs(self):
        return " "


class Span(Cell):
    def __init__(self, text):
        super().__init__()
        self.contents = "<span>%s</span>" % cgi.escape(str(text))

    def sortsAs(self):
        return self.contents


class Sequence(Cell):
    def __init__(self, elements):
        super().__init__()
        elements = [Cell.makeCell(x) for x in elements]

        self.elements = elements
        self.children = {"____c_%s__" % i: elements[i] for i in range(len(elements)) }

    def __add__(self, other):
        other = Cell.makeCell(other)
        if isinstance(other, Sequence):
            return Sequence(self.elements + other.elements)
        else:
            return Sequence(self.elements + [other])

    def recalculate(self):
        self.contents = "<div %s>" % self._divStyle() + "\n".join("____c_%s__" % i for i in range(len(self.elements))) + "</div>"

    def sortsAs(self):
        if self.elements:
            return self.elements[0].sortsAs()
        return None


class Columns(Cell):
    def __init__(self, *elements):
        super().__init__()
        elements = [Cell.makeCell(x) for x in elements]

        self.elements = elements
        self.children = {"____c_%s__" % i: elements[i] for i in range(len(elements)) }
        self.contents = (
            """
            <div class="container-fluid" __style__>
            <div class="row flex-nowrap">
                __contents__
            </div>
            </div>
            """
            .replace("__style__", self._divStyle())
            .replace(
                "__contents__",
                "\n".join(
                    """<div class="col-sm"> ____c_%s__ </div>""" % i
                    for i in range(len(elements)))
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

        self.contents = """
            <div>
                <span id="object_database_large_pending_download_text"></span>
            </div>
        """


class HeaderBar(Cell):
    def __init__(self, leftItems, centerItems=(), rightItems=()):
        super().__init__()
        self.leftItems = leftItems
        self.centerItems = centerItems
        self.rightItems = rightItems

        self.contents = """
            <div class="p-2 bg-light flex-container" style="display:flex;align-items:baseline">
                <div class="flex-item" style="flex-grow:0">
                    <div class="flex-container" style="display:flex;justify-content:flex-center;align-items:baseline">
                        %s
                    </div>
                </div>
                <div class="flex-item" style="flex-grow:1">
                    <div class="flex-container" style="display:flex;justify-content:center;align-items:baseline">
                        %s
                    </div>
                </div>
                <div class="flex-item" style="flex-grow:0">
                    <div class="flex-container" style="display:flex;justify-content:flex-center;align-items:baseline">
                        %s
                    </div>
                </div>
            </div>
        """ % (
            "".join(["<span class='flex-item px-3'>____left_%s__</span>" % i for i in range(len(self.leftItems))]),
            "".join(["<span class='flex-item px-3'>____center_%s__</span>" % i for i in range(len(self.centerItems))]),
            "".join(["<span class='flex-item px-3'>____right_%s__</span>" % i for i in range(len(self.rightItems))]),
        )

        self.children = {'____left_%s__' % i: self.leftItems[i] for i in range(len(self.leftItems))}
        self.children.update({'____center_%s__' % i: self.centerItems[i] for i in range(len(self.centerItems))})
        self.children.update({'____right_%s__' % i: self.rightItems[i] for i in range(len(self.rightItems))})


class Main(Cell):
    def __init__(self, child):
        super().__init__()

        self.contents = """
            <main class="py-md-2">
            <div class="container-fluid">
                ____child__
            </div>
            </main>
        """
        self.children = {'____child__': child}


class _NavTab(Cell):
    def __init__(self, slot, index, target, child):
        super().__init__()

        self.slot = slot
        self.index = index
        self.target = target
        self.child = child

    def recalculate(self):
        self.contents = ("""
            <li class="nav-item">
                <a class="nav-link __active__" role="tab"
                    onclick="websocket.send(JSON.stringify({'event':'click', 'ix': __ix__, 'target_cell': '__identity__'}))"
                    >
                    ____child__
                </a>
            </li>
            """.replace("__identity__", self.target)
               .replace("__ix__", str(self.index))
               .replace("__active__", "active" if self.index == self.slot.get() else "")
        )

        self.children['____child__'] = Cell.makeCell(self.child)


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
        self.children['____display__'] = Subscribed(lambda: self.headersAndChildren[self.whichSlot.get()][1])

        for i in range(len(self.headersAndChildren)):
            self.children['____header_{ix}__'.format(ix=i)] = _NavTab(self.whichSlot, i, self._identity, self.headersAndChildren[i][0])

        self.contents = (
            """
            <div class="container-fluid mb-3">
                 <ul class="nav nav-tabs" role="tablist">
                  __items__
                </ul>
                <div class="tab-content">
                  <div class="tab-pane fade show active" role="tabpanel">____display__
                </div>
            </div>
            """
            .replace(
                "__items__",
                "".join(
                    """ ____header___ix____ """.replace('__ix__', str(i))
                    for i in range(len(self.headersAndChildren))
                )
            ).replace("__identity__", self._identity)
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

            self.headersAndLambdas = [(header, makeCallback(header)) for header in headersAndLambdas]
        else:
            self.headersAndLambdas = headersAndLambdas

        self.title = Cell.makeCell(title)

    def sortsAs(self):
        return self.title.sortsAs()

    def recalculate(self):
        items = []

        self.children['____title__'] = self.title

        for i in range(len(self.headersAndLambdas)):
            header, onDropdown = self.headersAndLambdas[i]
            self.children["____child_%s__" % i] = Cell.makeCell(header)

            items.append(
                """
                    <a class='dropdown-item'
                        onclick="__onclick__"
                        >
                    ____child___ix____
                    </a>
                """.replace(
                    "__onclick__",
                    "websocket.send(JSON.stringify({'event':'menu', 'ix': __ix__, 'target_cell': '__identity__'}))"
                    if not isinstance(onDropdown, str) else
                    quoteForJs("window.location.href = '__url__'".replace("__url__", quoteForJs(onDropdown, "'")), '"')
                ).replace("__ix__", str(i)).replace("__identity__", self.identity)
            )

        self.contents = """
            <div class="btn-group">
                  <a role="button" class="btn btn-xs btn-outline-secondary">____title__</a>
                  <button class="btn btn-xs btn-outline-secondary dropdown-toggle dropdown-toggle-split" type="button"
                        id="___identity__-dropdownMenuButton" data-toggle="dropdown">
                  </button>
                  <div class="dropdown-menu">
                    __dropdown_items__
                  </div>
            </div>
            """.replace("__identity__", self.identity).replace("__dropdown_items__", "\n".join(items))

    def onMessage(self, msgFrame):
        fun = self.headersAndLambdas[msgFrame['ix']][1]
        fun()


class Container(Cell):
    def __init__(self, child=None):
        super().__init__()
        if child is None:
            self.contents = ""
            self.children = {}
        else:
            self.contents = "<div>____child__</div>"
            self.children = {"____child__": Cell.makeCell(child)}

    def setChild(self, child):
        self.setContents("<div>____child__</div>", {"____child__": Cell.makeCell(child)})

    def setContents(self, newContents, newChildren):
        self.contents = newContents
        self.children = newChildren
        self.markDirty()


class Scrollable(Container):
    def __init__(self, child=None):
        super().__init__(child)
        self.overflow('auto')


class RootCell(Container):
    def setRootSerializationContext(self, context):
        self.serializationContext = context

    @property
    def identity(self):
        return "page_root"

    def setChild(self, child):
        self.setContents("<div>____c__</div>", {"____c__": child})


class Traceback(Cell):
    def __init__(self, traceback):
        super().__init__()
        self.contents = """<div class='alert alert-primary'><pre>____child__</pre></alert>"""
        self.traceback = traceback
        self.children = {"____child__": Cell.makeCell(traceback)}

    def sortsAs(self):
        return self.traceback


class Code(Cell):
    def __init__(self, codeContents):
        super().__init__()
        self.contents = """<pre><code>____child__</code></pre>"""
        self.codeContents = codeContents
        self.children = {"____child__": Cell.makeCell(codeContents)}

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
        self.contents = """<div>____child__</div>"""

    def getChild(self):
        if type(self.obj) in ContextualDisplay._typeToDisplay:
            for context, dispFun in ContextualDisplay._typeToDisplay[type(self.obj)]:
                if context.matchesCell(self):
                    return dispFun(self.obj)

        return Traceback(f"Invalid object of type {type(self.obj)}")

    def recalculate(self):
        self.children = {"____child__": self.getChild()}

class Subscribed(Cell):
    def __init__(self, f):
        super().__init__()

        self.f = f

    def prepareForReuse(self):
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
            self.contents = """<div %s>____contents__</div>""" % self._divStyle()
            try:
                c = Cell.makeCell(self.f())
                if c.cells is not None:
                    c.prepareForReuse()
                self.children = {'____contents__': c}
            except SubscribeAndRetry:
                raise
            except Exception:
                self.children = {'____contents__': Traceback(traceback.format_exc())}
                self._logger.error("Subscribed inner function threw exception:\n%s", traceback.format_exc())

            self._resetSubscriptionsToViewReads(v)


class SubscribedSequence(Cell):
    def __init__(self, itemsFun, rendererFun, asColumns=False):
        super().__init__()

        self.itemsFun = itemsFun
        self.rendererFun = rendererFun
        self.existingItems = {}
        self.spine = []
        self.asColumns = asColumns

    def prepareForReuse(self):
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
                self._logger.error("Spine calc threw an exception:\n%s", traceback.format_exc())
                self.spine = []

            self._resetSubscriptionsToViewReads(v)

            new_children = {}
            for ix, rowKey in enumerate(self.spine):
                if rowKey in self.existingItems:
                    new_children["____child_%s__" % ix] = self.existingItems[rowKey]
                else:
                    try:
                        self.existingItems[rowKey] = new_children["____child_%s__" % ix] = self.makeCell(rowKey[0])
                    except SubscribeAndRetry:
                        raise
                    except Exception:
                        self.existingItems[rowKey] = new_children["____child_%s__" % ix] = Traceback(traceback.format_exc())

        self.children = new_children

        spineAsSet = set(self.spine)
        for i in list(self.existingItems):
            if i not in spineAsSet:
                del self.existingItems[i]

        if self.asColumns:
            self.contents = (
                """
                <div class="container-fluid" __style__>
                <div class="row flex-nowrap">
                    __contents__
                </div>
                </div>
                """
                .replace("__style__", self._divStyle())
                .replace(
                    "__contents__",
                    "\n".join(
                        f"""<div class="col-sm"> ____child_{i}__ </div>"""
                        for i in range(len(self.spine)))
                )
            )
        else:
            self.contents = """<div %s>%s</div>""" % (
                self._divStyle(),
                "\n".join(['____child_%s__' % i for i in range(len(self.spine))])
            )

class Popover(Cell):
    def __init__(self, contents, title, detail, width=400):
        super().__init__()

        self.width = width
        self.children = {
            '____contents__': Cell.makeCell(contents),
            '____detail__': Cell.makeCell(detail),
            '____title__': Cell.makeCell(title)
        }

    def recalculate(self):
        self.contents = """
            <div __style__>
            <a href="#popmain___identity__" data-toggle="popover" data-trigger="focus" data-bind="#pop___identity__" container="body" class="btn btn-xs" role="button">____contents__</a>
            <div style="display:none;">
              <div id="pop___identity__">
                <div class='data-placement'>bottom</div>
                <div class="data-title">____title__</div>
                <div class="data-content"><div style="width:__width__px">____detail__</div></div>
              </div>
            </div>

            </div>
            """.replace("__style__", self._divStyle()).replace("__identity__", self.identity).replace("__width__", str(self.width))

    def sortsAs(self):
        if '____title__' in self.children:
            return self.children['____title__'].sortsAs()


class Grid(Cell):
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
                self._logger.error("Row fun calc threw an exception:\n%s", traceback.format_exc())
                self.rows = []
            try:
                self.cols = augmentToBeUnique(self.colFun())
            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error("Col fun calc threw an exception:\n%s", traceback.format_exc())
                self.cols = []

            self._resetSubscriptionsToViewReads(v)

        new_children = {}
        seen = set()

        for col_ix, col in enumerate(self.cols):
            seen.add((None, col))
            if (None, col) in self.existingItems:
                new_children["____header_%s__" % (col_ix)] = self.existingItems[(None, col)]
            else:
                try:
                    self.existingItems[(None, col)] = new_children["____header_%s__" % col_ix] = Cell.makeCell(self.headerFun(col[0]))
                except SubscribeAndRetry:
                    raise
                except Exception:
                    self.existingItems[(None, col)] = new_children["____header_%s__" % col_ix] = Traceback(traceback.format_exc())

        if self.rowLabelFun is not None:
            for row_ix, row in enumerate(self.rows):
                seen.add((None, row))
                if (row, None) in self.existingItems:
                    new_children["____rowlabel_%s__" % (row_ix)] = self.existingItems[(row, None)]
                else:
                    try:
                        self.existingItems[(row, None)] = new_children["____rowlabel_%s__" % row_ix] = Cell.makeCell(self.rowLabelFun(row[0]))
                    except SubscribeAndRetry:
                        raise
                    except Exception:
                        self.existingItems[(row, None)] = new_children["____rowlabel_%s__" % row_ix] = Traceback(traceback.format_exc())

        seen = set()
        for row_ix, row in enumerate(self.rows):
            for col_ix, col in enumerate(self.cols):
                seen.add((row, col))
                if (row, col) in self.existingItems:
                    new_children["____child_%s_%s__" % (row_ix, col_ix)] = self.existingItems[(row, col)]
                else:
                    try:
                        self.existingItems[(row, col)] = new_children["____child_%s_%s__" % (row_ix, col_ix)] = Cell.makeCell(self.rendererFun(row[0], col[0]))
                    except SubscribeAndRetry:
                        raise
                    except Exception:
                        self.existingItems[(row, col)] = new_children["____child_%s_%s__" % (row_ix, col_ix)] = Traceback(traceback.format_exc())

        self.children = new_children

        for i in list(self.existingItems):
            if i not in seen:
                del self.existingItems[i]

        self.contents = (
            """
            <table class="table-hscroll table-sm table-striped">
            <thead><tr>""" + ("<th></th>" if self.rowLabelFun is not None else "") + """__headers__</tr></thead>
            <tbody>
            __rows__
            </tbody>
            </table>
            """
            .replace(
                "__headers__",
                "".join("<th>____header_%s__</th>" % (col_ix)
                        for col_ix in range(len(self.cols))))
            .replace(
                "__rows__",
                "\n".join(
                    "<tr>" +
                    ("<td>____rowlabel_%s__</td>" % row_ix if self.rowLabelFun is not None else "") +
                    "".join(
                        "<td>____child_%s_%s__</td>" % (row_ix, col_ix)
                        for col_ix in range(len(self.cols))
                    ) +
                    "</tr>"
                    for row_ix in range(len(self.rows))
                )
            )
        )


class SortWrapper:
    def __init__(self, x):
        self.x = x

    def __lt__(self, other):
        try:
            if type(self.x) is type(other.x):
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
            if type(self.x) is type(other.x):
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
        self.contents = (
            """
            <input __style__ type="text" id="text___identity__" onchange="
                websocket.send(JSON.stringify({'event':'click', 'target_cell': '__identity__', 'text': this.value}))
                "
                value="__contents__"
                __pat__
                __width__
                >
            """.replace("__style__", self._divStyle())
               .replace("__identity__", self.identity)
               .replace("__contents__", quoteForJs(self.slot.get(), '"'))
               .replace("__pat__", "" if not self.pattern else quoteForJs(self.pattern, '"'))
               .replace("__sytle__", self._divStyle())
        )

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
            self._logger.error("Failed to parse current page: %s", traceback.format_exc())

        return rows[page * self.maxRowsPerPage:(page+1) * self.maxRowsPerPage]

    def makeHeaderCell(self, col_ix):
        col = self.cols[col_ix]

        if col not in self.columnFilters:
            self.columnFilters[col] = Slot(None)

        def icon():
            if self.sortColumn.get() != col_ix:
                return ""
            return Octicon("arrow-up" if not self.sortColumnAscending.get() else "arrow-down")

        cell = Cell.makeCell(self.headerFun(col)).nowrap() + Padding() + Subscribed(icon).nowrap()

        def onClick():
            if self.sortColumn.get() == col_ix:
                self.sortColumnAscending.set(not self.sortColumnAscending.get())
            else:
                self.sortColumn.set(col_ix)
                self.sortColumnAscending.set(False)

        res = Clickable(cell, onClick, makeBold=True)

        if self.columnFilters[col].get() is None:
            res = res.nowrap() + Clickable(Octicon("search"), lambda: self.columnFilters[col].set("")).nowrap()
        else:
            res = res + SingleLineTextBox(self.columnFilters[col]).nowrap() + \
                Button(Octicon("x"), lambda: self.columnFilters[col].set(None), small=True)

        return Card(res, padding=1)

    def recalculate(self):
        with self.view() as v:
            try:
                self.cols = list(self.colFun())
            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error("Col fun calc threw an exception:\n%s", traceback.format_exc())
                self.cols = []

            try:
                self.unfilteredRows = list(self.rowFun())
                self.filteredRows = self.filterRows(self.unfilteredRows)
                self.rows = self.sortRows(self.filteredRows)

            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error("Row fun calc threw an exception:\n%s", traceback.format_exc())
                self.rows = []

            self._resetSubscriptionsToViewReads(v)

        new_children = {}
        seen = set()

        for col_ix, col in enumerate(self.cols):
            seen.add((None, col))
            if (None, col) in self.existingItems:
                new_children["____header_%s__" % (col_ix)] = self.existingItems[(None, col)]
            else:
                try:
                    self.existingItems[(None, col)] = new_children["____header_%s__" % col_ix] = self.makeHeaderCell(col_ix)
                except SubscribeAndRetry:
                    raise
                except Exception:
                    self.existingItems[(None, col)] = new_children["____header_%s__" % col_ix] = Traceback(traceback.format_exc())

        seen = set()
        for row_ix, row in enumerate(self.rows):
            for col_ix, col in enumerate(self.cols):
                seen.add((row, col))
                if (row, col) in self.existingItems:
                    new_children["____child_%s_%s__" % (row_ix, col_ix)] = self.existingItems[(row, col)]
                else:
                    try:
                        self.existingItems[(row, col)] = new_children["____child_%s_%s__" % (row_ix, col_ix)] = Cell.makeCell(self.rendererFun(row, col))
                    except SubscribeAndRetry:
                        raise
                    except Exception:
                        self.existingItems[(row, col)] = new_children["____child_%s_%s__" % (row_ix, col_ix)] = Traceback(traceback.format_exc())

        self.children = new_children

        for i in list(self.existingItems):
            if i not in seen:
                del self.existingItems[i]

        totalPages = ((len(self.filteredRows) - 1) // self.maxRowsPerPage + 1)

        rowDisplay = "____left__ ____right__ Page ____page__ of " + str(totalPages)
        if totalPages <= 1:
            self.children['____page__'] = Cell.makeCell(totalPages).nowrap()
        else:
            self.children['____page__'] = SingleLineTextBox(self.curPage, pattern="[0-9]+").width(10 * len(str(totalPages)) + 6).height(20).nowrap()
        if self.curPage.get() == "1":
            self.children['____left__'] = Octicon("triangle-left").nowrap().color("lightgray")
        else:
            self.children['____left__'] = Clickable(Octicon("triangle-left"), lambda: self.curPage.set(str(int(self.curPage.get())-1))).nowrap()

        if self.curPage.get() == str(totalPages):
            self.children['____right__'] = Octicon("triangle-right").nowrap().color("lightgray")
        else:
            self.children['____right__'] = Clickable(Octicon("triangle-right"), lambda: self.curPage.set(str(int(self.curPage.get())+1))).nowrap()

        self.contents = (
            """
            <table class="table-hscroll table-sm table-striped">
            <thead style="border-bottom: black;border-bottom-style:solid;border-bottom-width:thin;""><tr>""" +
            f'<th style="vertical-align:top"><div class="card"><div class="card-body p-1">{rowDisplay}</div></div></th>' +
            """
            __headers__</tr></thead>
            <tbody>
            __rows__
            </tbody>
            </table>
            """
            .replace(
                "__headers__",
                "".join('<th style="vertical-align:top">____header_%s__</th>' % (col_ix)
                        for col_ix in range(len(self.cols))))
            .replace(
                "__rows__",
                "\n".join(
                    "<tr>" +
                    ("<td>%s</td>" % (row_ix+1)) +
                    "".join(
                        "<td>____child_%s_%s__</td>" % (row_ix, col_ix)
                        for col_ix in range(len(self.cols))
                    ) +
                    "</tr>"
                    for row_ix in range(len(self.rows))
                )
            )
        )


class Clickable(Cell):
    def __init__(self, content, f, makeBold=False, makeUnderling=False):
        super().__init__()
        self.f = f
        self.content = Cell.makeCell(content)
        self.bold = makeBold

    def calculatedOnClick(self):
        if isinstance(self.f, str):
            return quoteForJs("window.location.href = '__url__'".replace("__url__", quoteForJs(self.f, "'")), '"')
        else:
            return "websocket.send(JSON.stringify({'event':'click', 'target_cell': '__identity__'}))".replace("__identity__", self.identity)

    def recalculate(self):
        self.children = {'____contents__': self.content}

        self.contents = (
            """
            <div onclick="__onclick__" __style__>
            ____contents__
            </div>"""
            .replace('__onclick__', self.calculatedOnClick())
            .replace(
                '__style__',
                self._divStyle("cursor:pointer;*cursor: hand" + (";font-weight:bold" if self.bold else ""))
            )
        )

    def sortsAs(self):
        return self.content.sortsAs()

    def onMessage(self, msgFrame):
        val = self.f()
        if isinstance(val, str):
            self.triggerPostscript(quoteForJs("window.location.href = '__url__'".replace("__url__", quoteForJs(val, "'")), '"'))

class Button(Clickable):
    def __init__(self, *args, small=False, active=True, style="primary", **kwargs):
        Clickable.__init__(self, *args, **kwargs)
        self.small = small
        self.active = active
        self.style = style

    def recalculate(self):
        self.children = {'____contents__': self.content}
        self.contents = (
            f"""
            <button
                class='btn btn{'-outline' if not self.active else ''}-{self.style} __size__'
                onclick="__onclick__"
                >
            ____contents__
            </button>"""
            .replace("__size__", "" if not self.small else "btn-xs")
            .replace("__size__", "" if not self.small else "btn-xs")
            .replace('__identity__', self.identity)
            .replace("__onclick__", self.calculatedOnClick())
        )

class ButtonGroup(Cell):
    def __init__(self, buttons):
        super().__init__()
        self.buttons = buttons

    def recalculate(self):
        self.children = {f'____{i}__': self.buttons[i] for i in range(len(self.buttons))}
        self.contents = (
            """
            <div class="btn-group" role="group">
                __buttons__
            </div>"""
            .replace("__buttons__", " ".join(f"____{i}__" for i in range(len(self.buttons))))
        )

class LoadContentsFromUrl(Cell):
    def __init__(self, targetUrl):
        Cell.__init__(self)
        self.targetUrl = targetUrl

    def recalculate(self):
        self.children = {}
        self.contents = """
            <div>
            <div id='loadtarget__identity__'></div>
            </div>
            """.replace('__identity__', self._identity)

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

def createTask(task):
    """Create a long-running task from within an executing cell. See Cells.createTask."""
    return _cur_cell.cell.cells.createTask(_cur_cell.cell, task)

def ensureSubscribedSchema(t, lazy=False):
    if not current_transaction().db().isSubscribedToSchema(t):
        raise SubscribeAndRetry(
            Timer("Subscribing to schema %s%s", t, " lazily" if lazy else "")(
                lambda db: db.subscribeToSchema(t, lazySubscription=lazy)
            )
        )


class Expands(Cell):
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
        self.contents = """
            <div __style__>
                <div onclick="websocket.send(JSON.stringify({'event':'click', 'target_cell': '__identity__'}))"
                        style="display:inline-block;vertical-align:top">
                    ____icon__
                </div>

                <div style="display:inline-block">
                    ____child__
                </div>
            </div>
            """.replace("__identity__", self.identity)
        self.contents = self.contents.replace("__style__", self._divStyle())
        self.children = {
            '____child__': self.open if self.isExpanded else self.closed,
            '____icon__': self.openedIcon if self.isExpanded else self.closedIcon
        }

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
            onTextChange=None
            ):
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
        self._slot = Slot((0, "")) #contains (current_iteration_number: int, text: str)
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
        newSlotState = (self._slot.getWithoutRegisteringDependency()[0]+1000000, contents)
        self._slot.set(newSlotState)
        self.sendCurrentStateToBrowser(newSlotState)

    def onMessage(self, msgFrame):
        if msgFrame['event'] == 'keybinding':
            self.keybindings[msgFrame['key']](msgFrame['buffer'], msgFrame['selection'])
        elif msgFrame['event'] == 'editing':
            if self.onTextChange:
                self._slot.set((self._slot.getWithoutRegisteringDependency()[0] + 1, msgFrame['buffer']))
                self.onTextChange(msgFrame['buffer'], msgFrame['selection'])

    def recalculate(self):
        self.children = {}
        self.contents = """
            <div __style__>
            <div id="editor__identity__" style="width:100%;height:100%;margin:auto;border:1px solid lightgray"></div>
            </div>
        """.replace("__identity__", self.identity).replace("__style__", self._divStyle()) + ";"

        self.postscript = """
            var editor = ace.edit("editor__identity__");
            aceEditors["editor__identity__"] = editor
            editor.last_edit_millis = Date.now()

            console.log("setting up editor with " )
            editor.setTheme("ace/theme/textmate");
            editor.session.setMode("ace/mode/python");
            editor.setAutoScrollEditorIntoView(true);
            editor.session.setUseSoftTabs(true);
            editor.setValue("__text__");
        """.replace("__text__", quoteForJs(self.initialText, '"'))

        if self.autocomplete:
            self.postscript += """
            editor.setOptions({enableBasicAutocompletion: true});
            editor.setOptions({enableLiveAutocompletion: true});
            """

        if self.fontSize is not None:
            self.postscript += """
            editor.setOption("fontSize", %s);
            """ % self.fontSize

        if self.minLines is not None:
            self.postscript += """
                editor.setOption("minLines", __minlines__);
            """.replace("__minlines__", str(self.minLines))

        if self.noScroll:
            self.postscript += """
                editor.setOption("maxLines", Infinity);
            """

        self.postscript += """
            editor.session.on('change', function(delta) {
                websocket.send(
                    JSON.stringify(
                        {'event': 'editor_change', 'target_cell': '__identity__', 'data': delta}
                        )
                    )
                //record that we just edited
                editor.last_edit_millis = Date.now()

                //schedule a function to run in 'SERVER_UPDATE_DELAY_MS'ms that will update the server,
                //but only if the user has stopped typing.
                SERVER_UPDATE_DELAY_MS = 200

                window.setTimeout(function() {
                    if (Date.now() - editor.last_edit_millis >= SERVER_UPDATE_DELAY_MS) {
                        //save our current state to the remote buffer
                        editor.current_iteration += 1;
                        editor.last_edit_millis = Date.now()
                        editor.last_edit_sent_text = editor.getValue()

                        websocket.send(
                            JSON.stringify(
                                {'event': 'editing', 'target_cell': '__identity__',
                                'buffer': editor.getValue(), 'selection': editor.selection.getRange(),
                                'iteration': editor.current_iteration
                                }
                                )
                            )
                    }
                }, SERVER_UPDATE_DELAY_MS + 2) //2ms for a little grace period.
            });
            """

        for kb in self.keybindings:
            self.postscript += """
            editor.commands.addCommand({
                name: 'cmd___key__',
                bindKey: {win: 'Ctrl-__key__',  mac: 'Command-__key__'},
                exec: function(editor) {
                    editor.current_iteration += 1;
                    editor.last_edit_millis = Date.now()
                    editor.last_edit_sent_text = editor.getValue()

                    websocket.send(
                        JSON.stringify(
                            {'event': 'keybinding', 'target_cell': '__identity__', 'key': '__key__',
                            'buffer': editor.getValue(), 'selection': editor.selection.getRange(),
                            'iteration': editor.current_iteration}
                            )
                        )
                },
                readOnly: true // false if this command should not apply in readOnly mode
            });
            """.replace("__key__", kb)

        self.postscript = self.postscript.replace("__identity__", self.identity)

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

                """.replace("__identity__", self.identity)
                   .replace("__text__", quoteForJs(newSlotState[1], '"'))
                   .replace("__iteration__", str(newSlotState[0]))
                )
        else:
            self.initialText = newSlotState[1]


class Sheet(Cell):
    """Make a nice spreadsheet viewer. The dataset needs to be static in this implementation."""

    def __init__(self, columnNames, rowCount, rowFun,
                 colWidth=200,
                 onCellDblClick=None
                  ):
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
        self.rowFun = rowFun  # for a row, the value of all the columns in a list.
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


            self._hookfns["onCellDblClick"] = _makeOnCellDblClick(onCellDblClick)


    def _addHandsontableOnCellDblClick(self):
        return """
        Handsontable.hooks.add(
            "beforeOnCellMouseDown",
            function(event,data) {
                handsOnObj = handsOnTables["__identity__"];
                if(handsOnObj.lastCellClicked.row == data.row &
                   handsOnObj.lastCellClicked.col == data.col){
                   handsOnObj.dblClicked = true;
                   setTimeout(function(){
                        handsOnObj = handsOnTables["__identity__"];
                        if(handsOnObj.dblClicked){
                            websocket.send(JSON.stringify(
                                {'event':'onCellDblClick',
                                 'target_cell': '__identity__',
                                 'row': data.row,
                                 'col': data.col
                                 }
                            ));
                        }
                        handsOnObj.lastCellClicked = {row: -100, col: -100};
                        handsOnObj.dblClicked = false;
                    },200);
                } else {
                handsOnObj.lastCellClicked = {row: data.row, col: data.col};
                setTimeout(function(){
                    handsOnObj = handsOnTables["__identity__"];
                    handsOnObj.lastCellClicked = {row: -100, col: -100};
                    handsOnObj.dblClicked = false;
                    },600);
                }
            },
            currentTable
        );
        Handsontable.hooks.add(
            "beforeOnCellContextMenu",
            function(event,data) {
                handsOnObj = handsOnTables["__identity__"];
                handsOnObj.dblClicked = false;
                handsOnObj.lastCellClicked = {row: -100, col: -100};
            },
            currentTable
        );
        Handsontable.hooks.add(
            "beforeContextMenuShow",
            function(event,data) {
                handsOnObj = handsOnTables["__identity__"];
                handsOnObj.dblClicked = false;
                handsOnObj.lastCellClicked = {row: -100, col: -100};
            },
            currentTable
        );

        """ .replace("__identity__", self._identity)

    def recalculate(self):
        self.contents = """
            <div>
                <div id="sheet__identity__" class='handsontable' __style__></div>
                ____error__
            </div>
            """.replace("__style__", self._divStyle()).replace("__identity__", self.identity)

        self.children = {
            '____error__': Subscribed(lambda: Traceback(self.error.get()) if self.error.get() is not None else Text(""))
        }

        self.postscript = ((
            """
            function model(opts) { return {} }

            function property(index) {
              return function (row) {
                return row[index]
              }
            }

            function SyntheticIntegerArray(size) {
                this.length = size
                this.cache = {}
                this.push = function() { }
                this.splice = function() {}

                this.slice = function(low, high) {
                    if (high === undefined) {
                        high = this.length
                    }

                    res = Array(high-low)
                    initLow = low
                    while (low < high) {
                        out = this.cache[low]
                        if (out === undefined) {
                            websocket.send(JSON.stringify(
                                {'event':'sheet_needs_data',
                                 'target_cell': '__identity__',
                                 'data': low
                                 }
                                ))
                            out = emptyRow
                        }
                        res[low-initLow] = out
                        low = low + 1
                    }

                    return res
                }
            }

            var data = new SyntheticIntegerArray(__rows__)
            var container = document.getElementById('sheet__identity__');

            var colnames = [__column_names__]
            var columns = []
            var emptyRow = []

            for (var i = 0; i < colnames.length; i++) {
                columns.push({data: property(i)})
                emptyRow.push("")
            }

            var currentTable = new Handsontable(container, {
                data: data,
                dataSchema: model,
                colHeaders: colnames,
                columns: columns,
                rowHeaders: true,
                rowHeaderWidth: 100,
                viewportRowRenderingOffset: 100,
                autoColumnSize: false,
                autoRowHeight: false,
                manualColumnResize: true,
                colWidths: __col_width__,
                rowHeights: 23,
                readOnly: true,
                ManualRowMove: false
                });
            handsOnTables["__identity__"] = {
                    table: currentTable,
                    lastCellClicked: {row: -100, col:-100},
                    dblClicked: true
            }
            """ +
            (self._addHandsontableOnCellDblClick() if "onCellDblClick" in self._hookfns else "")
            )
            .replace("__identity__", self._identity)
            .replace("__rows__", str(self.rowCount))
            .replace("__column_names__", ",".join('"%s"' % quoteForJs(x, '"') for x in self.columnNames))
            .replace("__col_width__", json.dumps(self.colWidth))
        )


    def onMessage(self, msgFrame):

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
        self.contents = """
            <div>
                <div __style__ id="plot__identity__"></div>
                ____chart_updater__
                ____error__
            </div>
            """.replace("__style__", self._divStyle()).replace("__identity__", self.identity)

        self.children = {
            '____chart_updater__': _PlotUpdater(self),
            '____error__': Subscribed(lambda: Traceback(self.error.get()) if self.error.get() is not None else Text(""))
        }

        self.postscript = """
            console.log("Creating a new plotly chart.")
            plotDiv = document.getElementById('plot__identity__');
            Plotly.plot(
                plotDiv,
                [],
                { margin: {t : 30, l: 30, r: 30, b:30 }
                },
                { scrollZoom: true, dragmode: 'pan', displaylogo: false, displayModeBar: 'hover',
                    modeBarButtons: [ ['pan2d'], ['zoom2d'], ['zoomIn2d'], ['zoomOut2d'] ] }
                );
            plotDiv.on('plotly_relayout',
                function(eventdata){
                    if (plotDiv.is_server_defined_move === true) {
                        return
                    }
                    //if we're sending a string, then its a date object, and we want to send
                    // a timestamp
                    if (typeof(eventdata['xaxis.range[0]']) === 'string') {
                        eventdata = Object.assign({},eventdata)
                        eventdata["xaxis.range[0]"] = Date.parse(eventdata["xaxis.range[0]"]) / 1000.0
                        eventdata["xaxis.range[1]"] = Date.parse(eventdata["xaxis.range[1]"]) / 1000.0
                    }

                    websocket.send(JSON.stringify(
                        {'event':'plot_layout',
                         'target_cell': '__identity__',
                         'data': eventdata
                         }
                        )
                    )
                });

            """.replace("__identity__", self._identity)

    def onMessage(self, msgFrame):
        d = msgFrame['data']
        curVal = self.curXYRanges.get() or ((None, None), (None, None))

        self.curXYRanges.set(
            ((d.get('xaxis.range[0]', curVal[0][0]), d.get('xaxis.range[1]', curVal[0][1])),
             (d.get('yaxis.range[0]', curVal[1][0]), d.get('yaxis.range[1]', curVal[1][1])))
        )

    def setXRange(self, low, high):
        self.curXYRanges.set(((low,high), self.curXYRanges.get()[1]))

        self.triggerPostscript(f"""
            plotDiv = document.getElementById('plot__identity__');
            newLayout = plotDiv.layout

            if (typeof(newLayout.xaxis.range[0]) === 'string') {{
                formatDate = function(d) {{
                    return (d.getYear() + 1900) + "-" + ("00" + (d.getMonth() + 1)).substr(-2) + "-" + ("00" + d.getDate()).substr(-2) + " " + ("00" + d.getHours()).substr(-2) + ":" + ("00" + d.getMinutes()).substr(-2) + ":" + ("00" + d.getSeconds()).substr(-2) + "." + ("000000" + d.getMilliseconds()).substr(-3)
                    }};

                newLayout.xaxis.range[0] = formatDate(new Date({low*1000}));
                newLayout.xaxis.range[1] = formatDate(new Date({high*1000}));
                newLayout.xaxis.autorange = false;

                console.log("set layout for __identity__ to " + formatDate(new Date({low*1000})) + " from " + {low*1000})

            }} else {{
                newLayout.xaxis.range[0] = {low};
                newLayout.xaxis.range[1] = {high};
                newLayout.xaxis.autorange = false;
            }}


            plotDiv.is_server_defined_move = true;
            Plotly.react(plotDiv, plotDiv.data, newLayout);
            plotDiv.is_server_defined_move = false;

            console.log("Range is now " + plotDiv.layout.xaxis.range[0])

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
            res = {'x': [float(x) for x in range(len(data))], 'y': [float(d) for d in data]}
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
            self.children = {}
            self.postscript = ""
            self.linePlot.error.set(None)

            try:
                jsonDataToDraw = self.calculatedDataJson()
                self.postscript = (
                    """
                    plotDiv = document.getElementById('plot__identity__');
                    data = __data__.map(mapPlotlyData)

                    Plotly.react(
                        plotDiv,
                        data,
                        plotDiv.layout,
                        );

                    """
                    .replace("__identity__", self.chartId)
                    .replace("__data__", json.dumps(jsonDataToDraw))
                )
            except SubscribeAndRetry:
                raise
            except Exception:
                self._logger.error(traceback.format_exc())
                self.linePlot.error.set(traceback.format_exc())
                self.postscript = """
                    plotDiv = document.getElementById('plot__identity__');
                    Plotly.purge(plotDiv)
                    """.replace("__identity__", self.chartId)

            self._resetSubscriptionsToViewReads(v)
