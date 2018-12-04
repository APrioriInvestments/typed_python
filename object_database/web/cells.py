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

def quoteForJs(string, quoteType):
    if quoteType == "'":
        return string.replace("\\", "\\\\").replace("'", "\\'")
    else:
        return string.replace("\\", "\\\\").replace('"', '\\"')

def multiReplace(msg, replacements):
    for k,v in replacements.items():
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
        #it's OK that we don't check if the bytes are written because we're just
        #trying to wake up the other side. If the operating system's buffer is full,
        #then that means the other side hasn't been clearing the bytes anyways,
        #and that it will come back around and read our data.
        if self.netChange > 2:
            return

        self.netChange += 1
        os.write(self.write_fd, b"\n")

class Cells:
    def __init__(self, db):
        self.db = db

        self.dirtyNodes = set()
        self.nodesNeedingBroadcast = set()
        self.root = RootCell()

        self.cells = {}
        self.cellsKnownChildren = {}

        self.nodesNeedingBroadcast.add(self.root)
        self.nodesToDiscard = set()

        self.transactionQueue = queue.Queue()
        self.gEventHasTransactions = GeventPipe()
        self.keysToCells = {}

        self._id = 0

        self._addCell(self.root, None)

        self.db._onTransaction.append(self._onTransaction)

    def _newID(self):
        self._id += 1
        return str(self._id)

    def triggerIfHasDirty(self):
        if self.dirtyNodes:
            self.gEventHasTransactions.trigger()

    def _onTransaction(self, *trans):
        self.transactionQueue.put(trans)
        self.gEventHasTransactions.trigger()

    def _handleTransaction(self, key_value, priors, set_adds, set_removes, transactionId):
        for k in list(key_value) + list(set_adds) + list(set_removes):
            if k in self.keysToCells:
                toDrop = []
                for cell in self.keysToCells[k]:
                    if not cell.garbageCollected:
                        cell.markDirty()
                    else:
                        toDrop.append(cell)
                for cell in toDrop:
                    self.keysToCells[k].discard(cell)
                if not self.keysToCells[k]:
                    del self.keysToCells[k]

    def _addCell(self, cell, parent):
        assert isinstance(cell, Cell)
        assert cell.cells is None

        cell.cells = self
        cell.parent = parent
        cell.level = parent.level + 1 if parent else 0

        self.cellsKnownChildren[cell.identity] = set()

        assert cell.identity not in self.cells
        self.cells[cell.identity] = cell

        self.dirtyNodes.add(cell)

        self.markForBroadcast(cell)

    def _cellOutOfScope(self, cell):
        for c in cell.children.values():
            self._cellOutOfScope(c)

        self.nodesToDiscard.add(cell)
        if cell.cells is not None:
            del self.cells[cell.identity]
            del self.cellsKnownChildren[cell.identity]

        cell.garbageCollected = True

    def markForBroadcast(self, node):
        assert node.cells is self

        self.nodesNeedingBroadcast.add(node)

    def renderMessages(self):
        self.recalculate()

        res = []

        cellsByLevel = {}

        for n in self.nodesNeedingBroadcast:
            if n not in self.nodesToDiscard:
                if n.level not in cellsByLevel:
                    cellsByLevel[n.level] = set()
                cellsByLevel[n.level].add(n)

        for level, cells in reversed(sorted(cellsByLevel.items())):
            for n in cells:
                res.append(self.updateMessageFor(n))

        for n in self.nodesToDiscard:
            if n.cells is not None:
                res.append({'id': n.identity, 'discard': True})

        self.nodesNeedingBroadcast = set()
        self.nodesToDiscard = set()

        return res

    def recalculate(self):
        #handle all the transactions so far
        old_queue = self.transactionQueue
        self.transactionQueue = queue.Queue()

        try:
            while True:
                self._handleTransaction(*old_queue.get_nowait())
        except queue.Empty:
            pass

        while self.dirtyNodes:
            n = self.dirtyNodes.pop()

            if not n.garbageCollected:
                self.markForBroadcast(n)

                origChildren = self.cellsKnownChildren[n.identity]

                try:
                    _cur_cell.cell = n
                    while True:
                        try:
                            n.recalculate()
                            break
                        except SubscribeAndRetry as e:
                            e.callback(self.db)
                except:
                    logging.error("Node %s had exception during recalculation:\n%s", n, traceback.format_exc())
                    logging.error("Subscribed cell threw an exception:\n%s", traceback.format_exc())
                    n.children = {'____contents__': Traceback(traceback.format_exc())}

                finally:
                    _cur_cell.cell = None

                newChildren = set(n.children.values())

                for c in newChildren.difference(origChildren):
                    assert isinstance(c, Cell)

                    if c.cells is not None and not c.prepareForReuse():
                        for k,v in n.children.items():
                            if v is c:
                                c = n.children[k] = Traceback("Cell %s being used twice in the tree" % c)

                    self._addCell(c, n)

                for c in origChildren.difference(newChildren):
                    self._cellOutOfScope(c)

                self.cellsKnownChildren[n.identity] = newChildren

    def markDirty(self, cell):
        assert not cell.garbageCollected

        self.dirtyNodes.add(cell)

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
        except:
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
    """Holds some arbitrary state for use in a session. Not mirrored in the DB."""
    def __init__(self, value=None):
        self._value = value
        self._subscribedCells = set()

    def setter(self, val):
        return lambda: self.set(val)

    def get(self):
        if _cur_cell.cell:
            self._subscribedCells.add(_cur_cell.cell)
        return self._value

    def set(self, val):
        if val != self._value:
            self._value = val
            for c in self._subscribedCells:
                c.markDirty()
            self._subscribedCells = set()

class Cell:
    def __init__(self):
        self.cells = None #will get set when its added to a 'Cells' object
        self.parent = None
        self.level = None
        self.children = {} #local node def to global node def
        self.contents = "" #some contents containing a local node def
        self._identity = None
        self.postscript = None
        self.garbageCollected = False
        self.subscriptions  = set()
        self._nowrap = False
        self._width = None
        self._color = None
        self._height = None

    def _resetSubscriptionsToViewReads(self, view):
        new_subscriptions = set(view._reads).union(set(view._indexReads))

        for k in new_subscriptions.difference(self.subscriptions):
            self.cells.keysToCells.setdefault(k, set()).add(self)

        for k in self.subscriptions.difference(new_subscriptions):
            self.cells.keysToCells.setdefault(k, set()).discard(self)

        self.subscriptions = new_subscriptions

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
            res.append("width:%spx" % self._width)

        if self._height is not None:
            res.append("height:%spx" % self._height)

        if self._color is not None:
            res.append("color:%s" % self._color)

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

    def height(self, height):
        self._height = height
        return self

    def color(self, color):
        self._color = color
        return self

    def prepareForReuse(self):
        if not self.garbageCollected:
            return False

        self.cells = None
        self.postscript = None
        self.garbageCollected = False
        self._identity = None
        self.parent = None

        return True

    @property
    def identity(self):
        if self._identity is None:
            assert self.cells is not None, "Can't ask for identity for %s as it's not part of a cells package" % self
            self._identity = self.cells._newID()
        return self._identity

    def markDirty(self):
        if not self.garbageCollected:
            self.cells.markDirty(self)

    def recalculate(self):
        pass

    @staticmethod
    def makeCell(x):
        if isinstance(x,(str, float, int, bool)):
            return Text(str(x), x)
        if x is None:
            return Span("")
        if isinstance(x, Cell):
            return x
        assert False, "don't know what to do with %s" % x

    def __add__(self, other):
        return Sequence([self, Cell.makeCell(other)])

class Card(Cell):
    def __init__(self, inner, padding=None):
        super().__init__()

        self.children = {"____contents__": Cell.makeCell(inner)}

        other = ""
        if padding:
            other += " p-" + str(padding)

        self.contents = """
        <div class='card'>
          <div class="card-body __other__">
            ____contents__
          </div>
        </div>
        """.replace('__other__', other)

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
        self.contents = "<div %s>" % self._divStyle() + "\n".join("____c_%s__" % i for i in range(len(elements))) + "</div>"

    def __add__(self, other):
        other = Cell.makeCell(other)
        if isinstance(other, Sequence):
            return Sequence(self.elements + other.elements)
        else:
            return Sequence(self.elements + [other])

    def sortsAs(self):
        if self.elements:
            return self.elements[0].sortsAs()
        return None

class HeaderBar(Cell):
    def __init__(self, navbarItems):
        super().__init__()

        self.contents = """
            <div class="p-2 bg-light mr-auto tl-navbar">
                %s
            </div>
        """ % "".join(["<span class='tl-navbar-item px-2'>____child_%s__</span>" % i for i in range(len(navbarItems))])

        self.children = {'____child_%s__' % i: navbarItems[i] for i in range(len(navbarItems))}

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

    def recalculate(self):
        items = []

        self.children['____display__'] = Subscribed(lambda: self.headersAndChildren[self.whichSlot.get()][1])

        for i in range(len(self.headersAndChildren)):
            self.children['____header_{ix}__'.format(ix=i)] = _NavTab(self.whichSlot, i, self._identity, self.headersAndChildren[i][0])

        self.contents = """
                <div class="container-fluid mb-3">
                     <ul class="nav nav-tabs" role="tablist">
                      __items__
                    </ul>
                    <div class="tab-content">
                      <div class="tab-pane fade show active" role="tabpanel">____display__
                    </div>
                </div>
                """.replace(
                    "__items__",
                    "".join(
                        """ ____header___ix____ """.replace('__ix__', str(i))
                            for i in range(len(self.headersAndChildren))
                        )
                    ).replace("__identity__", self._identity)

    def onMessage(self, msgFrame):
        self.whichSlot.set(int(msgFrame['ix']))

class Dropdown(Cell):
    def __init__(self, title, headersAndLambdas, singleLambda=None):
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
        t0 = time.time()
        tries = 0
        fun = self.headersAndLambdas[msgFrame['ix']][1]

        while True:
            try:
                with self.cells.db.transaction() as t:
                    fun()
                    return
            except RevisionConflictException as e:
                tries += 1
                if tries > MAX_TRIES or time.time() - t0 > MAX_TIMEOUT:
                    logging.error("Button click timed out. This should really fail.")
                    return
            except:
                logging.error("Exception in button logic:\n%s", traceback.format_exc())
                return

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

class RootCell(Container):
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

class Subscribed(Cell):
    def __init__(self, f):
        super().__init__()

        self.f = f

    def prepareForReuse(self):
        self.subscriptions = set()
        return super().prepareForReuse()

    def __repr__(self):
        return "Subscribed(%s)" % self.f

    def sortsAs(self):
        for c in self.children.values():
            return c.sortsAs()

        return Cell.makeCell(self.f()).sortsAs()

    def recalculate(self):
        with self.cells.db.view() as v:
            self.contents = """<div %s>____contents__</div>""" % self._divStyle()
            try:
                c = Cell.makeCell(self.f())
                if c.cells is not None:
                    c.prepareForReuse()
                self.children = {'____contents__': c}
            except SubscribeAndRetry:
                raise
            except:
                self.children = {'____contents__': Traceback(traceback.format_exc())}

            self._resetSubscriptionsToViewReads(v)

class SubscribedSequence(Cell):
    def __init__(self, itemsFun, rendererFun):
        super().__init__()

        self.itemsFun = itemsFun
        self.rendererFun = rendererFun

        self.subscriptions  = set()

        self.existingItems = {}
        self.spine = []

    def prepareForReuse(self):
        self.subscriptions = set()
        self.existingItems = {}
        self.spine = []
        return super().prepareForReuse()

    def sortsAs(self):
        if '____child_0__' in self.children:
            return self.children['____child_0__'].sortsAs()

    def recalculate(self):
        with self.cells.db.view() as v:
            try:
                self.spine = list(self.itemsFun())
            except SubscribeAndRetry:
                raise
            except:
                logging.error("Spine calc threw an exception:\n%s", traceback.format_exc())
                self.spine = []

            self._resetSubscriptionsToViewReads(v)

            new_children = {}
            for ix, s in enumerate(self.spine):
                if s in self.existingItems:
                    new_children["____child_%s__" % ix] = self.existingItems[s]
                else:
                    try:
                        self.existingItems[s] = new_children["____child_%s__" % ix] = self.rendererFun(s)
                    except SubscribeAndRetry:
                        raise
                    except:
                        self.existingItems[s] = new_children["____child_%s__" % ix] = Traceback(traceback.format_exc())

        self.children = new_children

        spineAsSet = set(self.spine)
        for i in list(self.existingItems):
            if i not in spineAsSet:
                del self.existingItems[i]

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

        self.subscriptions  = set()

        self.existingItems = {}
        self.rows = []
        self.cols = []

    def prepareForReuse(self):
        self.subscriptions = set()
        self.existingItems = []
        self.rows = []
        self.cols = []
        super().prepareForReuse()

    def recalculate(self):
        with self.cells.db.view() as v:
            try:
                self.rows = list(self.rowFun())
            except SubscribeAndRetry:
                raise
            except:
                logging.error("Row fun calc threw an exception:\n%s", traceback.format_exc())
                self.rows = []
            try:
                self.cols = list(self.colFun())
            except SubscribeAndRetry:
                raise
            except:
                logging.error("Col fun calc threw an exception:\n%s", traceback.format_exc())
                self.cols = []

            self._resetSubscriptionsToViewReads(v)

        new_children = {}
        seen = set()

        for col_ix, col in enumerate(self.cols):
            seen.add((None, col))
            if (None,col) in self.existingItems:
                new_children["____header_%s__" % (col_ix)] = self.existingItems[(None,col)]
            else:
                try:
                    self.existingItems[(None,col)] = new_children["____header_%s__" % col_ix] = Cell.makeCell(self.headerFun(col))
                except SubscribeAndRetry:
                    raise
                except:
                    self.existingItems[(None,col)] = new_children["____header_%s__" % col_ix] = Traceback(traceback.format_exc())

        if self.rowLabelFun is not None:
            for row_ix, row in enumerate(self.rows):
                seen.add((None, row))
                if (row, None) in self.existingItems:
                    new_children["____rowlabel_%s__" % (row_ix)] = self.existingItems[(row, None)]
                else:
                    try:
                        self.existingItems[(row, None)] = new_children["____rowlabel_%s__" % row_ix] = Cell.makeCell(self.rowLabelFun(row))
                    except SubscribeAndRetry:
                        raise
                    except:
                        self.existingItems[(row, None)] = new_children["____rowlabel_%s__" % row_ix] = Traceback(traceback.format_exc())

        seen = set()
        for row_ix, row in enumerate(self.rows):
            for col_ix, col in enumerate(self.cols):
                seen.add((row,col))
                if (row,col) in self.existingItems:
                    new_children["____child_%s_%s__" % (row_ix, col_ix)] = self.existingItems[(row,col)]
                else:
                    try:
                        self.existingItems[(row,col)] = new_children["____child_%s_%s__" % (row_ix, col_ix)] = Cell.makeCell(self.rendererFun(row,col))
                    except SubscribeAndRetry:
                        raise
                    except:
                        self.existingItems[(row,col)] = new_children["____child_%s_%s__" % (row_ix, col_ix)] = Traceback(traceback.format_exc())

        self.children = new_children

        for i in list(self.existingItems):
            if i not in seen:
                del self.existingItems[i]

        self.contents = """
            <table class="table-hscroll table-sm table-striped">
            <thead><tr>""" + ("<th></th>" if self.rowLabelFun is not None else "") + """__headers__</tr></thead>
            <tbody>
            __rows__
            </tbody>
            </table>
            """.replace("__headers__",
                "".join("<th>____header_%s__</th>" % (col_ix)
                            for col_ix in range(len(self.cols)))
                ).replace("__rows__",
                "\n".join("<tr>" +
                    ("<td>____rowlabel_%s__</td>" % row_ix if self.rowLabelFun is not None else "") +
                    "".join(
                        "<td>____child_%s_%s__</td>" % (row_ix, col_ix)
                            for col_ix in range(len(self.cols))
                        )
                    + "</tr>"
                        for row_ix in range(len(self.rows))
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
        except:
            try:
                return str(self.x) < str(self.other)
            except:
                return False

    def __eq__(self, other):
        try:
            if type(self.x) is type(other.x):
                return self.x == other.x
            else:
                return str(type(self.x)) == str(type(other.x))
        except:
            try:
                return str(self.x) == str(self.other)
            except:
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
               .replace("__contents__", quoteForJs(self.slot.get(),'"'))
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

        self.subscriptions  = set()

        self.existingItems = {}
        self.rows = []
        self.cols = []

        self.maxRowsPerPage = maxRowsPerPage

        self.curPage = Slot("1")
        self.sortColumn = Slot(None)
        self.sortColumnAscending = Slot(True)
        self.columnFilters = {}

    def prepareForReuse(self):
        self.subscriptions = set()
        self.existingItems = {}
        self.rows = []
        self.cols = []
        super().prepareForReuse()

    def cachedRenderFun(self, row, col):
        if (row, col) in self.existingItems:
            return self.existingItems[row,col]
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
                    except:
                        logging.error(traceback.format_exc())
                        keymemo[row] = SortWrapper(None)

                return keymemo[row]

            rows = sorted(rows,key=key)

            if not self.sortColumnAscending.get():
                rows = list(reversed(rows))

        page = 0
        try:
            page = max(0, int(self.curPage.get())-1)
            page = min(page, (len(rows) - 1) // self.maxRowsPerPage)
        except:
            logging.error("Failed to parse current page: %s", traceback.format_exc())

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
        with self.cells.db.view() as v:
            try:
                self.cols = list(self.colFun())
            except SubscribeAndRetry:
                raise
            except:
                logging.error("Col fun calc threw an exception:\n%s", traceback.format_exc())
                self.cols = []

            try:
                self.unfilteredRows = list(self.rowFun())
                self.filteredRows = self.filterRows(self.unfilteredRows)
                self.rows = self.sortRows(self.filteredRows)

            except SubscribeAndRetry:
                raise
            except:
                logging.error("Row fun calc threw an exception:\n%s", traceback.format_exc())
                self.rows = []

            self._resetSubscriptionsToViewReads(v)

        new_children = {}
        seen = set()

        for col_ix, col in enumerate(self.cols):
            seen.add((None, col))
            if (None,col) in self.existingItems:
                new_children["____header_%s__" % (col_ix)] = self.existingItems[(None,col)]
            else:
                try:
                    self.existingItems[(None,col)] = new_children["____header_%s__" % col_ix] = self.makeHeaderCell(col_ix)
                except SubscribeAndRetry:
                    raise
                except:
                    self.existingItems[(None,col)] = new_children["____header_%s__" % col_ix] = Traceback(traceback.format_exc())

        seen = set()
        for row_ix, row in enumerate(self.rows):
            for col_ix, col in enumerate(self.cols):
                seen.add((row,col))
                if (row,col) in self.existingItems:
                    new_children["____child_%s_%s__" % (row_ix, col_ix)] = self.existingItems[(row,col)]
                else:
                    try:
                        self.existingItems[(row,col)] = new_children["____child_%s_%s__" % (row_ix, col_ix)] = Cell.makeCell(self.rendererFun(row,col))
                    except SubscribeAndRetry:
                        raise
                    except:
                        self.existingItems[(row,col)] = new_children["____child_%s_%s__" % (row_ix, col_ix)] = Traceback(traceback.format_exc())

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

        self.contents = ("""
            <table class="table-hscroll table-sm table-striped">
            <thead style="border-bottom: black;border-bottom-style:solid;border-bottom-width:thin;""><tr>""" +
                ('<th style="vertical-align:top"><div class="card"><div class="card-body p-1">%s</div></div></th>' % rowDisplay) + """__headers__</tr></thead>
            <tbody>
            __rows__
            </tbody>
            </table>
            """.replace("__headers__",
                "".join('<th style="vertical-align:top">____header_%s__</th>' % (col_ix)
                            for col_ix in range(len(self.cols)))
                ).replace("__rows__",
                "\n".join("<tr>" +
                    ("<td>%s</td>" % (row_ix+1)) +
                    "".join(
                        "<td>____child_%s_%s__</td>" % (row_ix, col_ix)
                            for col_ix in range(len(self.cols))
                        )
                    + "</tr>"
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

        self.contents = """
            <div onclick="__onclick__" __style__>
            ____contents__
            </div>""".replace(
                '__onclick__', self.calculatedOnClick()
                ).replace(
                '__style__',
                self._divStyle("cursor:pointer;*cursor: hand" + (";font-weight:bold" if self.bold else ""))
                )

    def sortsAs(self):
        return self.content.sortsAs()

    def onMessage(self, msgFrame):
        t0 = time.time()
        tries = 0

        while True:
            try:
                with self.cells.db.transaction():
                    self.f()
                    return
            except RevisionConflictException as e:
                tries += 1
                if tries > MAX_TRIES or time.time() - t0 > MAX_TIMEOUT:
                    logging.error("Button click timed out. This should really fail.")
                    return
            except:
                logging.error("Exception in button logic:\n%s", traceback.format_exc())
                return

class Button(Clickable):
    def __init__(self, *args, small=False, **kwargs):
        Clickable.__init__(self, *args, **kwargs)
        self.small = small

    def recalculate(self):
        self.children = {'____contents__': self.content}
        self.contents = ("""
            <button
                class='btn btn-primary __size__'
                onclick="__onclick__"
                >
            ____contents__
            </button>"""
            .replace("__size__", "" if not self.small else "btn-xs")
            .replace('__identity__', self.identity)
            .replace("__onclick__", self.calculatedOnClick())
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

def ensureSubscribedSchema(t, lazy=False):
    if not current_transaction().db().isSubscribedToSchema(t):
        raise SubscribeAndRetry(
            Timer("Subscribing to schema %s%s", t, " lazily" if lazy else "")(
                lambda db: db.subscribeToSchema(t, lazySubscription=lazy)
                )
            )

class Expands(Cell):
    def __init__(self, closed, open, closedIcon=Octicon("diff-added)"), openedIcon=Octicon("diff-removed"), initialState=False):
        super().__init__()
        self.isExpanded = initialState
        self.closed = closed
        self.open = open
        self.openedIcon = openedIcon
        self.closedIcon = closedIcon

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

    def onMessage(self, msgFrame):
        self.isExpanded = not self.isExpanded
        self.markDirty()

class Plot(Cell):
    """Produce some reactive line plots."""
    def __init__(self, namedDataSubscriptions):
        """Initialize a line plot.

        namedDataSubscriptions: a map from plot name to a lambda function
            producing either an array, or {x: array, y: array}
        """
        super().__init__()

        self.namedDataSubscriptions = namedDataSubscriptions
        self.curXYRanges = Slot(None)

    def recalculate(self):
        self.contents = """
            <div __style__></div>
            ____chart_updater__

            """.replace("__style__", self._divStyle())

        self.children = {
            '____chart_updater__': _PlotUpdater(self)
            }

        self.postscript = """
            plotDiv = document.getElementById('__identity__');
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
        curVal = self.curXYRanges.get() or ((None, None), (None,None))

        self.curXYRanges.set(
            ((d.get('xaxis.range[0]', curVal[0][0]), d.get('xaxis.range[1]',curVal[0][1])),
             (d.get('yaxis.range[0]', curVal[1][0]), d.get('yaxis.range[1]',curVal[1][1])))
            )

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
            return [self.processSeries(callableOrData, name) for name, callableOrData in
                        series.items()]
        else:
            return [self.processSeries(callableOrData, None) for callableOrData in
                        series]

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

            for k,v in res.items():
                if isinstance(v, numpy.ndarray):
                    res[k] = v.astype('float64').tostring().hex()

        if name is not None:
            res['name'] = name

        return res

    def recalculate(self):
        with self.cells.db.view() as v:
            #we only exist to run our postscript
            self.contents = """<div style='display:none'>"""

            try:
                jsonDataToDraw = self.calculatedDataJson()

                self.postscript = """
                    plotDiv = document.getElementById('__identity__');
                    data = __data__.map(mapPlotlyData)

                    Plotly.react(
                        plotDiv,
                        data,
                        plotDiv.layout,
                        );

                    """.replace("__identity__", self.chartId).replace("__data__",
                        json.dumps(jsonDataToDraw)
                        )
            except SubscribeAndRetry:
                raise
            except:
                logging.error(traceback.format_exc())
                self.contents = """<div>____contents__</div>"""
                self.children = {'____contents__': Traceback(traceback.format_exc())}

            self._resetSubscriptionsToViewReads(v)
