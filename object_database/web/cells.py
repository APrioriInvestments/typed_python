import uuid
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

from object_database.view import RevisionConflictException 
from object_database.view import current_transaction

MAX_TIMEOUT = 1.0
MAX_TRIES = 10
MAX_FPS = 10

_cur_cell = threading.local()

class GeventPipe:
    """A simple mechanism for triggering the gevent webserver from a thread other than
    the webserver thread. Gevent itself expects everything to happen on greenelts. The
    database connection in the background is not based on gevent, so we cannot use any
    standard gevent-based event or queue objects from the db-trigger thread.
    """
    def __init__(self):
        self.read_fd, self.write_fd = os.pipe()
        self.write_f = os.fdopen(self.write_fd, "w")
        self.fileobj = gevent.fileobject.FileObjectPosix(self.read_fd, bufsize=0)

    def wait(self):
        self.fileobj.readline()

    def trigger(self):
        self.write_f.write("\n")
        self.write_f.flush()

class Cells:
    def __init__(self, db):
        self.db = db

        self.dirtyNodes = set()
        self.nodesNeedingBroadcast = set()
        self.root = RootCell()

        self.cells = {self.root.identity: self.root}
        self.nodesNeedingBroadcastOrder = [self.root]
        self.nodesNeedingBroadcast.add(self.root)

        self.transactionQueue = queue.Queue()
        self.db._onTransaction.append(self._onTransaction)
        self.gEventHasTransactions = GeventPipe()
        self.keysToCells = {}

        self.addCell(self.root)

    def triggerIfHasDirty(self):
        if self.dirtyNodes:
            self.gEventHasTransactions.trigger()

    def _onTransaction(self, *trans):
        self.transactionQueue.put(trans)
        self.gEventHasTransactions.trigger()

    def _handleTransaction(self, key_value, priors, set_adds, set_removes, transactionId):
        for k in list(key_value) + list(set_adds) + list(set_removes):
            for cell in self.keysToCells.get(k) or []:
                cell.markDirty()

    def addCell(self, cell):
        assert isinstance(cell, Cell)
        assert cell.cells is self or cell.cells is None, 'cell %s has cells %s != %s' % (cell, cell.cells, self)

        cell.cells = self

        if cell.identity in self.cells:
            assert self.cells[cell.identity] is cell
        else:
            self.cells[cell.identity] = cell
            self.dirtyNodes.add(cell)
            
            self.markForBroadcast(cell)

            for child in cell.children.values():
                assert isinstance(child, Cell), "Child of %s is %s, not a cell" % (cell, child)
                self.addCell(child)

                self.markForBroadcast(child)

    def markForBroadcast(self, node):
        if node not in self.nodesNeedingBroadcast:
            self.nodesNeedingBroadcast.add(node)
            self.nodesNeedingBroadcastOrder.append(node)

    def renderMessages(self):
        self.recalculate()

        res = []

        for n in self.nodesNeedingBroadcastOrder:
            res.append(self.updateMessageFor(n))

        self.nodesNeedingBroadcast = set()
        self.nodesNeedingBroadcastOrder = []

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
            
            self.markForBroadcast(n)

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
            finally:
                _cur_cell.cell = None

            for c in n.children.values():
                assert isinstance(c, Cell)
                self.addCell(c)

    def markDirty(self, cell):
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
            for k,v in formatArgs.items():
                contents = contents.replace(k, v)
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

    def get(self):
        self._subscribedCells.add(_cur_cell.cell)
        return self._value

    def set(self, val):
        if val != self._value:
            self._value = val
            for c in self._subscribedCells:
                c.markDirty()
            self._subscribedCells = set()

class Cell:
    def __init__(self, cells=None):
        self.cells = cells #will get set when its added to a 'Cells' object
        self.children = {} #local node def to global node def
        self.contents = "" #some contents containing a local node def
        self._identity = None
        self.postscript = None

    @property
    def identity(self):
        if self._identity is None:
            self._identity = str(uuid.uuid4()).replace("-","")
        return self._identity

    def markDirty(self):
        self.cells.markDirty(self)

    def recalculate(self):
        pass

    @staticmethod
    def makeCell(x):
        if isinstance(x,(str, float, int, bool)):
            return Text(str(x))
        if x is None:
            return Span("")
        if isinstance(x, Cell):
            return x
        assert False, "don't know what to do with %s" % x

    def __add__(self, other):
        return Sequence([self, Cell.makeCell(other)])

class Card(Cell):
    def __init__(self, inner):
        super().__init__()

        self.children = {"__contents__": Cell.makeCell(inner)}
        self.contents = """
        <div class='card'>
          <div class="card-body">
            __contents__
          </div>
        </div>
        """

class Octicon(Cell):
    def __init__(self, which):
        super().__init__()
        self.contents = '<span class="octicon octicon-%s" aria-hidden="true"></span>' % which

class Text(Cell):
    def __init__(self, text):
        super().__init__()
        self.contents = "<div>%s</div>" % cgi.escape(str(text))

class Span(Cell):
    def __init__(self, text):
        super().__init__()
        self.contents = "<span>%s</span>" % cgi.escape(str(text))

class Sequence(Cell):
    def __init__(self, elements):
        super().__init__()
        elements = [Cell.makeCell(x) for x in elements]

        self.elements = elements
        self.children = {"__c_%s__" % i: elements[i] for i in range(len(elements)) }
        self.contents = "<div>" + "\n".join("__c_%s__" % i for i in range(len(elements))) + "</div>"

    def __add__(self, other):
        other = Cell.makeCell(other)
        if isinstance(other, Sequence):
            return Sequence(self.elements + other.elements)
        else:
            return Sequence(self.elements + [other])

class HeaderBar(Cell):
    def __init__(self, navbarItems):
        super().__init__()

        self.contents = """
            <div class="p-2 bg-light mr-auto tl-navbar">
                %s
            </div>
        """ % "".join(["<span class='tl-navbar-item px-2'>__child_%s__</span>" % i for i in range(len(navbarItems))])

        self.children = {'__child_%s__' % i: navbarItems[i] for i in range(len(navbarItems))}

class Main(Cell):
    def __init__(self, child):
        super().__init__()

        self.contents = """
            <main class="py-md-2">
            <div class="container-fluid">
                __child__
            </div>
            </main>
        """
        self.children = {'__child__': child}

class Dropdown(Cell):
    def __init__(self, title, headersAndLambdas):
        super().__init__()

        self.headersAndLambdas = headersAndLambdas
        self.title = title

        items = []
        
        for i in range(len(headersAndLambdas)):
            header, _ = headersAndLambdas[i]
            self.children["__child_%s__" % i] = Cell.makeCell(header)

            items.append(
                """
                    <a class='dropdown-item' 
                        onclick="websocket.send(JSON.stringify({'event':'menu', 'ix': __ix__, 'target_cell': '__identity__'}))"
                        >
                    __child___ix____
                    </a>
                """.replace("__ix__", str(i)).replace("__identity__", self.identity)
                )

        self.contents = """
            <div class="btn-group">
                  <a role="button" class="btn btn-xs btn-outline-secondary" title="__title__">__title__</a>
                  <button class="btn btn-xs btn-outline-secondary dropdown-toggle dropdown-toggle-split" type="button" 
                        id="___identity__-dropdownMenuButton" data-toggle="dropdown">
                  </button>
                  <div class="dropdown-menu">
                    __dropdown_items__
                  </div>
            </div>
            """.replace("__title__", title).replace("__identity__", self.identity).replace("__dropdown_items__", "\n".join(items))

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

class Container(Cell):
    def __init__(self, child=None):
        super().__init__()
        if child is None:
            self.contents = ""
            self.children = {}
        else:
            self.contents = "<div>__child__</div>"
            self.children = {"__child__": Cell.makeCell(child)}

    def setChild(self, child):
        self.setContents("<div>__child__</div>", {"__child__": Cell.makeCell(child)})

    def setContents(self, newContents, newChildren):
        self.contents = newContents
        self.children = newChildren
        self.markDirty()

class RootCell(Container):
    @property
    def identity(self):
        return "page_root"

    def setChild(self, child):
        self.setContents("<div>__c__</div>", {"__c__": child})

class Traceback(Cell):
    def __init__(self, traceback):
        super().__init__()
        self.contents = """<div class='alert alert-primary'><pre>__child__</pre></alert>"""
        self.children = {"__child__": Cell.makeCell(traceback)}

class Subscribed(Cell):
    def __init__(self, f):
        super().__init__()

        self.f = f

        self.subscriptions  = set()

        self.context = None

    def recalculate(self):
        with self.cells.db.view() as v:
            self.contents = """<div>__contents__</div>"""
            try:
                self.children = {'__contents__': Cell.makeCell(self.f())}
            except SubscribeAndRetry:
                raise
            except:
                logging.error("Subscribed cell threw an exception:\n%s", traceback.format_exc())
                self.children = {'__contents__': Traceback(traceback.format_exc())}

            new_subscriptions = set(v._reads).union(set(v._indexReads))

            for k in new_subscriptions.difference(self.subscriptions):
                self.cells.keysToCells.setdefault(k, set()).add(self)

            for k in self.subscriptions.difference(new_subscriptions):
                self.cells.keysToCells.setdefault(k, set()).discard(self)

            self.subscriptions = new_subscriptions



class SubscribedSequence(Cell):
    def __init__(self, itemsFun, rendererFun):
        super().__init__()

        self.itemsFun = itemsFun
        self.rendererFun = rendererFun

        self.subscriptions  = set()

        self.existingItems = {}
        self.spine = []

    def recalculate(self):
        with self.cells.db.view() as v:
            try:
                self.spine = list(self.itemsFun())
            except SubscribeAndRetry:
                raise
            except:
                logging.error("Spine calc threw an exception:\n%s", traceback.format_exc())
                self.spine = []

            new_subscriptions = set(v._reads).union(set(v._indexReads))

            for k in new_subscriptions.difference(self.subscriptions):
                self.cells.keysToCells.setdefault(k, set()).add(self)

            for k in self.subscriptions.difference(new_subscriptions):
                self.cells.keysToCells.setdefault(k, set()).discard(self)

            self.subscriptions = new_subscriptions

        new_children = {}
        for ix, s in enumerate(self.spine):
            if s in self.existingItems:
                new_children["__child_%s__" % ix] = self.existingItems[s]
            else:
                try:
                    self.existingItems[s] = new_children["__child_%s__" % ix] = self.rendererFun(s)
                except SubscribeAndRetry:
                    raise
                except:
                    self.existingItems[s] = new_children["__child_%s__" % ix] = Traceback(traceback.format_exc())
        
        self.children = new_children

        spineAsSet = set(self.spine)
        for i in list(self.existingItems):
            if i not in spineAsSet:
                del self.existingItems[i]

        self.contents = """<div>%s</div>""" % "\n".join(['__child_%s__' % i for i in range(len(self.spine))])

class Popover(Cell):
    def __init__(self, contents, title, detail, width=400):
        super().__init__()
        self.children = {
            '__contents__': Cell.makeCell(contents),
            '__detail__': Cell.makeCell(detail),
            '__title__': Cell.makeCell(title)
            }

        self.contents = """
            <div>
            <a href="#popmain___identity__" data-toggle="popover" data-trigger="focus" data-bind="#pop___identity__" container="body" class="btn btn-xs" role="button">__contents__</a>
            <div style="display:none;">
              <div id="pop___identity__">
                <div class='data-placement'>bottom</div>
                <div class="data-title">__title__</div>
                <div class="data-content"><div style="width:__width__px">__detail__</div></div>
              </div>
            </div>

            </div>
            """.replace("__identity__", self.identity).replace("__width__", str(width))

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

            new_subscriptions = set(v._reads).union(set(v._indexReads))

            for k in new_subscriptions.difference(self.subscriptions):
                self.cells.keysToCells.setdefault(k, set()).add(self)

            for k in self.subscriptions.difference(new_subscriptions):
                self.cells.keysToCells.setdefault(k, set()).discard(self)

            self.subscriptions = new_subscriptions

        new_children = {}
        seen = set()

        for col_ix, col in enumerate(self.cols):
            seen.add((None, col))
            if (None,col) in self.existingItems:
                new_children["__header_%s__" % (col_ix)] = self.existingItems[(None,col)]
            else:
                try:
                    self.existingItems[(None,col)] = new_children["__header_%s__" % col_ix] = Cell.makeCell(self.headerFun(col))
                except SubscribeAndRetry:
                    raise
                except:
                    self.existingItems[(None,col)] = new_children["__header_%s__" % col_ix] = Traceback(traceback.format_exc())
        
        if self.rowLabelFun is not None:
            for row_ix, row in enumerate(self.rows):
                seen.add((None, row))
                if (row, None) in self.existingItems:
                    new_children["__rowlabel_%s__" % (row_ix)] = self.existingItems[(row, None)]
                else:
                    try:
                        self.existingItems[(row, None)] = new_children["__rowlabel_%s__" % row_ix] = Cell.makeCell(self.rowLabelFun(row))
                    except SubscribeAndRetry:
                        raise
                    except:
                        self.existingItems[(row, None)] = new_children["__rowlabel_%s__" % row_ix] = Traceback(traceback.format_exc())
        
        seen = set()
        for row_ix, row in enumerate(self.rows):
            for col_ix, col in enumerate(self.cols):
                seen.add((row,col))
                if (row,col) in self.existingItems:
                    new_children["__child_%s_%s__" % (row_ix, col_ix)] = self.existingItems[(row,col)]
                else:
                    try:
                        self.existingItems[(row,col)] = new_children["__child_%s_%s__" % (row_ix, col_ix)] = Cell.makeCell(self.rendererFun(row,col))
                    except SubscribeAndRetry:
                        raise
                    except:
                        self.existingItems[(row,col)] = new_children["__child_%s_%s__" % (row_ix, col_ix)] = Traceback(traceback.format_exc())
            
        self.children = new_children

        for i in list(self.existingItems):
            if i not in seen:
                del self.existingItems[i]

        self.contents = """
            <table class="table-hscroll table-sm table-striped">
            <tr>""" + ("<th></th>" if self.rowLabelFun is not None else "") + """__headers__</tr>
            __rows__
            </table>
            """.replace("__headers__",
                "".join("<th>__header_%s__</th>" % (col_ix)
                            for col_ix in range(len(self.cols)))
                ).replace("__rows__", 
                "\n".join("<tr>" + 
                    ("<td>__rowlabel_%s__</td>" % row_ix if self.rowLabelFun is not None else "") + 
                    "".join(
                        "<td>__child_%s_%s__</td>" % (row_ix, col_ix)
                            for col_ix in range(len(self.cols))
                        )
                    + "</tr>"
                        for row_ix in range(len(self.rows))
                    )
                )

class Clickable(Cell):
    def __init__(self, content, f):
        super().__init__()

        self.children = {'__contents__': Cell.makeCell(content)}
        self.contents = """
            <div onclick="websocket.send(JSON.stringify({'event':'click', 'target_cell': '__identity__'}))">
            __contents__
            </div>""".replace('__identity__', self.identity)

        self.f = f

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


class Button(Clickable):
    def __init__(self, content, f):
        super().__init__(content, f)

        self.contents = """
            <button 
                class='btn btn-primary' 
                onclick="websocket.send(JSON.stringify({'event':'click', 'target_cell': '__identity__'}))"
                >
            __contents__
            </button>""".replace('__identity__', self.identity)

class SubscribeAndRetry(Exception):
    def __init__(self, callback):
        super().__init__("SubscribeAndRetry")
        self.callback = callback

def ensureSubscribedType(t):
    if not current_transaction().db().isSubscribedToType(t):
        raise SubscribeAndRetry(lambda db: db.subscribeToType(t))

class Expands(Cell):
    def __init__(self, closed, open):
        super().__init__()
        self.isExpanded = False
        self.closed = closed
        self.open = open

    def recalculate(self):
        self.contents = """
            <div>
                <div onclick="websocket.send(JSON.stringify({'event':'click', 'target_cell': '__identity__'}))"
                        style="display:inline-block;vertical-align:top">
                    <span class="octicon octicon-diff-__which__" aria-hidden="true"></span>
                </div>

                <div style="display:inline-block">
                    __child__
                </div>

            </div>
            """.replace("__identity__", self.identity).replace("__which__", 'removed' if self.isExpanded else 'added')
        self.children = {'__child__': self.open if self.isExpanded else self.closed}

    def onMessage(self, msgFrame):
        self.isExpanded = not self.isExpanded
        self.markDirty()