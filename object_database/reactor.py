#   Copyright 2019 Nativepython authors
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

import traceback
import threading
import queue
import logging

from object_database.view import RevisionConflictException, ViewWatcher


class Reactor:
    """Reactor

    Repeatedly executes a function until it no longer produces a new writes.
    Waits for any new transactions to come in that would re-trigger it.

    The function should take no arguments, and should create a sequence of
    views and transactions (using only the database_connection specified in
    the reactor). If a function call produces no actual writes, then the
    reactor will go to sleep until one of the keys that was read in the most
    recent pass through the function is touched. Otherwise it will continue
    calling the reactor function.

    If the function throws a RevisionConflictException, we will retry the
    entire function from the beginning. If the function throws any other
    exception the Reactor will log an exception and exit its loop.

    You may also specify a periodic wakup time, which causes the reactor
    to run at least that frequently.

    Example:

        def consumeOne():
            with db.transactoon():
                t = T.lookupAny()
                if t is not None:
                    print("Consumed one")
                    t.delete()

        r = Reactor(db, consumeOne)
        r.start()

        ...

        r.stop()
    """
    class STOP:
        """singleton class to indicate that we should exit the loop."""
        pass

    def __init__(self, db, reactorFunction, maxSleepTime=None):
        self.db = db
        self.reactorFunction = reactorFunction
        self.maxSleepTime = maxSleepTime
        self.db.registerOnTransactionHandler(self._onTransaction)
        self._transactionQueue = queue.Queue()
        self._thread = threading.Thread(target=self.updateLoop)
        self._thread.daemon = True
        self._isStarted = False

    def start(self):
        self._isStarted = True
        self._thread.start()

    def stop(self):
        if not self._isStarted:
            return
        self._isStarted = False
        self._transactionQueue.put(Reactor.STOP)
        self._thread.join()

    def updateLoop(self):
        try:
            """Update as quickly as possible."""
            while self._isStarted:
                self._drainTransactionQueue()
                readKeys = self._calculate()
                if readKeys is not None:
                    self.blockUntilRecalculate(readKeys)
        except Exception:
            logging.error("Unexpected exception in Reactor loop:\n%s", traceback.format_exc())

    def blockUntilRecalculate(self, readKeys):
        while True:
            result = self._transactionQueue.get()
            if result is Reactor.STOP:
                return
            for key in result:
                if key in readKeys:
                    return

    def _drainTransactionQueue(self):
        self._transactionQueue = queue.Queue()

    def _calculate(self):
        """Calculate the reactor function.

        Returns 'None' if the function should be recalculated, otherwise
        the set of keys to check for updates and a transactionId
        """
        try:
            seenKeys = set()
            hadWrites = [False]

            def onViewClose(view, isException):
                if hadWrites[0]:
                    return

                if view._writes:
                    hadWrites[0] = True
                    return

                seenKeys.update(view._reads)
                seenKeys.update(view._indexReads)

            with ViewWatcher(onViewClose):
                self.reactorFunction()

            if hadWrites[0]:
                return None
            else:
                return seenKeys

        except RevisionConflictException as e:
            logging.getLogger(__name__).info(
                "Handled a revision conflict on key %s in %s. Retrying." % (e, self.reactorFunction.__name__)
            )
            return None

    def _onTransaction(self, key_value, priors, set_adds, set_removes, transactionId):
        self._transactionQueue.put(list(key_value) + list(set_adds) + list(set_removes))
