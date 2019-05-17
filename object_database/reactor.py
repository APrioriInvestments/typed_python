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
import time

from object_database.view import RevisionConflictException, ViewWatcher


class Timeout:
    """Singleton used to indicate that the reactor timed out."""
    pass


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

    Reactors can run inside a thread, using 'start/stop' semantics, or
    synchronously, where the client calls 'next', which returns the
    next function call result, or Timeout if the reactor doesn't want
    to retrigger within the timeout. This can be useful for watching for
    a condition.

    Finally, you may call 'blockUntilTrue' if you want to wait until
    the function returns a non-false value.

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
        r.teardown()

        # alternatively

        r1 = Reactor(db, consumeOne)
        r1.next(timeout=1.0)
        r1.next(timeout=1.0)

        ...

    """
    class STOP:
        """singleton class to indicate that we should exit the loop."""
        pass

    def __init__(self, db, reactorFunction, maxSleepTime=None):
        self.db = db
        self.reactorFunction = reactorFunction
        self.maxSleepTime = maxSleepTime

        self._transactionQueue = queue.Queue()
        self._thread = threading.Thread(target=self.updateLoop)
        self._thread.daemon = True
        self._isStarted = False
        self._lastReadKeys = None

        # grab a transaction handler. We need to ensure this is the same object
        # when we deregister it.
        self.transactionHandler = self._onTransaction
        self.db.registerOnTransactionHandler(self.transactionHandler)

    def start(self):
        self._isStarted = True
        self._thread.start()

    def stop(self):
        if not self._isStarted:
            return
        self._isStarted = False
        self._transactionQueue.put(Reactor.STOP)
        self._thread.join()

    def teardown(self):
        """Remove this reactor. Clients should _always_ call this when they are done."""
        self.db.dropTransactionHandler(self.transactionHandler)

    def blockUntilTrue(self, timeout=None):
        """Block until the reactor function returns 'True'.

        Returns True if we succeed, False if we exceed the threshold.
        """
        if timeout is None:
            while not self.next():
                pass

            return True
        else:
            curTime = time.time()
            timeThreshold = curTime + timeout

            while curTime < timeThreshold:
                result = self.next(timeout=timeThreshold - curTime)

                if result is Timeout:
                    return False

                if result:
                    return True

                curTime = time.time()

            return False

    def next(self, timeout=None):
        if self._isStarted:
            raise Exception("Can't call 'next' if the reactor is being used in threaded mode.")

        if self._lastReadKeys is not None:
            if not self._blockUntilRecalculate(self._lastReadKeys, timeout=timeout):
                return Timeout

        self._drainTransactionQueue()
        result, self._lastReadKeys = self._calculate(catchRevisionConflicts=False)

        return result

    def updateLoop(self):
        try:
            """Update as quickly as possible."""
            while self._isStarted:
                self._drainTransactionQueue()
                _, readKeys = self._calculate(catchRevisionConflicts=True)
                if readKeys is not None:
                    self._blockUntilRecalculate(readKeys, self.maxSleepTime)

        except Exception:
            logging.error("Unexpected exception in Reactor loop:\n%s", traceback.format_exc())

    def _blockUntilRecalculate(self, readKeys, timeout):
        """Wait until we're triggered, or hit a timeout.

        Returns:
            True if we were triggered by a key update, False otherwise.
        """
        if not readKeys and timeout is None:
            raise Exception("Reactor would block forever.")

        curTime = time.time()
        finalTime = curTime + (timeout if timeout is not None else 10**8)

        while curTime < finalTime:
            try:
                result = self._transactionQueue.get(timeout=finalTime - curTime)
            except queue.Empty:
                return False

            if result is Reactor.STOP:
                return False
            for key in result:
                if key in readKeys:
                    return True

            curTime = time.time()

        return False

    def _drainTransactionQueue(self):
        self._transactionQueue = queue.Queue()

    def _calculate(self, catchRevisionConflicts):
        """Calculate the reactor function.

        Returns:
            (functionResult, keySetOrNone)

            keySetOrNone will be 'None' if the function should be
            recalculated, otherwise the set of keys to check for updates and a
            transactionId.

            functionResult will be the actual result of the function,
            or None if it threw a RevisionConflictException
        """
        try:
            seenKeys = set()
            hadWrites = [False]

            def onViewClose(view, isException):
                if hadWrites[0]:
                    return

                if view._view.extractWrites():
                    hadWrites[0] = True
                    return

                seenKeys.update(view._view.extractReads())
                seenKeys.update(view._view.extractIndexReads())

            with ViewWatcher(onViewClose):
                logging.getLogger(__name__).info("Reactor %s recalculating", self.reactorFunction)
                functionResult = self.reactorFunction()

            if hadWrites[0]:
                return functionResult, None
            else:
                return functionResult, seenKeys

        except RevisionConflictException as e:
            if not catchRevisionConflicts:
                raise

            logging.getLogger(__name__).info(
                "Handled a revision conflict on key %s in %s. Retrying." % (e, self.reactorFunction.__name__)
            )
            return None, None

    def _onTransaction(self, key_value, set_adds, set_removes, transactionId):
        self._transactionQueue.put(list(key_value) + list(set_adds) + list(set_removes))
