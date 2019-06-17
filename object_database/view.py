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

import logging
import threading
import queue
import time

import object_database._types as _types

LOG_SLOW_COMMIT_THRESHOLD = 1.0


class DisconnectedException(Exception):
    pass


class RevisionConflictException(Exception):
    pass


class FieldNotDefaultInitializable(Exception):
    pass


class ObjectDoesntExistException(Exception):
    def __init__(self, obj):
        super().__init__("%s(%s)" % (type(obj).__qualname__, obj._identity))
        self.obj = obj


class MaskView:
    def __enter__(self):
        if hasattr(_cur_view, 'view'):
            self.view = _cur_view.view
            del _cur_view.view
        else:
            self.view = None

    def __exit__(self, *args):
        if self.view is not None:
            _cur_view.view = self.view


def revisionConflictRetry(f):
    MAX_TRIES = 100

    def inner(*args, **kwargs):
        tries = 0
        while tries < MAX_TRIES:
            try:
                return f(*args, **kwargs)
            except RevisionConflictException as e:
                logging.getLogger(__name__).info(
                    "Handled a revision conflict on key %s in %s. Retrying." % (e, f.__name__)
                )
                tries += 1

        raise RevisionConflictException()

    inner.__name__ = f.__name__
    return inner


# thread local variable containing the current view or transaction as property 'view',
# if there is one.
_cur_view = threading.local()


class View(object):
    _writeable = False

    def __init__(self, db, transaction_id):
        object.__init__(self)
        self._db = db
        self._transaction_num = transaction_id
        self.serializationContext = db.serializationContext
        self._view = _types.View(db._connection_state, transaction_id, self._writeable)
        self._confirmCommitCallback = None
        self._logger = logging.getLogger(__name__)

    def db(self):
        return self._db

    def setSerializationContext(self, serializationContext):
        self.serializationContext = serializationContext.withoutCompression()
        self._view.setSerializationContext(self.serializationContext)
        return self

    def transaction_id(self):
        return self._transaction_num

    def getFieldReads(self):
        return self._view.extractReads()

    def getFieldWrites(self):
        return self._view.extractWrites()

    def getIndexReads(self):
        return self._view.extractIndexReads()

    def commit(self):
        if not self._writeable:
            raise Exception("Views are static. Please open a transaction.")

        reads = self._view.extractReads()
        writes = self._view.extractWrites()
        indexReads = self._view.extractIndexReads()
        setAdds = self._view.extractSetAdds()
        setRemoves = self._view.extractSetRemoves()

        if writes:
            tid = self._transaction_num

            if self._confirmCommitCallback is None:
                result_queue = queue.Queue()

                confirmCallback = result_queue.put
            else:
                confirmCallback = self._confirmCommitCallback

            self._db._createTransaction(
                writes,
                {k: v for k, v in setAdds.items() if v},
                {k: v for k, v in setRemoves.items() if v},
                reads,
                indexReads,
                tid,
                confirmCallback
            )

            if not self._confirmCommitCallback:
                # this is the synchronous case - we want to wait for the confirm
                t0 = time.time()

                res = result_queue.get()

                if time.time() - t0 > LOG_SLOW_COMMIT_THRESHOLD:
                    self._logger.info(
                        "Committing %s writes and %s set changes took %.1f seconds",
                        len(writes), len(setAdds) + len(setRemoves), time.time() - t0
                    )

                if res.matches.Success:
                    return
                if res.matches.Disconnected:
                    raise DisconnectedException()
                if res.matches.RevisionConflict:
                    if hasattr(res.key, 'fieldId'):
                        fieldId = res.key.fieldId
                        fieldDef = self._db._field_id_to_field_def.get(fieldId)
                    else:
                        fieldDef = None
                    raise RevisionConflictException(res.key, fieldDef)

                assert False, "unknown transaction result: " + str(res)

    def nocommit(self):
        class Scope:
            def __enter__(scope):
                assert not hasattr(_cur_view, 'view')
                _cur_view.view = self
                self._view.enter()

            def __exit__(scope, *args):
                del _cur_view.view
                self._view.exit()

        return Scope()

    def __enter__(self):
        assert not hasattr(_cur_view, 'view')
        _cur_view.view = self
        self._view.enter()
        return self

    def __exit__(self, type, val, tb):
        del _cur_view.view
        self._view.exit()

        if hasattr(_cur_view, "watchers"):
            for watcher in _cur_view.watchers:
                watcher.callback(self, type is None)

        if type is None and self._writeable:
            self.commit()


class ViewWatcher:
    """Get a chance to look at any view or transaction being __exit__ed.

    All views or clients below us in the call stack will see this object and
    call 'callback' with the view and a boolean indicating whether they are
    exiting with an exception.
    """
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        if not hasattr(_cur_view, 'watchers'):
            _cur_view.watchers = set()
        _cur_view.watchers.add(self)

    def __exit__(self, *args):
        _cur_view.watchers.discard(self)


class Transaction(View):
    _writeable = True

    def onConfirmed(self, callback):
        """Set a callback function to be called on the main event thread with a boolean indicating
        whether the transaction was accepted."""
        self._confirmCommitCallback = callback

        return self

    def noconfirm(self):
        """Indicate that the transaction should return immediately without a round-trip to
        confirm that it was successful."""
        def ignoreConfirmResult(result):
            pass
        self._confirmCommitCallback = ignoreConfirmResult

        return self


def current_transaction():
    if not hasattr(_cur_view, "view"):
        return None
    return _cur_view.view
