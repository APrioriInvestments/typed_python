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

import threading
import queue


class BytecountLimitedQueue(object):
    """Looks like a Queue, but limits the total number of bytes in the queue.

    We parametrize the Queue with a function that decides how many bytes are in a
    given message.
    """
    def __init__(self, bytecountFunction, maxBytes=None):
        self._bytecountFunction = bytecountFunction
        self._canPushCondition = threading.Condition(threading.Lock())
        self._underlyingQueue = queue.Queue()
        self.totalBytes = 0
        self.maxBytes = None

    def pendingCount(self):
        return self._underlyingQueue.qsize()

    def setMaxBytes(self, bytecount):
        self.maxBytes = bytecount

        with self._canPushCondition:
            self._canPushCondition.notify_all()

    def put(self, msg, block=True, allowWriteWhileOverLimit=False):
        with self._canPushCondition:
            msgLen = self._bytecountFunction(msg)

            if not allowWriteWhileOverLimit:
                while self.isBlocked():
                    if not block:
                        raise queue.Full()

                    self._canPushCondition.wait()

            self.totalBytes += msgLen
            self._underlyingQueue.put(msg)

    def isBlocked(self):
        return self.maxBytes is not None and self.totalBytes >= self.maxBytes

    def get(self, timeout=None):
        msg = self._underlyingQueue.get(timeout=timeout)

        with self._canPushCondition:
            blocked = self.isBlocked()

            self.totalBytes -= self._bytecountFunction(msg)

            if blocked != self.isBlocked():
                self._canPushCondition.notify_all()

        return msg
