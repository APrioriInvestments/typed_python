#   Copyright 2017-2019 typed_python Authors
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

import unittest
import threading
import time
import queue

from flaky import flaky
from typed_python.typed_queue import TypedQueue
from typed_python import ListOf, Entrypoint, Tuple
from typed_python._types import refcount


class TypedQueueTests(unittest.TestCase):
    def test_basic(self):
        queue = TypedQueue(float)()

        queue.put(1.0)

        self.assertEqual(queue.get(), 1.0)
        self.assertEqual(queue.getNonblocking(), None)

        queue.put(2.0)
        queue.put(3.0)
        self.assertEqual(queue.get(), 2.0)
        queue.put(4.0)
        self.assertEqual(queue.get(), 3.0)
        self.assertEqual(queue.get(), 4.0)
        self.assertEqual(queue.getNonblocking(), None)

        self.assertEqual(len(queue), 0)

        queue.put(1.0)
        queue.put(2.0)

        self.assertEqual(len(queue), 2)
        self.assertEqual(queue.peek(), 1.0)
        self.assertEqual(queue.get(), 1.0)

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.peek(), 2.0)
        self.assertEqual(queue.get(), 2.0)

        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.peek(), None)
        self.assertEqual(queue.getNonblocking(), None)

    def test_threading(self):
        queue1 = TypedQueue(float)()
        queue2 = TypedQueue(float)()

        def pong():
            queue2.put(queue1.get())

        thread1 = threading.Thread(target=pong)
        thread1.start()

        queue1.put(10)
        self.assertEqual(queue2.get(), 10)

        thread1.join()

    @flaky(max_runs=3, min_passes=1)
    def test_queue_perf(self):
        untypedQueue1 = queue.Queue()
        untypedQueue2 = queue.Queue()
        shouldExitUntyped = [False]

        def pongUntyped():
            while not shouldExitUntyped[0]:
                untypedQueue2.put(untypedQueue1.get())

        threadUntyped = threading.Thread(target=pongUntyped)
        threadUntyped.start()

        def putFloatsUntyped(q, count):
            for i in range(count):
                q.put(i)

        def getFloatsUntyped(q, count):
            res = 0.0

            for i in range(count):
                res += q.get()

            return res

        t0 = time.time()
        putFloatsUntyped(untypedQueue1, 10000)
        getFloatsUntyped(untypedQueue2, 10000)
        print("took ", time.time() - t0, " to do 10k untyped")
        timeForUntyped1mm = (time.time() - t0) * 100

        shouldExitUntyped[0] = True
        untypedQueue1.put(10)
        threadUntyped.join()

        shouldExit = ListOf(bool)([False])
        queue1 = TypedQueue(float)()
        queue2 = TypedQueue(float)()

        @Entrypoint
        def pong():
            while not shouldExit[0]:
                queue2.putMany(queue1.getMany(1, 1000))

        thread1 = threading.Thread(target=pong)
        thread1.start()

        @Entrypoint
        def putFloats(q, count):
            floats = ListOf(float)()

            for i in range(count):
                floats.append(i)

            q.putMany(floats)

        @Entrypoint
        def getFloats(q, count):
            res = 0.0
            i = 0

            while i < count:
                elts = q.getMany(1, 1000)
                for elt in elts:
                    res += elt
                    i += 1

            return res

        # prime the compiler
        putFloats(queue1, 1000)
        getFloats(queue2, 1000)

        t0 = time.time()
        putFloats(queue1, 1000000)
        getFloats(queue2, 1000000)
        speedup = timeForUntyped1mm / (time.time() - t0)
        print("took ", time.time() - t0, " to do 1mm typed, which is ", speedup, " times faster")

        shouldExit[0] = True
        queue1.put(1.0)
        thread1.join()

    def test_create_in_compiler(self):
        def f():
            q = TypedQueue(float)()
            q.put(10)
            q.get()
            q.put(10)
            return q.get()

        self.assertEqual(f(), 10)
        self.assertEqual(Entrypoint(f)(), 10)

    def test_create_in_compiler_and_use_in_other_thread(self):
        @Entrypoint
        def f():
            return TypedQueue(float)()

        q = f()

        @Entrypoint
        def otherThread(q):
            q.put(10.0)

        otherThread(q)

        t = threading.Thread(target=otherThread, args=(q,), daemon=True)
        t.start()

        self.assertEqual(q.get(), 10)

        t.join()

    def test_many_append(self):
        @Entrypoint
        def f():
            return TypedQueue(Tuple(int, int))()

        q = f()

        @Entrypoint
        def otherThread(q):
            for i in range(1000000):
                q.put(Tuple(int, int)((i, i)))

        otherThread(q)

    def test_typed_queue_refcounts(self):
        x = TypedQueue(ListOf(int))()
        a = ListOf(int)()

        x.put(a)
        x.peek()
        x.get()

        assert refcount(a) == 1
