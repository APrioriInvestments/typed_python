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

from typed_python.typed_queue import TypedQueue


class TypedQueueTests(unittest.TestCase):
    def test_basic(self):
        queue = TypedQueue(float)()

        queue.put(1.0)

        self.assertEqual(queue.get(), 1.0)
        self.assertEqual(queue.get(), None)

        queue.put(2.0)
        queue.put(3.0)
        self.assertEqual(queue.get(), 2.0)
        queue.put(4.0)
        self.assertEqual(queue.get(), 3.0)
        self.assertEqual(queue.get(), 4.0)
        self.assertEqual(queue.get(), None)

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
        self.assertEqual(queue.get(), None)
