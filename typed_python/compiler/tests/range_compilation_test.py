#   Copyright 2017-2020 typed_python Authors
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

from typed_python import ListOf, Compiled, Entrypoint
import unittest


class TestRangeCompilation(unittest.TestCase):
    def test_sum_with_range(self):
        @Compiled
        def sumWithRange(x: int):
            res = 0
            for i in range(x):
                res += (i+1)
            return res

        for i in range(10):
            self.assertEqual(sumWithRange(i), sum(range(i+1)))

    def test_range_with_two_values(self):
        @Compiled
        def sumWithRangePair(x: int, y: int):
            res = 0
            for i in range(x, y):
                res += i
            return res

        for i in range(30):
            for i2 in range(30):
                self.assertEqual(sumWithRangePair(i, i2), sum(range(i, i2)))

    def test_range_repeat(self):
        @Compiled
        def repeat(array: ListOf(int), times: int):
            """Concatenate a 'array' to itself 'times' times."""
            out = ListOf(int)()
            out.resize(len(array) * times)

            i = 0

            for o in range(times * len(array)):
                out[o] = array[i]

                i += 1

                if i >= len(array):
                    i = 0

            return out

        aList = ListOf(int)([1, 2, 3])

        self.assertEqual(repeat(aList, 0), type(aList)())
        self.assertEqual(repeat(aList, 1), aList)
        self.assertEqual(repeat(aList, 2), aList + aList)
        self.assertEqual(repeat(aList, 3), aList + aList + aList)

    def test_range_type_str(self):
        def f(x):
            return str(type(x))

        x = range(10)

        r1 = f(x)
        r2 = Entrypoint(f)(x)
        self.assertEqual(r1, r2)
