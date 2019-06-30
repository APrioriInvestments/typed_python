#   Coyright 2017-2019 Nativepython Authors
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

from typed_python import Dict, ListOf
import typed_python._types as _types
from nativepython import SpecializedEntrypoint
import unittest


class TestDictCompilation(unittest.TestCase):
    def test_can_copy_dict(self):
        @SpecializedEntrypoint
        def f(x: Dict(int, int)):
            y = x
            return y

        self.assertEqual(f({1: 2}), {1: 2})

        @SpecializedEntrypoint
        def reversed(x: ListOf(Dict(int, int))):
            res = ListOf(Dict(int, int))()

            i = len(x) - 1
            while i >= 0:
                res.append(x[i])
                i -= 1

            return res

        for length in range(100):
            dicts = [{x: x * 2 + 1} for x in range(length)]

            aList = ListOf(Dict(int, int))(dicts)

            refcounts = [_types.refcount(x) for x in aList]
            aListRev = reversed(aList)
            self.assertEqual(aListRev, list(reversed(dicts)))
            aListRev = None

            refcounts2 = [_types.refcount(x) for x in aList]

            self.assertEqual(refcounts, refcounts2)

    def test_dict_length(self):
        @SpecializedEntrypoint
        def dict_len(x):
            return len(x)

        x = Dict(int, int)({1: 2})

        self.assertEqual(dict_len(x), 1)
        x[2] = 3

        self.assertEqual(dict_len(x), 2)

        del x[1]

        self.assertEqual(dict_len(x), 1)

    def test_dict_getitem(self):
        @SpecializedEntrypoint
        def dict_getitem(x, y):
            return x[y]

        x = Dict(int, int)()

        x[1] = 2

        self.assertEqual(dict_getitem(x, 1), 2)

        with self.assertRaisesRegex(Exception, "Key doesn't exist"):
            dict_getitem(x, 2)
