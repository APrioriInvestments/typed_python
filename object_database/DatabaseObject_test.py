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

import unittest
from object_database.schema import Schema


class DatabaseObjectTests(unittest.TestCase):
    def test_database_object(self):
        s = Schema("testschema")

        @s.define
        class T:
            val = int

            @property
            def aProperty(self):
                return self._identity + 1

            @staticmethod
            def aStaticMethod(x):
                return x + 1

            def aMethod(self, x):
                return self._identity + x

        self.assertEqual(T.fromIdentity(123)._identity, 123)
        self.assertEqual(T.fromIdentity(123).aProperty, 124)
        self.assertEqual(T.fromIdentity(123).aStaticMethod(10), 11)
        self.assertEqual(T.fromIdentity(123).aMethod(10), 133)

        with self.assertRaisesRegex(Exception, "without an active"):
            T.fromIdentity(123).val
