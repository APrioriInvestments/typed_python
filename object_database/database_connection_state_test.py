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

from object_database.schema import ObjectFieldId, IndexId
from object_database.test_util import currentMemUsageMb
from object_database._types import DatabaseConnectionState


class DatabaseConnectionStateTests(unittest.TestCase):
    def test_memory_growth_transactions(self):
        connectionState = DatabaseConnectionState()

        m0 = currentMemUsageMb()

        for i in range(20000):
            connectionState.incomingTransaction(
                i,
                {ObjectFieldId(objId=0, fieldId=0, isIndexValue=False): b" " * 10000},
                {IndexId(fieldId=0, indexValue=b" " * 10000): (i,)},
                {IndexId(fieldId=0, indexValue=b" " * 10000): (i-1,)} if i > 0 else {},
            )

        self.assertLess(currentMemUsageMb() - m0, 1)

    def test_memory_growth_transactions_changing_values(self):
        connectionState = DatabaseConnectionState()

        m0 = currentMemUsageMb()

        for i in range(20000):
            connectionState.incomingTransaction(
                i,
                {ObjectFieldId(objId=0, fieldId=0, isIndexValue=False): b" " * i},
                {IndexId(fieldId=0, indexValue=b" " * i): (0,)},
                {IndexId(fieldId=0, indexValue=b" " * (i-1)): (0,)} if i > 0 else {},
            )

        self.assertLess(currentMemUsageMb() - m0, 1)
