#   Copyright 2018 Braxton Mckee
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

from typed_python._types import NativeType

import unittest
import time

class NativeTypesTests(unittest.TestCase):
    def test_objects_are_singletons(self):
        self.assertTrue(NativeType.Int8() is NativeType.Int8())
        self.assertTrue(NativeType.NoneType() is NativeType.NoneType())

    def test_object_bytecounts(self):
        self.assertEqual(NativeType.NoneType().bytecount(), 0)
        self.assertEqual(NativeType.Int8().bytecount(), 1)

    def test_type_stringification(self):
        for t in ['Int8', 'NoneType']:
            self.assertEqual(str(getattr(NativeType,t)()), t)
