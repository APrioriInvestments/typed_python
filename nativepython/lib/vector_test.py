#   Copyright 2017 Braxton Mckee
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

import nativepython.runtime as runtime
import nativepython.util as util
import nativepython.lib.vector as vector
import unittest
import ast
import time

@util.typefun
def typename(t):
	return str(t)

class VectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = runtime.Runtime.singleton()

    def test_instantiation(self):
        v = vector.Vector(util.Int64)

        vec_type = self.runtime.wrap(v)
        vec = vec_type()
        vec.append(20)

        self.assertTrue(len(vec) == 1)
        self.assertEqual(vec[0], 20)

        vec.append(22)
        self.assertTrue(len(vec) == 2)
        self.assertTrue(vec[0] == 20)
        self.assertTrue(vec[1] == 22)
