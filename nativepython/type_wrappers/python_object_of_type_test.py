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

from typed_python import *
from nativepython.runtime import Runtime
import unittest
import time

class TestPythonObjectOfType(unittest.TestCase):
    def test_can_pass_object_in_and_out(self):
        @TypedFunction
        def f(x):
            return x

        r = Runtime.singleton()
        r.compile(f)

        for thing in [0,10,f,str]:
            self.assertIs(f(thing), thing)