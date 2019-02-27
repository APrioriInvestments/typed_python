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
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or imp lied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typed_python import *
from nativepython.runtime import Runtime
import unittest


def Compiled(f):
    f = Function(f)
    return Runtime.singleton().compile(f)


class HoldsAnA:
    def __init__(self, a):
        self.a = a


class TestPythonObjectOfTypeCompilation(unittest.TestCase):
    def test_can_pass_object_in_and_out(self):
        @Compiled
        def f(x):
            return x

        for thing in [0, 10, f, str]:
            self.assertIs(f(thing), thing)

    def test_can_assign(self):
        @Compiled
        def f(x):
            y = x
            return y

        for thing in [0, 10, f, str, HoldsAnA(10)]:
            for i in range(10000):
                self.assertIs(f(thing), thing)

    def test_member_access(self):
        @Compiled
        def f(x):
            return x.a

        x = '1'
        self.assertIs(f(HoldsAnA(x)), x)
        with self.assertRaises(Exception):
            f(10)
