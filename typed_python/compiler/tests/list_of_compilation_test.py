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

import numpy
import psutil
import time
import typed_python._types as _types
import unittest

from flaky import flaky
from typed_python import ListOf, Function, TupleOf, OneOf, Compiled, Entrypoint
from typed_python import UInt8, UInt16, UInt32, Class, Final, Member


class TestListOfCompilation(unittest.TestCase):
    def checkFunction(self, f, argsToCheck):
        f_fast = Compiled(f)

        t_py = 0.0
        t_fast = 0.0
        for a in argsToCheck:
            t0 = time.time()
            fastval = f_fast(*a)
            t1 = time.time()
            slowval = f(*a)
            t2 = time.time()

            t_py += t2-t1
            t_fast += t1-t0

            self.assertEqual(fastval, slowval)
        return t_py, t_fast

