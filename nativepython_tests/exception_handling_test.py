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
import nativepython.type_model as type_model
import nativepython_tests.counter as counter
import unittest
import ast
import time

Int8 = type_model.Int8
Counter = counter.Counter
CounterInc = counter.CounterInc

@type_model.cls
class FancyException:
    def __types__(cls):
        cls.holder = CounterInc
        cls.x = int

    def __init__(self, counter, x):
        self.holder = CounterInc(util.addr(counter))
        self.x = x

class ExceptionHandlingTests(unittest.TestCase):
    @property
    def runtime(self):
        return runtime.Runtime.singleton()

    def test_raises(self):
        def f():
            raise 10
                        
        with self.assertRaises(runtime.RuntimeException):
            self.runtime.wrap(f)()

    def test_exceptions_basic(self):
        def thrower():
            try:
                try:
                    raise Int8.pointer(0xdeadbeef + 1)
                except Int8.pointer as x:
                    raise Int8.pointer(x - 1)
            except type_model.Int8 as x2:
                return

        def f():
            res = 0
            
            try:
                thrower()
            except Int8.pointer as x3:
                if int(x3) == 0xdeadbeef:
                    res = 1

            return res
                        
        self.assertEqual(self.runtime.wrap(f)(), 1)

    def test_exceptions_object_lifetime(self):
        def f():
            c = Counter()

            try:
                raise CounterInc(util.addr(c))
            except CounterInc as e:
                if c.count != 1:
                    return 1

            if c.count != 0:
                return 2

            try:
                raise CounterInc(util.addr(c))
            except CounterInc:
                if c.count != 0:
                    return 6

            try:
                raise CounterInc(util.addr(c))
            except:
                if c.count != 0:
                    return 3

            if c.count != 0:
                return 5

            try:
                try:
                    raise CounterInc(util.addr(c))
                except CounterInc as e:
                    raise e
            except CounterInc:
                if c.count != 0:
                    return 4

            return 0
                        
        self.assertEqual(self.runtime.wrap(f)(), 0)
