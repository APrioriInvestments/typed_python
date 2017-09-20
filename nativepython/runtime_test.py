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
import unittest
import ast
import time

def g(a):
    return a+2

class Simple:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def f(self):
        return self

class PythonNativeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = runtime.Runtime()

    def test_conversion(self):
        def f(a):
            return g(a)+g(1)

        self.assertTrue(self.runtime.wrap(f)(10) == f(10))

    def test_holding_objects(self):
        try:
            res = self.runtime.wrap(Simple)(10, 12.0)

            self.assertEqual(res.x, 10)

            count = len(self.runtime.functions_by_name)

            #access the property several more times
            for i in xrange(10):
                res.x
        
            #and verify we're not compiling new code every time
            self.assertEqual(count, len(self.runtime.functions_by_name))
        except KeyboardInterrupt:
            import traceback
            traceback.print_exc()
            raise

    def test_returning_function_results(self):
        res = self.runtime.wrap(Simple)(10, 12.0)
        
        res2 = res.f()

        self.assertEqual(res2.x, res.x)

    def test_printf(self):
        def f():
            util.printf("hello\n")
                        
        self.runtime.wrap(f)()

    def test_self_pointer(self):
        addr = util.addr

        class A:
            def __init__(self):
                self.ptr = int(addr(self))
                
            def __copy_constructor__(self, other):
                self.ptr = int(addr(self))
                
            def __assign__(self, other):
                self.ptr = int(addr(self))
                
            def compare(self):
                return (self.ptr == int(addr(self)), self.ptr, int(addr(self)))

        def g(an_A):
            return an_A.compare()
            
        def f():
            an_a = A()
            return g(an_a)
        
        res = self.runtime.wrap(f)()

        self.assertTrue(res.f0, (res.f0, res.f1, res.f2))

    def test_decltype(self):
        addr = util.addr
        typeof = util.typeof

        def f():
            return typeof
        
        result = self.runtime.wrap(f)()

        self.assertIs(result, typeof)
 
    def test_pass_by_ref(self):
        ref = util.ref

        def increment(x):
            x = x + 1

        def no_ref(x):
            increment(x)
            return x

        def with_ref(x):
            increment(ref(x))
            return x

        self.assertEqual(self.runtime.wrap(no_ref)(0), 0)
        self.assertEqual(self.runtime.wrap(with_ref)(0), 1)
 
    def test_pass_by_ref_internal(self):
        ref = util.ref

        def increment(x):
            x = x + 1

        def increment_2(x):
            #because we're not passing by 'ref' again, the increment should
            #have no effect
            increment(x)

        def with_ref(x):
            increment_2(ref(x))
            return x

        self.assertEqual(self.runtime.wrap(with_ref)(0), 0)
