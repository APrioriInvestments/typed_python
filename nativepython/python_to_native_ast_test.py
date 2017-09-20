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

import nativepython.python_to_native_ast as python_to_native_ast
import nativepython.native_ast as native_ast
import nativepython.util as util
import nativepython.llvm_compiler as llvm_compiler
import unittest
import ast
import time

def g(a):
    return a+2

class PythonToNativeAstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = llvm_compiler.Compiler()
        cls.converter = python_to_native_ast.Converter()

    def compile(self, f):
        f_target = self.converter.convert(f, [python_to_native_ast.Float64])

        functions = self.converter.extract_new_function_definitions()

        return self.compiler.add_functions(functions)[f_target.name]

    def test_conversion(self):
        def f(a):
            return g(a)+g(1)

        self.assertTrue(self.compile(f)(10) == f(10))

    def test_branching(self):
        def f(a):
            if a > 0:
                return g(a)+g(1)
            else:
                return -3.0

        self.assertTrue(self.compile(f)(10) == f(10))

    def test_branching_2(self):
        def f(a):
            if a > 0:
                return g(a)+g(1)

            return -3.0
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10) == f(10))
        self.assertTrue(f_comp(-10) == f(-10))

    def test_assignment(self):
        def f(a):
            x = a + 1
            return x
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10) == f(10))
        self.assertTrue(f_comp(-10) == f(-10))

    def test_while(self):
        def f(a):
            x = a
            res = 0.0
            while x < 1000:
                x = x + 1
                res = res + x
            return res
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10) == f(10))
        self.assertTrue(f_comp(-10) == f(-10))

    def test_pointers(self):
        def increment(a):
            a[0] = a[0] + 1

        def f(a):
            increment(util.addr(a))
            return a
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10) == 11)

    def test_conversion(self):
        def f(a):
            return int(a+.5)
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10) == f(10))

    def test_negation(self):
        def f(a):
            b = 3
            b = -b
            return -a + b
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10.5) == f(10.5))

    def test_malloc_free(self):
        int64 = util.Int64

        def f(a):
            x = int64.pointer(util.malloc(int64.sizeof * 3))
            x[0] = 10
            x[1] = 11
            (x+2)[0] = 12
            
            y = x[0] + x[1] + (x+3)[-1]

            util.free(x)
            return y
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10) == 33)

    def test_structs(self):
        def f(a):
            x = (util.Struct()
                .with_field("a",util.Int64)
                .with_field("b",util.Float64)
                )(a,a)

            x.b = x.b + 2.3

            return x.a + x.b
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10.5) == 10 + 10.5 + 2.3)

    def test_classes(self):
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def f(self, z):
                return self.x + self.y + z

        def f(a):
            b = A(a,1)
            return b.f(a)
        
        f_comp = self.compile(f)

        self.assertTrue(f_comp(10.5) == 22)

    def test_constructors_and_destructors(self):
        class A:
            def __init__(self, x):
                self.x = x

            def __copy_constructor__(self, other):
                self.x = other.x

            def __destructor__(self):
                pass

            def __assign__(self, other):
                self.x = other.x

        def g(c):
            return A(c)
            
        def f(a):
            res = g(a)
            
            return res.x
        
        f_comp = self.compile(f)

        self.assertEqual(f_comp(10), 10)

    def test_constructors_and_destructors_2(self):
        class Counter:
            def __init__(self):
                self.alive = 0
                self.total = 0

            def inc(self):
                self.alive += 1
                self.total += 1

            def dec(self):
                self.alive -= 1

        class A:
            def __init__(self, c, x):
                self.x = x
                self.c = c
                self.c.inc()
                
            def __copy_constructor__(self, other):
                self.x = other.x
                self.c = other.c
                self.c.inc()
                
            def __destructor__(self):
                self.c.dec()

            def __assign__(self, other):
                if self.c:
                    self.c.dec()
                other.c.inc()

                self.c = other.c
                self.x = other.x

        def h(c):
            return A(c, 2)

        def i(c):
            return (A(c, 2), A(c,3))

        def g(c):
            (A(c, 2), A(c,3))
            return A(c,2)

            
        def f(a):
            c = Counter()
            res = g(util.addr(c))
            return c.alive
        
        f_comp = self.compile(f)

        self.assertEqual(f_comp(10), 1)
