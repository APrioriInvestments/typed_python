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
import typed_python._types as _types
from nativepython.runtime import Runtime
import unittest
import time

def add(x:int,y:int)->int:
    return x+y
def sub(x:int,y:int)->int:
    return x-y
def mul(x:int,y:int)->int:
    return x*y
def div(x:int,y:int)->float:
    return x/y
def mod(x:int,y:int)->int:
    return x%y
def lshift(x:int,y:int)->int:
    return x << y
def rshift(x:int,y:int)->int:
    return x >> y
def pow(x:int,y:int)->int:
    return x ** y
def bitxor(x:int,y:int)->int:
    return x ^ y
def bitand(x:int,y:int)->int:
    return x&y
def bitor(x:int,y:int)->int:
    return x|y
def less(x:int,y:int)->bool:
    return x<y
def greater(x:int,y:int)->bool:
    return x>y
def lessEq(x:int,y:int)->bool:
    return x<=y
def greaterEq(x:int,y:int)->bool:
    return x>=y
def eq(x:int,y:int)->bool:
    return x == y
def neq(x:int,y:int)->bool:
    return x != y

class TestArithmeticCompilation(unittest.TestCase):
    def test_runtime_singleton(self):
        self.assertTrue(Runtime.singleton() is Runtime.singleton())

    def test_compile_simple(self):
        @TypedFunction
        def f(x: int) -> int:
            return x+x+x

        r = Runtime.singleton()
        r.compile(f.overloads[0])

        self.assertEqual(f(20), 60)
        self.assertEqual(f(10), 30)

    def test_binary_operators_int(self):
        r  = Runtime.singleton()

        failures = 0

        for f in [add,sub,mul,div,mod,lshift,rshift,
                    pow,bitxor,bitand,bitor,less,
                    greater,lessEq,greaterEq,eq,neq]:
            if f in [pow]:
                lvals = range(-5,5)
                rvals = range(5)
            else:
                lvals = list(range(-20,20))
                rvals = lvals

            f_fast = r.compile(f)

            for val1 in lvals:
                for val2 in rvals:
                    try:
                        pyVal = f(val1,val2)
                    except:
                        pyVal = "Exception"

                    try:
                        llvmVal = f_fast(val1,val2)
                    except:
                        llvmVal = "Exception"

                    if type(pyVal) is not type(llvmVal) or pyVal != llvmVal:
                        print("FAILURE", f, val1, val2, pyVal, llvmVal)
                        failures += 1

        self.assertEqual(failures, 0)

    def checkFunctionOfIntegers(self, f):
        r  = Runtime.singleton()

        f_fast = r.compile(f)

        for i in range(100):
            self.assertEqual(f_fast(i), f(i))


    def test_assignment_of_pod(self):
        def f(x: int) -> int:
            y = x
            return y

        self.checkFunctionOfIntegers(f)

    def test_simple_loop(self):
        def f(x: int) -> int:
            y = 0
            while x > 0:
                x = x - 1
                y = y + x
            return y

        self.checkFunctionOfIntegers(f)

    def test_call_other_typed_function(self):
        def g(x: int) -> int:
            return x+1

        def f(x: int) -> int:
            return g(x+2)

        self.checkFunctionOfIntegers(f)

    def test_basic_type_conversion(self):
        def f(x: int) -> int:
            y = 1.5
            return int(y)

        self.checkFunctionOfIntegers(f)

    def test_integers_in_closures(self):
        y = 2
        def f(x: int) -> int:
            return x+y

        self.checkFunctionOfIntegers(f)
