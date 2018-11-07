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


def add(x,y):
    return x+y
def sub(x,y):
    return x-y
def mul(x,y):
    return x*y
def div(x,y):
    return x/y
def mod(x,y):
    return x%y
def lshift(x,y):
    return x << y
def rshift(x,y):
    return x >> y
def pow(x,y):
    return x ** y
def bitxor(x,y):
    return x ^ y
def bitand(x,y):
    return x&y
def bitor(x,y):
    return x|y
def less(x,y):
    return x<y
def greater(x,y):
    return x>y
def lessEq(x,y):
    return x<=y
def greaterEq(x,y):
    return x>=y
def eq(x,y):
    return x == y
def neq(x,y):
    return x != y

class TestRuntime(unittest.TestCase):
    def test_runtime_singleton(self):
        self.assertTrue(Runtime.singleton() is Runtime.singleton())

    def test_compile_simple(self):
        def f(x):
            return x+x+x

        r = Runtime.singleton()

        f_fast = r.compile(f, (int,))

        self.assertEqual(f_fast(10), 30)

    def test_binary_operators_int(self):
        r  = Runtime.singleton()

        failures = 0

        for f in [add,sub,mul,div,mod,lshift,rshift,
                    #pow, -- doesn't work yet
                    bitxor,bitand,bitor,less,greater,lessEq,greaterEq,eq,neq]:
            if f in [mod]:
                lvals = [0,1,10,100]
                rvals = [1,2,10,100]
            elif f in [div]:
                lvals = [-100, -10, -1, 0,1,10,100]
                rvals = [1,2,10,100, -1, -2, -10]
            elif f in [lshift,rshift]:
                lvals = [0,1,10]
                rvals = [1,2,10]
            else:
                lvals = [-1,0,1,10,100]
                rvals = [-1,0,1,10,100]

            f_fast = r.compile(f, (int,int))
            for val1 in lvals:
                for val2 in rvals:
                    try:
                        pyVal = f(val1,val2)
                    except:
                        pyVal = "Exception"

                    llvmVal = f_fast(val1,val2)

                    if type(pyVal) is not type(llvmVal) or pyVal != llvmVal:
                        print("FAILURE", f, val1, val2, pyVal, llvmVal)
                        failures += 1

        self.assertEqual(failures, 0)
