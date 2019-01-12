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

def Compiled(f):
    f = TypedFunction(f)
    return Runtime.singleton().compile(f)

class AClass(Class):
    x = Member(int)
    y = Member(float)
    z = Member(TupleOf(int))

    def f(self):
        return self.x + self.y

class AClassWithAnotherClass(Class):
    x = Member(int)
    y = Member(float)
    ac = Member(AClass)

class TestClassCompilationCompilation(unittest.TestCase):
    def test_class_attribute(self):
        a = AClass(x=10,y=20.5,z=(1,2,3))

        @Compiled
        def getX(a: AClass) -> int:
            return a.x

        @Compiled
        def getY(a: AClass) -> float:
            return a.y
        
        @Compiled
        def getZ(a: AClass) -> TupleOf(int):
            return a.z

        self.assertEqual(getX(a), a.x)
        self.assertEqual(getY(a), a.y)
        self.assertEqual(getZ(a), a.z)
                
    def test_class_set_attribute(self):
        a = AClass()

        aTupleOfInt = TupleOf(int)((1,2,3))

        @Compiled
        def setX(a: AClass, x: int) -> None:
            a.x = x

        @Compiled
        def setY(a: AClass, y: float) -> None:
            a.y = y

        @Compiled
        def setZ(a: AClass, z: TupleOf(int)) -> None:
            a.z = z

        setX(a, 20)
        setY(a, 20.5)
        setZ(a, aTupleOfInt)

        self.assertEqual(a.x, 20)
        self.assertEqual(a.y, 20.5)
        self.assertEqual(a.z, aTupleOfInt)

        self.assertEqual(_types.refcount(aTupleOfInt), 2)

        a.z = (1,2,3,4)

        self.assertEqual(_types.refcount(aTupleOfInt), 1)

        a.z = aTupleOfInt

        self.assertEqual(_types.refcount(aTupleOfInt), 2)

        a = None

        self.assertEqual(_types.refcount(aTupleOfInt), 1)

    def test_class_uninitialized_attribute(self):
        @Compiled
        def set(ac: AClassWithAnotherClass, a: AClass) -> None:
            ac.ac = a

        @Compiled
        def get(ac: AClassWithAnotherClass) -> AClass:
            return ac.ac

        ac1 = AClass(x=1)
        ac2 = AClass(x=2)

        anAWithAClass = AClassWithAnotherClass(ac=ac1)

        self.assertEqual(_types.refcount(ac1), 2)
        self.assertEqual(_types.refcount(ac2), 1)
        self.assertEqual(anAWithAClass.ac.x, 1)

        set(anAWithAClass, ac2)
        self.assertEqual(_types.refcount(ac1), 1)
        self.assertEqual(_types.refcount(ac2), 2)
        self.assertEqual(anAWithAClass.ac.x, 2)

        anAWithAClass = AClassWithAnotherClass()
        self.assertEqual(_types.refcount(ac1), 1)
        self.assertEqual(_types.refcount(ac2), 1)
        
        with self.assertRaises(Exception):
            get(anAWithAClass)

        
        set(anAWithAClass, ac1)
        self.assertEqual(_types.refcount(ac1), 2)
        self.assertEqual(_types.refcount(ac2), 1)
        
        self.assertEqual(get(anAWithAClass).x, 1)
        self.assertEqual(get(anAWithAClass).x, 1)
        self.assertEqual(get(anAWithAClass).x, 1)
        self.assertEqual(get(anAWithAClass).x, 1)
        self.assertEqual(_types.refcount(ac1), 2)
        
        set(anAWithAClass, ac2)
        self.assertEqual(_types.refcount(ac1), 1)
        self.assertEqual(_types.refcount(ac2), 2)
        self.assertEqual(get(anAWithAClass).x, 2)











