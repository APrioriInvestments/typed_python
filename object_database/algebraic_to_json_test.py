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

from typed_python import Alternative, TupleOf, OneOf, sha_hash, Class, NamedTuple
from object_database.algebraic_to_json import Encoder
import unittest

opcode = Alternative("Opcode", Add={}, Sub={})

expr = Alternative("Expr")
expr.define(
    Constant = {'value': int},
    ConstantBytes = {'value': bytes},
    ConstantStr = {'value': str},
    Binop = {"opcode": opcode, 'l': expr, 'r': expr},
    Add = {'l': expr, 'r': expr},
    Sub = {'l': expr, 'r': expr},
    Mul = {'l': expr, 'r': expr},
    Many = {'vals': TupleOf(expr)},
    Possibly = {'val': OneOf(None, expr)}
    )

c10 = expr.Constant(value=10)
c20 = expr.Constant(value=20)
withbytes = expr.ConstantBytes(value=sha_hash("something").digest)
withstr = expr.ConstantStr(value="asdf")

a = expr.Add(l=c10,r=c20)
bin_a = expr.Binop(opcode=opcode.Sub(), l=c10, r=c20)
several = expr.Many([c10, c20, a, expr.Possibly(None), expr.Possibly(c20), bin_a])

class NamedTupleSubclass(NamedTuple(x=int, y=float)):
    pass

ntc = NamedTupleSubclass(x=20, y = 20.2)

class HeldClass(Class):
    h = float

    def __init__(self):
        self.h = 0.0

    def __init__(self, h):
        self.h = h

    def __eq__(self, other):
        return self.h == other.h

class AClass(Class):
    x = int
    y = TupleOf(int)
    z = HeldClass

    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

hClass = HeldClass(2.0)
aClass = AClass(10, (10,20), HeldClass(2.0))

class AlgebraicToJsonTests(unittest.TestCase):
    def test_roundtrip(self):
        e = Encoder()

        for item in [c10, c20, a, bin_a, several, withbytes, withstr]:
            self.assertEqual(
                item, 
                e.from_json(e.to_json(expr, item), expr)
                )

        for item in [hClass, ntc]:
            self.assertEqual(
                item, 
                e.from_json(e.to_json(type(item), item), type(item))
                )

    def test_convert_sha_hash_bytes(self):
        e = Encoder()
        def checkRoundtrip(b):
            self.assertEqual(e.from_json(e.to_json(bytes, b), bytes), b)

        for i in range(1000):
            checkRoundtrip(sha_hash("this is some sha hash" + str(i)).digest)

