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

from typed_python import Alternative, TupleOf, OneOf
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
withbytes = expr.ConstantBytes(value=b"asdf\x80")
withstr = expr.ConstantStr(value="asdf")

a = expr.Add(l=c10,r=c20)
bin_a = expr.Binop(opcode=opcode.Sub(), l=c10, r=c20)
several = expr.Many([c10, c20, a, expr.Possibly(None), expr.Possibly(c20), bin_a])

class AlgebraicToJsonTests(unittest.TestCase):
    def test_roundtrip(self):
        e = Encoder()

        for item in [c10, c20, a, bin_a, several, withbytes, withstr]:
            self.assertEqual(
                item, 
                e.from_json(e.to_json(expr, item), expr)
                )
