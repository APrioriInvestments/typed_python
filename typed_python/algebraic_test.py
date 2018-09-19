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

from typed_python.algebraic import Alternative
from typed_python.types import ListOf, TupleOf, OneOf, TypeConvert, ConstDict

import unittest

expr = Alternative("Expr")
expr.define(
    Constant={'value': int},
    Add={'l': expr, 'r': expr},
    Sub={'l': expr, 'r': expr},
    Mul={'l': expr, 'r': expr}
    )

class AlgebraicTests(unittest.TestCase):
    def test_basic(self):
        X = Alternative('X', A = {}, B = {})

        xa = X.A()
        xb = X.B()

        self.assertTrue(xa.matches.A)
        self.assertFalse(xa.matches.B)

        self.assertTrue(xb.matches.B)
        self.assertFalse(xb.matches.A)

    def test_conversion(self):
        X = Alternative('X')
        X.define(A = {'x': int}, B = {'x': X, 'y': int})

        X2 = Alternative('X')
        X2.define(A = {'x': int}, B = {'x': X2, 'y': int})
        
        x1 = X.B(x=X.A(x=10),y=20)
        x2 = X2.B(x=X.A(x=10),y=20)
        x3 = X2.B(x=X2.A(x=10),y=20)

        for possibleType in [X, X2]:
            for possibleValue in [x1,x2,x3]:
                c = TypeConvert(possibleType, possibleValue, True)
                self.assertTrue(c.matches.B)
                self.assertTrue(c.x.matches.A)
                self.assertTrue(c.x.x is 10)
                self.assertTrue(c.y is 20)

        C1 = ConstDict(str, X)
        C2 = ConstDict(str, X2)

        c1 = C1({'a': x1})
        c2 = C2({'b': x2})

        TypeConvert(C2, c1, True)
        TypeConvert(C1, c2, True)


    def test_field_lookup(self):
        X = Alternative('X', A = {'a': int}, B = {'b': float})

        self.assertEqual(X.A(10).a, 10)
        with self.assertRaises(AttributeError):
            X.A(10).b

        self.assertEqual(X.B(11.0).b, 11.0)
        with self.assertRaises(AttributeError):
            X.B(11.0).a
    
    def test_tuples(self):
        X = Alternative('X')
        X.define(
            A = {'val': int},
            B = {'val': TupleOf(X)}
            )

        xa = X.A(10)
        xb = X.B([xa, X.A(11)])

        self.assertTrue(xa.matches.A)
        self.assertTrue(xb.matches.B)
        self.assertTrue(isinstance(xb.val, TupleOf))
        self.assertTrue(len(xb.val) == 2)

    def test_stringification(self):
        self.assertEqual(
            repr(expr.Add(l = expr.Constant(10), r = expr.Constant(20))),
            "Expr.Add(l=Expr.Constant(value=10),r=Expr.Constant(value=20))"
            )

    def test_isinstance(self):
        self.assertTrue(isinstance(expr.Constant(10), expr))
        self.assertTrue(isinstance(expr.Constant(10), expr.Constant))

    def test_coercion(self):
        Sub = Alternative('Sub', I={}, S={})

        with self.assertRaises(Exception):
            Sub.I(Sub.S)

        X = Alternative('X', A={'val': Sub})

        X.A(val=Sub.S())
        with self.assertRaises(Exception):
            X.A(val=Sub.S)

    def test_cant_assign(self):
        e = expr.Constant(10)
        with self.assertRaises(Exception):
            e.value = 20

    def test_null(self):
        Sub = Alternative("Sub", I={})
        X = Alternative('X', I={'val': OneOf(Sub, None)})

        self.assertTrue(isinstance(X.I(Sub.I()).val, Sub))
        self.assertTrue(X.I(None).val is None)

    def test_equality(self):
        for i in range(10):
            self.assertEqual(hash(expr.Constant(i)), hash(expr.Constant(i)))
            self.assertEqual(expr.Constant(i), expr.Constant(i))
            self.assertNotEqual(expr.Constant(i), expr.Constant(i+1))

    def test_algebraics_in_dicts(self):
        d = {}
        for i in range(10):
            d[expr.Constant(i)] = i
            d[expr.Add(l=expr.Constant(i),r=expr.Constant(i+1))] = 2*i+1
            
        for i in range(10):
            self.assertEqual(d[expr.Constant(i)], i)
            self.assertEqual(d[expr.Add(l=expr.Constant(i),r=expr.Constant(i+1))], 2*i+1)
