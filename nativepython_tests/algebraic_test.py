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

import nativepython.algebraic as Algebraic
import unittest

expr = Algebraic.Alternative("Expr")
expr.Constant = {'value': int}
expr.Add = {'l': expr, 'r': expr}
expr.Sub = {'l': expr, 'r': expr}
expr.Mul = {'l': expr, 'r': expr}

class AlgebraicTests(unittest.TestCase):
    def test_basic(self):
        X = Algebraic.Alternative('X', A = {}, B = {})

        xa = X.A()
        xb = X.B()

        self.assertTrue(xa.matches.A)
        self.assertFalse(xa.matches.B)

        self.assertTrue(xb.matches.B)
        self.assertFalse(xb.matches.A)

        with self.assertRaises(AttributeError):
            xb.matches.C

    def test_field_lookup(self):
        X = Algebraic.Alternative('X', A = {'a': int}, B = {'b': float})

        self.assertEqual(X.A(10).a, 10)
        with self.assertRaises(AttributeError):
            X.A(10).b

        self.assertEqual(X.B(11.0).b, 11.0)
        with self.assertRaises(AttributeError):
            X.B(11.0).a
    
    def test_lists(self):
        X = Algebraic.Alternative('X')
        X.A = {'val': int}
        X.B = {'val': Algebraic.List(X)}

        xa = X.A(10)
        xb = X.B([xa, X.A(11)])

        self.assertTrue(xa.matches.A)
        self.assertTrue(xb.matches.B)
        self.assertTrue(isinstance(xb.val, tuple))
        self.assertTrue(len(xb.val) == 2)

    def test_stringification(self):
        self.assertEqual(
            repr(expr.Add(l = expr(10), r = expr(20))),
            "Expr.Add(l=Expr.Constant(value=10),r=Expr.Constant(value=20))"
            )

    def test_isinstance(self):
        self.assertTrue(isinstance(expr(10), Algebraic.AlternativeInstance))
        self.assertTrue(isinstance(expr(10), expr.Constant))

    def test_coercion(self):
        Sub = Algebraic.Alternative('Sub', I={}, S={})

        with self.assertRaises(Exception):
            Sub.I(Sub.S)

        X = Algebraic.Alternative('X', A={'val': Sub})

        X.A(val=Sub.S())
        with self.assertRaises(Exception):
            X.A(val=Sub.S)

    def test_coercion_null(self):
        Sub = Algebraic.Alternative('Sub', I={}, S={})
        X = Algebraic.Alternative('X', I={'val': Algebraic.Nullable(Sub)})

        self.assertTrue(X(Sub.I()).val.matches.Value)

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
