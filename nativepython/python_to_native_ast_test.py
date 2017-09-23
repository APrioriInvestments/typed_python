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

import nativepython.type_model as type_model
import nativepython.runtime as runtime
import nativepython.python_to_native_ast as python_to_native_ast
import nativepython.native_ast as native_ast
import nativepython.util as util
import nativepython.python.inspect_override as inspect_override
import nativepython.llvm_compiler as llvm_compiler
import nativepython.test_config as test_config
import unittest
import ast
import time
import numpy

TEST_SEED = 1

def g(a):
    return a+2

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
        if self.c:
            self.c.inc()
        
    def __destructor__(self):
        if self.c:
            self.c.dec()

    def __assign__(self, other):
        if self.c:
            self.c.dec()
        if other.c:
            other.c.inc()

        self.c = other.c
        self.x = other.x

def generate_functions(seed, count, add_printfs=False):
    numpy.random.seed(seed)

    def choice(array):
        if isinstance(array, int):
            array = range(array)

        array = sorted(list(array))

        return array[numpy.random.choice(range(len(array)))]

    def choice_sq(low, high):
        x = numpy.random.uniform()
        x = x * x
        return low + int(x * (high-low))

    def random_function(out_t, signature, function_signatures):
        variable_types = {"arg%s" % i:signature[i] for i in xrange(len(signature))}

        def random_expression_of_type(target_t, allow_refs, depth=0):
            tries = 0
            while True:
                tries += 1
                assert tries < 100, "can't make an expression of type " + str(target_t)

                v,t,ref = random_expression(allow_refs, depth)
                if target_t == t:
                    return v,t,ref

        def random_expression(allow_refs, depth=0):
            x = numpy.random.uniform()

            if x < .02 and depth < 3:
                #access a field
                expr, t, referenceable = random_expression(allow_refs, depth+1)
                if t == A:
                    return "(%s).x" % expr, int, referenceable
                if t == (A,A):
                    return "(%s).f%s" % (expr, choice(range(2))), A, referenceable
                return expr + "+ %s" % (choice([-2,-1,0,1,2])) , t, False

            if x < .04 and allow_refs and depth < 3:
                #take reference
                expr, t, referenceable = random_expression(allow_refs, depth+1)
                if referenceable:
                    return "util.ref(%s)" % expr, t, referenceable
                return expr, t, referenceable

            if x < .06 and depth < 3:
                #make a bigger type
                expr, t, referenceable = random_expression(allow_refs, depth+1)
                if t == int:
                    return "A(c,%s)" % expr, A, True
                if t == A:
                    e2, _,_ = random_expression_of_type(A, allow_refs, depth+1)
                    return "(%s,%s)" % (expr, e2), (A,A), False
                return expr, t, referenceable

            if x < .1 and depth < 3:
                #branch
                int_e,_,_ = random_expression_of_type(int, allow_refs, depth+1)

                true_e,true_t,_ = random_expression(allow_refs, depth+1)
                false_e,false_t,_ = \
                    random_expression_of_type(true_t, allow_refs, depth+1)

                return "((%s) if (%s) else (%s))" % (true_e, int_e, false_e), true_t, False

            if x < .2 and function_signatures and depth < 3:
                #function call
                fname, (signature, out_t) = choice(list(function_signatures.iteritems()))

                args = [random_expression_of_type(t, allow_refs, depth+1)[0] for t in signature]

                return "(%s(c,%s))" % (fname, ",".join(args)), out_t, False

            if x < .6 and variable_types:
                #pick a variable
                varname, t = choice(sorted(list(variable_types.iteritems())))
                return varname, t, True

            #random integer
            which = choice([int, A, (A,A)])
            if which == int:
                return str(choice(range(3))), int, False
            if which == A:
                return "A(c,%s)" % str(choice(range(3))), A, False
            if which == (A,A):
                return "(A(c,%s),A(c,%s))" % (choice(3), choice(3)), (A,A), False

        def random_statement(allow):
            if add_printfs:
                return "util.printf(\"line __line__\\n\")\n" + random_statement_(allow)
            else:
                return "\n" + random_statement_(allow)

        def random_statement_(allow_new_variables):
            x = numpy.random.uniform()

            if x < .05:
                #naked expression
                return random_expression(True)[0]

            if x < .2:
                #if block
                true_statements = "\n".join(
                    random_statement(False) for _ in 
                        xrange(choice_sq(1,3))
                    )
                false_statements = "\n".join(
                    random_statement(False) for _ in 
                        xrange(choice_sq(1,3))
                    )
                cond = random_expression_of_type(int, True)[0]

                return "if %s:\n%s\nelse:\n%s" % (
                    cond, 
                    native_ast.indent(true_statements), 
                    native_ast.indent(false_statements)
                    )

            if x < .4 and allow_new_variables:
                #new variables
                expr, t, referenceable = random_expression(True)
                new_var = "var_%s" % len(variable_types)
                variable_types[new_var] = t
                return "%s=%s" % (new_var, expr)

            if x < .6 and variable_types:
                #assignment
                varname, t = choice(sorted(list(variable_types.iteritems())))

                expr = random_expression_of_type(t, True)[0]

                return "%s=%s" % (varname, expr)

            
            return "return (%s)" % random_expression_of_type(out_t, False)[0]

        body = [random_statement(True)]
        while body[-1].split("\n")[-1][:3] != "ret":
            body.append(random_statement(True))

        body_text = "\n".join(body)

        fname = "f%s" % len(function_signatures)
        argtuple = ",".join(["c"]+["arg%s" % i for i in xrange(len(signature))])
        def type_name(t):
            if t is int:
                return "int"
            if t is A:
                return "A"
            if t == (A,A):
                return "(A,A)"
        return fname, "def %s(%s):  #%s\n%s" % (
            fname, 
            argtuple, 
            str([type_name(t) for t in signature]) + " -> " + type_name(out_t),
            native_ast.indent(body_text)
            )

    fdefs = []
    signatures = {}
    while len(fdefs) < count:
        out_t = choice([int, A, (A,A)])
        signature = [choice([int, A, (A,A)]) for _ in 
                            xrange(choice(3))]

        fname, body = random_function(out_t, signature, signatures)

        signatures[fname] = (tuple(signature), out_t)

        fdefs.append(body)

    all_defs = "\n".join(fdefs)
    all_defs = all_defs.split("\n")
    for i in xrange(len(all_defs)):
        all_defs[i] = all_defs[i].replace("__line__", str(i+1))
    all_defs = "\n".join(all_defs)

    inspect_override.pathExistsOnDiskCache_["<<<FAKE>>>"] = True
    inspect_override.linesCache_["<<<FAKE>>>"] = [x+'\n' for x in all_defs.split("\n")]
    
    locals_and_globals = {'A': A,'util': util}

    code = compile(all_defs, "<<<FAKE>>>", 'exec')
    exec code in locals_and_globals

    return {fname: locals_and_globals[fname] for fname in signatures}, all_defs, signatures


class PythonToNativeAstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = runtime.Runtime.singleton().compiler
        cls.converter = runtime.Runtime.singleton().converter

    def convert_expression(self, lambda_func):
        return self.converter.convert_lambda_as_expression(lambda_func)

    def compile(self, f):
        f_target = self.converter.convert(f, [util.Float64])

        functions = self.converter.extract_new_function_definitions()

        return self.compiler.add_functions(functions)[f_target.name]

    def test_typefuncs_1(self):
        self.assertTrue(self.convert_expression(lambda: 3).expr.matches.Constant)

    def test_typefuncs_2(self):
        self.assertTrue(
            self.convert_expression(lambda: util.typeof(3).reference).expr_type
                .python_object_representation == type_model.Int64.reference
            )
        self.assertTrue(
            self.convert_expression(lambda: util.typeof(3+3).reference).expr_type
                .python_object_representation == type_model.Int64.reference
            )

    def test_typefuncs_3(self):
        def f():
            output_type = util.typeof(3)
            return output_type.reference

        self.assertTrue(
            self.convert_expression(lambda: f()).expr_type
                .python_object_representation == type_model.Int64.reference
            )

    def test_typeof(self):
        def g(x):
            return x + 1

        class RefToSelf:
            def __init__(self, x):
                self.x = x

            def returns_ref_to_self(self):
                return util.ref(self)

            def returns_copy_of_self(self):
                return self

        def makes_ref(x):
            return RefToSelf(x)

        def all_pass():
            util.assert_types_same(util.typeof(3), type_model.Int64)
            util.assert_types_same(util.typeof(g(3)), type_model.Int64)

            x = 3
            util.assert_types_same(util.typeof(x), type_model.Int64.reference)
            
            aRefToSelf = RefToSelf(0)

            util.assert_types_same(
                util.typeof(aRefToSelf),
                util.typeof(aRefToSelf.returns_ref_to_self())
                )

            util.assert_types_same(
                util.typeof(RefToSelf(0)),
                util.typeof(aRefToSelf.returns_copy_of_self())
                )


            return 0

        self.convert_expression(lambda: all_pass())

        with self.assertRaises(python_to_native_ast.ConversionException):
            self.convert_expression(
                lambda: util.assert_types_same(util.typeof(10), type_model.Int64.reference)
                )

    def test_agressive_branch_pruning(self):
        def g():
            if False:
                return 1
            else:
                return 0.0

        self.convert_expression(lambda: g())

    def test_agressive_branch_pruning_2(self):
        def g():
            if True:
                return 1
            return 0.0

        self.convert_expression(lambda: g())

    def test_simple(self):
        def f(a):
            return a+1

        self.assertTrue(self.compile(f)(10) == f(10))

    def test_simple_2(self):
        def f(a):
            b = a + 1
            return b

        self.assertTrue(self.compile(f)(10) == f(10))

    def test_simple_3(self):
        def f(a):
            if a > 0:
                b = a + 1
                return b
            else:
                return b

        with self.assertRaises(python_to_native_ast.ConversionException):
            self.compile(f)

    def test_simple_4(self):
        def g(a):
            a = a + 1

        def f(a):
            #a is passed as a reference to 'g'
            g(a)
            return a

        self.assertTrue(self.compile(f)(10) == 11)

    def test_simple_5(self):
        def returns(a):
            return a

        def g(a):
            a = a + 1

        def f(a):
            #the result of 'returns' is a value, not a reference
            g(returns(a))
            return a

        self.assertTrue(self.compile(f)(10) == 10)

    def test_simple_6(self):
        def returns(a):
            return util.ref(a)

        def g2(a):
            a = a + 1

        def f(a):
            g2(returns(a))
            return a

        self.assertEqual(self.compile(f)(10), 11)

    def test_simple_7(self):
        def returns(a):
            return util.ref(a)

        def g2(a):
            a = a + 1

        def f(a):
            x = 10
            y = util.ref(x)
            y = 20
            return x

        self.assertEqual(self.compile(f)(10), 20)

    def test_conversion(self):
        def f(a):
            return g(a)+g(1)

        self.assertTrue(self.compile(f)(10) == f(10))

    def test_branching_1(self):
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

    def test_references_to_temporaries(self):
        def test_references_to_temporaries_increment(a):
            a = a + 1

        def f(a):
            test_references_to_temporaries_increment(1.0+2.0)
            test_references_to_temporaries_increment(a)
            return a
        
        f_target = self.converter.convert(f, [util.Float64])
        functions = self.converter.extract_new_function_definitions()
        self.compiler.add_functions(functions)

        subfuncs = [f for f in functions.keys() if 'test_references_to_temporaries' in f]

        #ensure we only have one function
        self.assertEqual(len(subfuncs),1, subfuncs)

    def test_calling_functions_with_contentless_args(self):
        def caller(f,a):
            return f(a)

        def g(x):
            return x + 1

        def f(a):
            return caller(g,a)
        
        f_target = self.converter.convert(f, [util.Float64])
        functions = self.converter.extract_new_function_definitions()
        self.compiler.add_functions(functions)

        subfuncs = [functions[f] for f in functions.keys() if 'caller' in f]
        assert len(subfuncs) == 1

        type_of_f = subfuncs[0].args[0][1]

        self.assertEqual(type_of_f, native_ast.Type.Struct(()))

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

    def test_constructors_and_destructors_1(self):
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
        def h(c):
            return A(c, 2)

        def i(c):
            return (A(c, 2), A(c,3))

        def g(c):
            (A(c, 2), A(c,3))
            x = h(c)
            y = i(c)
            z = i(c).f0
            return z

        def f(a):
            c = Counter()
            res = g(util.addr(c))
            return c.alive
        
        f_comp = self.compile(f)

        self.assertEqual(f_comp(10), 1)

    def test_constructors_and_destructors_3(self):
        def g(c,arg0):  #['A'] -> int
            if (arg0).x == 0:
                return 1234

            var_1=(A(c,2),A(c,2))
            return 1

        def f(a):
            c = Counter()
            g(util.addr(c), A(util.addr(c), 0))
            return c.alive
        
        f_comp = self.compile(f)

        self.assertEqual(f_comp(10), 0)

    def test_constructors_and_destructors_4(self):
        def g(c):  #[] -> (A,A)
            x = ((((A(c,2),A(c,2))) if (0) else 
                    ((((A(c,1),A(c,2))) if (0) else ((A(c,0),A(c,0)))))))
            return x

        def f(a):
            c = Counter()
            g(util.addr(c))
            return c.alive
        
        f_comp = self.compile(f)

        self.assertEqual(f_comp(10), 0)

    def test_constructors_and_destructors_5(self):
        def returns_args(*args):
            return args
        def g(c):  #[] -> (A,A)
            z = returns_args(A(c,2), A(c,3))
            return z[0]

        def f(a):
            c = Counter()
            g(util.addr(c))
            return c.alive
        
        f_comp = self.compile(f)

        self.assertEqual(f_comp(10), 0)

        
    def test_constructors_and_destructors_fuzz(self):
        #we want to generate some random functions. Signatures always take a 'c', and can take
        #int, A, or (A,A), and can return same. Our goal is to verify constructor semantics in a
        #wide variety of situations.

        deep = test_config.tests_are_deep

        for i in xrange(2 if not deep else 20):
            functions, text, signatures = generate_functions(TEST_SEED + i, 4 if not deep else 16)

            for fname,signature in signatures.iteritems():
                if len(signature[0]) in (0,1):
                    def make_f_empty(f):
                        def caller(a):
                            c = Counter()
                            f(util.addr(c))
                            return c.alive
                        return caller

                    def make_f_int(f):
                        def caller(a):
                            c = Counter()
                            f(util.addr(c), 0)
                            return c.alive
                        return caller

                    def make_f_A(f):
                        def caller(a):
                            c = Counter()
                            f(util.addr(c), A(util.addr(c),0))
                            return c.alive
                        return caller
                    
                    def make_f_AA(f):
                        def caller(a):
                            c = Counter()
                            f(util.addr(c), (A(util.addr(c),0),A(util.addr(c),1)) )
                            return c.alive
                        return caller
                    
                    makers = {
                        (int,):make_f_int, 
                        (A,):make_f_A, 
                        ((A,A),): make_f_AA, 
                        (): make_f_empty
                        }

                    maker = makers[signature[0]]
                    f = maker(functions[fname])
                    f_comp = self.compile(f)
                    
                    result = f_comp(10)
                    assert result==0, "%s produced %s in \n\n%s" % (fname, result, text)

            