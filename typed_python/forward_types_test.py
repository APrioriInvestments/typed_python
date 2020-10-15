#   Copyright 2017-2019 typed_python Authors
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
import unittest

from typed_python import (
    TupleOf, OneOf, Alternative, Class, Member, Function,
    Forward, NamedTuple, Tuple, Dict, ListOf, ConstDict, Module
)

from typed_python._types import recursiveTypeGroup, isRecursive


class ForwardTypesTests(unittest.TestCase):
    def test_basic_forward_type_resolution(self):
        f = Forward("f")
        T = TupleOf(f)

        f.define(int)

        self.assertEqual(T.__name__, "TupleOf(int)")

        self.assertEqual(f.get(), int)

    def test_class_in_forward(self):
        class C(Class):
            x = Member(int)

        Fwd = Forward("Forward")
        Fwd = Fwd.define(OneOf(None, C, TupleOf(Fwd), ListOf(Fwd), ConstDict(str, Fwd)))

        Fwd(C())

    def test_recursive_forwards(self):
        Value = Forward("Value")
        Value.define(OneOf(
            None,
            ConstDict(str, Value)
        ))

    def test_forward_type_resolution_sequential(self):
        F0 = Forward("f0")
        T0 = TupleOf(F0)

        F1 = Forward("f1")
        T1 = TupleOf(F1)

        F1.define(T0)
        F0.define(int)

        self.assertEqual(T1.__name__, "TupleOf(TupleOf(int))")

    def test_recursive_alternative(self):
        List = Forward("List")
        List = List.define(Alternative(
            "List",
            Node={'head': int, 'tail': List },
            Leaf={},
            unpack=lambda self: () if self.matches.Leaf else (self.head,) + self.tail.unpack()
        ))

        # ensure recursive implementation actually works
        lst = List.Leaf()

        for i in range(100):
            lst = List.Node(head=i, tail=lst)

        self.assertEqual(list(lst.unpack()), list(reversed(range(100))))

    def test_mutually_recursive_classes(self):
        B0 = Forward("B")

        class A(Class):
            bvals = Member(TupleOf(B0))

        class B(Class):
            avals = Member(TupleOf(A))

        B0 = B0.define(B)
        a = A()
        b = B()

        a.bvals = (b,)
        b.avals = (a,)

        self.assertTrue(a.bvals[0].avals[0] == a)

    def test_recursives_held_infinitely_throws(self):
        X = Forward("X")

        with self.assertRaisesRegex(Exception, "type-containment cycle"):
            X = X.define(NamedTuple(x=X))

        with self.assertRaisesRegex(Exception, "type-containment cycle"):
            X = X.define(OneOf(None, X))

        with self.assertRaisesRegex(Exception, "type-containment cycle"):
            X = X.define(Tuple(X))

    def test_tuple_of_one_of(self):
        X = Forward("X")
        T = TupleOf(OneOf(None, X))

        X = X.define(T)

        str(X)

        anX = X( (None,) )

        self.assertTrue("Unresolved" not in str(X))

        anotherX = X( (anX, anX) )

        self.assertEqual(anotherX[0], anX)

    def test_deep_forwards_work(self):
        X = Forward("X")
        X = X.define(TupleOf(TupleOf(TupleOf(TupleOf(OneOf(None, X))))))

        str(X)

        anX = X( ((((None,),),),) )

        anotherX = X( ((((anX,),),),) )

        self.assertEqual(anotherX[0][0][0][0], anX)

    def test_recursive_dicts(self):
        D = Forward("D")
        D = D.define(Dict(int, OneOf(int, D)))

        dInst = D()
        dInst[10] = dInst
        dInst[20] = 20

        self.assertEqual(dInst[10][10][10][20], 20)

        # stringifying it shouldn't blow up
        str(dInst)

        self.assertEqual(dInst, dInst[10])

    def test_forward_types_not_instantiatable(self):
        F = Forward("int")
        F.define(int)

        with self.assertRaisesRegex(Exception, "Can't construct"):
            F(10)

    def test_recursive_alternatives(self):
        X = Forward("X")
        X = X.define(
            Alternative(
                "X",
                A=dict(x=X, y=int),
                B=dict()
            )
        )

        anX = X.A(x=X.B(), y=21)

        self.assertEqual(anX.y, 21)
        self.assertTrue(anX.x.matches.B)

    def test_recursive_oneof(self):
        OneOfTupleOfSelf = Forward("OneOfTupleOfSelf")
        OneOfTupleOfSelf = OneOfTupleOfSelf.define(OneOf(None, TupleOf(OneOfTupleOfSelf)))

        self.assertEqual(OneOfTupleOfSelf.__qualname__, "OneOfTupleOfSelf")
        self.assertEqual(OneOf(None, TupleOf(OneOfTupleOfSelf)).__qualname__, "OneOfTupleOfSelf")

        TO = Forward("TO")
        TO = TO.define(TupleOf(OneOf(None, TO)))
        self.assertEqual(TO.__qualname__, "TO")
        self.assertIs(TO.ElementType.Types[1], TO)
        self.assertTrue(isRecursive(TO.ElementType))
        self.assertEqual(TO.ElementType.__qualname__, "OneOf(None, TO)")

    def test_forward_types_identity_container(self):
        for ContainerKind in [TupleOf]:
            module = Module("M")
            module.RecursiveThing = OneOf(None, ContainerKind(module.RecursiveThing))

            aRecursiveThing = module.RecursiveThing((None, (None,)))

            self.assertEqual(type(aRecursiveThing), module.RecursiveThing.Types[1])
            self.assertEqual(type(aRecursiveThing), ContainerKind(module.RecursiveThing))

    def test_forward_types_identity_dict(self):
        for DictKind in [Dict, ConstDict]:
            module = Module("M")
            module.RecursiveThing = OneOf(None, DictKind(int, module.RecursiveThing), DictKind(module.RecursiveThing, int))

            aRecursiveThing = module.RecursiveThing({1: None})
            self.assertEqual(type(aRecursiveThing), DictKind(int, module.RecursiveThing))

            aRecursiveThing = module.RecursiveThing({None: 1})
            self.assertEqual(type(aRecursiveThing), DictKind(module.RecursiveThing, int))

    def test_forward_types_twice(self):
        module = Module("M")
        module.RecursiveThing = OneOf(
            None,
            Dict(int, TupleOf(module.RecursiveThing)),
            Dict(str, TupleOf(module.RecursiveThing))
        )

        self.assertEqual(
            module.RecursiveThing.Types[1].ValueType, TupleOf(module.RecursiveThing)
        )
        self.assertEqual(
            module.RecursiveThing.Types[2].ValueType, TupleOf(module.RecursiveThing)
        )

    def test_forwards_resolve_to_forward_in_cycle(self):
        Fs = [Forward(f"f{i}") for i in range(4)]

        Fs[0].define(Fs[1])
        Fs[2].define(Fs[3])
        Fs[1].define(Fs[2])

        with self.assertRaisesRegex(TypeError, "Can't resolve a forward to itself!"):
            Fs[3].define(Fs[0])

    def test_forwards_resolve_to_forward(self):
        Intermediate1 = Forward("Intermediate1")
        Intermediate2 = Forward("Intermediate2")

        Intermediate1.define(Intermediate2)

        T0 = Tuple(Intermediate1, Intermediate1)
        T1 = Tuple(Intermediate1, int)

        Intermediate2.define(Tuple(float, float))

        self.assertEqual(T0, Tuple(Tuple(float, float), Tuple(float, float)))
        self.assertEqual(T1, Tuple(Tuple(float, float), int))

        T0(((1, 2), (3, 4)))

    def test_check_recursive_group(self):
        module = Module("M")

        @module.define
        class A(Class):
            b = Member(OneOf(None, module.B))

        @module.define
        class B(Class):
            a = Member(OneOf(None, module.A))

        self.assertEqual(recursiveTypeGroup(A), recursiveTypeGroup(B))

        self.assertTrue(A in recursiveTypeGroup(A))
        self.assertTrue(B in recursiveTypeGroup(A))

    def test_call_function_with_unresolved_forward_fails(self):
        X = Forward("X")

        @Function
        def f() -> X:
            return 0

        with self.assertRaisesRegex(Exception, "unresolved forwards"):
            f()

        X.define(int)

        f()
