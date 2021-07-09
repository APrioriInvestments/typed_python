from typed_python import ModuleRepresentation, SerializationContext
from typed_python import sha_hash, identityHash, ListOf

import tempfile
import unittest
import os


def evaluateInto(module, code, codeDir=None):
    if codeDir is None:
        exec(code, module.getDict())
    else:
        filename = os.path.abspath(
            os.path.join(codeDir, sha_hash(code).hexdigest)
        )

        try:
            os.makedirs(os.path.dirname(filename))
        except OSError:
            pass

        with open(filename, "wb") as codeFile:
            codeFile.write(code.encode("utf8"))

        exec(compile(code, filename, "exec"), module.getDict())

    module.update()
    return module.getDict()


class TestModuleRepresentation(unittest.TestCase):
    def test_construction(self):
        mr = ModuleRepresentation("module")

        assert mr.getDict()['__name__'] == 'module'

    def test_addExternal(self):
        mr = ModuleRepresentation("module")

        assert mr.getDict()['__name__'] == 'module'

        mr.addExternal("hi", "bye")

        assert mr.getDict()['hi'] == 'bye'

    def test_canSeeRecursives(self):
        mr = ModuleRepresentation("module")

        evaluateInto(mr, "x = lambda: 10\n")

        assert not mr.getVisibleNames('x')

        evaluateInto(mr, "x = lambda: x + 10\n")
        assert mr.getVisibleNames('x') == ['x']

        evaluateInto(mr, "x = (lambda: x + 10, 1)\n")
        assert mr.getVisibleNames('x') == ['x']

        evaluateInto(mr, "x = [lambda: x + 10]\n")
        assert mr.getVisibleNames('x') == ['x']

        evaluateInto(mr, "x = set([lambda: x + 10])\n")
        assert mr.getVisibleNames('x') == ['x']

        evaluateInto(mr, "x = {'a': lambda: x + 10}\n")
        assert mr.getVisibleNames('x') == ['x']

        evaluateInto(
            mr,
            "class C:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "x = C(lambda: x + 10)\n"
        )
        assert mr.getVisibleNames('x') == ['x']

    def test_canSeeMutuallyRecursive(self):
        mr = ModuleRepresentation("module")

        evaluateInto(mr, "x = lambda: y\n")

        assert mr.getVisibleNames('x') == ['y']
        assert not mr.getVisibleNames('y')

        evaluateInto(mr, "y = lambda: x\n")

        assert mr.getVisibleNames('x') == ['y']
        assert mr.getVisibleNames('y') == ['x']

    def test_otherModulesAreExternal(self):
        mr = ModuleRepresentation("module")

        evaluateInto(mr, "import numpy")
        evaluateInto(mr, "array = numpy.array")

        assert len(mr.getExternalReferences('numpy')) == 1
        assert len(mr.getExternalReferences('array')) == 1

    def test_copy_into_basic(self):
        mr = ModuleRepresentation("module")

        evaluateInto(mr, "y = 10")

        mr2 = ModuleRepresentation("module")
        mr.copyInto(mr2, ['y'])

        assert mr2.getDict()['y'] == 10

    def test_copy_into_other_module(self):
        mr = ModuleRepresentation("module")

        evaluateInto(mr, "def f(): return y")
        evaluateInto(mr, "y = 10")

        assert mr.getDict()['f']() == 10

        mr2 = ModuleRepresentation("module")

        # give mr2 a complete copy of the variables in mr
        mr.copyInto(mr2, ['f', 'y'])

        # it should also evaluate correctly
        assert mr2.getDict()['f']() == 10

        # but now mr and mr2 should be independent
        evaluateInto(mr, "y = 20")
        assert mr.getDict()['f']() == 20
        assert mr2.getDict()['f']() == 10

    def test_duplication_of_mutually_recursive_functions(self):
        mr = ModuleRepresentation("module")

        evaluateInto(mr, "def f(): return g")
        evaluateInto(mr, "def g(): return f")

        mr2 = ModuleRepresentation("module")

        mr.copyInto(mr2, ['f', 'g'])

        assert mr.getDict()['f'].__globals__ is mr.getDict()
        assert mr.getDict()['g'].__globals__ is mr.getDict()

        assert mr2.getDict()['f'].__globals__ is mr2.getDict()
        assert mr2.getDict()['g'].__globals__ is mr2.getDict()

    def test_duplication_of_classes(self):
        mr = ModuleRepresentation("module")

        evaluateInto(mr, "y = 10")
        evaluateInto(
            mr,
            "class C:\n"
            "    def __init__(self, x):\n"
            "        self.x = x\n"
            "    def f(self):\n"
            "        return self.x + y\n"
            "    @property\n"
            "    def prop(self):\n"
            "        return y\n"
            "    @staticmethod\n"
            "    def staticMeth():\n"
            "        return y\n"
            "    @classmethod\n"
            "    def classMeth(cls):\n"
            "        return y\n"
        )
        evaluateInto(
            mr,
            "c = C(10)"
        )

        mr2 = ModuleRepresentation("module")

        # give mr2 a complete copy of the variables in mr
        mr.copyInto(mr2, ['C', 'y', 'c'])

        # 'C' should be a new type
        assert mr.getDict()['C'] is not mr2.getDict()['C']

        # each instance of 'c' should have the right type
        assert type(mr.getDict()['c']) is mr.getDict()['C']
        assert type(mr2.getDict()['c']) is not mr.getDict()['C']
        assert type(mr2.getDict()['c']) is mr2.getDict()['C']

        # it should also evaluate correctly
        assert mr.getDict()['C'](10).f() == 20
        assert mr.getDict()['C'](10).prop == 10
        assert mr.getDict()['C'].classMeth() == 10
        assert mr.getDict()['C'].staticMeth() == 10
        assert mr.getDict()['c'].f() == 20

        assert mr2.getDict()['C'](10).f() == 20
        assert mr2.getDict()['c'].f() == 20
        assert mr2.getDict()['C'](10).prop == 10
        assert mr2.getDict()['C'].classMeth() == 10
        assert mr2.getDict()['C'].staticMeth() == 10

        # but now mr and mr2 should be independent
        evaluateInto(mr, "y = 11")

        assert mr.getDict()['C'](10).f() == 21
        assert mr.getDict()['c'].f() == 21
        assert mr.getDict()['C'](10).prop == 11
        assert mr.getDict()['C'].classMeth() == 11
        assert mr.getDict()['C'].staticMeth() == 11

        assert mr2.getDict()['C'](10).f() == 20
        assert mr2.getDict()['c'].f() == 20
        assert mr2.getDict()['C'](10).prop == 10
        assert mr2.getDict()['C'].classMeth() == 10
        assert mr2.getDict()['C'].staticMeth() == 10

    def test_tp_functions(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(mr, "from typed_python import Entrypoint", td)
            evaluateInto(
                mr,
                "@Entrypoint\n"
                "def f(x):\n"
                "    return g(x)\n",
                td
            )

            assert mr.getVisibleNames('f') == ['g']

            # verify we can execute and compile this, and that it throws
            # because 'g' is not defined

            with self.assertRaisesRegex(NameError, "name 'g' is not defined"):
                mr.getDict()['f'](10)

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['f'])

            evaluateInto(
                mr2,
                "def g(x):\n"
                "    return x + 1\n",
                td
            )

            assert mr2.getDict()['f'](10) == 11

            assert identityHash(mr.getDict()['f']) != identityHash(mr2.getDict()['f'])

    def test_tp_classes(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(mr, "from typed_python import Class, Member, Final", td)
            evaluateInto(
                mr,
                "class C(Class, Final):\n"
                "    m = Member(int)\n"
                "    def __init__(self, x):\n"
                "        self.m = x\n"
                "    def f(self):\n"
                "        return g(self.m)\n"
                "    @staticmethod\n"
                "    def staticF(x):\n"
                "        return g2(x)\n"
                "    @property\n"
                "    def propF(self):\n"
                "        return g3(self.m)\n",
                td
            )

            assert mr.getVisibleNames('C') == ['g', 'g2', 'g3']

            # verify we can execute and compile this, and that it throws
            # because 'g' is not defined

            with self.assertRaisesRegex(NameError, "name 'g' is not defined"):
                mr.getDict()['C'](10).f()

            with self.assertRaisesRegex(NameError, "name 'g2' is not defined"):
                mr.getDict()['C'](10).staticF(10)

            with self.assertRaisesRegex(NameError, "name 'g3' is not defined"):
                mr.getDict()['C'](10).propF

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['C'])

            evaluateInto(
                mr2,
                "def g(x):\n"
                "    return x + 1\n"
                "def g2(x):\n"
                "    return x + 2\n"
                "def g3(x):\n"
                "    return x + 3\n",
                td
            )

            assert mr2.getDict()['C'](10).f() == 11
            assert mr2.getDict()['C'](10).staticF(10) == 12
            assert mr2.getDict()['C'](10).propF == 13

    def test_tp_class_hierarchy(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(mr, "from typed_python import Class, Member, Final", td)
            evaluateInto(
                mr,
                "class Base(Class):\n"
                "    def f(self) -> int:\n"
                "        return Base.g()\n"
                "    @staticmethod\n"
                "    def g():\n"
                "        return 2\n"
                "class Child(Base):\n"
                "    def f(self) -> int:\n"
                "        return 3\n",
                td
            )

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['Base', 'Child'])

            assert mr2.getDict()['Child'].BaseClasses[0] is not mr.getDict()['Base']
            assert mr2.getDict()['Child'].BaseClasses[0] is mr2.getDict()['Base']

    def test_tp_class_mutually_recursive_child_and_base(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "from typed_python import Forward, Class\n"
                "Base = Forward('Optimizer')\n"
                "@Base.define\n"
                "class Base(Class):\n"
                "    def __add__(self, x: Base) -> Base:\n"
                "        return Child()\n",
                td
            )

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['Base', 'Class'])

            evaluateInto(
                mr2,
                "class Child(Base):\n"
                "    def __add__(self, x: Base) -> Base:\n"
                "        return Child()\n",
                td
            )

            Child = mr2.getDict()['Child']
            Base = mr2.getDict()['Base']

            assert type(Base() + Base()) is Child

    def test_copy_into_doesnt_duplicate_unnecessarily(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "list1 = [1, 2, 3]\n"
                "def f(): return list2\n"
                "list2 = [f]\n",
                td
            )

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['list1', 'list2', 'f'])

            assert mr.getDict()['list1'] is mr2.getDict()['list1']
            assert mr.getDict()['list2'] is not mr2.getDict()['list2']

    def test_duplicated_class_object_module_and_name(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "class C:\n"
                "    'docstring'\n"
                "    def f(self): return 10\n"
                "    def g(self): return C\n",
                td
            )

            assert mr.getVisibleNames('C') == ['C']

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['C'])

            C1 = mr.getDict()['C']
            C2 = mr2.getDict()['C']

            assert C1.__doc__ == C2.__doc__
            assert C1.__module__ == C2.__module__

    def test_class_method_closures_get_copied(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "class C:\n"
                "    def c(self):\n"
                "        return __class__\n"
                "    def c2(self):\n"
                "        return C\n"
                "    def c3(self):\n"
                "        return x\n"
                "x = 10\n",
                td
            )

            assert mr.getVisibleNames('C') == ['C', 'x']

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['C', 'x'])

            C1 = mr.getDict()['C']
            C2 = mr2.getDict()['C']

            assert C1 is not C2

            assert C1().c() is C1

            assert C2().c2() is C2
            assert C2().c() is C2
            assert C2().c() is not C1

    def test_copy_in_native_base_class(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "class Base:\n"
                "    pass\n"
                "class Child(Base):\n"
                "    def f(self):\n"
                "        return x\n",
                td
            )

            assert mr.getDict()['Base'] not in mr.getInternalReferences('Child')

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['Child'])

            Base = mr.getDict()['Base']
            Child = mr2.getDict()['Child']

            assert issubclass(Child, Base)

    def test_copy_in_tp_base_class(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "from typed_python import Class\n"
                "class Base(Class):\n"
                "    pass\n"
                "class Child(Base):\n"
                "    def f(self) -> int:\n"
                "        return x\n",
                td
            )

            assert mr.getDict()['Base'] not in mr.getInternalReferences('Child')

            mr2 = ModuleRepresentation("module")

            # copy into mr2
            mr.copyInto(mr2, ['Child'])

            Base = mr.getDict()['Base']
            Child = mr2.getDict()['Child']

            assert issubclass(Child, Base)

    def test_copying_classes_with_methods_is_transitive(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "class C:\n"
                "    def f(self):\n"
                "        self.f = x\n",
                td
            )

            mr2 = ModuleRepresentation("module")
            mr.copyInto(mr2, ['C'])

            assert mr2.getDict()['C'].f.__globals__ is mr2.getDict()

            mr3 = ModuleRepresentation("module")
            mr2.copyInto(mr3, ['C'])

            assert mr3.getDict()['C'].f.__globals__ is not mr2.getDict()
            assert mr3.getDict()['C'].f.__globals__ is mr3.getDict()

    def test_assign_to_global_scope_updates_class_and_subclass(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "class Base:\n"
                "    def __init__(self, x):\n"
                "        self.x = x + g()\n"
                "    def asChild(self):\n"
                "        return Child(self.x)\n",
                td
            )

            mr2 = ModuleRepresentation("module")
            mr.copyInto(mr2, ['Base'])

            assert mr2.getDict()['Base'].__init__.__globals__ is mr2.getDict()

            evaluateInto(
                mr2,
                "class Child(Base):\n"
                "    def __init__(self, x):\n"
                "        super().__init__(x + 100 + g())\n",
                td
            )

            mr3 = ModuleRepresentation("module")

            # copy into mr2
            mr2.copyInto(mr3, ['Child', 'Base'])

            Base = mr3.getDict()['Base']

            assert Base.__init__.__globals__ is mr3.getDict()

    def test_duplicated_child_class_interchangeable(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")
            mr2 = ModuleRepresentation("module")

            for m in [mr, mr2]:
                evaluateInto(
                    m,
                    "from typed_python import Class\n"
                    "class Base(Class):\n"
                    "    pass\n"
                    "class Child(Base):\n"
                    "    pass\n",
                    td
                )

            Base1 = mr.getDict()['Base']
            Child1 = mr.getDict()['Child']

            Base2 = mr2.getDict()['Base']
            Child2 = mr2.getDict()['Child']

            sc = SerializationContext()

            # round-trip serialize these two, which places them in the main memo
            sc.deserialize(sc.serialize(Child1))
            sc.deserialize(sc.serialize(Base1))

            ChildDup2 = sc.deserialize(sc.serialize(Child2))

            # we should see that ChildDup2 is actually equal to Child1 because of the memo
            assert ChildDup2 is Child1

            # but that shouldn't prevent us from using it interchangeably because
            # its the 'type identical'
            ListOf(Base2)([Child1()])

    def test_serialization_robust_to_module_ordering(self):
        with tempfile.TemporaryDirectory() as td:
            mr = ModuleRepresentation("module")

            evaluateInto(
                mr,
                "from typed_python import NotCompiled\n"
                "x = 1\n"
                "y = 1\n"
                "def f():\n"
                "    return x + y\n"
                "@NotCompiled\n"
                "def fTyped():\n"
                "    return x + y\n",
                td
            )

            f = mr.getDict()['f']
            fTyped = mr.getDict()['fTyped']

            sc = (
                SerializationContext()
                .withoutCompression()
                .withoutLineInfoEncoded()
                .withSerializeHashSequence()
            )

            globalDict = mr.getDict()

            # serialize once with 'x' before 'y'
            del globalDict['x']
            del globalDict['y']
            globalDict['x'] = 1
            globalDict['y'] = 1
            assert list(globalDict).index('x') < list(globalDict).index('y')
            serialization1 = sc.serialize((f, fTyped))

            # serialize again with 'x' after 'y'
            del globalDict['x']
            del globalDict['y']
            globalDict['y'] = 1
            globalDict['x'] = 1
            assert list(globalDict).index('x') > list(globalDict).index('y')
            serialization2 = sc.serialize((f, fTyped))

            # bytes on the wire shouldn't depend on this
            assert serialization1 == serialization2
