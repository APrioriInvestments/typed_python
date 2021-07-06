from typed_python import ModuleRepresentation
from typed_python import sha_hash, identityHash
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

            print(mr2.getDict()['Child'].BaseClasses)

            assert mr2.getDict()['Child'].BaseClasses[0] is not mr.getDict()['Base']
            assert mr2.getDict()['Child'].BaseClasses[0] is mr2.getDict()['Base']
