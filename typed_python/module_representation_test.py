from typed_python import ModuleRepresentation
import unittest


def evaluateInto(module, code):
    exec(code, module.getDict())
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
        
        evaluateInto(mr, 
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

        import numpy
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
