import numpy

from typed_python import ListOf
from typed_python._types import CompilerVisiblePyObj


def test_make_cvpo_basic():
    assert CompilerVisiblePyObj.create(int).kind == 'Type'

    assert CompilerVisiblePyObj.create((1, 2, 3)).kind == 'PyTuple'
    assert CompilerVisiblePyObj.create((1, 2, 3)).elements[0].kind == 'Instance'

    assert CompilerVisiblePyObj.create({1, 2, 3}).kind == 'PySet'
    assert CompilerVisiblePyObj.create({1, 2, 3}).elements[0].kind == 'Instance'

    assert CompilerVisiblePyObj.create([1, 2, 3]).kind == 'PyList'
    assert CompilerVisiblePyObj.create([1, 2, 3]).elements[0].kind == 'Instance'

    assert CompilerVisiblePyObj.create({1: 2}).kind == 'PyDict'
    assert CompilerVisiblePyObj.create({'1': 2}).elements[0].kind == 'Instance'
    assert CompilerVisiblePyObj.create({'1': 2}).keys[0].kind == 'String'

    assert CompilerVisiblePyObj.create(1).kind == 'Instance'
    assert CompilerVisiblePyObj.create(1).instance == 1

    assert CompilerVisiblePyObj.create('1').kind == 'String'
    assert CompilerVisiblePyObj.create('1').stringValue == '1'

    assert CompilerVisiblePyObj.create(ListOf(int)((1, 2, 3))).kind == 'Instance'
    assert CompilerVisiblePyObj.create(ListOf(int)((1, 2, 3))).instance[1] == 2


def test_cvpo_function():
    y = 10

    def f(x: int, z=20):
        return 10 + y

    cvpo = CompilerVisiblePyObj.create(f)

    assert cvpo.kind == 'PyFunction'
    assert cvpo.name == 'f'
    assert cvpo.moduleName == 'typed_python.compiler_visible_py_obj_test'
    assert cvpo.func_name.kind == 'String'
    assert cvpo.func_module.kind == 'String'
    assert cvpo.func_closure.kind == 'PyTuple'
    assert cvpo.func_closure.elements[0].kind == 'PyCell'
    assert cvpo.func_closure.elements[0].cell_contents.pyobj == 10
    assert cvpo.func_globals.kind == "PyModuleDict"
    assert cvpo.func_globals.name == 'typed_python.compiler_visible_py_obj_test'
    assert cvpo.func_annotations.pyobj['x'] is int
    assert cvpo.func_defaults.pyobj == (20,)
    assert not hasattr(cvpo, 'func_kwdefaults')

    assert cvpo.func_code.kind == 'PyCodeObject'
    assert cvpo.func_code.co_argcount == 2
    assert cvpo.func_code.co_kwonlyargcount == 0
    assert cvpo.func_code.co_flags == f.__code__.co_flags
    assert cvpo.func_code.co_posonlyargcount == f.__code__.co_posonlyargcount
    assert cvpo.func_code.co_nlocals == f.__code__.co_nlocals
    assert cvpo.func_code.co_stacksize == f.__code__.co_stacksize
    assert cvpo.func_code.co_firstlineno == f.__code__.co_firstlineno
    assert cvpo.func_code.co_code.pyobj == f.__code__.co_code
    assert cvpo.func_code.co_consts.pyobj == f.__code__.co_consts
    assert cvpo.func_code.co_names.pyobj == f.__code__.co_names
    assert cvpo.func_code.co_varnames.pyobj == f.__code__.co_varnames
    assert cvpo.func_code.co_freevars.pyobj == f.__code__.co_freevars
    assert cvpo.func_code.co_cellvars.pyobj == f.__code__.co_cellvars
    assert cvpo.func_code.co_name.pyobj == f.__code__.co_name
    assert cvpo.func_code.co_filename.pyobj == f.__code__.co_filename


def test_cvpo_numpy_internals():
    assert CompilerVisiblePyObj.create(numpy.array).kind == 'NamedPyObject'

    # in theory, we could do better than this by using the reduce pathway.
    # however, that would involve calling into arbitrary python code during walk time
    # which is dangerous because it can (a) modify objects during the walk and
    # (b) it can release the GIL allowing other threads to run (who might then
    # modify objects) all of which make it impossible to get a good trace of the
    # graph.
    assert CompilerVisiblePyObj.create(numpy.sin).kind == 'ArbitraryPyObject'
    assert CompilerVisiblePyObj.create(numpy.array([1])).kind == 'ArbitraryPyObject'


def test_cvpo_builtins():
    assert CompilerVisiblePyObj.create(__builtins__).kind == 'PyModuleDict'
    assert CompilerVisiblePyObj.create(__builtins__).module_dict_of.kind == 'PyModule'
    assert CompilerVisiblePyObj.create(set).kind == 'NamedPyObject'
    assert CompilerVisiblePyObj.create(print).kind == 'NamedPyObject'
    assert CompilerVisiblePyObj.create(len).kind == 'NamedPyObject'
    assert CompilerVisiblePyObj.create(range).kind == 'NamedPyObject'


def test_cvpo_class():
    class C:
        def f(self):
            return gInst

        def fMeth(self):
            return gMeth

        @property
        def aProp(self):
            return 20

    class G(C):
        def __init__(self):
            self.y = 23

        @staticmethod
        def staticMeth():
            return 10

        @classmethod
        def clsMeth(cls):
            return 10

    gInst = G()
    gMeth = gInst.f

    cvpo = CompilerVisiblePyObj.create(C)

    assert cvpo.kind == 'PyClass'
    assert cvpo.name == 'C'
    assert cvpo.moduleName == 'typed_python.compiler_visible_py_obj_test'
    assert cvpo.cls_dict.kind == 'PyClassDict'

    assert cvpo.cls_dict.byKey['aProp'].kind == 'PyProperty'
    assert cvpo.cls_dict.byKey['aProp'].prop_get.kind == 'PyFunction'

    assert cvpo.cls_dict.byKey['f'].kind == 'PyFunction'

    gMethPO = cvpo.cls_dict.byKey['fMeth'].func_closure.elements[0].cell_contents
    assert gMethPO.kind == 'PyBoundMethod'
    assert gMethPO.meth_self.pyobj is gInst
    assert gMethPO.meth_func.kind == 'PyFunction'

    gInstPO = cvpo.cls_dict.byKey['f'].func_closure.elements[0].cell_contents
    assert gInstPO.pyobj is gInst
    assert gInstPO.inst_dict.kind == 'PyDict'
    gInstPO.inst_dict.byKey['y'].kind == 'Instance'

    GPO = gInstPO.inst_type
    assert GPO.kind == 'PyClass'
    assert GPO.cls_bases.pyobj == (C,)

    # we can't actually tell that this is a class dict because
    # its not the base class
    assert GPO.cls_dict.kind == 'PyDict'

    assert GPO.cls_dict.byKey['staticMeth'].kind == 'PyStaticMethod'
    assert GPO.cls_dict.byKey['staticMeth'].meth_func.kind == 'PyFunction'

    assert GPO.cls_dict.byKey['clsMeth'].kind == 'PyClassMethod'
    assert GPO.cls_dict.byKey['clsMeth'].meth_func.kind == 'PyFunction'


def test_cvpo_rehydration():
    assert CompilerVisiblePyObj.create(1, False).pyobj == 1
    assert CompilerVisiblePyObj.create('1', False).pyobj == '1'
    assert CompilerVisiblePyObj.create(b'1', False).pyobj == b'1'
    assert CompilerVisiblePyObj.create([1, 2, 'hi'], False).pyobj == [1, 2, 'hi']
    assert CompilerVisiblePyObj.create((1, 2, 'hi'), False).pyobj == (1, 2, 'hi')
    assert CompilerVisiblePyObj.create({1, 2}, False).pyobj == {1, 2}
    assert CompilerVisiblePyObj.create({1: 2, '3': 4}, False).pyobj == {1: 2, '3': 4}
    assert CompilerVisiblePyObj.create(print, False).pyobj is print

    def f():
        return 1

    assert CompilerVisiblePyObj.create(f, False).pyobj() == 1

    class C:
        @property
        def aProp(self):
            return 10

        @staticmethod
        def aSM(x):
            return x + 1

        @classmethod
        def aCM(x):
            return x + 1

        def getCInst(self):
            return cInst

        def getCInstMeth(self):
            return cInstMeth

    cInst = C()
    cInstMeth = cInst.getCInstMeth

    C2cvpo = CompilerVisiblePyObj.create(C, False)
    C2 = C2cvpo.pyobj

    assert C2 is not C
    assert C2.__name__ == C.__name__
    assert C2.__module__ == C.__module__
    assert C2.getCInst is not C.getCInst

    c2Inst = C2cvpo.cls_dict.byKey['getCInst'].func_closure.elements[0].cell_contents.pyobj
    c2InstMeth = C2cvpo.cls_dict.byKey['getCInstMeth'].func_closure.elements[0].cell_contents.pyobj

    assert C2().getCInst() is c2Inst
    assert C2().getCInstMeth() is c2InstMeth
    assert c2InstMeth() is c2InstMeth
