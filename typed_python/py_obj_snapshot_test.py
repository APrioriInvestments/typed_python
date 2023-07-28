import numpy

from typed_python import ListOf
from typed_python._types import PyObjSnapshot


def test_make_snapshot_basic():
    assert PyObjSnapshot.create(int).kind == 'Type'

    assert PyObjSnapshot.create((1, 2, 3)).kind == 'PyTuple'
    assert PyObjSnapshot.create((1, 2, 3)).elements[0].kind == 'Instance'

    assert PyObjSnapshot.create({1, 2, 3}).kind == 'PySet'
    assert PyObjSnapshot.create({1, 2, 3}).elements[0].kind == 'Instance'

    assert PyObjSnapshot.create([1, 2, 3]).kind == 'PyList'
    assert PyObjSnapshot.create([1, 2, 3]).elements[0].kind == 'Instance'

    assert PyObjSnapshot.create({1: 2}).kind == 'PyDict'
    assert PyObjSnapshot.create({'1': 2}).elements[0].kind == 'Instance'
    assert PyObjSnapshot.create({'1': 2}).keys[0].kind == 'String'

    assert PyObjSnapshot.create(1).kind == 'Instance'
    assert PyObjSnapshot.create(1).instance == 1

    assert PyObjSnapshot.create('1').kind == 'String'
    assert PyObjSnapshot.create('1').stringValue == '1'

    assert PyObjSnapshot.create(ListOf(int)((1, 2, 3))).kind == 'Instance'
    assert PyObjSnapshot.create(ListOf(int)((1, 2, 3))).instance[1] == 2


def test_snapshot_function():
    y = 10

    def f(x: int, z=20):
        return 10 + y

    snapshot = PyObjSnapshot.create(f)

    assert snapshot.kind == 'PyFunction'
    assert snapshot.name == 'f'
    assert snapshot.moduleName == 'typed_python.py_obj_snapshot_test'
    assert snapshot.func_name.kind == 'String'
    assert snapshot.func_module.kind == 'String'
    assert snapshot.func_closure.kind == 'PyTuple'
    assert snapshot.func_closure.elements[0].kind == 'PyCell'
    assert snapshot.func_closure.elements[0].cell_contents.pyobj == 10
    assert snapshot.func_globals.kind == "PyModuleDict"
    assert snapshot.func_globals.name == 'typed_python.py_obj_snapshot_test'
    assert snapshot.func_annotations.pyobj['x'] is int
    assert snapshot.func_defaults.pyobj == (20,)
    assert not hasattr(snapshot, 'func_kwdefaults')

    assert snapshot.func_code.kind == 'PyCodeObject'
    assert snapshot.func_code.co_argcount == 2
    assert snapshot.func_code.co_kwonlyargcount == 0
    assert snapshot.func_code.co_flags == f.__code__.co_flags
    assert snapshot.func_code.co_posonlyargcount == f.__code__.co_posonlyargcount
    assert snapshot.func_code.co_nlocals == f.__code__.co_nlocals
    assert snapshot.func_code.co_stacksize == f.__code__.co_stacksize
    assert snapshot.func_code.co_firstlineno == f.__code__.co_firstlineno
    assert snapshot.func_code.co_code.pyobj == f.__code__.co_code
    assert snapshot.func_code.co_consts.pyobj == f.__code__.co_consts
    assert snapshot.func_code.co_names.pyobj == f.__code__.co_names
    assert snapshot.func_code.co_varnames.pyobj == f.__code__.co_varnames
    assert snapshot.func_code.co_freevars.pyobj == f.__code__.co_freevars
    assert snapshot.func_code.co_cellvars.pyobj == f.__code__.co_cellvars
    assert snapshot.func_code.co_name.pyobj == f.__code__.co_name
    assert snapshot.func_code.co_filename.pyobj == f.__code__.co_filename


def test_snapshot_numpy_internals():
    assert PyObjSnapshot.create(numpy.array).kind == 'NamedPyObject'

    # in theory, we could do better than this by using the reduce pathway.
    # however, that would involve calling into arbitrary python code during walk time
    # which is dangerous because it can (a) modify objects during the walk and
    # (b) it can release the GIL allowing other threads to run (who might then
    # modify objects) all of which make it impossible to get a good trace of the
    # graph.
    assert PyObjSnapshot.create(numpy.sin).kind == 'ArbitraryPyObject'
    assert PyObjSnapshot.create(numpy.array([1])).kind == 'ArbitraryPyObject'


def test_snapshot_builtins():
    assert PyObjSnapshot.create(__builtins__).kind == 'PyModuleDict'
    assert PyObjSnapshot.create(__builtins__).module_dict_of.kind == 'PyModule'
    assert PyObjSnapshot.create(set).kind == 'NamedPyObject'
    assert PyObjSnapshot.create(print).kind == 'NamedPyObject'
    assert PyObjSnapshot.create(len).kind == 'NamedPyObject'
    assert PyObjSnapshot.create(range).kind == 'NamedPyObject'


def test_snapshot_class():
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

    snapshot = PyObjSnapshot.create(C)

    assert snapshot.kind == 'PyClass'
    assert snapshot.name == 'C'
    assert snapshot.moduleName == 'typed_python.py_obj_snapshot_test'
    assert snapshot.cls_dict.kind == 'PyClassDict'

    assert snapshot.cls_dict.byKey['aProp'].kind == 'PyProperty'
    assert snapshot.cls_dict.byKey['aProp'].prop_get.kind == 'PyFunction'

    assert snapshot.cls_dict.byKey['f'].kind == 'PyFunction'

    gMethPO = snapshot.cls_dict.byKey['fMeth'].func_closure.elements[0].cell_contents
    assert gMethPO.kind == 'PyBoundMethod'
    assert gMethPO.meth_self.pyobj is gInst
    assert gMethPO.meth_func.kind == 'PyFunction'

    gInstPO = snapshot.cls_dict.byKey['f'].func_closure.elements[0].cell_contents
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


def test_snapshot_rehydration():
    assert PyObjSnapshot.create(1, False).pyobj == 1
    assert PyObjSnapshot.create('1', False).pyobj == '1'
    assert PyObjSnapshot.create(b'1', False).pyobj == b'1'
    assert PyObjSnapshot.create([1, 2, 'hi'], False).pyobj == [1, 2, 'hi']
    assert PyObjSnapshot.create((1, 2, 'hi'), False).pyobj == (1, 2, 'hi')
    assert PyObjSnapshot.create({1, 2}, False).pyobj == {1, 2}
    assert PyObjSnapshot.create({1: 2, '3': 4}, False).pyobj == {1: 2, '3': 4}
    assert PyObjSnapshot.create(print, False).pyobj is print

    def f():
        return 1

    assert PyObjSnapshot.create(f, False).pyobj() == 1

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

    C2snapshot = PyObjSnapshot.create(C, False)
    C2 = C2snapshot.pyobj

    assert C2 is not C
    assert C2.__name__ == C.__name__
    assert C2.__module__ == C.__module__
    assert C2.getCInst is not C.getCInst

    c2Inst = C2snapshot.cls_dict.byKey['getCInst'].func_closure.elements[0].cell_contents.pyobj
    c2InstMeth = C2snapshot.cls_dict.byKey['getCInstMeth'].func_closure.elements[0].cell_contents.pyobj

    assert C2().getCInst() is c2Inst
    assert C2().getCInstMeth() is c2InstMeth
    assert c2InstMeth() is c2InstMeth


def test_snapshot_graph():
    def f():
        return f

    s1 = PyObjSnapshot.create(f)
    s2 = PyObjSnapshot.create(f)

    assert s1.graph is not s2.graph
    assert s1.kind == 'PyFunction'
    assert s1.func_closure.graph is s1.graph
    assert s1.func_closure.graph is not s2.func_closure.graph
