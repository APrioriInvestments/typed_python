#include "Python.h"
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "Type.hpp"

class Arena {
public:
    std::vector<char> m_data;
};


struct native_instance_wrapper {
    PyObject_HEAD
  
    Type* mType;

    std::shared_ptr<Arena> mArenaPtr;

    size_t mArenaOffset;

    static PyMethodDef* typeMethods() {
        static PyMethodDef typed_python_TypeMethods[] = {
            {NULL, NULL}
        };

        return typed_python_TypeMethods;
    };

    static PyTypeObject* typeObj(Type* inType) { 
        static std::mutex mutex;
        static std::map<Type*, PyTypeObject*> types;

        std::lock_guard<std::mutex> lock(mutex);

        auto it = types.find(inType);
        if (it != types.end()) {
            return it->second;
        }

        types[inType] = new PyTypeObject {
                PyVarObject_HEAD_INIT(NULL, 0)
                inType->name().c_str(),    /* tp_name */
                sizeof(native_instance_wrapper), /* tp_basicsize */
                0,                         // tp_itemsize
                0,                         // tp_dealloc
                0,                         // tp_print
                0,                         // tp_getattr
                0,                         // tp_setattr
                0,                         // tp_reserved
                0,                         // tp_repr
                0,                         // tp_as_number
                0,                         // tp_as_sequence
                0,                         // tp_as_mapping
                0,                         // tp_hash
                0,                         // tp_call
                0,                         // tp_str
                0,                         // tp_getattro
                0,                         // tp_setattro
                0,                         // tp_as_buffer
                Py_TPFLAGS_DEFAULT,        // tp_flags
                0,                         // tp_doc
                0,                         // traverseproc tp_traverse;
                0,                         // inquiry tp_clear;
                0,                         // richcmpfunc tp_richcompare;
                0,                         // Py_ssize_t tp_weaklistoffset;
                0,                         // getiterfunc tp_iter;
                0,                         // iternextfunc tp_iternext;
                typeMethods(),             // struct PyMethodDef *tp_methods;
                0,                         // struct PyMemberDef *tp_members;
                0,                         // struct PyGetSetDef *tp_getset;
                0,                         // struct _typeobject *tp_base;
                0,                         // PyObject *tp_dict;
                0,                         // descrgetfunc tp_descr_get;
                0,                         // descrsetfunc tp_descr_set;
                0,                         // Py_ssize_t tp_dictoffset;
                0,                         // initproc tp_init;
                0,                         // allocfunc tp_alloc;
                0,                         // newfunc tp_new;
                0,                         // freefunc tp_free; /* Low-level free-memory routine */
                0,                         // inquiry tp_is_gc; /* For PyObject_IS_GC */
                0,                         // PyObject *tp_bases;
                0,                         // PyObject *tp_mro; /* method resolution order */
                0,                         // PyObject *tp_cache;
                0,                         // PyObject *tp_subclasses;
                0,                         // PyObject *tp_weaklist;
                0,                         // destructor tp_del;
                0,                         // unsigned int tp_version_tag;
                0,                         // destructor tp_finalize;
            };

        return types[inType];
    }
};

struct native_type_wrapper {
    PyObject_HEAD
  
    Type* mType;

    static PyObject* wrap(Type* t) {
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);

        //these python objects are singletons
        static std::map<Type*, PyObject*> pyObjMap;

        auto it = pyObjMap.find(t);

        if (it != pyObjMap.end()) {
            Py_INCREF(it->second);
            return it->second;
        }

        pyObjMap[t] = typeObj()->tp_alloc(typeObj(), 0);

        ((native_type_wrapper*)pyObjMap[t])->mType = t;

        Py_INCREF(pyObjMap[t]);
        return pyObjMap[t];

    }

    static PyObject* str(PyObject* self) {
        return PyUnicode_FromString(((native_type_wrapper*)self)->mType->name().c_str());
    }

    static PyObject *Int8(PyObject* nullValue, PyObject* args) {
        return wrap(::Int8::Make());
    }
    static PyObject *None(PyObject* nullValue, PyObject* args) {
        return wrap(::None::Make());
    }

    static PyObject *bytecount(PyObject* t, PyObject* args) {
        try {
            return PyLong_FromLong(((native_type_wrapper*)t)->mType->bytecount());
        }
        catch (std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return NULL;
        }
    }

    static PyMethodDef* typeMethods() {
        static PyMethodDef typed_python_TypeMethods[] = {
            {"NoneType", (PyCFunction)native_type_wrapper::None, METH_STATIC | METH_NOARGS, NULL},
            {"Int8", (PyCFunction)native_type_wrapper::Int8, METH_STATIC | METH_NOARGS, NULL},
            {"bytecount", (PyCFunction)native_type_wrapper::bytecount, METH_NOARGS, NULL},
            {NULL, NULL}
        };

        return typed_python_TypeMethods;
    };

    static PyTypeObject* typeObj() { 
        static PyTypeObject type = {
                PyVarObject_HEAD_INIT(NULL, 0)
                "NativeType",/* tp_name */
                sizeof(native_type_wrapper), /* tp_basicsize */
                0,                         // tp_itemsize
                0,                         // tp_dealloc
                0,                         // tp_print
                0,                         // tp_getattr
                0,                         // tp_setattr
                0,                         // tp_reserved
                0,                         // tp_repr
                0,                         // tp_as_number
                0,                         // tp_as_sequence
                0,                         // tp_as_mapping
                0,                         // tp_hash
                0,                         // tp_call
                native_type_wrapper::str,  // tp_str
                0,                         // tp_getattro
                0,                         // tp_setattro
                0,                         // tp_as_buffer
                Py_TPFLAGS_DEFAULT,        // tp_flags
                0,                         // tp_doc
                0,                         // traverseproc tp_traverse;
                0,                         // inquiry tp_clear;
                0,                         // richcmpfunc tp_richcompare;
                0,                         // Py_ssize_t tp_weaklistoffset;
                0,                         // getiterfunc tp_iter;
                0,                         // iternextfunc tp_iternext;
                typeMethods(),             // struct PyMethodDef *tp_methods;
                0,                         // struct PyMemberDef *tp_members;
                0,                         // struct PyGetSetDef *tp_getset;
                0,                         // struct _typeobject *tp_base;
                0,                         // PyObject *tp_dict;
                0,                         // descrgetfunc tp_descr_get;
                0,                         // descrsetfunc tp_descr_set;
                0,                         // Py_ssize_t tp_dictoffset;
                0,                         // initproc tp_init;
                0,                         // allocfunc tp_alloc;
                0,                         // newfunc tp_new;
                0,                         // freefunc tp_free; /* Low-level free-memory routine */
                0,                         // inquiry tp_is_gc; /* For PyObject_IS_GC */
                0,                         // PyObject *tp_bases;
                0,                         // PyObject *tp_mro; /* method resolution order */
                0,                         // PyObject *tp_cache;
                0,                         // PyObject *tp_subclasses;
                0,                         // PyObject *tp_weaklist;
                0,                         // destructor tp_del;
                0,                         // unsigned int tp_version_tag;
                0,                         // destructor tp_finalize;
            };

        return &type;
    }
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

class module_state {
public:
};

static PyMethodDef module_methods[] = {
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_types",
    NULL,
    sizeof(struct module_state),
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__types(void)
{
    PyObject *module = PyModule_Create(&moduledef);

    if (module == NULL)
        return NULL;

    new (GETSTATE(module)) module_state();

    if (PyType_Ready(native_type_wrapper::typeObj()) < 0)
        return NULL;

    Py_INCREF(native_type_wrapper::typeObj());
    PyModule_AddObject(module, "NativeType", (PyObject *)native_type_wrapper::typeObj());

    return module;
}
