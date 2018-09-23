#include "Python.h"
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "Type.hpp"

class TupleOfLayout {
public:
    size_t count;
    unsigned char data[0];

    unsigned char* ptr(Type* t, size_t ix) {
        return &data[t->bytecount() * ix];
    }

    static TupleOfLayout* alloc(size_t count, size_t bytes_per_record) {
        TupleOfLayout* ptr = (TupleOfLayout*)malloc(sizeof(TupleOfLayout) + count * bytes_per_record);
        ptr->count = count;
        return ptr;
    }
};

class Arena {
public:
    std::vector<unsigned char> m_data;

    template<class T>
    T& asType() {
        assert(sizeof(T) == m_data.size());
        return *(T*)&m_data[0];
    }
};

//extension of PyTypeObject that stashes a Type* on the end.
struct NativeTypeWrapper {
    PyTypeObject typeObj;
    Type* mType;
};

struct native_instance_wrapper {
    PyObject_HEAD
  
    Type* mType;

    std::shared_ptr<Arena> mArenaPtr;

    size_t mArenaOffset;

    static PyObject* bytecount(PyObject* o) {
        NativeTypeWrapper* w = (NativeTypeWrapper*)o;
        return PyLong_FromLong(w->mType->bytecount());
    }

    static PyMethodDef* typeMethods() {
        static PyMethodDef typed_python_TypeMethods[] = {
            {"bytecount", (PyCFunction)native_instance_wrapper::bytecount, METH_CLASS | METH_NOARGS, NULL},
            {NULL, NULL}
        };

        return typed_python_TypeMethods;
    };

    static void tp_dealloc(PyObject* self) {
        native_instance_wrapper* wrapper = (native_instance_wrapper*)self;

        wrapper->mArenaPtr.~shared_ptr();

        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static bool initialize_targetFromPython(Type* eltType, unsigned char* tgt, PyObject* o) {
        if (eltType->getTypeCategory() == Type::TypeCategory::catInt64) {
            if (PyLong_Check(o)) {
                ((int64_t*)tgt)[0] = PyLong_AsLong(o);
                return true;
            }
        }


        return false;
    }

    static PyObject* extractPythonObject(unsigned char* data, Type* eltType) {
        if (eltType->getTypeCategory() == Type::TypeCategory::catInt64) {
            return PyLong_FromLong(*(int64_t*)data);
        }

        PyErr_SetString(PyExc_TypeError, "cant convert this back to python");
        return NULL;
    }

    static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
        NativeTypeWrapper* t = (NativeTypeWrapper*)subtype;

        std::shared_ptr<Arena> arena(new Arena());
        arena->m_data.resize(t->mType->bytecount());

        native_instance_wrapper* self = (native_instance_wrapper*)subtype->tp_alloc(subtype, 0);
        self->mType = t->mType;
        self->mArenaOffset = 0;
        new (&self->mArenaPtr) std::shared_ptr<Arena>(arena);

        Type::TypeCategory cat = t->mType->getTypeCategory();

        if (cat == Type::TypeCategory::catTupleOf) {
            if (PyTuple_Size(args) != 1) {
                PyErr_SetString(PyExc_TypeError, "error out because we should only have one argument and leaking an arena pointer because we are sloppy.");
                return NULL;
            }
            PyObject* argTuple = PyTuple_GetItem(args, 0);
            if (!PyTuple_Check(argTuple)) {
                PyErr_SetString(PyExc_TypeError, "error out because we wanted a tuple, and leaking an arena pointer because we are sloppy.");
                return NULL;
            }

            Type* heldType = ((TupleOf*)t->mType)->getEltType();
            auto& layout = arena->asType<TupleOfLayout*>();

            layout = TupleOfLayout::alloc(PyTuple_Size(argTuple), heldType->bytecount());

            for (long k = 0; k < PyTuple_Size(argTuple); k++) {
                if (!initialize_targetFromPython(heldType, layout->ptr(heldType, k), PyTuple_GetItem(argTuple,k))) {
                    PyErr_SetString(PyExc_TypeError, "error out because we tried and failed to initialize a tuple elt, and leak an arena pointer due to sloppiness.");
                    return NULL;
                }
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "error out and leaking an arena pointer because we are sloppy.");
            return NULL;
        }

        return (PyObject*)self;
    }

    static Py_ssize_t sq_length(native_instance_wrapper* w) {
        if (w->mType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            return w->mArenaPtr->asType<TupleOfLayout*>()->count;
        }

        PyErr_SetString(PyExc_TypeError, "not a __len__'able thing.");
    }

    static PyObject* sq_item(native_instance_wrapper* w, Py_ssize_t ix) {
        if (w->mType->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            TupleOfLayout* tup = w->mArenaPtr->asType<TupleOfLayout*>();
            if (ix >= tup->count) {
                PyErr_SetString(PyExc_IndexError, "out of bounds");
                return NULL;
            }

            Type* eltType = (Type*)((TupleOf*)w->mType)->getEltType();
            return extractPythonObject(tup->ptr(eltType, ix), eltType);
        }

        PyErr_SetString(PyExc_TypeError, "not a __getitem__'able thing.");
        return NULL;
    }

    static PyTypeObject* typeObj(Type* inType) { 
        static std::mutex mutex;
        static std::map<Type*, NativeTypeWrapper*> types;

        std::lock_guard<std::mutex> lock(mutex);

        auto it = types.find(inType);
        if (it != types.end()) {
            return (PyTypeObject*)it->second;
        }

        types[inType] = new NativeTypeWrapper { {
                PyVarObject_HEAD_INIT(NULL, 0)
                inType->name().c_str(),    /* tp_name */
                sizeof(native_instance_wrapper), /* tp_basicsize */
                0,                         // tp_itemsize
                native_instance_wrapper::tp_dealloc,// tp_dealloc
                0,                         // tp_print
                0,                         // tp_getattr
                0,                         // tp_setattr
                0,                         // tp_reserved
                0,                         // tp_repr
                0,                         // tp_as_number
                new PySequenceMethods{
                 (lenfunc)native_instance_wrapper::sq_length,
                 0,
                 0,
                 (ssizeargfunc)native_instance_wrapper::sq_item,
                 0,
                 0,
                 0,
                 0},                       // tp_as_sequence
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
                native_instance_wrapper::tp_new,// newfunc tp_new;
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
                }, inType
                };

        PyType_Ready((PyTypeObject*)types[inType]);
        return (PyTypeObject*)types[inType];
    }
};

Type* unwrapTypeArgToTypePtr(PyObject* typearg) {
    if ((PyTypeObject*)typearg == &PyLong_Type) {
        return Int64::Make();
    }
    if ((PyTypeObject*)typearg == &PyFloat_Type) {
        return Float64::Make();
    }


    PyErr_SetString(PyExc_TypeError, "Cannot convert argument to a native type.");
    return NULL;
}

PyObject *TupleOf(PyObject* nullValue, PyObject* args) {
    std::vector<Type*> types;
    for (long k = 0; k < PyTuple_Size(args); k++) {
        types.push_back(unwrapTypeArgToTypePtr(PyTuple_GetItem(args,k)));
        if (not types.back()) {
            return NULL;
        }
    }
    
    if (types.size() != 1) {
        PyErr_SetString(PyExc_TypeError, "TupleOf takes 1 positional argument.");
        return NULL;
    }

    PyObject* typeObj = (PyObject*)native_instance_wrapper::typeObj(TupleOf::Make(types[0]));

    Py_INCREF(typeObj);
    return typeObj;
}

PyObject *Int8(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::Int8::Make());
    Py_INCREF(res);
    return res;
}

PyObject *NoneType(PyObject* nullValue, PyObject* args) {
    PyObject* res = (PyObject*)native_instance_wrapper::typeObj(::None::Make());
    Py_INCREF(res);
    return res;
}

static PyMethodDef module_methods[] = {
    {"NoneType", (PyCFunction)NoneType, METH_VARARGS, NULL},
    {"Int8", (PyCFunction)Int8, METH_VARARGS, NULL},
    {"TupleOf", (PyCFunction)TupleOf, METH_VARARGS, NULL},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_types",
    NULL,
    0,
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

    return module;
}
