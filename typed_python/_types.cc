#include "Python.h"
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "Type.hpp"
#include <iostream>

//extension of PyTypeObject that stashes a Type* on the end.
struct NativeTypeWrapper {
    PyTypeObject typeObj;
    Type* mType;
};

struct native_instance_wrapper {
    PyObject_HEAD
  
    Type* getType() const {
        return ((const NativeTypeWrapper*)((PyObject*)this)->ob_type)->mType;
    }

    int64_t mIsInitialized;
    uint8_t data[0];

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

        if (wrapper->mIsInitialized) {
            wrapper->getType()->destroy(wrapper->data);
        }

        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static void copy_initialize(Type* eltType, unsigned char* tgt, PyObject* pyRepresentation) {
        Type::TypeCategory cat = eltType->getTypeCategory();

        if (cat == Type::TypeCategory::catInt64) {
            if (PyLong_Check(pyRepresentation)) {
                ((int64_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an int64 from an instance of " + 
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (pyRepresentation->ob_type == typeObj(eltType)) {
            //it's already the right kind of instance
            eltType->copy_constructor(tgt, ((native_instance_wrapper*)pyRepresentation)->data);
            return;
        }

        if (cat == Type::TypeCategory::catTupleOf) {
            if (PyTuple_Check(pyRepresentation)) {
                ((TupleOf*)eltType)->constructor(tgt, PyTuple_Size(pyRepresentation), 
                    [&](uint8_t* eltPtr, int64_t k) {
                        copy_initialize(((TupleOf*)eltType)->getEltType(), eltPtr, PyTuple_GetItem(pyRepresentation,k));
                        }
                    );
                return;
            }
            if (PyList_Check(pyRepresentation)) {
                ((TupleOf*)eltType)->constructor(tgt, PyList_Size(pyRepresentation), 
                    [&](uint8_t* eltPtr, int64_t k) {
                        copy_initialize(((TupleOf*)eltType)->getEltType(), eltPtr, PyList_GetItem(pyRepresentation,k));
                        }
                    );
                return;
            }
        }

        if (cat == Type::TypeCategory::catInt64) {
            if (PyLong_Check(pyRepresentation)) {
                ((int64_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
        }

        throw std::logic_error("Couldn't initialize internal elt.");
    }

    static PyObject* extractPythonObject(unsigned char* data, Type* eltType) {
        if (eltType->getTypeCategory() == Type::TypeCategory::catInt64) {
            return PyLong_FromLong(*(int64_t*)data);
        }

        native_instance_wrapper* self = (native_instance_wrapper*)typeObj(eltType)->tp_alloc(typeObj(eltType), 0);

        try {
            self->mIsInitialized = false;

            eltType->copy_constructor(self->data, data);

            self->mIsInitialized = true;

            return (PyObject*)self;
        } catch(std::exception& e) {
            typeObj(self->getType())->tp_dealloc((PyObject*)self);

            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }

    static void initialize(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs) {
        Type::TypeCategory cat = t->getTypeCategory();

        if (cat == Type::TypeCategory::catTupleOf) {
            if (PyTuple_Size(args) != 1) {
                throw std::runtime_error("wrong argument count");
            }
            PyObject* argTuple = PyTuple_GetItem(args, 0);

            copy_initialize(t, data, argTuple);

            return;
        }

        throw std::logic_error("Can't initialize " + t->name() + " from python yet.");
    }

    static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
        NativeTypeWrapper* t = (NativeTypeWrapper*)subtype;

        native_instance_wrapper* self = (native_instance_wrapper*)subtype->tp_alloc(subtype, 0);

        try {
            initialize(self->data, t->mType, args, kwds);
            self->mIsInitialized = true;
            return (PyObject*)self;
        } catch(std::exception& e) {
            subtype->tp_dealloc((PyObject*)self);
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }

    static Py_ssize_t sq_length(native_instance_wrapper* w) {
        if (w->getType()->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            return ((TupleOf*)w->getType())->count(w->data);
        }

        PyErr_SetString(PyExc_TypeError, "not a __len__'able thing.");
    }

    static PyObject* sq_item(native_instance_wrapper* w, Py_ssize_t ix) {
        if (w->getType()->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            int64_t count = ((TupleOf*)w->getType())->count(w->data);

            if (ix >= count || ix < 0) {
                PyErr_SetString(PyExc_IndexError, "out of bounds");
                return NULL;
            }

            Type* eltType = (Type*)((TupleOf*)w->getType())->getEltType();
            return extractPythonObject(
                ((TupleOf*)w->getType())->eltPtr(w->data, ix), 
                eltType
                );
        }

        PyErr_SetString(PyExc_TypeError, "not a __getitem__'able thing.");
        return NULL;
    }

    static PyTypeObject* typeObj(Type* inType) {
        if (!inType->getTypeRep()) {
            inType->setTypeRep(typeObjInternal(inType));
        }
            
        return inType->getTypeRep();
    }

    static PySequenceMethods* sequenceMethods() {
        static PySequenceMethods* res = 
            new PySequenceMethods {
                (lenfunc)native_instance_wrapper::sq_length,
                0,
                0,
                (ssizeargfunc)native_instance_wrapper::sq_item,
                0,
                0,
                0,
                0
                };

        return res;
    }

    static Type* extractTypeFrom(PyTypeObject* typeObj) {
        if (typeObj->tp_as_sequence == sequenceMethods()) {
            return ((NativeTypeWrapper*)typeObj)->mType;
        }

        return nullptr;
    }

    static PyTypeObject* typeObjInternal(Type* inType) {
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
                sizeof(native_instance_wrapper) + inType->bytecount(),       /* tp_basicsize */
                0,                         // tp_itemsize
                native_instance_wrapper::tp_dealloc,// tp_dealloc
                0,                         // tp_print
                0,                         // tp_getattr
                0,                         // tp_setattr
                0,                         // tp_reserved
                0,                         // tp_repr
                0,                         // tp_as_number
                sequenceMethods(),         // tp_as_sequence
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

    Type* res = native_instance_wrapper::extractTypeFrom((PyTypeObject*)typearg);
    if (res) {
        return res;
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
