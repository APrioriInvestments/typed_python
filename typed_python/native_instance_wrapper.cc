#include <Python.h>
#include "Type.hpp"
#include "_runtime.h"
#include "native_instance_wrapper.h"

// static
bool native_instance_wrapper::guaranteeForwardsResolved(Type* t) {
    try {
        guaranteeForwardsResolvedOrThrow(t);
        return true;
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return false;
    }
}

// static
void native_instance_wrapper::guaranteeForwardsResolvedOrThrow(Type* t) {
    t->guaranteeForwardsResolved([&](PyObject* o) {
        PyObject* result = PyObject_CallFunctionObjArgs(o, NULL);
        if (!result) {
            PyErr_Clear();
            throw std::runtime_error("Python type callback threw an exception.");
        }

        if (!PyType_Check(result)) {
            Py_DECREF(result);
            throw std::runtime_error("Python type callback didn't return a type: got " +
                std::string(result->ob_type->tp_name));
        }

        Type* resType = unwrapTypeArgToTypePtr(result);
        Py_DECREF(result);

        if (!resType) {
            throw std::runtime_error("Python type callback didn't return a native type: got " +
                std::string(result->ob_type->tp_name));
        }

        return resType;
    });
}

instance_ptr native_instance_wrapper::dataPtr() {
    return mContainingInstance.data();
}

// static
PyObject* native_instance_wrapper::constDictItems(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);

    native_instance_wrapper* w = (native_instance_wrapper*)o;

    if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

        self->mIteratorOffset = 0;
        self->mIteratorFlag = 2;
        self->mIsMatcher = false;

        self->initialize([&](instance_ptr data) {
            self_type->copy_constructor(data, w->dataPtr());
        });


        return (PyObject*)self;
    }

    PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
    return NULL;
}

// static
PyObject* native_instance_wrapper::constDictKeys(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

        self->mIteratorOffset = 0;
        self->mIteratorFlag = 0;
        self->mIsMatcher = false;

        self->initialize([&](instance_ptr data) {
            self_type->copy_constructor(data, w->dataPtr());
        });


        return (PyObject*)self;
    }

    PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
    return NULL;
}

// static
PyObject* native_instance_wrapper::constDictValues(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

        self->mIteratorOffset = 0;
        self->mIteratorFlag = 1;
        self->mIsMatcher = false;

        self->initialize([&](instance_ptr data) {
            self_type->copy_constructor(data, w->dataPtr());
        });


        return (PyObject*)self;
    }

    PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
    return NULL;
}

//static
PyObject* native_instance_wrapper::undefinedBehaviorException() {
    static PyObject* module = PyImport_ImportModule("typed_python.internals");
    static PyObject* t = PyObject_GetAttrString(module, "UndefinedBehaviorException");
    return t;
}

//static
PyObject* native_instance_wrapper::listSetSizeUnsafe(PyObject* o, PyObject* args) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;
    ListOf* listT = (ListOf*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 1 || !PyLong_Check(PyTuple_GetItem(args, 0))) {
        PyErr_SetString(PyExc_TypeError, "ListOf.getUnsafe takes one integer argument");
        return NULL;
    }

    int64_t ix = PyLong_AsLong(PyTuple_GetItem(args,0));

    if (ix < 0) {
        PyErr_SetString(undefinedBehaviorException(), "setSizeUnsafe passed negative index");
        return NULL;
    }

    listT->setSizeUnsafe(self_w->dataPtr(), ix);

    return incref(Py_None);
}

//static
PyObject* native_instance_wrapper::listPointerUnsafe(PyObject* o, PyObject* args) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;
    ListOf* listT = (ListOf*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 1 || !PyLong_Check(PyTuple_GetItem(args, 0))) {
        PyErr_SetString(PyExc_TypeError, "ListOf.pointerUnsafe takes one integer argument");
        return NULL;
    }

    int64_t ix = PyLong_AsLong(PyTuple_GetItem(args,0));

    void* ptr = (void*)listT->eltPtr(self_w->dataPtr(), ix);

    return extractPythonObject((instance_ptr)&ptr, PointerTo::Make(listT->getEltType()));
}

// static
PyObject* native_instance_wrapper::listAppend(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "ListOf.append takes one argument");
        return NULL;
    }

    PyObject* value = PyTuple_GetItem(args, 0);

    native_instance_wrapper* self_w = (native_instance_wrapper*)o;
    native_instance_wrapper* value_w = (native_instance_wrapper*)value;

    Type* self_type = extractTypeFrom(o->ob_type);
    Type* value_type = extractTypeFrom(value->ob_type);

    ListOf* listT = (ListOf*)self_type;
    Type* eltType = listT->getEltType();

    if (value_type == eltType) {
        native_instance_wrapper* value_w = (native_instance_wrapper*)value;

        listT->append(self_w->dataPtr(), value_w->dataPtr());
    } else {
        instance_ptr tempObj = (instance_ptr)malloc(eltType->bytecount());
        try {
            copyConstructFromPythonInstance(eltType, tempObj, value);
        } catch(std::exception& e) {
            free(tempObj);
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }

        listT->append(self_w->dataPtr(), tempObj);

        eltType->destroy(tempObj);

        free(tempObj);
    }

    return incref(Py_None);
}


// static
PyObject* native_instance_wrapper::listReserved(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "ListOf.reserved takes no arguments");
        return NULL;
    }

    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    Type* self_type = extractTypeFrom(o->ob_type);

    ListOf* listT = (ListOf*)self_type;

    return PyLong_FromLong(listT->reserved(self_w->dataPtr()));
}

PyObject* native_instance_wrapper::listReserve(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "ListOf.append takes one argument");
        return NULL;
    }

    PyObject* pyReserveSize = PyTuple_GetItem(args, 0);

    if (!PyLong_Check(pyReserveSize)) {
        PyErr_SetString(PyExc_TypeError, "ListOf.append takes an integer");
        return NULL;
    }

    int size = PyLong_AsLong(pyReserveSize);

    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    Type* self_type = extractTypeFrom(o->ob_type);

    ListOf* listT = (ListOf*)self_type;

    listT->reserve(self_w->dataPtr(), size);

    return incref(Py_None);
}

PyObject* native_instance_wrapper::listClear(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "ListOf.clear takes no arguments");
        return NULL;
    }

    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    Type* self_type = extractTypeFrom(o->ob_type);

    ListOf* listT = (ListOf*)self_type;

    listT->resize(self_w->dataPtr(), 0);

    return incref(Py_None);
}

PyObject* native_instance_wrapper::listResize(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 1 && PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "ListOf.append takes one argument");
        return NULL;
    }

    PyObject* pySize = PyTuple_GetItem(args, 0);

    if (!PyLong_Check(pySize)) {
        PyErr_SetString(PyExc_TypeError, "ListOf.append takes an integer");
        return NULL;
    }

    int size = PyLong_AsLong(pySize);

    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    Type* self_type = extractTypeFrom(o->ob_type);

    ListOf* listT = (ListOf*)self_type;

    if (listT->count(self_w->dataPtr()) > size) {
        listT->resize(self_w->dataPtr(), size);
    } else {
        if (PyTuple_Size(args) == 2) {
            Type* eltType = listT->getEltType();

            instance_ptr tempObj = (instance_ptr)malloc(eltType->bytecount());

            try {
                copyConstructFromPythonInstance(eltType, tempObj, PyTuple_GetItem(args, 1));
            } catch(std::exception& e) {
                free(tempObj);
                PyErr_SetString(PyExc_TypeError, e.what());
                return NULL;
            }

            listT->resize(self_w->dataPtr(), size, tempObj);

            eltType->destroy(tempObj);

            free(tempObj);
        } else {
            if (!listT->getEltType()->is_default_constructible()) {
                PyErr_SetString(PyExc_TypeError, "Cannot increase the size of this list without an object to copy in because the"
                    " element type is not copy-constructible");
                return NULL;
            }

            listT->resize(self_w->dataPtr(), size);
        }
    }

    return incref(Py_None);
}


PyObject* native_instance_wrapper::listPop(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "ListOf.pop takes zero or one argument");
        return NULL;
    }

    int which = -1;

    if (PyTuple_Size(args)) {
        PyObject* pySize = PyTuple_GetItem(args, 0);

        if (!PyLong_Check(pySize)) {
            PyErr_SetString(PyExc_TypeError, "ListOf.append takes an integer");
            return NULL;
        }

        which = PyLong_AsLong(pySize);
    }

    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    Type* self_type = extractTypeFrom(o->ob_type);

    ListOf* listT = (ListOf*)self_type;

    int listSize = listT->count(self_w->dataPtr());

    if (listSize == 0) {
        PyErr_SetString(PyExc_TypeError, "pop from empty list");
        return NULL;
    }

    if (which < 0) {
        which += listSize;
    }

    if (which < 0 || which >= listSize) {
        PyErr_SetString(PyExc_IndexError, "pop index out of range");
        return NULL;
    }

    PyObject* result = extractPythonObject(
            listT->eltPtr(self_w->dataPtr(), which),
            listT->getEltType()
            );

    listT->remove(self_w->dataPtr(), which);

    return result;
}

//static
PyObject* native_instance_wrapper::pointerInitialize(PyObject* o, PyObject* args) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.initialize takes zero or one argument");
        return NULL;
    }

    instance_ptr target = (instance_ptr)*(void**)self_w->dataPtr();

    if (PyTuple_Size(args) == 0) {
        if (!pointerT->getEltType()->is_default_constructible()) {
            PyErr_Format(PyExc_TypeError, "%s is not default initializable", pointerT->getEltType()->name().c_str());
            return NULL;
        }

        pointerT->getEltType()->constructor(target);
        return incref(Py_None);
    } else {
        try {
            copyConstructFromPythonInstance(pointerT->getEltType(), target, PyTuple_GetItem(args, 0));
            return incref(Py_None);
        } catch(std::exception& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        }
    }
}

//static
PyObject* native_instance_wrapper::pointerSet(PyObject* o, PyObject* args) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.set takes one argument");
        return NULL;
    }

    instance_ptr target = (instance_ptr)*(void**)self_w->dataPtr();

    instance_ptr tempObj = (instance_ptr)malloc(pointerT->getEltType()->bytecount());
    try {
        copyConstructFromPythonInstance(pointerT->getEltType(), tempObj, PyTuple_GetItem(args, 0));
    } catch(std::exception& e) {
        free(tempObj);
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }

    pointerT->getEltType()->assign(target, tempObj);
    pointerT->getEltType()->destroy(tempObj);
    free(tempObj);

    return incref(Py_None);
}

//static
PyObject* native_instance_wrapper::pointerGet(PyObject* o, PyObject* args) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.get takes one argument");
        return NULL;
    }

    instance_ptr target = (instance_ptr)*(void**)self_w->dataPtr();

    return extractPythonObject(target, pointerT->getEltType());
}

//static
PyObject* native_instance_wrapper::pointerCast(PyObject* o, PyObject* args) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.cast takes one argument");
        return NULL;
    }

    Type* targetType = native_instance_wrapper::unwrapTypeArgToTypePtr(PyTuple_GetItem(args, 0));

    if (!targetType) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.cast requires a type argument");
        return NULL;
    }

    Type* newType = PointerTo::Make(targetType);

    return extractPythonObject(self_w->dataPtr(), newType);
}

// static
PyObject* native_instance_wrapper::constDictGet(PyObject* o, PyObject* args) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    if (PyTuple_Size(args) < 1 || PyTuple_Size(args) > 2) {
        PyErr_SetString(PyExc_TypeError, "ConstDict.get takes one or two arguments");
        return NULL;
    }

    PyObject* item = PyTuple_GetItem(args,0);
    PyObject* ifNotFound = (PyTuple_Size(args) == 2 ? PyTuple_GetItem(args,1) : Py_None);

    Type* self_type = extractTypeFrom(o->ob_type);
    Type* item_type = extractTypeFrom(item->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        ConstDict* dict_t = (ConstDict*)self_type;

        if (item_type == dict_t->keyType()) {
            native_instance_wrapper* item_w = (native_instance_wrapper*)item;

            instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), item_w->dataPtr());

            if (!i) {
                Py_INCREF(ifNotFound);
                return ifNotFound;
            }

            return extractPythonObject(i, dict_t->valueType());
        } else {
            instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
            try {
                copyConstructFromPythonInstance(dict_t->keyType(), tempObj, item);
            } catch(std::exception& e) {
                free(tempObj);
                PyErr_SetString(PyExc_TypeError, e.what());
                return NULL;
            }

            instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), tempObj);

            dict_t->keyType()->destroy(tempObj);
            free(tempObj);

            if (!i) {
                Py_INCREF(ifNotFound);
                return ifNotFound;
            }

            return extractPythonObject(i, dict_t->valueType());
        }

        PyErr_SetString(PyExc_TypeError, "Invalid ConstDict lookup type");
        return NULL;
    }

    PyErr_SetString(PyExc_TypeError, "Wrong type!");
    return NULL;
}

// static
PyMethodDef* native_instance_wrapper::typeMethods(Type* t) {
    if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
        return new PyMethodDef [5] {
            {"get", (PyCFunction)native_instance_wrapper::constDictGet, METH_VARARGS, NULL},
            {"items", (PyCFunction)native_instance_wrapper::constDictItems, METH_NOARGS, NULL},
            {"keys", (PyCFunction)native_instance_wrapper::constDictKeys, METH_NOARGS, NULL},
            {"values", (PyCFunction)native_instance_wrapper::constDictValues, METH_NOARGS, NULL},
            {NULL, NULL}
        };
    }

    if (t->getTypeCategory() == Type::TypeCategory::catListOf) {
        return new PyMethodDef [12] {
            {"append", (PyCFunction)native_instance_wrapper::listAppend, METH_VARARGS, NULL},
            {"clear", (PyCFunction)native_instance_wrapper::listClear, METH_VARARGS, NULL},
            {"reserved", (PyCFunction)native_instance_wrapper::listReserved, METH_VARARGS, NULL},
            {"reserve", (PyCFunction)native_instance_wrapper::listReserve, METH_VARARGS, NULL},
            {"resize", (PyCFunction)native_instance_wrapper::listResize, METH_VARARGS, NULL},
            {"pop", (PyCFunction)native_instance_wrapper::listPop, METH_VARARGS, NULL},
            {"setSizeUnsafe", (PyCFunction)native_instance_wrapper::listSetSizeUnsafe, METH_VARARGS, NULL},
            {"pointerUnsafe", (PyCFunction)native_instance_wrapper::listPointerUnsafe, METH_VARARGS, NULL},
            {NULL, NULL}
        };
    }

    if (t->getTypeCategory() == Type::TypeCategory::catPointerTo) {
        return new PyMethodDef [5] {
            {"initialize", (PyCFunction)native_instance_wrapper::pointerInitialize, METH_VARARGS, NULL},
            {"set", (PyCFunction)native_instance_wrapper::pointerSet, METH_VARARGS, NULL},
            {"get", (PyCFunction)native_instance_wrapper::pointerGet, METH_VARARGS, NULL},
            {"cast", (PyCFunction)native_instance_wrapper::pointerCast, METH_VARARGS, NULL},
            {NULL, NULL}
        };
    }

    return new PyMethodDef [2] {
        {NULL, NULL}
    };
};

// static
void native_instance_wrapper::tp_dealloc(PyObject* self) {
    native_instance_wrapper* wrapper = (native_instance_wrapper*)self;

    if (wrapper->mIsInitialized) {
        wrapper->mContainingInstance.~Instance();
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

// static
bool native_instance_wrapper::pyValCouldBeOfType(Type* t, PyObject* pyRepresentation) {
    guaranteeForwardsResolvedOrThrow(t);

    if (t->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
        int isinst = PyObject_IsInstance(pyRepresentation, (PyObject*)((PythonObjectOfType*)t)->pyType());

        if (isinst == -1) {
            isinst = 0;
            PyErr_Clear();
        }

        return isinst > 0;
    }

    if (t->getTypeCategory() == Type::TypeCategory::catValue) {
        Value* valType = (Value*)t;
        if (compare_to_python(valType->value().type(), valType->value().data(), pyRepresentation, true) == 0) {
            return true;
        } else {
            return false;
        }
    }

    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    if (argType) {
        return argType->isBinaryCompatibleWith(argType);
    }

    if (t->getTypeCategory() == Type::TypeCategory::catNamedTuple ||
            t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
            t->getTypeCategory() == Type::TypeCategory::catListOf ||
            t->getTypeCategory() == Type::TypeCategory::catTuple
            ) {
        return PyTuple_Check(pyRepresentation) || PyList_Check(pyRepresentation) || PyDict_Check(pyRepresentation);
    }

    if (t->getTypeCategory() == Type::TypeCategory::catFloat64 ||
            t->getTypeCategory() == Type::TypeCategory::catFloat32)  {
        return PyFloat_Check(pyRepresentation);
    }

    if (t->getTypeCategory() == Type::TypeCategory::catInt64 ||
            t->getTypeCategory() == Type::TypeCategory::catInt32 ||
            t->getTypeCategory() == Type::TypeCategory::catInt16 ||
            t->getTypeCategory() == Type::TypeCategory::catInt8 ||
            t->getTypeCategory() == Type::TypeCategory::catUInt64 ||
            t->getTypeCategory() == Type::TypeCategory::catUInt32 ||
            t->getTypeCategory() == Type::TypeCategory::catUInt16 ||
            t->getTypeCategory() == Type::TypeCategory::catUInt8
            )  {
        return PyLong_CheckExact(pyRepresentation);
    }

    if (t->getTypeCategory() == Type::TypeCategory::catBool) {
        return PyBool_Check(pyRepresentation);
    }

    if (t->getTypeCategory() == Type::TypeCategory::catString) {
        return PyUnicode_Check(pyRepresentation);
    }

    if (t->getTypeCategory() == Type::TypeCategory::catBytes) {
        return PyBytes_Check(pyRepresentation);
    }

    return true;
}

// static
void native_instance_wrapper::copyConstructFromPythonInstance(Type* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
    guaranteeForwardsResolvedOrThrow(eltType);

    Type* argType = extractTypeFrom(pyRepresentation->ob_type);

    if (argType && argType->isBinaryCompatibleWith(eltType)) {
        //it's already the right kind of instance
        eltType->copy_constructor(tgt, ((native_instance_wrapper*)pyRepresentation)->dataPtr());
        return;
    }

    Type::TypeCategory cat = eltType->getTypeCategory();

    if (cat == Type::TypeCategory::catPythonSubclass) {
        copyConstructFromPythonInstance((Type*)eltType->getBaseType(), tgt, pyRepresentation);
        return;
    }

    if (cat == Type::TypeCategory::catPythonObjectOfType) {
        int isinst = PyObject_IsInstance(pyRepresentation, (PyObject*)((PythonObjectOfType*)eltType)->pyType());
        if (isinst == -1) {
            isinst = 0;
            PyErr_Clear();
        }

        if (!isinst) {
            throw std::logic_error("Object of type " + std::string(pyRepresentation->ob_type->tp_name) +
                    " is not an instance of " + ((PythonObjectOfType*)eltType)->pyType()->tp_name);
        }

        Py_INCREF(pyRepresentation);
        ((PyObject**)tgt)[0] = pyRepresentation;
        return;
    }

    if (cat == Type::TypeCategory::catValue) {
        Value* v = (Value*)eltType;

        const Instance& elt = v->value();

        if (compare_to_python(elt.type(), elt.data(), pyRepresentation, false) != 0) {
            throw std::logic_error("Can't initialize a " + eltType->name() + " from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        } else {
            //it's the value we want
            return;
        }
    }

    if (cat == Type::TypeCategory::catOneOf) {
        OneOf* oneOf = (OneOf*)eltType;

        for (long k = 0; k < oneOf->getTypes().size(); k++) {
            Type* subtype = oneOf->getTypes()[k];

            if (pyValCouldBeOfType(subtype, pyRepresentation)) {
                try {
                    copyConstructFromPythonInstance(subtype, tgt+1, pyRepresentation);
                    *(uint8_t*)tgt = k;
                    return;
                } catch(...) {
                }
            }
        }

        throw std::logic_error("Can't initialize a " + eltType->name() + " from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
        return;
    }

    if (cat == Type::TypeCategory::catNone) {
        if (pyRepresentation == Py_None) {
            return;
        }
        throw std::logic_error("Can't initialize a None from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catInt64) {
        if (PyLong_Check(pyRepresentation)) {
            ((int64_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize an int64 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catInt32) {
        if (PyLong_Check(pyRepresentation)) {
            ((int32_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize an int32 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catInt16) {
        if (PyLong_Check(pyRepresentation)) {
            ((int16_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize an int16 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catInt8) {
        if (PyLong_Check(pyRepresentation)) {
            ((int8_t*)tgt)[0] = PyLong_AsLong(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize an int8 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catUInt64) {
        if (PyLong_Check(pyRepresentation)) {
            ((uint64_t*)tgt)[0] = PyLong_AsUnsignedLong(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize an uint64 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catUInt32) {
        if (PyLong_Check(pyRepresentation)) {
            ((uint32_t*)tgt)[0] = PyLong_AsUnsignedLong(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize an uint32 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catUInt16) {
        if (PyLong_Check(pyRepresentation)) {
            ((uint16_t*)tgt)[0] = PyLong_AsUnsignedLong(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize an uint16 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catUInt8) {
        if (PyLong_Check(pyRepresentation)) {
            ((uint8_t*)tgt)[0] = PyLong_AsUnsignedLong(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize an uint8 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }
    if (cat == Type::TypeCategory::catBool) {
        if (PyLong_Check(pyRepresentation)) {
            ((bool*)tgt)[0] = PyLong_AsLong(pyRepresentation) != 0;
            return;
        }
        throw std::logic_error("Can't initialize a Bool from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catString) {
        if (PyUnicode_Check(pyRepresentation)) {
            auto kind = PyUnicode_KIND(pyRepresentation);
            assert(
                kind == PyUnicode_1BYTE_KIND ||
                kind == PyUnicode_2BYTE_KIND ||
                kind == PyUnicode_4BYTE_KIND
                );
            String().constructor(
                tgt,
                kind == PyUnicode_1BYTE_KIND ? 1 :
                kind == PyUnicode_2BYTE_KIND ? 2 :
                                                4,
                PyUnicode_GET_LENGTH(pyRepresentation),
                kind == PyUnicode_1BYTE_KIND ? (const char*)PyUnicode_1BYTE_DATA(pyRepresentation) :
                kind == PyUnicode_2BYTE_KIND ? (const char*)PyUnicode_2BYTE_DATA(pyRepresentation) :
                                               (const char*)PyUnicode_4BYTE_DATA(pyRepresentation)
                );
            return;
        }
        throw std::logic_error("Can't initialize a String from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catBytes) {
        if (PyBytes_Check(pyRepresentation)) {
            Bytes().constructor(
                tgt,
                PyBytes_GET_SIZE(pyRepresentation),
                PyBytes_AsString(pyRepresentation)
                );
            return;
        }
        throw std::logic_error("Can't initialize a Bytes object from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catFloat64) {
        if (PyLong_Check(pyRepresentation)) {
            ((double*)tgt)[0] = PyLong_AsLong(pyRepresentation);
            return;
        }
        if (PyFloat_Check(pyRepresentation)) {
            ((double*)tgt)[0] = PyFloat_AsDouble(pyRepresentation);
            return;
        }
        throw std::logic_error("Can't initialize a float64 from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    if (cat == Type::TypeCategory::catConstDict) {
        if (PyDict_Check(pyRepresentation)) {
            ConstDict* dictType = ((ConstDict*)eltType);
            dictType->constructor(tgt, PyDict_Size(pyRepresentation), false);

            try {
                PyObject *key, *value;
                Py_ssize_t pos = 0;

                int i = 0;

                while (PyDict_Next(pyRepresentation, &pos, &key, &value)) {
                    copyConstructFromPythonInstance(dictType->keyType(), dictType->kvPairPtrKey(tgt, i), key);
                    try {
                        copyConstructFromPythonInstance(dictType->valueType(), dictType->kvPairPtrValue(tgt, i), value);
                    } catch(...) {
                        dictType->keyType()->destroy(dictType->kvPairPtrKey(tgt,i));
                        throw;
                    }
                    dictType->incKvPairCount(tgt);
                    i++;
                }

                dictType->sortKvPairs(tgt);
            } catch(...) {
                dictType->destroy(tgt);
                throw;
            }
            return;
        }

        throw std::logic_error("Couldn't initialize internal elt of type " + eltType->name()
                + " with a " + pyRepresentation->ob_type->tp_name);
    }

    if (cat == Type::TypeCategory::catTupleOf || cat == Type::TypeCategory::catListOf) {
        if (PyTuple_Check(pyRepresentation)) {
            ((TupleOrListOf*)eltType)->constructor(tgt, PyTuple_Size(pyRepresentation),
                [&](uint8_t* eltPtr, int64_t k) {
                    copyConstructFromPythonInstance(((TupleOrListOf*)eltType)->getEltType(), eltPtr, PyTuple_GetItem(pyRepresentation,k));
                    }
                );
            return;
        }
        if (PyList_Check(pyRepresentation)) {
            ((TupleOrListOf*)eltType)->constructor(tgt, PyList_Size(pyRepresentation),
                [&](uint8_t* eltPtr, int64_t k) {
                    copyConstructFromPythonInstance(((TupleOrListOf*)eltType)->getEltType(), eltPtr, PyList_GetItem(pyRepresentation,k));
                    }
                );
            return;
        }
        if (PySet_Check(pyRepresentation)) {
            if (PySet_Size(pyRepresentation) == 0) {
                ((TupleOrListOf*)eltType)->constructor(tgt);
                return;
            }

            PyObject *iterator = PyObject_GetIter(pyRepresentation);

            ((TupleOrListOf*)eltType)->constructor(tgt, PySet_Size(pyRepresentation),
                [&](uint8_t* eltPtr, int64_t k) {
                    PyObject* item = PyIter_Next(iterator);
                    copyConstructFromPythonInstance(((TupleOrListOf*)eltType)->getEltType(), eltPtr, item);
                    Py_DECREF(item);
                    }
                );

            Py_DECREF(iterator);

            return;
        }

        throw std::logic_error("Couldn't initialize internal elt of type " + eltType->name()
                + " with a " + pyRepresentation->ob_type->tp_name);
    }

    if (eltType->isComposite()) {
        if (PyTuple_Check(pyRepresentation)) {
            if (((CompositeType*)eltType)->getTypes().size() != PyTuple_Size(pyRepresentation)) {
                throw std::runtime_error("Wrong number of arguments to construct " + eltType->name());
            }

            ((CompositeType*)eltType)->constructor(tgt,
                [&](uint8_t* eltPtr, int64_t k) {
                    copyConstructFromPythonInstance(((CompositeType*)eltType)->getTypes()[k], eltPtr, PyTuple_GetItem(pyRepresentation,k));
                    }
                );
            return;
        }
        if (PyList_Check(pyRepresentation)) {
            if (((CompositeType*)eltType)->getTypes().size() != PyList_Size(pyRepresentation)) {
                throw std::runtime_error("Wrong number of arguments to construct " + eltType->name());
            }

            ((CompositeType*)eltType)->constructor(tgt,
                [&](uint8_t* eltPtr, int64_t k) {
                    copyConstructFromPythonInstance(((CompositeType*)eltType)->getTypes()[k], eltPtr, PyList_GetItem(pyRepresentation,k));
                    }
                );
            return;
        }
    }

    if (eltType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
        NamedTuple* namedTupleT = (NamedTuple*)eltType;

        if (PyDict_Check(pyRepresentation)) {
            if (namedTupleT->getTypes().size() < PyDict_Size(pyRepresentation)) {
                throw std::runtime_error("Couldn't initialize type of " + eltType->name() + " because supplied dictionary had too many items");
            }
            long actuallyUsed = 0;

            namedTupleT->constructor(tgt,
                [&](uint8_t* eltPtr, int64_t k) {
                    const std::string& name = namedTupleT->getNames()[k];
                    Type* t = namedTupleT->getTypes()[k];

                    PyObject* o = PyDict_GetItemString(pyRepresentation, name.c_str());
                    if (o) {
                        copyConstructFromPythonInstance(t, eltPtr, o);
                        actuallyUsed++;
                    }
                    else if (eltType->is_default_constructible()) {
                        t->constructor(eltPtr);
                    } else {
                        throw std::logic_error("Can't default initialize argument " + name);
                    }
                });

            if (actuallyUsed != PyDict_Size(pyRepresentation)) {
                throw std::runtime_error("Couldn't initialize type of " + eltType->name() + " because supplied dictionary had unused arguments");
            }

            return;
        }
    }

    throw std::logic_error("Couldn't initialize internal elt of type " + eltType->name() + " from " + pyRepresentation->ob_type->tp_name);
}

// static
void native_instance_wrapper::initializeClassWithDefaultArguments(Class* cls, uint8_t* data, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args)) {
        PyErr_Format(PyExc_TypeError,
            "default __init__ for instances of '%s' doesn't accept positional arguments.",
            cls->name().c_str()
            );
        throw PythonExceptionSet();
    }

    if (!kwargs) {
        return;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(kwargs, &pos, &key, &value)) {
        int res = classInstanceSetAttributeFromPyObject(cls, data, key, value);

        if (res != 0) {
            throw PythonExceptionSet();
        }
    }
}

// static
void native_instance_wrapper::constructFromPythonArguments(uint8_t* data, Type* t, PyObject* args, PyObject* kwargs) {
    guaranteeForwardsResolvedOrThrow(t);

    Type::TypeCategory cat = t->getTypeCategory();

    if (cat == Type::TypeCategory::catPythonSubclass) {
        constructFromPythonArguments(data, (Type*)t->getBaseType(), args, kwargs);
        return;
    }

    if (cat == Type::TypeCategory::catConcreteAlternative) {
        ConcreteAlternative* alt = (ConcreteAlternative*)t;
        alt->constructor(data, [&](instance_ptr p) {
            if ((kwargs == nullptr || PyDict_Size(kwargs) == 0) && PyTuple_Size(args) == 1) {
                //construct an alternative from a single argument.
                //if it's a binary compatible subtype of the alternative we're constructing, then
                //invoke the copy constructor.
                PyObject* arg = PyTuple_GetItem(args, 0);
                Type* argType = extractTypeFrom(arg->ob_type);

                if (argType && argType->isBinaryCompatibleWith(alt)) {
                    //it's already the right kind of instance, so we can copy-through the underlying element
                    alt->elementType()->copy_constructor(p, alt->eltPtr(((native_instance_wrapper*)arg)->dataPtr()));
                    return;
                }

                //otherwise, if we have exactly one subelement, attempt to construct from that
                if (alt->elementType()->getTypeCategory() != Type::TypeCategory::catNamedTuple) {
                    throw std::runtime_error("ConcreteAlternatives are supposed to only contain NamedTuples");
                }

                NamedTuple* alternativeEltType = (NamedTuple*)alt->elementType();

                if (alternativeEltType->getTypes().size() != 1) {
                    throw std::logic_error("Can't initialize " + t->name() + " with positional arguments because it doesn't have only one field.");
                }

                native_instance_wrapper::copyConstructFromPythonInstance(alternativeEltType->getTypes()[0], p, arg);
            } else if (PyTuple_Size(args) == 0) {
                //construct an alternative from Kwargs
                constructFromPythonArguments(p, alt->elementType(), args, kwargs);
            } else {
                throw std::logic_error("Can only initialize " + t->name() + " from python with kwargs or a single in-place argument");
            }
        });
        return;
    }

    if (cat == Type::TypeCategory::catClass) {
        Class* classT = (Class*)t;

        classT->constructor(data);

        auto it = classT->getMemberFunctions().find("__init__");
        if (it == classT->getMemberFunctions().end()) {
            //run the default constructor
            initializeClassWithDefaultArguments(classT, data, args, kwargs);
            return;
        }

        Function* initMethod = it->second;

        PyObject* selfAsObject = native_instance_wrapper::initialize(classT, [&](instance_ptr selfData) {
            classT->copy_constructor(selfData, data);
        });

        PyObject* targetArgTuple = PyTuple_New(PyTuple_Size(args)+1);

        PyTuple_SetItem(targetArgTuple, 0, selfAsObject); //steals the reference to the new 'selfAsObject'

        for (long k = 0; k < PyTuple_Size(args); k++) {
            PyTuple_SetItem(targetArgTuple, k+1, incref(PyTuple_GetItem(args, k))); //have to incref because of stealing
        }

        bool threw = false;
        bool ran = false;

        for (const auto& overload: initMethod->getOverloads()) {
            std::pair<bool, PyObject*> res = tryToCallOverload(overload, nullptr, targetArgTuple, kwargs);
            if (res.first) {
                //res.first is true if we matched and tried to call this function
                if (res.second) {
                    //don't need the result.
                    Py_DECREF(res.second);
                    ran = true;
                } else {
                    //it threw an exception
                    ran = true;
                    threw = true;
                }

                break;
            }
        }

        Py_DECREF(targetArgTuple);

        if (!ran) {
            throw std::runtime_error("Cannot find a valid overload of __init__ with these arguments.");
        }

        if (threw) {
            throw PythonExceptionSet();
        }

        return;
    }

    if (kwargs == NULL) {
        if (args == NULL || PyTuple_Size(args) == 0) {
            if (t->is_default_constructible()) {
                t->constructor(data);
                return;
            }
        }

        if (PyTuple_Size(args) == 1) {
            PyObject* argTuple = PyTuple_GetItem(args, 0);

            copyConstructFromPythonInstance(t, data, argTuple);

            return;
        }

        throw std::logic_error("Can't initialize " + t->name() + " with these in-place arguments.");
    } else {
        if (cat == Type::TypeCategory::catNamedTuple) {
            long actuallyUsed = 0;

            CompositeType* compositeT = ((CompositeType*)t);

            compositeT->constructor(
                data,
                [&](uint8_t* eltPtr, int64_t k) {
                    Type* eltType = compositeT->getTypes()[k];
                    PyObject* o = PyDict_GetItemString(kwargs, compositeT->getNames()[k].c_str());
                    if (o) {
                        copyConstructFromPythonInstance(eltType, eltPtr, o);
                        actuallyUsed++;
                    }
                    else if (eltType->is_default_constructible()) {
                        eltType->constructor(eltPtr);
                    } else {
                        throw std::logic_error("Can't default initialize argument " + compositeT->getNames()[k]);
                    }
                });

            if (actuallyUsed != PyDict_Size(kwargs)) {
                throw std::runtime_error("Couldn't initialize type of " + t->name() + " because supplied dictionary had unused arguments");
            }

            return;
        }

        throw std::logic_error("Can't initialize " + t->name() + " from python with kwargs.");
    }
}


/**
 *  produce the pythonic representation of this object. for things like integers, string, etc,
 *  convert them back to their python-native form. otherwise, a pointer back into a native python
 *  structure
 */
// static
PyObject* native_instance_wrapper::extractPythonObject(instance_ptr data, Type* eltType) {
    if (eltType->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
        PyObject* res = *(PyObject**)data;
        Py_INCREF(res);
        return res;
    }
    if (eltType->getTypeCategory() == Type::TypeCategory::catValue) {
        Value* valueType = (Value*)eltType;
        return extractPythonObject(valueType->value().data(), valueType->value().type());
    }
    if (eltType->getTypeCategory() == Type::TypeCategory::catNone) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (eltType->getTypeCategory() == Type::TypeCategory::catInt64) {
        return PyLong_FromLong(*(int64_t*)data);
    }
    if (eltType->getTypeCategory() == Type::TypeCategory::catBool) {
        PyObject* res = *(bool*)data ? Py_True : Py_False;
        Py_INCREF(res);
        return res;
    }
    if (eltType->getTypeCategory() == Type::TypeCategory::catFloat64) {
        return PyFloat_FromDouble(*(double*)data);
    }
    if (eltType->getTypeCategory() == Type::TypeCategory::catFloat32) {
        return PyFloat_FromDouble(*(float*)data);
    }
    if (eltType->getTypeCategory() == Type::TypeCategory::catBytes) {
        return PyBytes_FromStringAndSize(
            (const char*)Bytes().eltPtr(data, 0),
            Bytes().count(data)
            );
    }
    if (eltType->getTypeCategory() == Type::TypeCategory::catString) {
        int bytes_per_codepoint = String().bytes_per_codepoint(data);

        return PyUnicode_FromKindAndData(
            bytes_per_codepoint == 1 ? PyUnicode_1BYTE_KIND :
            bytes_per_codepoint == 2 ? PyUnicode_2BYTE_KIND :
                                       PyUnicode_4BYTE_KIND,
            String().eltPtr(data, 0),
            String().count(data)
            );
    }

    if (eltType->getTypeCategory() == Type::TypeCategory::catOneOf) {
        std::pair<Type*, instance_ptr> child = ((OneOf*)eltType)->unwrap(data);
        return extractPythonObject(child.second, child.first);
    }

    Type* concreteT = eltType->pickConcreteSubclass(data);

    try {
        return native_instance_wrapper::initialize(concreteT, [&](instance_ptr selfData) {
            concreteT->copy_constructor(selfData, data);
        });
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
}

// static
PyObject* native_instance_wrapper::tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) {
    Type* eltType = extractTypeFrom(subtype);

    if (!guaranteeForwardsResolved(eltType)) { return nullptr; }

    if (isSubclassOfNativeType(subtype)) {
        native_instance_wrapper* self = (native_instance_wrapper*)subtype->tp_alloc(subtype, 0);

        try {
            self->mIteratorOffset = -1;
            self->mIsMatcher = false;

            self->initialize([&](instance_ptr data) {
                constructFromPythonArguments(data, eltType, args, kwds);
            });

            return (PyObject*)self;
        } catch(std::exception& e) {
            subtype->tp_dealloc((PyObject*)self);

            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        } catch(PythonExceptionSet& e) {
            subtype->tp_dealloc((PyObject*)self);
            return NULL;
        }

        // not reachable
        assert(false);

    } else {
        instance_ptr tgt = (instance_ptr)malloc(eltType->bytecount());

        try {
            constructFromPythonArguments(tgt, eltType, args, kwds);
        } catch(std::exception& e) {
            free(tgt);
            PyErr_SetString(PyExc_TypeError, e.what());
            return NULL;
        } catch(PythonExceptionSet& e) {
            free(tgt);
            return NULL;
        }

        PyObject* result = extractPythonObject(tgt, eltType);

        eltType->destroy(tgt);
        free(tgt);

        return result;
    }
}

// static
Py_ssize_t native_instance_wrapper::sq_length(PyObject* o) {
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    Type* t = extractTypeFrom(o->ob_type);

    if (t->getTypeCategory() == Type::TypeCategory::catTupleOf || t->getTypeCategory() == Type::TypeCategory::catListOf) {
        return ((TupleOrListOf*)t)->count(w->dataPtr());
    }
    if (t->isComposite()) {
        return ((CompositeType*)t)->getTypes().size();
    }
    if (t->getTypeCategory() == Type::TypeCategory::catString) {
        return String().count(w->dataPtr());
    }
    if (t->getTypeCategory() == Type::TypeCategory::catBytes) {
        return Bytes().count(w->dataPtr());
    }

    return 0;
}

// static
PyObject* native_instance_wrapper::nb_rshift(PyObject* lhs, PyObject* rhs) {
    std::pair<bool, PyObject*> res = checkForPyOperator(lhs, rhs, "__rshift__");
    if (res.first) {
        return res.second;
    }

    PyErr_Format(
        PyExc_TypeError,
        "Unsupported operand type(s) for >>: %S and %S",
        lhs->ob_type,
        rhs->ob_type,
        NULL
        );

    return NULL;
}

// static
std::pair<bool, PyObject*> native_instance_wrapper::checkForPyOperator(PyObject* lhs, PyObject* rhs, const char* op) {
    Type* lhs_type = extractTypeFrom(lhs->ob_type);

    Alternative* alt = nullptr;
    if (lhs_type && lhs_type->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        alt = ((ConcreteAlternative*)lhs_type)->getAlternative();
    }
    if (lhs_type && lhs_type->getTypeCategory() == Type::TypeCategory::catAlternative) {
        alt = ((Alternative*)lhs_type);
    }
    if (alt) {
        auto it = alt->getMethods().find(op);
        if (it != alt->getMethods().end()) {
            Function* f = it->second;

            PyObject* argTuple = PyTuple_Pack(2, lhs, rhs);

            for (const auto& overload: f->getOverloads()) {
                std::pair<bool, PyObject*> res = tryToCallOverload(overload, nullptr, argTuple, nullptr);
                if (res.first) {
                    Py_DECREF(argTuple);
                    return res;
                }

            Py_DECREF(argTuple);
            }
        }
    }

    return std::pair<bool, PyObject*>(false, NULL);
}

// static
PyObject* native_instance_wrapper::nb_add(PyObject* lhs, PyObject* rhs) {
    Type* lhs_type = extractTypeFrom(lhs->ob_type);

    if (lhs_type->getTypeCategory() == Type::TypeCategory::catPointerTo && PyLong_Check(rhs)) {
        int64_t ix = PyLong_AsLong(rhs);
        void* output;
        ((PointerTo*)lhs_type)->offsetBy((instance_ptr)&output, ((native_instance_wrapper*)lhs)->dataPtr(), ix);
        return extractPythonObject((instance_ptr)&output, lhs_type);
    }

    std::pair<bool, PyObject*> res = checkForPyOperator(lhs, rhs, "__add__");
    if (res.first) {
        return res.second;
    }

    PyErr_Format(
        PyExc_TypeError,
        "Unsupported operand type(s) for +: %S and %S",
        lhs->ob_type,
        rhs->ob_type,
        NULL
        );

    return NULL;
}

// static
PyObject* native_instance_wrapper::nb_subtract(PyObject* lhs, PyObject* rhs) {
    Type* lhs_type = extractTypeFrom(lhs->ob_type);
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    if (lhs_type) {
        native_instance_wrapper* w_lhs = (native_instance_wrapper*)lhs;

        if (lhs_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            ConstDict* dict_t = (ConstDict*)lhs_type;

            Type* tupleOfKeysType = dict_t->tupleOfKeysType();

            if (lhs_type == tupleOfKeysType) {
                native_instance_wrapper* w_rhs = (native_instance_wrapper*)rhs;

                native_instance_wrapper* self =
                    (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                self->initialize([&](instance_ptr data) {
                    ((ConstDict*)lhs_type)->subtractTupleOfKeysFromDict(w_lhs->dataPtr(), w_rhs->dataPtr(), data);
                });

                return (PyObject*)self;
            } else {
                //attempt to convert rhs to a relevant dict type.
                instance_ptr tempObj = (instance_ptr)malloc(tupleOfKeysType->bytecount());

                try {
                    copyConstructFromPythonInstance(tupleOfKeysType, tempObj, rhs);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return NULL;
                }

                native_instance_wrapper* self =
                    (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                self->initialize([&](instance_ptr data) {
                    ((ConstDict*)lhs_type)->subtractTupleOfKeysFromDict(w_lhs->dataPtr(), tempObj, data);
                });

                tupleOfKeysType->destroy(tempObj);

                free(tempObj);

                return (PyObject*)self;
            }
        }
    }

    std::pair<bool, PyObject*> res = checkForPyOperator(lhs, rhs, "__sub__");
    if (res.first) {
        return res.second;
    }

    PyErr_Format(
        PyExc_TypeError,
        "Unsupported operand type(s) for -: %S and %S",
        lhs->ob_type,
        rhs->ob_type,
        NULL
        );

    return NULL;
}

// static
PyObject* native_instance_wrapper::sq_concat(PyObject* lhs, PyObject* rhs) {
    Type* lhs_type = extractTypeFrom(lhs->ob_type);
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    if (lhs_type) {
        native_instance_wrapper* w_lhs = (native_instance_wrapper*)lhs;

        if (lhs_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
            if (lhs_type == rhs_type) {
                native_instance_wrapper* w_rhs = (native_instance_wrapper*)rhs;

                native_instance_wrapper* self =
                    (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                self->initialize([&](instance_ptr data) {
                    ((ConstDict*)lhs_type)->addDicts(w_lhs->dataPtr(), w_rhs->dataPtr(), data);
                });

                return (PyObject*)self;
            } else {
                //attempt to convert rhs to a relevant dict type.
                instance_ptr tempObj = (instance_ptr)malloc(lhs_type->bytecount());

                try {
                    copyConstructFromPythonInstance(lhs_type, tempObj, rhs);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return NULL;
                }

                native_instance_wrapper* self =
                    (native_instance_wrapper*)typeObj(lhs_type)->tp_alloc(typeObj(lhs_type), 0);

                self->initialize([&](instance_ptr data) {
                    ((ConstDict*)lhs_type)->addDicts(w_lhs->dataPtr(), tempObj, data);
                });

                lhs_type->destroy(tempObj);

                free(tempObj);

                return (PyObject*)self;
            }
        }
        if (lhs_type->getTypeCategory() == Type::TypeCategory::catTupleOf || lhs_type->getTypeCategory() == Type::TypeCategory::catListOf) {
            //TupleOrListOf(X) + TupleOrListOf(X) fastpath
            if (lhs_type == rhs_type) {
                native_instance_wrapper* w_rhs = (native_instance_wrapper*)rhs;

                TupleOrListOf* tupT = (TupleOrListOf*)lhs_type;
                Type* eltType = tupT->getEltType();
                native_instance_wrapper* self =
                    (native_instance_wrapper*)typeObj(tupT)
                        ->tp_alloc(typeObj(tupT), 0);

                int count_lhs = tupT->count(w_lhs->dataPtr());
                int count_rhs = tupT->count(w_rhs->dataPtr());

                self->initialize([&](instance_ptr data) {
                    tupT->constructor(data, count_lhs + count_rhs,
                        [&](uint8_t* eltPtr, int64_t k) {
                            eltType->copy_constructor(
                                eltPtr,
                                k < count_lhs ? tupT->eltPtr(w_lhs->dataPtr(), k) :
                                    tupT->eltPtr(w_rhs->dataPtr(), k - count_lhs)
                                );
                            }
                        );
                });

                return (PyObject*)self;
            }
            //generic path to add any kind of iterable.
            if (PyObject_Length(rhs) != -1) {
                TupleOrListOf* tupT = (TupleOrListOf*)lhs_type;
                Type* eltType = tupT->getEltType();

                native_instance_wrapper* self =
                    (native_instance_wrapper*)typeObj(tupT)
                        ->tp_alloc(typeObj(tupT), 0);

                int count_lhs = tupT->count(w_lhs->dataPtr());
                int count_rhs = PyObject_Length(rhs);

                try {
                    self->initialize([&](instance_ptr data) {
                        tupT->constructor(data, count_lhs + count_rhs,
                            [&](uint8_t* eltPtr, int64_t k) {
                                if (k < count_lhs) {
                                    eltType->copy_constructor(
                                        eltPtr,
                                        tupT->eltPtr(w_lhs->dataPtr(), k)
                                        );
                                } else {
                                    PyObject* kval = PyLong_FromLong(k - count_lhs);
                                    PyObject* o = PyObject_GetItem(rhs, kval);
                                    Py_DECREF(kval);

                                    if (!o) {
                                        throw InternalPyException();
                                    }

                                    try {
                                        copyConstructFromPythonInstance(eltType, eltPtr, o);
                                    } catch(...) {
                                        Py_DECREF(o);
                                        throw;
                                    }

                                    Py_DECREF(o);
                                }
                            });
                    });
                } catch(std::exception& e) {
                    typeObj(tupT)->tp_dealloc((PyObject*)self);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return NULL;
                }

                return (PyObject*)self;
            }
        }
    }

    PyErr_SetString(
        PyExc_TypeError,
        (std::string("cannot concatenate ") + lhs->ob_type->tp_name + " and "
                + rhs->ob_type->tp_name).c_str()
        );
    return NULL;
}

// static
PyObject* native_instance_wrapper::sq_item(PyObject* o, Py_ssize_t ix) {
    native_instance_wrapper* w = (native_instance_wrapper*)o;
    Type* t = extractTypeFrom(o->ob_type);

    if (t->getTypeCategory() == Type::TypeCategory::catTupleOf || t->getTypeCategory() == Type::TypeCategory::catListOf) {
        int64_t count = ((TupleOrListOf*)t)->count(w->dataPtr());

        if (ix < 0) {
            ix += count;
        }

        if (ix >= count || ix < 0) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }

        Type* eltType = (Type*)((TupleOrListOf*)t)->getEltType();
        return extractPythonObject(
            ((TupleOrListOf*)t)->eltPtr(w->dataPtr(), ix),
            eltType
            );
    }

    if (t->isComposite()) {
        auto compType = (CompositeType*)t;

        if (ix < 0 || ix >= (int64_t)compType->getTypes().size()) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }

        Type* eltType = compType->getTypes()[ix];

        return extractPythonObject(
            compType->eltPtr(w->dataPtr(), ix),
            eltType
            );
    }

    if (t->getTypeCategory() == Type::TypeCategory::catBytes) {
        if (ix < 0 || ix >= (int64_t)Bytes().count(w->dataPtr())) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }

        return PyBytes_FromStringAndSize(
            (const char*)Bytes().eltPtr(w->dataPtr(), ix),
            1
            );
    }
    if (t->getTypeCategory() == Type::TypeCategory::catString) {
        if (ix < 0 || ix >= (int64_t)String().count(w->dataPtr())) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }

        int bytes_per_codepoint = String().bytes_per_codepoint(w->dataPtr());

        return PyUnicode_FromKindAndData(
            bytes_per_codepoint == 1 ? PyUnicode_1BYTE_KIND :
            bytes_per_codepoint == 2 ? PyUnicode_2BYTE_KIND :
                                       PyUnicode_4BYTE_KIND,
            String().eltPtr(w->dataPtr(), ix),
            1
            );
    }

    PyErr_SetString(PyExc_TypeError, "not a __getitem__'able thing.");
    return NULL;
}

// static
PyTypeObject* native_instance_wrapper::typeObj(Type* inType) {
    if (!inType->getTypeRep()) {
        inType->setTypeRep(typeObjInternal(inType));
    }

    return inType->getTypeRep();
}

// static
PySequenceMethods* native_instance_wrapper::sequenceMethodsFor(Type* t) {
    if (    t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
            t->getTypeCategory() == Type::TypeCategory::catListOf ||
            t->getTypeCategory() == Type::TypeCategory::catTuple ||
            t->getTypeCategory() == Type::TypeCategory::catNamedTuple ||
            t->getTypeCategory() == Type::TypeCategory::catString ||
            t->getTypeCategory() == Type::TypeCategory::catBytes ||
            t->getTypeCategory() == Type::TypeCategory::catConstDict) {
        PySequenceMethods* res =
            new PySequenceMethods {0,0,0,0,0,0,0,0};

        if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
            res->sq_contains = (objobjproc)native_instance_wrapper::sq_contains;
        } else {
            res->sq_length = (lenfunc)native_instance_wrapper::sq_length;
            res->sq_item = (ssizeargfunc)native_instance_wrapper::sq_item;
        }

        res->sq_concat = native_instance_wrapper::sq_concat;

        return res;
    }

    return 0;
}

// static
PyNumberMethods* native_instance_wrapper::numberMethods(Type* t) {
    return new PyNumberMethods {
            //only enable this for the types that it operates on. Otherwise it disables the concatenation functions
            //we should probably just unify them
            t->getTypeCategory() == Type::TypeCategory::catConcreteAlternative ||
                t->getTypeCategory() == Type::TypeCategory::catAlternative ||
                t->getTypeCategory() == Type::TypeCategory::catPointerTo
                ? nb_add : 0, //binaryfunc nb_add
            nb_subtract, //binaryfunc nb_subtract
            0, //binaryfunc nb_multiply
            0, //binaryfunc nb_remainder
            0, //binaryfunc nb_divmod
            0, //ternaryfunc nb_power
            0, //unaryfunc nb_negative
            0, //unaryfunc nb_positive
            0, //unaryfunc nb_absolute
            0, //inquiry nb_bool
            0, //unaryfunc nb_invert
            0, //binaryfunc nb_lshift
            nb_rshift, //binaryfunc nb_rshift
            0, //binaryfunc nb_and
            0, //binaryfunc nb_xor
            0, //binaryfunc nb_or
            0, //unaryfunc nb_int
            0, //void *nb_reserved
            0, //unaryfunc nb_float
            0, //binaryfunc nb_inplace_add
            0, //binaryfunc nb_inplace_subtract
            0, //binaryfunc nb_inplace_multiply
            0, //binaryfunc nb_inplace_remainder
            0, //ternaryfunc nb_inplace_power
            0, //binaryfunc nb_inplace_lshift
            0, //binaryfunc nb_inplace_rshift
            0, //binaryfunc nb_inplace_and
            0, //binaryfunc nb_inplace_xor
            0, //binaryfunc nb_inplace_or
            0, //binaryfunc nb_floor_divide
            0, //binaryfunc nb_true_divide
            0, //binaryfunc nb_inplace_floor_divide
            0, //binaryfunc nb_inplace_true_divide
            0, //unaryfunc nb_index
            0, //binaryfunc nb_matrix_multiply
            0  //binaryfunc nb_inplace_matrix_multiply
            };
}

// static
Py_ssize_t native_instance_wrapper::mp_length(PyObject* o) {
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    Type* t = extractTypeFrom(o->ob_type);

    if (t->getTypeCategory() == Type::TypeCategory::catConstDict) {
        return ((ConstDict*)t)->size(w->dataPtr());
    }

    if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
        return ((TupleOf*)t)->count(w->dataPtr());
    }

    if (t->getTypeCategory() == Type::TypeCategory::catListOf) {
        return ((ListOf*)t)->count(w->dataPtr());
    }

    return 0;
}

// static
int native_instance_wrapper::sq_contains(PyObject* o, PyObject* item) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    Type* self_type = extractTypeFrom(o->ob_type);
    Type* item_type = extractTypeFrom(item->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        ConstDict* dict_t = (ConstDict*)self_type;

        if (item_type == dict_t->keyType()) {
            native_instance_wrapper* item_w = (native_instance_wrapper*)item;

            instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), item_w->dataPtr());

            if (!i) {
                return 0;
            }

            return 1;
        } else {
            instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
            try {
                copyConstructFromPythonInstance(dict_t->keyType(), tempObj, item);
            } catch(std::exception& e) {
                free(tempObj);
                PyErr_SetString(PyExc_TypeError, e.what());
                return -1;
            }

            instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), tempObj);

            dict_t->keyType()->destroy(tempObj);
            free(tempObj);

            if (!i) {
                return 0;
            }

            return 1;
        }
    }

    return 0;
}

int native_instance_wrapper::mp_ass_subscript(PyObject* o, PyObject* item, PyObject* value) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    Type* self_type = extractTypeFrom(o->ob_type);
    Type* value_type = extractTypeFrom(value->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catListOf) {
        ListOf* listT = (ListOf*)self_type;
        Type* eltType = listT->getEltType();

        if (PyLong_Check(item)) {
            int64_t ix = PyLong_AsLong(item);
            int64_t count = ((TupleOrListOf*)self_type)->count(self_w->dataPtr());

            if (ix < 0) {
                ix += count;
            }

            if (ix >= count || ix < 0) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return -1;
            }

            if (value_type == eltType) {
                native_instance_wrapper* value_w = (native_instance_wrapper*)value;

                eltType->assign(
                    listT->eltPtr(self_w->dataPtr(), ix),
                    value_w->dataPtr()
                    );
            } else {
                instance_ptr tempObj = (instance_ptr)malloc(eltType->bytecount());
                try {
                    copyConstructFromPythonInstance(eltType, tempObj, value);
                } catch(std::exception& e) {
                    free(tempObj);
                    PyErr_SetString(PyExc_TypeError, e.what());
                    return -1;
                }

                eltType->assign(
                    listT->eltPtr(self_w->dataPtr(), ix),
                    tempObj
                    );

                eltType->destroy(tempObj);

                free(tempObj);

                return 0;
            }
        }
    }

    PyErr_Format(PyExc_TypeError, "'%s' object does not support item assignment", o->ob_type->tp_name);
    return -1;
}

// static
PyObject* native_instance_wrapper::mp_subscript(PyObject* o, PyObject* item) {
    native_instance_wrapper* self_w = (native_instance_wrapper*)o;

    Type* self_type = extractTypeFrom(o->ob_type);
    Type* item_type = extractTypeFrom(item->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        ConstDict* dict_t = (ConstDict*)self_type;

        if (item_type == dict_t->keyType()) {
            native_instance_wrapper* item_w = (native_instance_wrapper*)item;

            instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), item_w->dataPtr());

            if (!i) {
                PyErr_SetObject(PyExc_KeyError, item);
                return NULL;
            }

            return extractPythonObject(i, dict_t->valueType());
        } else {
            instance_ptr tempObj = (instance_ptr)malloc(dict_t->keyType()->bytecount());
            try {
                copyConstructFromPythonInstance(dict_t->keyType(), tempObj, item);
            } catch(std::exception& e) {
                free(tempObj);
                PyErr_SetString(PyExc_TypeError, e.what());
                return NULL;
            }

            instance_ptr i = dict_t->lookupValueByKey(self_w->dataPtr(), tempObj);

            dict_t->keyType()->destroy(tempObj);
            free(tempObj);

            if (!i) {
                PyErr_SetObject(PyExc_KeyError, item);
                return NULL;
            }

            return extractPythonObject(i, dict_t->valueType());
        }

        PyErr_SetString(PyExc_TypeError, "Invalid ConstDict lookup type");
        return NULL;
    }

    if (self_type->getTypeCategory() == Type::TypeCategory::catTupleOf ||
            self_type->getTypeCategory() == Type::TypeCategory::catListOf
            ) {
        if (PySlice_Check(item)) {
            TupleOrListOf* tupType = (TupleOrListOf*)self_type;

            Py_ssize_t start,stop,step,slicelength;
            if (PySlice_GetIndicesEx(item, tupType->count(self_w->dataPtr()), &start,
                        &stop, &step, &slicelength) == -1) {
                return NULL;
            }

            Type* eltType = tupType->getEltType();

            native_instance_wrapper* result =
                (native_instance_wrapper*)typeObj(tupType)->tp_alloc(typeObj(tupType), 0);

            result->initialize([&](instance_ptr data) {
                tupType->constructor(data, slicelength,
                    [&](uint8_t* eltPtr, int64_t k) {
                        eltType->copy_constructor(
                            eltPtr,
                            tupType->eltPtr(self_w->dataPtr(), start + k * step)
                            );
                        }
                    );
            });

            return (PyObject*)result;
        }

        if (PyLong_Check(item)) {
            return sq_item((PyObject*)self_w, PyLong_AsLong(item));
        }
    }

    PyErr_SetObject(PyExc_KeyError, item);
    return NULL;
}

// static
PyMappingMethods* native_instance_wrapper::mappingMethods(Type* t) {
    static PyMappingMethods* res =
        new PyMappingMethods {
            native_instance_wrapper::mp_length, //mp_length
            native_instance_wrapper::mp_subscript, //mp_subscript
            native_instance_wrapper::mp_ass_subscript //mp_ass_subscript
            };

    if (t->getTypeCategory() == Type::TypeCategory::catConstDict ||
        t->getTypeCategory() == Type::TypeCategory::catTupleOf ||
        t->getTypeCategory() == Type::TypeCategory::catListOf) {
        return res;
    }

    return 0;
}

// static
PyBufferProcs* native_instance_wrapper::bufferProcs() {
    static PyBufferProcs* procs = new PyBufferProcs { 0, 0 };
    return procs;
}

/**
    Determine if a given PyTypeObject* is one of our types.

    We are using pointer-equality with the tp_as_buffer function pointer
    that we set on our types. This should be safe because:
    - No other type can be pointing to it, and
    - All of our types point to the unique instance of PyBufferProcs
*/
// static
inline bool native_instance_wrapper::isNativeType(PyTypeObject* typeObj) {
    return typeObj->tp_as_buffer == bufferProcs();
}

/**
 *  Return true if the given PyTypeObject* is a subclass of a NativeType.
 *  This will return false when called with a native type
*/
// static
bool native_instance_wrapper::isSubclassOfNativeType(PyTypeObject* typeObj) {
    if (isNativeType(typeObj)) {
        return false;
    }

    while (typeObj) {
        if (isNativeType(typeObj)) {
            return true;
        }
        typeObj = typeObj->tp_base;
    }
    return false;
}

// static
Type* native_instance_wrapper::extractTypeFrom(PyTypeObject* typeObj, bool exact /*=false*/) {
    if (exact && isSubclassOfNativeType(typeObj)) {
        return PythonSubclass::Make(extractTypeFrom(typeObj), typeObj);
    }

    while (!exact && typeObj->tp_base && !isNativeType(typeObj)) {
        typeObj = typeObj->tp_base;
    }

    if (isNativeType(typeObj)) {
        return ((NativeTypeWrapper*)typeObj)->mType;
    } else {
        return nullptr;
    }

}

PyTypeObject* native_instance_wrapper::typeObjInternal(Type* inType) {
    static std::recursive_mutex mutex;
    static std::map<Type*, NativeTypeWrapper*> types;

    std::lock_guard<std::recursive_mutex> lock(mutex);

    auto it = types.find(inType);
    if (it != types.end()) {
        return (PyTypeObject*)it->second;
    }

    types[inType] = new NativeTypeWrapper { {
            PyVarObject_HEAD_INIT(NULL, 0)              // TYPE (c.f., Type Objects)
            .tp_name = inType->name().c_str(),          // const char*
            .tp_basicsize = sizeof(native_instance_wrapper),    // Py_ssize_t
            .tp_itemsize = 0,                           // Py_ssize_t
            .tp_dealloc = native_instance_wrapper::tp_dealloc,  // destructor
            .tp_print = 0,                              // printfunc
            .tp_getattr = 0,                            // getattrfunc
            .tp_setattr = 0,                            // setattrfunc
            .tp_as_async = 0,                           // PyAsyncMethods*
            .tp_repr = tp_repr,                         // reprfunc
            .tp_as_number = numberMethods(inType),      // PyNumberMethods*
            .tp_as_sequence = sequenceMethodsFor(inType),   // PySequenceMethods*
            .tp_as_mapping = mappingMethods(inType),    // PyMappingMethods*
            .tp_hash = tp_hash,                         // hashfunc
            .tp_call = tp_call,                         // ternaryfunc
            .tp_str = tp_str,                           // reprfunc
            .tp_getattro = native_instance_wrapper::tp_getattro,    // getattrofunc
            .tp_setattro = native_instance_wrapper::tp_setattro,    // setattrofunc
            .tp_as_buffer = bufferProcs(),              // PyBufferProcs*
            .tp_flags = typeCanBeSubclassed(inType) ?
                Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE
            :   Py_TPFLAGS_DEFAULT,                     // unsigned long
            .tp_doc = 0,                                // const char*
            .tp_traverse = 0,                           // traverseproc
            .tp_clear = 0,                              // inquiry
            .tp_richcompare = tp_richcompare,           // richcmpfunc
            .tp_weaklistoffset = 0,                     // Py_ssize_t
            .tp_iter = inType->getTypeCategory() == Type::TypeCategory::catConstDict ?
                native_instance_wrapper::tp_iter
            :   0,                                      // getiterfunc tp_iter;
            .tp_iternext = native_instance_wrapper::tp_iternext,// iternextfunc
            .tp_methods = typeMethods(inType),          // struct PyMethodDef*
            .tp_members = 0,                            // struct PyMemberDef*
            .tp_getset = 0,                             // struct PyGetSetDef*
            .tp_base = 0,                               // struct _typeobject*
            .tp_dict = PyDict_New(),                    // PyObject*
            .tp_descr_get = 0,                          // descrgetfunc
            .tp_descr_set = 0,                          // descrsetfunc
            .tp_dictoffset = 0,                         // Py_ssize_t
            .tp_init = 0,                               // initproc
            .tp_alloc = 0,                              // allocfunc
            .tp_new = native_instance_wrapper::tp_new,  // newfunc
            .tp_free = 0,                               // freefunc /* Low-level free-memory routine */
            .tp_is_gc = 0,                              // inquiry  /* For PyObject_IS_GC */
            .tp_bases = 0,                              // PyObject*
            .tp_mro = 0,                                // PyObject* /* method resolution order */
            .tp_cache = 0,                              // PyObject*
            .tp_subclasses = 0,                         // PyObject*
            .tp_weaklist = 0,                           // PyObject*
            .tp_del = 0,                                // destructor
            .tp_version_tag = 0,                        // unsigned int
            .tp_finalize = 0,                           // destructor
            }, inType
            };

    // at this point, the dictionary has an entry, so if we recurse back to this function
    // we will return the correct entry.
    if (inType->getBaseType()) {
        types[inType]->typeObj.tp_base = typeObjInternal((Type*)inType->getBaseType());
        Py_INCREF(types[inType]->typeObj.tp_base);
    }

    PyType_Ready((PyTypeObject*)types[inType]);

    PyDict_SetItemString(
        types[inType]->typeObj.tp_dict,
        "__typed_python_category__",
        categoryToPyString(inType->getTypeCategory())
        );

    PyDict_SetItemString(
        types[inType]->typeObj.tp_dict,
        "__typed_python_basetype__",
        inType->getBaseType() ?
            (PyObject*)typeObjInternal(inType->getBaseType())
        :   Py_None
        );

    mirrorTypeInformationIntoPyType(inType, &types[inType]->typeObj);

    return (PyTypeObject*)types[inType];
}

/**
 *  Return 0 if successful and -1 if it failed
 */
// static
int native_instance_wrapper::classInstanceSetAttributeFromPyObject(Class* cls, instance_ptr data, PyObject* attrName, PyObject* attrVal) {
    int i = cls->memberNamed(PyUnicode_AsUTF8(attrName));

    if (i < 0) {
        auto it = cls->getClassMembers().find(PyUnicode_AsUTF8(attrName));
        if (it == cls->getClassMembers().end()) {
            PyErr_Format(
                PyExc_AttributeError,
                "'%s' object has no attribute '%S' and cannot add attributes to instances of this type",
                cls->name().c_str(), attrName
            );
        } else {
            PyErr_Format(
                PyExc_AttributeError,
                "Cannot modify read-only class member '%S' of instance of type '%s'",
                attrName, cls->name().c_str()
            );
        }
        return -1;
    }

    Type* eltType = cls->getMemberType(i);

    Type* attrType = extractTypeFrom(attrVal->ob_type);

    if (eltType == attrType) {
        native_instance_wrapper* item_w = (native_instance_wrapper*)attrVal;

        cls->setAttribute(data, i, item_w->dataPtr());

        return 0;
    } else {
        instance_ptr tempObj = (instance_ptr)malloc(eltType->bytecount());
        try {
            copyConstructFromPythonInstance(eltType, tempObj, attrVal);
        } catch(std::exception& e) {
            free(tempObj);
            PyErr_SetString(PyExc_TypeError, e.what());
            return -1;
        }


        cls->setAttribute(data, i, tempObj);

        eltType->destroy(tempObj);
        free(tempObj);

        return 0;
    }
}


// static
int native_instance_wrapper::tp_setattro(PyObject *o, PyObject* attrName, PyObject* attrVal) {
    if (!PyUnicode_Check(attrName)) {
        PyErr_Format(
            PyExc_AttributeError,
            "Cannot set attribute '%S' on instance of type '%S'. Attribute does not resolve to a string",
            attrName, o->ob_type
        );
        return -1;
    }

    Type* type = extractTypeFrom(o->ob_type);
    Type::TypeCategory cat = type->getTypeCategory();

    if (cat == Type::TypeCategory::catClass) {
        native_instance_wrapper* self_w = (native_instance_wrapper*)o;

        return classInstanceSetAttributeFromPyObject((Class*)type, self_w->dataPtr(), attrName, attrVal);
    } else if (cat == Type::TypeCategory::catNamedTuple ||
               cat == Type::TypeCategory::catConcreteAlternative) {
        PyErr_Format(
            PyExc_AttributeError,
            "Cannot set attributes on instance of type '%S' because it is immutable",
            o->ob_type
        );
        return -1;
    } else {
        PyErr_Format(
            PyExc_AttributeError,
            "Instances of type '%S' do not accept attributes",
            attrName, o->ob_type
        );
        return -1;
    }
}

// static
std::pair<bool, PyObject*> native_instance_wrapper::tryToCallOverload(const Function::Overload& f, PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* targetArgTuple = PyTuple_New(PyTuple_Size(args)+(self?1:0));
    Function::Matcher matcher(f);

    int write_slot = 0;

    if (self) {
        Py_INCREF(self);
        PyTuple_SetItem(targetArgTuple, write_slot++, self);
        matcher.requiredTypeForArg(nullptr);
    }

    for (long k = 0; k < PyTuple_Size(args); k++) {
        PyObject* elt = PyTuple_GetItem(args, k);

        //what type would we need for this unnamed arg?
        Type* targetType = matcher.requiredTypeForArg(nullptr);

        if (!matcher.stillMatches()) {
            Py_DECREF(targetArgTuple);
            return std::make_pair(false, nullptr);
        }

        if (!targetType) {
            Py_INCREF(elt);
            PyTuple_SetItem(targetArgTuple, write_slot++, elt);
        }
        else {
            try {
                PyObject* targetObj =
                    native_instance_wrapper::initializePythonRepresentation(targetType, [&](instance_ptr data) {
                        copyConstructFromPythonInstance(targetType, data, elt);
                    });

                PyTuple_SetItem(targetArgTuple, write_slot++, targetObj);
            } catch(...) {
                //not a valid conversion, but keep going
                Py_DECREF(targetArgTuple);
                return std::make_pair(false, nullptr);
            }
        }
    }

    PyObject* newKwargs = nullptr;

    if (kwargs) {
        newKwargs = PyDict_New();

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                Py_DECREF(targetArgTuple);
                Py_DECREF(newKwargs);
                PyErr_SetString(PyExc_TypeError, "Keywords arguments must be strings.");
                return std::make_pair(false, nullptr);
            }

            //what type would we need for this unnamed arg?
            Type* targetType = matcher.requiredTypeForArg(PyUnicode_AsUTF8(key));

            if (!matcher.stillMatches()) {
                Py_DECREF(targetArgTuple);
                Py_DECREF(newKwargs);
                return std::make_pair(false, nullptr);
            }

            if (!targetType) {
                PyDict_SetItem(newKwargs, key, value);
            }
            else {
                try {
                    PyObject* convertedValue = native_instance_wrapper::initializePythonRepresentation(targetType, [&](instance_ptr data) {
                        copyConstructFromPythonInstance(targetType, data, value);
                    });

                    PyDict_SetItem(newKwargs, key, convertedValue);
                    Py_DECREF(convertedValue);
                } catch(...) {
                    //not a valid conversion
                    Py_DECREF(targetArgTuple);
                    Py_DECREF(newKwargs);
                    return std::make_pair(false, nullptr);
                }
            }
        }
    }

    if (!matcher.definitelyMatches()) {
        Py_DECREF(targetArgTuple);
        return std::make_pair(false, nullptr);
    }

    PyObject* result;

    bool hadNativeDispatch = false;

    if (!native_dispatch_disabled) {
        auto tried_and_result = dispatchFunctionCallToNative(f, targetArgTuple, newKwargs);
        hadNativeDispatch = tried_and_result.first;
        result = tried_and_result.second;
    }

    if (!hadNativeDispatch) {
        result = PyObject_Call((PyObject*)f.getFunctionObj(), targetArgTuple, newKwargs);
    }

    Py_DECREF(targetArgTuple);
    if (newKwargs) {
        Py_DECREF(newKwargs);
    }

    //exceptions pass through directly
    if (!result) {
        return std::make_pair(true, result);
    }

    //force ourselves to convert to the native type
    if (f.getReturnType()) {
        try {
            PyObject* newRes = native_instance_wrapper::initializePythonRepresentation(f.getReturnType(), [&](instance_ptr data) {
                    copyConstructFromPythonInstance(f.getReturnType(), data, result);
                });
            Py_DECREF(result);
            return std::make_pair(true, newRes);
        } catch (std::exception& e) {
            Py_DECREF(result);
            PyErr_SetString(PyExc_TypeError, e.what());
            return std::make_pair(true, (PyObject*)nullptr);
        }
    }

    return std::make_pair(true, result);
}

// static
std::pair<bool, PyObject*> native_instance_wrapper::dispatchFunctionCallToNative(const Function::Overload& overload, PyObject* argTuple, PyObject *kwargs) {
    for (const auto& spec: overload.getCompiledSpecializations()) {
        auto res = dispatchFunctionCallToCompiledSpecialization(overload, spec, argTuple, kwargs);
        if (res.first) {
            return res;
        }
    }

    return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
}

std::pair<bool, PyObject*> native_instance_wrapper::dispatchFunctionCallToCompiledSpecialization(
                                                        const Function::Overload& overload,
                                                        const Function::CompiledSpecialization& specialization,
                                                        PyObject* argTuple,
                                                        PyObject *kwargs
                                                        ) {
    Type* returnType = specialization.getReturnType();

    if (!returnType) {
        throw std::runtime_error("Malformed function specialization: missing a return type.");
    }

    if (PyTuple_Size(argTuple) != overload.getArgs().size() || overload.getArgs().size() != specialization.getArgTypes().size()) {
        return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
    }

    if (kwargs && PyDict_Size(kwargs)) {
        return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
    }

    std::vector<Instance> instances;

    for (long k = 0; k < overload.getArgs().size(); k++) {
        auto arg = overload.getArgs()[k];
        Type* argType = specialization.getArgTypes()[k];

        if (arg.getIsKwarg() || arg.getIsStarArg()) {
            return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
        }

        try {
            instances.push_back(
                Instance::createAndInitialize(argType, [&](instance_ptr p) {
                    copyConstructFromPythonInstance(argType, p, PyTuple_GetItem(argTuple, k));
                })
                );
            }
        catch(...) {
            //failed to convert
            return std::pair<bool, PyObject*>(false, (PyObject*)nullptr);
        }

    }

    try {
        Instance result = Instance::createAndInitialize(returnType, [&](instance_ptr returnData) {
            std::vector<instance_ptr> args;
            for (auto& i: instances) {
                args.push_back(i.data());
            }

            specialization.getFuncPtr()(returnData, &args[0]);
        });

        return std::pair<bool, PyObject*>(true, (PyObject*)extractPythonObject(result.data(), result.type()));
    } catch(...) {
        const char* e = nativepython_runtime_get_stashed_exception();
        if (!e) {
            e = "Generated code threw an unknown exception.";
        }

        PyErr_Format(PyExc_TypeError, e);
        return std::pair<bool, PyObject*>(true, (PyObject*)nullptr);
    }
}

// static
PyObject* native_instance_wrapper::tp_call(PyObject* o, PyObject* args, PyObject* kwargs) {
    Type* self_type = extractTypeFrom(o->ob_type);
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    if (self_type->getTypeCategory() == Type::TypeCategory::catFunction) {
        Function* methodType = (Function*)self_type;

        for (const auto& overload: methodType->getOverloads()) {
            std::pair<bool, PyObject*> res = tryToCallOverload(overload, nullptr, args, kwargs);
            if (res.first) {
                return res.second;
            }
        }

        PyErr_Format(PyExc_TypeError, "'%s' cannot find a valid overload with these arguments", o->ob_type->tp_name);
        return 0;
    }


    if (self_type->getTypeCategory() == Type::TypeCategory::catBoundMethod) {
        BoundMethod* methodType = (BoundMethod*)self_type;

        Function* f = methodType->getFunction();
        Type* c = methodType->getFirstArgType();

        PyObject* objectInstance = native_instance_wrapper::initializePythonRepresentation(c, [&](instance_ptr d) {
            c->copy_constructor(d, w->dataPtr());
        });

        for (const auto& overload: f->getOverloads()) {
            std::pair<bool, PyObject*> res = tryToCallOverload(overload, objectInstance, args, kwargs);
            if (res.first) {
                Py_DECREF(objectInstance);
                return res.second;
            }
        }

        Py_DECREF(objectInstance);
        PyErr_Format(PyExc_TypeError, "'%s' cannot find a valid overload with these arguments", o->ob_type->tp_name);
        return 0;
    }

    PyErr_Format(PyExc_TypeError, "'%s' object is not callable", o->ob_type->tp_name);
    return 0;
}

// static
PyObject* native_instance_wrapper::tp_getattro(PyObject *o, PyObject* attrName) {
    if (!PyUnicode_Check(attrName)) {
        PyErr_SetString(PyExc_AttributeError, "attribute is not a string");
        return NULL;
    }

    char *attr_name = PyUnicode_AsUTF8(attrName);

    Type* t = extractTypeFrom(o->ob_type);

    native_instance_wrapper* w = (native_instance_wrapper*)o;

    Type::TypeCategory cat = t->getTypeCategory();

    if (w->mIsMatcher) {
        PyObject* res;

        if (cat == Type::TypeCategory::catAlternative) {
            Alternative* a = (Alternative*)t;
            if (a->subtypes()[a->which(w->dataPtr())].first == attr_name) {
                res = Py_True;
            } else {
                res = Py_False;
            }
        } else {
            ConcreteAlternative* a = (ConcreteAlternative*)t;
            if (a->getAlternative()->subtypes()[a->which()].first == attr_name) {
                res = Py_True;
            } else {
                res = Py_False;
            }
        }

        Py_INCREF(res);
        return res;
    }

    if (cat == Type::TypeCategory::catAlternative ||
            cat == Type::TypeCategory::catConcreteAlternative) {
        if (strcmp(attr_name,"matches") == 0) {
            native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

            self->mIteratorOffset = -1;
            self->mIsMatcher = true;

            self->initialize([&](instance_ptr data) {
                t->copy_constructor(data, w->dataPtr());
            });

            return (PyObject*)self;
        }

        //see if its a method
        Alternative* toCheck =
            (Alternative*)(cat == Type::TypeCategory::catConcreteAlternative ? t->getBaseType() : t)
            ;

        auto it = toCheck->getMethods().find(attr_name);
        if (it != toCheck->getMethods().end()) {
            return PyMethod_New((PyObject*)it->second->getOverloads()[0].getFunctionObj(), o);
        }
    }

    if (t->getTypeCategory() == Type::TypeCategory::catClass) {
        Class* nt = (Class*)t;
        for (long k = 0; k < nt->getMembers().size();k++) {
            if (nt->getMemberName(k) == attr_name) {
                Type* eltType = nt->getMemberType(k);

                if (!nt->checkInitializationFlag(w->dataPtr(),k)) {
                    PyErr_Format(
                        PyExc_AttributeError,
                        "Attribute '%S' is not initialized",
                        attrName
                    );
                    return NULL;
                }

                return extractPythonObject(
                    nt->eltPtr(w->dataPtr(), k),
                    eltType
                    );
            }
        }

        {
            auto it = nt->getMemberFunctions().find(attr_name);
            if (it != nt->getMemberFunctions().end()) {
                BoundMethod* bm = BoundMethod::Make(nt, it->second);

                return native_instance_wrapper::initializePythonRepresentation(bm, [&](instance_ptr data) {
                    bm->copy_constructor(data, w->dataPtr());
                });
            }
        }

        {
            auto it = nt->getClassMembers().find(attr_name);
            if (it != nt->getClassMembers().end()) {
                PyObject* res = it->second;
                Py_INCREF(res);
                return res;
            }
        }
    }

    PyObject* result = getattr(t, w->dataPtr(), attr_name);

    if (result) {
        return result;
    }

    return PyObject_GenericGetAttr(o, attrName);
}

// static
PyObject* native_instance_wrapper::getattr(Type* type, instance_ptr data, char* attr_name) {
    if (type->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        ConcreteAlternative* t = (ConcreteAlternative*)type;

        return getattr(
            t->getAlternative()->subtypes()[t->which()].second,
            t->getAlternative()->eltPtr(data),
            attr_name
            );
    }
    if (type->getTypeCategory() == Type::TypeCategory::catAlternative) {
        Alternative* t = (Alternative*)type;

        return getattr(
            t->subtypes()[t->which(data)].second,
            t->eltPtr(data),
            attr_name
            );
    }

    if (type->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
        NamedTuple* nt = (NamedTuple*)type;
        for (long k = 0; k < nt->getNames().size();k++) {
            if (nt->getNames()[k] == attr_name) {
                return extractPythonObject(
                    nt->eltPtr(data, k),
                    nt->getTypes()[k]
                    );
            }
        }
    }

    return NULL;
}

// static
Py_hash_t native_instance_wrapper::tp_hash(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    int32_t h = self_type->hash32(w->dataPtr());
    if (h == -1) {
        h = -2;
    }

    return h;
}

// static
char native_instance_wrapper::compare_to_python(Type* t, instance_ptr self, PyObject* other, bool exact) {
    if (t->getTypeCategory() == Type::TypeCategory::catValue) {
        Value* valType = (Value*)t;
        return compare_to_python(valType->value().type(), valType->value().data(), other, exact);
    }

    Type* otherT = extractTypeFrom(other->ob_type);

    if (otherT) {
        if (otherT < t) {
            return 1;
        }
        if (otherT > t) {
            return -1;
        }
        return t->cmp(self, ((native_instance_wrapper*)other)->dataPtr());
    }

    if (t->getTypeCategory() == Type::TypeCategory::catOneOf) {
        std::pair<Type*, instance_ptr> child = ((OneOf*)t)->unwrap(self);
        return compare_to_python(child.first, child.second, other, exact);
    }

    if (other == Py_None) {
        return (t->getTypeCategory() == Type::TypeCategory::catNone ? 0 : 1);
    }

    if (PyBool_Check(other)) {
        int64_t other_l = other == Py_True ? 1 : 0;
        int64_t self_l;

        if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            self_l = (*(bool*)self) ? 1 : 0;
        } else {
            return -1;
        }

        if (other_l < self_l) { return -1; }
        if (other_l > self_l) { return 1; }
        return 0;
    }

    if (PyLong_Check(other)) {
        int64_t other_l = PyLong_AsLong(other);
        int64_t self_l;

        if (t->getTypeCategory() == Type::TypeCategory::catInt64) {
            self_l = (*(int64_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catInt32) {
            self_l = (*(int32_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catInt16) {
            self_l = (*(int16_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catInt8) {
            self_l = (*(int8_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catBool) {
            self_l = (*(bool*)self) ? 1 : 0;
        } else if (t->getTypeCategory() == Type::TypeCategory::catUInt64) {
            self_l = (*(uint64_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catUInt32) {
            self_l = (*(uint32_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catUInt16) {
            self_l = (*(uint16_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catUInt8) {
            self_l = (*(uint8_t*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catFloat32) {
            if (exact) {
                return -1;
            }
            if (other_l < *(float*)self) { return -1; }
            if (other_l > *(float*)self) { return 1; }
            return 0;
        } else if (t->getTypeCategory() == Type::TypeCategory::catFloat64) {
            if (exact) {
                return -1;
            }
            if (other_l < *(double*)self) { return -1; }
            if (other_l > *(double*)self) { return 1; }
            return 0;
        } else {
            return -1;
        }

        if (other_l < self_l) { return -1; }
        if (other_l > self_l) { return 1; }
        return 0;
    }

    if (PyFloat_Check(other)) {
        double other_d = PyFloat_AsDouble(other);
        double self_d;

        if (t->getTypeCategory() == Type::TypeCategory::catFloat32) {
            self_d = (*(float*)self);
        } else if (t->getTypeCategory() == Type::TypeCategory::catFloat64) {
            self_d = (*(double*)self);
        } else {
            if (exact) {
                return -1;
            }
            if (t->getTypeCategory() == Type::TypeCategory::catInt64) {
                self_d = (*(int64_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catInt32) {
                self_d = (*(int32_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catInt16) {
                self_d = (*(int16_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catInt8) {
                self_d = (*(int8_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catBool) {
                self_d = (*(bool*)self) ? 1 : 0;
            } else if (t->getTypeCategory() == Type::TypeCategory::catUInt64) {
                self_d = (*(uint64_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catUInt32) {
                self_d = (*(uint32_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catUInt16) {
                self_d = (*(uint16_t*)self);
            } else if (t->getTypeCategory() == Type::TypeCategory::catUInt8) {
                self_d = (*(uint8_t*)self);
            } else {
                return -1;
            }
        }

        if (other_d < self_d) { return -1; }
        if (other_d > self_d) { return 1; }
        return 0;
    }

    if (PyTuple_Check(other)) {
        if (t->getTypeCategory() == Type::TypeCategory::catTupleOf) {
            TupleOf* tupT = (TupleOf*)t;
            int lenO = PyTuple_Size(other);
            int lenS = tupT->count(self);
            for (long k = 0; k < lenO && k < lenS; k++) {
                char res = compare_to_python(tupT->getEltType(), tupT->eltPtr(self, k), PyTuple_GetItem(other,k), exact);
                if (res) {
                    return res;
                }
            }

            if (lenS < lenO) { return -1; }
            if (lenS > lenO) { return 1; }
            return 0;
        }
        if (t->getTypeCategory() == Type::TypeCategory::catTuple ||
                    t->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
            CompositeType* tupT = (CompositeType*)t;
            int lenO = PyTuple_Size(other);
            int lenS = tupT->getTypes().size();

            for (long k = 0; k < lenO && k < lenS; k++) {
                char res = compare_to_python(tupT->getTypes()[k], tupT->eltPtr(self, k), PyTuple_GetItem(other,k), exact);
                if (res) {
                    return res;
                }
            }

            if (lenS < lenO) { return -1; }
            if (lenS > lenO) { return 1; }

            return 0;
        }
    }

    if (PyList_Check(other)) {
        if (t->getTypeCategory() == Type::TypeCategory::catListOf) {
            ListOf* listT = (ListOf*)t;
            int lenO = PyList_Size(other);
            int lenS = listT->count(self);
            for (long k = 0; k < lenO && k < lenS; k++) {
                char res = compare_to_python(listT->getEltType(), listT->eltPtr(self, k), PyList_GetItem(other,k), exact);
                if (res) {
                    return res;
                }
            }

            if (lenS < lenO) { return -1; }
            if (lenS > lenO) { return 1; }
            return 0;
        }
    }

    if (PyUnicode_Check(other) && t->getTypeCategory() == Type::TypeCategory::catString) {
        auto kind = PyUnicode_KIND(other);
        int bytesPer = kind == PyUnicode_1BYTE_KIND ? 1 :
            kind == PyUnicode_2BYTE_KIND ? 2 : 4;

        if (bytesPer != ((String*)t)->bytes_per_codepoint(self)) {
            return -1;
        }

        if (PyUnicode_GET_LENGTH(other) != ((String*)t)->count(self)) {
            return -1;
        }

        return memcmp(
            kind == PyUnicode_1BYTE_KIND ? (const char*)PyUnicode_1BYTE_DATA(other) :
            kind == PyUnicode_2BYTE_KIND ? (const char*)PyUnicode_2BYTE_DATA(other) :
                                           (const char*)PyUnicode_4BYTE_DATA(other),
            ((String*)t)->eltPtr(self, 0),
            PyUnicode_GET_LENGTH(other) * bytesPer
            ) == 0 ? 0 : 1;
    }
    if (PyBytes_Check(other) && t->getTypeCategory() == Type::TypeCategory::catBytes) {
        if (PyBytes_GET_SIZE(other) != ((Bytes*)t)->count(self)) {
            return -1;
        }

        return memcmp(
            PyBytes_AsString(other),
            ((Bytes*)t)->eltPtr(self, 0),
            PyBytes_GET_SIZE(other)
            ) == 0 ? 0 : 1;
    }

    return -1;
}

// static
PyObject* native_instance_wrapper::tp_richcompare(PyObject *a, PyObject *b, int op) {
    Type* own = extractTypeFrom(a->ob_type);
    Type* other = extractTypeFrom(b->ob_type);


    if (!other) {
        char cmp = compare_to_python(own, ((native_instance_wrapper*)a)->dataPtr(), b, false);

        PyObject* res;
        if (op == Py_EQ) {
            res = cmp == 0 ? Py_True : Py_False;
        } else if (op == Py_NE) {
            res = cmp != 0 ? Py_True : Py_False;
        } else {
            PyErr_SetString(PyExc_TypeError, "invalid comparison");
            return NULL;
        }

        Py_INCREF(res);

        return res;
    } else {
        char cmp = 0;

        if (own == other) {
            cmp = own->cmp(((native_instance_wrapper*)a)->dataPtr(), ((native_instance_wrapper*)b)->dataPtr());
        } else if (own < other) {
            cmp = -1;
        } else {
            cmp = 1;
        }

        PyObject* res;

        if (op == Py_LT) {
            res = (cmp < 0 ? Py_True : Py_False);
        } else if (op == Py_LE) {
            res = (cmp <= 0 ? Py_True : Py_False);
        } else if (op == Py_EQ) {
            res = (cmp == 0 ? Py_True : Py_False);
        } else if (op == Py_NE) {
            res = (cmp != 0 ? Py_True : Py_False);
        } else if (op == Py_GT) {
            res = (cmp > 0 ? Py_True : Py_False);
        } else if (op == Py_GE) {
            res = (cmp >= 0 ? Py_True : Py_False);
        } else {
            res = Py_NotImplemented;
        }

        Py_INCREF(res);

        return res;
    }
}

// static
PyObject* native_instance_wrapper::tp_iter(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    if (self_type && self_type->getTypeCategory() == Type::TypeCategory::catConstDict) {
        native_instance_wrapper* self = (native_instance_wrapper*)o->ob_type->tp_alloc(o->ob_type, 0);

        self->mIteratorOffset = 0;
        self->mIteratorFlag = w->mIteratorFlag;
        self->mIsMatcher = false;

        self->initialize([&](instance_ptr data) {
            self_type->copy_constructor(data, w->dataPtr());
        });

        return (PyObject*)self;
    }

    PyErr_SetString(PyExc_TypeError, ("Cannot iterate an instance of " + self_type->name()).c_str());
    return NULL;
}

// static
PyObject* native_instance_wrapper::tp_iternext(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    if (self_type->getTypeCategory() != Type::TypeCategory::catConstDict) {
        return NULL;
    }

    ConstDict* dict_t = (ConstDict*)self_type;

    if (w->mIteratorOffset >= dict_t->size(w->dataPtr())) {
        return NULL;
    }

    w->mIteratorOffset++;

    if (w->mIteratorFlag == 2) {
        auto t1 = extractPythonObject(
                dict_t->kvPairPtrKey(w->dataPtr(), w->mIteratorOffset-1),
                dict_t->keyType()
                );
        auto t2 = extractPythonObject(
                dict_t->kvPairPtrValue(w->dataPtr(), w->mIteratorOffset-1),
                dict_t->valueType()
                );

        auto res = PyTuple_Pack(2, t1, t2);

        Py_DECREF(t1);
        Py_DECREF(t2);

        return res;
    } else if (w->mIteratorFlag == 1) {
        return extractPythonObject(
            dict_t->kvPairPtrValue(w->dataPtr(), w->mIteratorOffset-1),
            dict_t->valueType()
            );
    } else {
        return extractPythonObject(
            dict_t->kvPairPtrKey(w->dataPtr(), w->mIteratorOffset-1),
            dict_t->keyType()
            );
    }
}

// static
PyObject* native_instance_wrapper::tp_repr(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);
    native_instance_wrapper* w = (native_instance_wrapper*)o;

    std::ostringstream str;
    ReprAccumulator accumulator(str);

    str << std::showpoint;

    self_type->repr(w->dataPtr(), accumulator);

    return PyUnicode_FromString(str.str().c_str());
}

// static
PyObject* native_instance_wrapper::tp_str(PyObject *o) {
    Type* self_type = extractTypeFrom(o->ob_type);

    if (self_type->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        self_type = self_type->getBaseType();
    }

    if (self_type->getTypeCategory() == Type::TypeCategory::catAlternative) {
        Alternative* a = (Alternative*)self_type;
        auto it = a->getMethods().find("__str__");
        if (it != a->getMethods().end()) {
            return PyObject_CallFunctionObjArgs(
                (PyObject*)it->second->getOverloads()[0].getFunctionObj(),
                o,
                NULL
                );
        }
    }

    return tp_repr(o);
}

// static
bool native_instance_wrapper::typeCanBeSubclassed(Type* t) {
    return t->getTypeCategory() == Type::TypeCategory::catNamedTuple;
}

// static
void native_instance_wrapper::mirrorTypeInformationIntoPyType(Type* inType, PyTypeObject* pyType) {
    if (inType->getTypeCategory() == Type::TypeCategory::catAlternative) {
        Alternative* alt = (Alternative*)inType;

        PyObject* alternatives = PyTuple_New(alt->subtypes().size());

        for (long k = 0; k < alt->subtypes().size(); k++) {
            ConcreteAlternative* concrete = ConcreteAlternative::Make(alt, k);

            PyDict_SetItemString(
                pyType->tp_dict,
                alt->subtypes()[k].first.c_str(),
                (PyObject*)typeObjInternal(concrete)
                );

            PyTuple_SetItem(alternatives, k, incref((PyObject*)typeObjInternal(concrete)));
        }

        PyDict_SetItemString(
            pyType->tp_dict,
            "__typed_python_alternatives__",
            alternatives
            );

        Py_DECREF(alternatives);
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative) {
        ConcreteAlternative* alt = (ConcreteAlternative*)inType;

        PyDict_SetItemString(
            pyType->tp_dict,
            "Index",
            PyLong_FromLong(alt->which())
            );

        PyDict_SetItemString(
            pyType->tp_dict,
            "ElementType",
            (PyObject*)typeObjInternal(alt->elementType())
            );
    }


    if (inType->getTypeCategory() == Type::TypeCategory::catBoundMethod) {
        BoundMethod* methodT = (BoundMethod*)inType;

        PyDict_SetItemString(pyType->tp_dict, "FirstArgType", typePtrToPyTypeRepresentation(methodT->getFirstArgType()));
        PyDict_SetItemString(pyType->tp_dict, "Function", typePtrToPyTypeRepresentation(methodT->getFunction()));
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catClass) {
        Class* classT = (Class*)inType;

        PyObject* types = PyTuple_New(classT->getMembers().size());
        for (long k = 0; k < classT->getMembers().size(); k++) {
            PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(std::get<1>(classT->getMembers()[k]))));
        }

        PyObject* names = PyTuple_New(classT->getMembers().size());
        for (long k = 0; k < classT->getMembers().size(); k++) {
            PyObject* namePtr = PyUnicode_FromString(std::get<0>(classT->getMembers()[k]).c_str());
            PyTuple_SetItem(names, k, namePtr);
        }

        PyObject* defaults = PyDict_New();
        for (long k = 0; k < classT->getMembers().size(); k++) {

            if (classT->getHeldClass()->memberHasDefaultValue(k)) {
                const Instance& i = classT->getHeldClass()->getMemberDefaultValue(k);

                PyObject* defaultVal = native_instance_wrapper::extractPythonObject(i.data(), i.type());

                PyDict_SetItemString(
                    defaults,
                    classT->getHeldClass()->getMemberName(k).c_str(),
                    defaultVal
                    );

                Py_DECREF(defaultVal);
            }
        }

        PyObject* memberFunctions = PyDict_New();
        for (auto p: classT->getMemberFunctions()) {
            PyDict_SetItemString(memberFunctions, p.first.c_str(), typePtrToPyTypeRepresentation(p.second));
            PyDict_SetItemString(pyType->tp_dict, p.first.c_str(), typePtrToPyTypeRepresentation(p.second));
        }

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "HeldClass", typePtrToPyTypeRepresentation(classT->getHeldClass()));
        PyDict_SetItemString(pyType->tp_dict, "MemberTypes", types);
        PyDict_SetItemString(pyType->tp_dict, "MemberNames", names);
        PyDict_SetItemString(pyType->tp_dict, "MemberDefaultValues", defaults);

        PyDict_SetItemString(pyType->tp_dict, "MemberFunctions", memberFunctions);

        for (auto nameAndObj: ((Class*)inType)->getClassMembers()) {
            PyDict_SetItemString(
                pyType->tp_dict,
                nameAndObj.first.c_str(),
                nameAndObj.second
                );
        }

        for (auto nameAndObj: ((Class*)inType)->getStaticFunctions()) {
            PyDict_SetItemString(
                pyType->tp_dict,
                nameAndObj.first.c_str(),
                native_instance_wrapper::initializePythonRepresentation(nameAndObj.second, [&](instance_ptr data){
                    //nothing to do - functions like this are just types.
                })
                );
        }
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catTupleOf ||
                    inType->getTypeCategory() == Type::TypeCategory::catListOf) {
        TupleOf* tupleOfType = (TupleOf*)inType;

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(
                pyType->tp_dict,
                "ElementType",
                typePtrToPyTypeRepresentation(tupleOfType->getEltType())
                );
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catPointerTo) {
        PointerTo* pointerT = (PointerTo*)inType;

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(
                pyType->tp_dict,
                "ElementType",
                typePtrToPyTypeRepresentation(pointerT->getEltType())
                );
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catConstDict) {
        ConstDict* constDictT = (ConstDict*)inType;

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "KeyType",
                typePtrToPyTypeRepresentation(constDictT->keyType())
                );
        PyDict_SetItemString(pyType->tp_dict, "ValueType",
                typePtrToPyTypeRepresentation(constDictT->valueType())
                );
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catNamedTuple) {
        NamedTuple* tupleT = (NamedTuple*)inType;

        PyObject* types = PyTuple_New(tupleT->getTypes().size());
        for (long k = 0; k < tupleT->getTypes().size(); k++) {
            PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(tupleT->getTypes()[k])));
        }

        PyObject* names = PyTuple_New(tupleT->getNames().size());
        for (long k = 0; k < tupleT->getNames().size(); k++) {
            PyObject* namePtr = PyUnicode_FromString(tupleT->getNames()[k].c_str());
            PyTuple_SetItem(names, k, namePtr);
        }

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "ElementTypes", types);
        PyDict_SetItemString(pyType->tp_dict, "ElementNames", names);

        Py_DECREF(names);
        Py_DECREF(types);
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catOneOf) {
        OneOf* oneOfT = (OneOf*)inType;

        PyObject* types = PyTuple_New(oneOfT->getTypes().size());
        for (long k = 0; k < oneOfT->getTypes().size(); k++) {
            PyTuple_SetItem(types, k, incref(typePtrToPyTypeRepresentation(oneOfT->getTypes()[k])));
        }

        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "Types", types);
        Py_DECREF(types);
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catTuple) {
        Tuple* tupleT = (Tuple*)inType;

        PyObject* res = PyTuple_New(tupleT->getTypes().size());
        for (long k = 0; k < tupleT->getTypes().size(); k++) {
            PyTuple_SetItem(res, k, incref(typePtrToPyTypeRepresentation(tupleT->getTypes()[k])));
        }
        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "ElementTypes", res);
    }

    if (inType->getTypeCategory() == Type::TypeCategory::catFunction) {
        //expose a list of overloads
        PyObject* overloads = createOverloadPyRepresentation((Function*)inType);

        PyDict_SetItemString(
                pyType->tp_dict,
                "overloads",
                overloads
                );

        Py_DECREF(overloads);
    }
}

// static
PyTypeObject* native_instance_wrapper::getObjectAsTypeObject() {
    static PyObject* module = PyImport_ImportModule("typed_python.internals");
    static PyObject* t = PyObject_GetAttrString(module, "object");
    return (PyTypeObject*)t;
}

// static
PyObject* native_instance_wrapper::createOverloadPyRepresentation(Function* f) {
    static PyObject* internalsModule = PyImport_ImportModule("typed_python.internals");

    if (!internalsModule) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals");
    }

    static PyObject* funcOverload = PyObject_GetAttrString(internalsModule, "FunctionOverload");

    if (!funcOverload) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals.FunctionOverload");
    }

    PyObject* overloadTuple = PyTuple_New(f->getOverloads().size());

    for (long k = 0; k < f->getOverloads().size(); k++) {
        auto& overload = f->getOverloads()[k];

        PyObject* pyIndex = PyLong_FromLong(k);

        PyObject* pyOverloadInst = PyObject_CallFunctionObjArgs(
            funcOverload,
            typePtrToPyTypeRepresentation(f),
            pyIndex,
            (PyObject*)overload.getFunctionObj(),
            overload.getReturnType() ? (PyObject*)typePtrToPyTypeRepresentation(overload.getReturnType()) : Py_None,
            NULL
            );

        Py_DECREF(pyIndex);

        if (pyOverloadInst) {
            for (auto arg: f->getOverloads()[k].getArgs()) {
                PyObject* res = PyObject_CallMethod(pyOverloadInst, "addArg", "sOOOO",
                    arg.getName().c_str(),
                    arg.getDefaultValue() ? PyTuple_Pack(1, arg.getDefaultValue()) : Py_None,
                    arg.getTypeFilter() ? (PyObject*)typePtrToPyTypeRepresentation(arg.getTypeFilter()) : Py_None,
                    arg.getIsStarArg() ? Py_True : Py_False,
                    arg.getIsKwarg() ? Py_True : Py_False
                    );

                if (!res) {
                    PyErr_PrintEx(0);
                } else {
                    Py_DECREF(res);
                }
            }

            PyTuple_SetItem(overloadTuple, k, pyOverloadInst);
        } else {
            PyErr_PrintEx(0);
            Py_INCREF(Py_None);
            PyTuple_SetItem(overloadTuple, k, Py_None);
        }
    }

    return overloadTuple;
}

// static
Type* native_instance_wrapper::pyFunctionToForward(PyObject* arg) {
    static PyObject* internalsModule = PyImport_ImportModule("typed_python.internals");

    if (!internalsModule) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals");
    }

    static PyObject* forwardToName = PyObject_GetAttrString(internalsModule, "forwardToName");

    if (!forwardToName) {
        throw std::runtime_error("Internal error: couldn't find typed_python.internals.makeFunction");
    }

    PyObject* fRes = PyObject_CallFunctionObjArgs(forwardToName, arg, NULL);

    std::string fwdName;

    if (!fRes) {
        fwdName = "<Internal Error>";
        PyErr_Clear();
    } else {
        if (!PyUnicode_Check(fRes)) {
            fwdName = "<Internal Error>";
            Py_DECREF(fRes);
        } else {
            fwdName = PyUnicode_AsUTF8(fRes);
            Py_DECREF(fRes);
        }
    }

    Py_INCREF(arg);
    return new Forward(arg, fwdName);
}

/**
 *  We are doing this here rather than in Type because we want to create a singleton PyUnicode
 *  object for each type category to make this function ultra fast.
 */
// static
PyObject* native_instance_wrapper::categoryToPyString(Type::TypeCategory cat) {
    if (cat == Type::TypeCategory::catNone) { static PyObject* res = PyUnicode_FromString("None"); return res; }
    if (cat == Type::TypeCategory::catBool) { static PyObject* res = PyUnicode_FromString("Bool"); return res; }
    if (cat == Type::TypeCategory::catUInt8) { static PyObject* res = PyUnicode_FromString("UInt8"); return res; }
    if (cat == Type::TypeCategory::catUInt16) { static PyObject* res = PyUnicode_FromString("UInt16"); return res; }
    if (cat == Type::TypeCategory::catUInt32) { static PyObject* res = PyUnicode_FromString("UInt32"); return res; }
    if (cat == Type::TypeCategory::catUInt64) { static PyObject* res = PyUnicode_FromString("UInt64"); return res; }
    if (cat == Type::TypeCategory::catInt8) { static PyObject* res = PyUnicode_FromString("Int8"); return res; }
    if (cat == Type::TypeCategory::catInt16) { static PyObject* res = PyUnicode_FromString("Int16"); return res; }
    if (cat == Type::TypeCategory::catInt32) { static PyObject* res = PyUnicode_FromString("Int32"); return res; }
    if (cat == Type::TypeCategory::catInt64) { static PyObject* res = PyUnicode_FromString("Int64"); return res; }
    if (cat == Type::TypeCategory::catString) { static PyObject* res = PyUnicode_FromString("String"); return res; }
    if (cat == Type::TypeCategory::catBytes) { static PyObject* res = PyUnicode_FromString("Bytes"); return res; }
    if (cat == Type::TypeCategory::catFloat32) { static PyObject* res = PyUnicode_FromString("Float32"); return res; }
    if (cat == Type::TypeCategory::catFloat64) { static PyObject* res = PyUnicode_FromString("Float64"); return res; }
    if (cat == Type::TypeCategory::catValue) { static PyObject* res = PyUnicode_FromString("Value"); return res; }
    if (cat == Type::TypeCategory::catOneOf) { static PyObject* res = PyUnicode_FromString("OneOf"); return res; }
    if (cat == Type::TypeCategory::catTupleOf) { static PyObject* res = PyUnicode_FromString("TupleOf"); return res; }
    if (cat == Type::TypeCategory::catPointerTo) { static PyObject* res = PyUnicode_FromString("PointerTo"); return res; }
    if (cat == Type::TypeCategory::catListOf) { static PyObject* res = PyUnicode_FromString("ListOf"); return res; }
    if (cat == Type::TypeCategory::catNamedTuple) { static PyObject* res = PyUnicode_FromString("NamedTuple"); return res; }
    if (cat == Type::TypeCategory::catTuple) { static PyObject* res = PyUnicode_FromString("Tuple"); return res; }
    if (cat == Type::TypeCategory::catConstDict) { static PyObject* res = PyUnicode_FromString("ConstDict"); return res; }
    if (cat == Type::TypeCategory::catAlternative) { static PyObject* res = PyUnicode_FromString("Alternative"); return res; }
    if (cat == Type::TypeCategory::catConcreteAlternative) { static PyObject* res = PyUnicode_FromString("ConcreteAlternative"); return res; }
    if (cat == Type::TypeCategory::catPythonSubclass) { static PyObject* res = PyUnicode_FromString("PythonSubclass"); return res; }
    if (cat == Type::TypeCategory::catBoundMethod) { static PyObject* res = PyUnicode_FromString("BoundMethod"); return res; }
    if (cat == Type::TypeCategory::catClass) { static PyObject* res = PyUnicode_FromString("Class"); return res; }
    if (cat == Type::TypeCategory::catHeldClass) { static PyObject* res = PyUnicode_FromString("HeldClass"); return res; }
    if (cat == Type::TypeCategory::catFunction) { static PyObject* res = PyUnicode_FromString("Function"); return res; }
    if (cat == Type::TypeCategory::catForward) { static PyObject* res = PyUnicode_FromString("Forward"); return res; }
    if (cat == Type::TypeCategory::catPythonObjectOfType) { static PyObject* res = PyUnicode_FromString("PythonObjectOfType"); return res; }

    static PyObject* res = PyUnicode_FromString("Unknown");
    return res;
}

// static
Instance native_instance_wrapper::unwrapPyObjectToInstance(PyObject* inst) {
    if (inst == Py_None) {
        return Instance();
    }
    if (PyBool_Check(inst)) {
        return Instance::create(inst == Py_True);
    }
    if (PyLong_Check(inst)) {
        return Instance::create(PyLong_AsLong(inst));
    }
    if (PyFloat_Check(inst)) {
        return Instance::create(PyFloat_AsDouble(inst));
    }
    if (PyBytes_Check(inst)) {
        return Instance::createAndInitialize(
            Bytes::Make(),
            [&](instance_ptr i) {
                Bytes::Make()->constructor(i, PyBytes_GET_SIZE(inst), PyBytes_AsString(inst));
            }
        );
    }
    if (PyUnicode_Check(inst)) {
        auto kind = PyUnicode_KIND(inst);
        assert(
            kind == PyUnicode_1BYTE_KIND ||
            kind == PyUnicode_2BYTE_KIND ||
            kind == PyUnicode_4BYTE_KIND
            );
        int64_t bytesPerCodepoint =
            kind == PyUnicode_1BYTE_KIND ? 1 :
            kind == PyUnicode_2BYTE_KIND ? 2 :
                                           4 ;

        int64_t count = PyUnicode_GET_LENGTH(inst);

        const char* data =
            kind == PyUnicode_1BYTE_KIND ? (char*)PyUnicode_1BYTE_DATA(inst) :
            kind == PyUnicode_2BYTE_KIND ? (char*)PyUnicode_2BYTE_DATA(inst) :
                                           (char*)PyUnicode_4BYTE_DATA(inst);

        return Instance::createAndInitialize(
            String::Make(),
            [&](instance_ptr i) {
                String::Make()->constructor(i, bytesPerCodepoint, count, data);
            }
        );

    }

    assert(!PyErr_Occurred());
    PyErr_Format(
        PyExc_TypeError,
        "Cannot convert %S to an Instance "
        "(only None, int, bool, bytes, and str are supported currently).",
        inst
    );

    return Instance();  // when failed, return a None instance
}

// static
Type* native_instance_wrapper::tryUnwrapPyInstanceToValueType(PyObject* typearg) {
    Instance inst = unwrapPyObjectToInstance(typearg);

    if (!PyErr_Occurred()) {
        return Value::Make(inst);
    }
    PyErr_Clear();

    Type* nativeType = native_instance_wrapper::extractTypeFrom(typearg->ob_type);
    if (nativeType) {
        return Value::Make(
            Instance::create(
                nativeType,
                ((native_instance_wrapper*)typearg)->dataPtr()
            )
        );
    }
    return nullptr;
}

//static
PyObject* native_instance_wrapper::typePtrToPyTypeRepresentation(Type* t) {
    return (PyObject*)typeObjInternal(t);
}

// static
Type* native_instance_wrapper::tryUnwrapPyInstanceToType(PyObject* arg) {
    if (PyType_Check(arg)) {
        Type* possibleType = native_instance_wrapper::unwrapTypeArgToTypePtr(arg);
        if (!possibleType) {
            return NULL;
        }
        return possibleType;
    }

    if (arg == Py_None) {
        return None::Make();
    }

    if (PyFunction_Check(arg)) {
        return pyFunctionToForward(arg);
    }

    return  native_instance_wrapper::tryUnwrapPyInstanceToValueType(arg);
}

// static
Type* native_instance_wrapper::unwrapTypeArgToTypePtr(PyObject* typearg) {
    if (PyType_Check(typearg)) {
        PyTypeObject* pyType = (PyTypeObject*)typearg;

        if (pyType == &PyLong_Type) {
            return Int64::Make();
        }
        if (pyType == &PyFloat_Type) {
            return Float64::Make();
        }
        if (pyType == Py_None->ob_type) {
            return None::Make();
        }
        if (pyType == &PyBool_Type) {
            return Bool::Make();
        }
        if (pyType == &PyBytes_Type) {
            return Bytes::Make();
        }
        if (pyType == &PyUnicode_Type) {
            return String::Make();
        }

        if (native_instance_wrapper::isSubclassOfNativeType(pyType)) {
            Type* nativeT = native_instance_wrapper::extractTypeFrom(pyType);

            if (!nativeT) {
                PyErr_SetString(PyExc_TypeError,
                    ("Type " + std::string(pyType->tp_name) + " looked like a native type subclass, but has no base").c_str()
                    );
                return NULL;
            }

            //this is now a permanent object
            Py_INCREF(typearg);

            return PythonSubclass::Make(nativeT, pyType);
        } else {
            Type* res = native_instance_wrapper::extractTypeFrom(pyType);
            if (res) {
                // we have a native type -> return it
                return res;
            } else {
                // we have a python type -> wrap it
                return PythonObjectOfType::Make(pyType);
            }
        }

    }
    // else: typearg is not a type -> it is a value
    Type* valueType = native_instance_wrapper::tryUnwrapPyInstanceToValueType(typearg);

    if (valueType) {
        return valueType;
    }

    if (PyFunction_Check(typearg)) {
        return pyFunctionToForward(typearg);
    }


    PyErr_Format(PyExc_TypeError, "Cannot convert %S to a native type.", typearg);
    return NULL;
}
