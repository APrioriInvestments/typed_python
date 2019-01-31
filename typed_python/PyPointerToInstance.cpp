#include "PyPointerToInstance.hpp"

PointerTo* PyPointerToInstance::type() {
    return (PointerTo*)extractTypeFrom(((PyObject*)this)->ob_type);
}

//static
PyObject* PyPointerToInstance::pointerInitialize(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;
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
PyObject* PyPointerToInstance::pointerSet(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;
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
PyObject* PyPointerToInstance::pointerGet(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.get takes one argument");
        return NULL;
    }

    instance_ptr target = (instance_ptr)*(void**)self_w->dataPtr();

    return extractPythonObject(target, pointerT->getEltType());
}

//static
PyObject* PyPointerToInstance::pointerCast(PyObject* o, PyObject* args) {
    PyInstance* self_w = (PyInstance*)o;
    PointerTo* pointerT = (PointerTo*)extractTypeFrom(o->ob_type);

    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.cast takes one argument");
        return NULL;
    }

    Type* targetType = PyPointerToInstance::unwrapTypeArgToTypePtr(PyTuple_GetItem(args, 0));

    if (!targetType) {
        PyErr_SetString(PyExc_TypeError, "PointerTo.cast requires a type argument");
        return NULL;
    }

    Type* newType = PointerTo::Make(targetType);

    return extractPythonObject(self_w->dataPtr(), newType);
}

PyObject* PyPointerToInstance::pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
    if (strcmp(op, "__add__") == 0) {
        int64_t ix = PyLong_AsLong(rhs);
        void* output;

        type()->offsetBy((instance_ptr)&output, dataPtr(), ix);

        return extractPythonObject((instance_ptr)&output, type());
    }

    return ((PyInstance*)this)->pyOperatorConcrete(rhs, op, opErr);
}