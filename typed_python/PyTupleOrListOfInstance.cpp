#include "PyTupleOrListOfInstance.hpp"

TupleOrListOf* PyTupleOrListOfInstance::type() {
    return (TupleOrListOf*)extractTypeFrom(((PyObject*)this)->ob_type);
}

TupleOf* PyTupleOfInstance::type() {
    return (TupleOf*)extractTypeFrom(((PyObject*)this)->ob_type);
}

ListOf* PyListOfInstance::type() {
    return (ListOf*)extractTypeFrom(((PyObject*)this)->ob_type);
}

PyObject* PyTupleOrListOfInstance::sq_concat_concrete(PyObject* rhs) {
    Type* rhs_type = extractTypeFrom(rhs->ob_type);

    //TupleOrListOf(X) + TupleOrListOf(X) fastpath
    if (type() == rhs_type) {
        PyTupleOrListOfInstance* w_rhs = (PyTupleOrListOfInstance*)rhs;

        Type* eltType = type()->getEltType();

        return PyInstance::initialize(type(), [&](instance_ptr data) {
            int count_lhs = type()->count(dataPtr());
            int count_rhs = type()->count(w_rhs->dataPtr());

            type()->constructor(data, count_lhs + count_rhs,
                [&](uint8_t* eltPtr, int64_t k) {
                    eltType->copy_constructor(
                        eltPtr,
                        k < count_lhs ? type()->eltPtr(dataPtr(), k) :
                            type()->eltPtr(w_rhs->dataPtr(), k - count_lhs)
                        );
                    }
                );
            });
    }

    //generic path to add any kind of iterable.
    if (PyObject_Length(rhs) != -1) {
        Type* eltType = type()->getEltType();

        return PyInstance::initialize(type(), [&](instance_ptr data) {
            int count_lhs = type()->count(dataPtr());
            int count_rhs = PyObject_Length(rhs);

            type()->constructor(data, count_lhs + count_rhs,
                [&](uint8_t* eltPtr, int64_t k) {
                    if (k < count_lhs) {
                        eltType->copy_constructor(
                            eltPtr,
                            type()->eltPtr(dataPtr(), k)
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
    }

    PyErr_SetString(
        PyExc_TypeError,
        (std::string("cannot concatenate ") + type()->name() + " and "
                + rhs->ob_type->tp_name).c_str()
        );

    return NULL;
}

//static
PyObject* PyListOfInstance::listSetSizeUnsafe(PyObject* o, PyObject* args) {
    PyListOfInstance* self_w = (PyListOfInstance*)o;

    if (PyTuple_Size(args) != 1 || !PyLong_Check(PyTuple_GetItem(args, 0))) {
        PyErr_SetString(PyExc_TypeError, "ListOf.getUnsafe takes one integer argument");
        return NULL;
    }

    int64_t ix = PyLong_AsLong(PyTuple_GetItem(args,0));

    if (ix < 0) {
        PyErr_SetString(undefinedBehaviorException(), "setSizeUnsafe passed negative index");
        return NULL;
    }

    self_w->type()->setSizeUnsafe(self_w->dataPtr(), ix);

    return incref(Py_None);
}

//static
PyObject* PyListOfInstance::listPointerUnsafe(PyObject* o, PyObject* args) {
    PyListOfInstance* self_w = (PyListOfInstance*)o;

    if (PyTuple_Size(args) != 1 || !PyLong_Check(PyTuple_GetItem(args, 0))) {
        PyErr_SetString(PyExc_TypeError, "ListOf.pointerUnsafe takes one integer argument");
        return NULL;
    }

    int64_t ix = PyLong_AsLong(PyTuple_GetItem(args,0));

    void* ptr = (void*)self_w->type()->eltPtr(self_w->dataPtr(), ix);

    return extractPythonObject((instance_ptr)&ptr, PointerTo::Make(self_w->type()->getEltType()));
}

// static
PyObject* PyListOfInstance::listAppend(PyObject* o, PyObject* args) {
    try {
        if (PyTuple_Size(args) != 1) {
            PyErr_SetString(PyExc_TypeError, "ListOf.append takes one argument");
            return NULL;
        }

        PyObject* value = PyTuple_GetItem(args, 0);

        PyListOfInstance* self_w = (PyListOfInstance*)o;
        PyInstance* value_w = (PyInstance*)value;

        Type* value_type = extractTypeFrom(value->ob_type);

        Type* eltType = self_w->type()->getEltType();

        if (value_type == eltType) {
            PyInstance* value_w = (PyInstance*)value;

            self_w->type()->append(self_w->dataPtr(), value_w->dataPtr());
        } else {
            Instance temp(eltType, [&](instance_ptr data) {
                copyConstructFromPythonInstance(eltType, data, value);
            });

            self_w->type()->append(self_w->dataPtr(), temp.data());
        }

        return incref(Py_None);
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
}

// static
PyObject* PyListOfInstance::listReserved(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "ListOf.reserved takes no arguments");
        return NULL;
    }

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    return PyLong_FromLong(self_w->type()->reserved(self_w->dataPtr()));
}

PyObject* PyListOfInstance::listReserve(PyObject* o, PyObject* args) {
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

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    self_w->type()->reserve(self_w->dataPtr(), size);

    return incref(Py_None);
}

PyObject* PyListOfInstance::listClear(PyObject* o, PyObject* args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "ListOf.clear takes no arguments");
        return NULL;
    }

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    self_w->type()->resize(self_w->dataPtr(), 0);

    return incref(Py_None);
}

PyObject* PyListOfInstance::listResize(PyObject* o, PyObject* args) {
    try {
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

        PyListOfInstance* self_w = (PyListOfInstance*)o;
        Type* eltType = self_w->type()->getEltType();

        if (self_w->type()->count(self_w->dataPtr()) > size) {
            self_w->type()->resize(self_w->dataPtr(), size);
        } else {
            if (PyTuple_Size(args) == 2) {
                Instance temp(eltType, [&](instance_ptr data) {
                    copyConstructFromPythonInstance(eltType, data, PyTuple_GetItem(args, 1));
                });

                self_w->type()->resize(self_w->dataPtr(), size, temp.data());
            } else {
                if (!self_w->type()->getEltType()->is_default_constructible()) {
                    PyErr_SetString(PyExc_TypeError, "Cannot increase the size of this list without an object to copy in because the"
                        " element type is not copy-constructible");
                    return NULL;
                }

                self_w->type()->resize(self_w->dataPtr(), size);
            }
        }

        return incref(Py_None);
    } catch(std::exception& e) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return NULL;
    }
}

PyObject* PyListOfInstance::listPop(PyObject* o, PyObject* args) {
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

    PyListOfInstance* self_w = (PyListOfInstance*)o;

    int listSize = self_w->type()->count(self_w->dataPtr());

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
            self_w->type()->eltPtr(self_w->dataPtr(), which),
            self_w->type()->getEltType()
            );

    self_w->type()->remove(self_w->dataPtr(), which);

    return result;
}

PyObject* PyTupleOrListOfInstance::sq_item_concrete(Py_ssize_t ix) {
    int64_t count = type()->count(dataPtr());

    if (ix < 0) {
        ix += count;
    }

    if (ix >= count || ix < 0) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    Type* eltType = type()->getEltType();

    return extractPythonObject(
        type()->eltPtr(dataPtr(), ix),
        eltType
        );
}

Py_ssize_t PyTupleOrListOfInstance::mp_and_sq_length_concrete() {
    return type()->count(dataPtr());
}


