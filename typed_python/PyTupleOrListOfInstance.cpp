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


