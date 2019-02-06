#include "PyCompositeTypeInstance.hpp"

CompositeType* PyCompositeTypeInstance::type() {
    return (CompositeType*)extractTypeFrom(((PyObject*)this)->ob_type);
}

Tuple* PyTupleInstance::type() {
    return (Tuple*)extractTypeFrom(((PyObject*)this)->ob_type);
}

NamedTuple* PyNamedTupleInstance::type() {
    return (NamedTuple*)extractTypeFrom(((PyObject*)this)->ob_type);
}

PyObject* PyCompositeTypeInstance::sq_item_concrete(Py_ssize_t ix) {
    if (ix < 0 || ix >= (int64_t)type()->getTypes().size()) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    Type* eltType = type()->getTypes()[ix];

    return extractPythonObject(type()->eltPtr(dataPtr(), ix), eltType);
}


Py_ssize_t PyCompositeTypeInstance::mp_and_sq_length_concrete() {
    return type()->getTypes().size();
}


void PyNamedTupleInstance::copyConstructFromPythonInstanceConcrete(NamedTuple* namedTupleT, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
    if (PyDict_Check(pyRepresentation)) {
        if (namedTupleT->getTypes().size() < PyDict_Size(pyRepresentation)) {
            throw std::runtime_error("Couldn't initialize type of " + namedTupleT->name() + " because supplied dictionary had too many items");
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
                else if (namedTupleT->is_default_constructible()) {
                    t->constructor(eltPtr);
                } else {
                    throw std::logic_error("Can't default initialize argument " + name);
                }
            });

        if (actuallyUsed != PyDict_Size(pyRepresentation)) {
            throw std::runtime_error("Couldn't initialize type of " + namedTupleT->name() + " because supplied dictionary had unused arguments");
        }

        return;
    }

    PyInstance::copyConstructFromPythonInstanceConcrete(namedTupleT, tgt, pyRepresentation, isExplicit);
}

void PyNamedTupleInstance::constructFromPythonArgumentsConcrete(NamedTuple* namedTupleT, uint8_t* data, PyObject* args, PyObject* kwargs) {
    long actuallyUsed = 0;

    if (kwargs) {
        namedTupleT->constructor(
            data,
            [&](uint8_t* eltPtr, int64_t k) {
                Type* eltType = namedTupleT->getTypes()[k];
                PyObject* o = PyDict_GetItemString(kwargs, namedTupleT->getNames()[k].c_str());
                if (o) {
                    copyConstructFromPythonInstance(eltType, eltPtr, o);
                    actuallyUsed++;
                }
                else if (eltType->is_default_constructible()) {
                    eltType->constructor(eltPtr);
                } else {
                    throw std::logic_error("Can't default initialize argument " + namedTupleT->getNames()[k]);
                }
            });

        if (actuallyUsed != PyDict_Size(kwargs)) {
            namedTupleT->destroy(data);
            throw std::runtime_error("Couldn't initialize type of " + namedTupleT->name() + " because supplied dictionary had unused arguments");
        }

        return;
    }

    return PyInstance::constructFromPythonArgumentsConcrete(namedTupleT, data, args, kwargs);
}