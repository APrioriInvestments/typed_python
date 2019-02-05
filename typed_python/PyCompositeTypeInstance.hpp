#pragma once

#include "PyInstance.hpp"

class PyCompositeTypeInstance : public PyInstance {
public:
    typedef CompositeType modeled_type;

    CompositeType* type();

    PyObject* sq_item_concrete(Py_ssize_t ix);

    Py_ssize_t mp_and_sq_length_concrete();

    static void copyConstructFromPythonInstanceConcrete(CompositeType* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
        if (PyTuple_Check(pyRepresentation)) {
            if (eltType->getTypes().size() != PyTuple_Size(pyRepresentation)) {
                throw std::runtime_error("Wrong number of arguments to construct " + eltType->name());
            }

            eltType->constructor(tgt,
                [&](uint8_t* eltPtr, int64_t k) {
                    copyConstructFromPythonInstance(eltType->getTypes()[k], eltPtr, PyTuple_GetItem(pyRepresentation,k));
                    }
                );
            return;
        }
        if (PyList_Check(pyRepresentation)) {
            if (eltType->getTypes().size() != PyList_Size(pyRepresentation)) {
                throw std::runtime_error("Wrong number of arguments to construct " + eltType->name());
            }

            eltType->constructor(tgt,
                [&](uint8_t* eltPtr, int64_t k) {
                    copyConstructFromPythonInstance(eltType->getTypes()[k], eltPtr, PyList_GetItem(pyRepresentation,k));
                    }
                );
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return PyTuple_Check(pyRepresentation) || PyList_Check(pyRepresentation) || PyDict_Check(pyRepresentation);
    }
};

class PyTupleInstance : public PyCompositeTypeInstance {
public:
    typedef Tuple modeled_type;

    Tuple* type();
};

class PyNamedTupleInstance : public PyCompositeTypeInstance {
public:
    typedef NamedTuple modeled_type;

    NamedTuple* type();

    static void copyConstructFromPythonInstanceConcrete(NamedTuple* namedTupleT, instance_ptr tgt, PyObject* pyRepresentation) {
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

        PyInstance::copyConstructFromPythonInstanceConcrete(namedTupleT, tgt, pyRepresentation);
    }
};
