#pragma once

#include "PyInstance.hpp"

class PyCompositeTypeInstance : public PyInstance {
public:
    typedef CompositeType modeled_type;

    CompositeType* type();

    PyObject* sq_item_concrete(Py_ssize_t ix);

    Py_ssize_t mp_and_sq_length_concrete();

    static void copyConstructFromPythonInstanceConcrete(CompositeType* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
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

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, isExplicit);
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

    static void copyConstructFromPythonInstanceConcrete(NamedTuple* namedTupleT, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit);

    static void constructFromPythonArgumentsConcrete(NamedTuple* t, uint8_t* data, PyObject* args, PyObject* kwargs);
};
