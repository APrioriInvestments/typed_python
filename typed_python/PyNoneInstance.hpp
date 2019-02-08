#pragma once

#include "PyInstance.hpp"

class PyNoneInstance : public PyInstance {
public:
    typedef None modeled_type;

    static void copyConstructFromPythonInstanceConcrete(None* oneOf, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        if (pyRepresentation == Py_None) {
            return;
        }
        throw std::logic_error("Can't initialize a None from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }

    static PyObject* extractPythonObjectConcrete(None* valueType, instance_ptr data) {
        return incref(Py_None);
    }

    static bool compare_to_python_concrete(None* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        return cmpResultToBoolForPyOrdering(
            pyComparisonOp,
            other == Py_None ? 0 : 1
            );
    }
};

