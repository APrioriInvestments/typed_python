#pragma once

#include "PyInstance.hpp"

class PyPythonSubclassInstance : public PyInstance {
public:
    typedef PythonSubclass modeled_type;

    static void copyConstructFromPythonInstanceConcrete(PythonSubclass* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        copyConstructFromPythonInstance(eltType->getBaseType(), tgt, pyRepresentation);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }

    static void constructFromPythonArgumentsConcrete(Type* t, uint8_t* data, PyObject* args, PyObject* kwargs) {
        constructFromPythonArguments(data, (Type*)t->getBaseType(), args, kwargs);
    }
};

