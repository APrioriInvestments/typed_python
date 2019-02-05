#pragma once

#include "PyInstance.hpp"

class PyPythonSubclassInstance : public PyInstance {
public:
    typedef PythonSubclass modeled_type;

    static void copyConstructFromPythonInstanceConcrete(PythonSubclass* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
        copyConstructFromPythonInstance(eltType->getBaseType(), tgt, pyRepresentation);
    }
};

