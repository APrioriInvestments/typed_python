#pragma once

#include "PyInstance.hpp"

class PyBytesInstance : public PyInstance {
public:
    typedef Bytes modeled_type;

    static void copyConstructFromPythonInstanceConcrete(Bytes* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
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

};

