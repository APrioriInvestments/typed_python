#pragma once

#include "PyInstance.hpp"

class PyPythonObjectOfTypeInstance : public PyInstance {
public:
    typedef PythonObjectOfType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(PythonObjectOfType* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
        int isinst = PyObject_IsInstance(pyRepresentation, (PyObject*)eltType->pyType());
        if (isinst == -1) {
            isinst = 0;
            PyErr_Clear();
        }

        if (!isinst) {
            throw std::logic_error("Object of type " + std::string(pyRepresentation->ob_type->tp_name) +
                    " is not an instance of " + ((PythonObjectOfType*)eltType)->pyType()->tp_name);
        }

        Py_INCREF(pyRepresentation);
        ((PyObject**)tgt)[0] = pyRepresentation;
        return;
    }

};

