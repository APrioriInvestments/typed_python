#pragma once

#include "PyInstance.hpp"

class PyClassInstance : public PyInstance {
public:
    typedef Class modeled_type;

    Class* type();

    static void initializeClassWithDefaultArguments(Class* cls, uint8_t* data, PyObject* args, PyObject* kwargs);

    static int classInstanceSetAttributeFromPyObject(Class* cls, uint8_t* data, PyObject* attrName, PyObject* attrVal);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }

    PyObject* mp_subscript_concrete(PyObject* item);

    Py_ssize_t mp_and_sq_length_concrete();

    std::pair<bool, PyObject*> callMemberFunction(const char* name, PyObject* arg=nullptr);
};
