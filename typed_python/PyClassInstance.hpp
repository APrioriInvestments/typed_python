#pragma once

#include "PyInstance.hpp"

class PyClassInstance : public PyInstance {
public:
    typedef Class modeled_type;

    Class* type();

    static void initializeClassWithDefaultArguments(Class* cls, uint8_t* data, PyObject* args, PyObject* kwargs);

    static int classInstanceSetAttributeFromPyObject(Class* cls, uint8_t* data, PyObject* attrName, PyObject* attrVal);

};
