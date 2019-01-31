#pragma once

#include "PyInstance.hpp"

class PyClassInstance : public PyInstance {
public:
    Class* type();

    static void initializeClassWithDefaultArguments(Class* cls, uint8_t* data, PyObject* args, PyObject* kwargs);

    static int classInstanceSetAttributeFromPyObject(Class* cls, uint8_t* data, PyObject* attrName, PyObject* attrVal);

};
