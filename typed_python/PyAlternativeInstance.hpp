#pragma once

#include "PyInstance.hpp"

class PyAlternativeInstance : public PyInstance {
public:
    typedef Alternative modeled_type;

    Alternative* type();

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* ternary, const char* op, const char* opErr);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr);

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    static void mirrorTypeInformationIntoPyTypeConcrete(Alternative* alt, PyTypeObject* pyType);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }
};

class PyConcreteAlternativeInstance : public PyInstance {
public:
    typedef ConcreteAlternative modeled_type;

    ConcreteAlternative* type();

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* ternary, const char* op, const char* opErr);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr);

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(ConcreteAlternative* alt, PyTypeObject* pyType);

    static void constructFromPythonArgumentsConcrete(ConcreteAlternative* t, uint8_t* data, PyObject* args, PyObject* kwargs);

};
