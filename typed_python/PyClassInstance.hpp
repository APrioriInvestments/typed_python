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

    static void constructFromPythonArgumentsConcrete(Class* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    PyObject* mp_subscript_concrete(PyObject* item);

    Py_ssize_t mp_and_sq_length_concrete();

    std::pair<bool, PyObject*> callMemberFunction(const char* name, PyObject* arg0=nullptr, PyObject* arg1=nullptr);

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    PyObject* pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErr);

    PyObject* pyTernaryUnaryOperatorConcrete(PyObject* rhs, PyObject* ternaryArg, const char* op, const char* opErr);

    static void mirrorTypeInformationIntoPyTypeConcrete(Class* classT, PyTypeObject* pyType);
};
