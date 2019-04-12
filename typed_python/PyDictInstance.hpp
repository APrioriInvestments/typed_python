#pragma once

#include "PyInstance.hpp"

class PyDictInstance : public PyInstance {
public:
    typedef DictType modeled_type;

    DictType* type();

    PyObject* tp_iter_concrete();

    PyObject* tp_iternext_concrete();

    Py_ssize_t mp_and_sq_length_concrete();

    int sq_contains_concrete(PyObject* item);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    PyObject* mp_subscript_concrete(PyObject* item);

    int mp_ass_subscript_concrete(PyObject* item, PyObject* value);

    int mp_ass_subscript_concrete_typed(instance_ptr key, instance_ptr value);

    int mp_ass_subscript_concrete_keytyped(PyObject* pyKey, instance_ptr key, PyObject* value);

    static PyObject* dictItems(PyObject *o);

    static PyObject* dictKeys(PyObject *o);

    static PyObject* dictValues(PyObject *o);

    static PyObject* dictGet(PyObject* o, PyObject* args);

    static PyMethodDef* typeMethodsConcrete();

    static void mirrorTypeInformationIntoPyTypeConcrete(DictType* dictT, PyTypeObject* pyType);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }

    static void copyConstructFromPythonInstanceConcrete(DictType* dictType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit);
};



