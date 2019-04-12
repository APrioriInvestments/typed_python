#pragma once

#include "PyInstance.hpp"

class PyConstDictInstance : public PyInstance {
public:
    typedef ConstDictType modeled_type;

    ConstDictType* type();

    PyObject* tp_iter_concrete();

    PyObject* tp_iternext_concrete();

    Py_ssize_t mp_and_sq_length_concrete();

    int sq_contains_concrete(PyObject* item);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    PyObject* mp_subscript_concrete(PyObject* item);

    static PyObject* constDictItems(PyObject *o);

    static PyObject* constDictKeys(PyObject *o);

    static PyObject* constDictValues(PyObject *o);

    static PyObject* constDictGet(PyObject* o, PyObject* args);

    static PyMethodDef* typeMethodsConcrete();

    static void mirrorTypeInformationIntoPyTypeConcrete(ConstDictType* constDictT, PyTypeObject* pyType);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }

    static void copyConstructFromPythonInstanceConcrete(ConstDictType* dictType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        if (PyDict_Check(pyRepresentation)) {
            dictType->constructor(tgt, PyDict_Size(pyRepresentation), false);

            try {
                PyObject *key, *value;
                Py_ssize_t pos = 0;

                int i = 0;

                while (PyDict_Next(pyRepresentation, &pos, &key, &value)) {
                    copyConstructFromPythonInstance(dictType->keyType(), dictType->kvPairPtrKey(tgt, i), key);
                    try {
                        copyConstructFromPythonInstance(dictType->valueType(), dictType->kvPairPtrValue(tgt, i), value);
                    } catch(...) {
                        dictType->keyType()->destroy(dictType->kvPairPtrKey(tgt,i));
                        throw;
                    }
                    dictType->incKvPairCount(tgt);
                    i++;
                }

                dictType->sortKvPairs(tgt);
            } catch(...) {
                dictType->destroy(tgt);
                throw;
            }
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(dictType, tgt, pyRepresentation, isExplicit);
    }

};



