/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

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

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static void mirrorTypeInformationIntoPyTypeConcrete(ConstDictType* constDictT, PyTypeObject* pyType);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, bool isExplicit) {
        if (!isExplicit) {
            if (PyDict_Check(pyRepresentation)) {
                return true;
            }

            return false;
        }

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



