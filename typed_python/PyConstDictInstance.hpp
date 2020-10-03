/******************************************************************************
   Copyright 2017-2019 typed_python Authors

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

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    PyObject* mp_subscript_concrete(PyObject* item);

    static PyObject* constDictItems(PyObject *o);

    static PyObject* constDictKeys(PyObject *o);

    static PyObject* constDictValues(PyObject *o);

    static PyObject* constDictGet(PyObject* o, PyObject* args);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static void mirrorTypeInformationIntoPyTypeConcrete(ConstDictType* constDictT, PyTypeObject* pyType);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level);

    static bool compare_to_python_concrete(ConstDictType* dictType, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    static void copyConstructFromPythonInstanceConcrete(ConstDictType* dictType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);

    bool compare_as_iterator_to_python_concrete(PyObject* other, int pyComparisonOp);

    PyObject* tp_repr_concrete();

    PyObject* tp_str_concrete();
};
