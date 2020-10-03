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

class PyDictInstance : public PyInstance {
public:
    typedef DictType modeled_type;

    DictType* type();

    PyObject* tp_iter_concrete();

    PyObject* tp_iternext_concrete();

    Py_ssize_t mp_and_sq_length_concrete();

    int sq_contains_concrete(PyObject* item);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    PyObject* mp_subscript_concrete(PyObject* item);

    int mp_ass_subscript_concrete(PyObject* item, PyObject* value);

    int mp_ass_subscript_concrete_typed(instance_ptr key, instance_ptr value);

    int mp_ass_subscript_concrete_keytyped(PyObject* pyKey, instance_ptr key, PyObject* value);

    static PyObject* dictItems(PyObject *o);

    static PyObject* dictKeys(PyObject *o);

    static PyObject* dictValues(PyObject *o);

    static PyObject* dictGet(PyObject* o, PyObject* args);

    static PyObject* dictUpdate(PyObject* o, PyObject* args);

    static PyObject* dictClear(PyObject* o);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static void mirrorTypeInformationIntoPyTypeConcrete(DictType* dictT, PyTypeObject* pyType);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return true;
    }

    static void copyConstructFromPythonInstanceConcrete(DictType* dictType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);

    static void constructFromPythonArgumentsConcrete(DictType* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    static bool compare_to_python_concrete(DictType* listT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    bool compare_as_iterator_to_python_concrete(PyObject* other, int pyComparisonOp);

    // override tp_repr so that we can check if the object is 'keys', 'values' or 'items'
    PyObject* tp_repr_concrete();

    PyObject* tp_str_concrete();

    /**
     * Function implementing python's dict::setdefault.
     *
     * https://docs.python.org/3/library/stdtypes.html#dict.setdefault
     *
     * setdefault(key[, default])
     *      If key is in the dictionary, return its value.
     *      If not, insert key with a value of default and return default. default defaults to None.
     *
     */
    static PyObject* setDefault(PyObject* o, PyObject* args);

    static PyObject* pop(PyObject* o, PyObject* args);
};
