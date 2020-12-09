/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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

class PyTupleOrListOfInstance : public PyInstance {
public:
    typedef TupleOrListOfType modeled_type;

    TupleOrListOfType* type();

    PyObject* sq_item_concrete(Py_ssize_t ix);

    Py_ssize_t mp_and_sq_length_concrete();

    PyObject* mp_subscript_concrete(PyObject* item);

    static void copyConstructFromPythonInstanceConcrete(
        TupleOrListOfType* tupT, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level
    );

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    PyObject* pyOperatorAdd(PyObject* rhs, const char* op, const char* opErr, bool reversed);

    PyObject* pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErrRep);

    static PyObject* pointerUnsafe(PyObject* o, PyObject* args);

    static PyObject* toArray(PyObject* o, PyObject* args);

    static PyObject* toBytes(PyObject* o, PyObject* args);

    static PyObject* fromBytes(PyObject* o, PyObject* args, PyObject* kwds);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level);

    static void mirrorTypeInformationIntoPyTypeConcrete(TupleOrListOfType* inType, PyTypeObject* pyType);
};

class PyListOfInstance : public PyTupleOrListOfInstance {
public:
    typedef ListOfType modeled_type;

    ListOfType* type();

    static PyObject* listAppend(PyObject* o, PyObject* args);

    static PyObject* listAppendDirect(PyObject* o, PyObject* args);

    static PyObject* listExtend(PyObject* o, PyObject* args);

    static PyObject* listResize(PyObject* o, PyObject* args);

    static PyObject* listReserve(PyObject* o, PyObject* args);

    static PyObject* listClear(PyObject* o, PyObject* args);

    static PyObject* listReserved(PyObject* o, PyObject* args);

    static PyObject* listPop(PyObject* o, PyObject* args);

    static PyObject* listSetSizeUnsafe(PyObject* o, PyObject* args);

    static PyObject* listTranspose(PyObject* o, PyObject* args);

    int mp_ass_subscript_concrete(PyObject* item, PyObject* value);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static void constructFromPythonArgumentsConcrete(ListOfType* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    static bool compare_to_python_concrete(ListOfType* listT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);
};

class PyTupleOfInstance : public PyTupleOrListOfInstance {
public:
    typedef TupleOfType modeled_type;

    TupleOfType* type();

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static bool compare_to_python_concrete(TupleOfType* tupT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);
};
