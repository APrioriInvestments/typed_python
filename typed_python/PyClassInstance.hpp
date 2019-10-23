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

class PyClassInstance : public PyInstance {
public:
    typedef Class modeled_type;

    Class* type();

    static void initializeClassWithDefaultArguments(Class* cls, uint8_t* data, PyObject* args, PyObject* kwargs);

    static int classInstanceSetAttributeFromPyObject(Class* cls, uint8_t* data, PyObject* attrName, PyObject* attrVal);

    static bool pyValCouldBeOfTypeConcrete(Class* type, PyObject* pyRepresentation, bool isExplicit);

    static PyObject* extractPythonObjectConcrete(Type* eltType, instance_ptr data);

    static void copyConstructFromPythonInstanceConcrete(Class* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit);

    static void constructFromPythonArgumentsConcrete(Class* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    std::pair<bool, PyObject*> callMemberFunction(const char* name, PyObject* arg0=nullptr, PyObject* arg1=nullptr, PyObject* arg2=nullptr);

    int64_t tryCallHashMemberFunction();

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    int sq_contains_concrete(PyObject* item);

    Py_ssize_t mp_and_sq_length_concrete();

    PyObject* mp_subscript_concrete(PyObject* item);
    int mp_ass_subscript_concrete(PyObject* item, PyObject* v);

    PyObject* tp_iter_concrete();

    PyObject* tp_iternext_concrete();

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    PyObject* pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErr);

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* ternaryArg, const char* op, const char* opErr);

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    static void mirrorTypeInformationIntoPyTypeConcrete(Class* classT, PyTypeObject* pyType);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    static PyMethodDef* typeMethodsConcrete(Type* t);
private:
    static PyObject* clsFormat(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsBytes(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsDir(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsReversed(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsComplex(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsRound(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsTrunc(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsFloor(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsCeil(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsEnter(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* clsExit(PyObject* o, PyObject* args, PyObject* kwargs);
};
