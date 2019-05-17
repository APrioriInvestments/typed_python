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

class PyPointerToInstance : public PyInstance {
public:
    typedef PointerTo modeled_type;

    PointerTo* type();

    PyObject* sq_item_concrete(Py_ssize_t ix);

    static PyObject* pointerInitialize(PyObject* o, PyObject* args);

    static PyObject* pointerSet(PyObject* o, PyObject* args);

    static PyObject* pointerGet(PyObject* o, PyObject* args);

    static PyObject* pointerCast(PyObject* o, PyObject* args);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    static void mirrorTypeInformationIntoPyTypeConcrete(PointerTo* pointerT, PyTypeObject* pyType);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, bool isExplicit) {
        return true;
    }

    static PyMethodDef* typeMethodsConcrete();
};
