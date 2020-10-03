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

class PyRefToInstance : public PyInstance {
public:
    typedef RefTo modeled_type;

    RefTo* type();

    static void mirrorTypeInformationIntoPyTypeConcrete(RefTo* pointerT, PyTypeObject* pyType);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return true;
    }

    std::pair<bool, PyObject*> callMemberFunction(const char* name, PyObject* arg0=nullptr, PyObject* arg1=nullptr, PyObject* arg2=nullptr);

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    static PyMethodDef* typeMethodsConcrete(Type* t);
};
