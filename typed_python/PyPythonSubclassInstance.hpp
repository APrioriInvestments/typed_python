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

class PyPythonSubclassInstance : public PyInstance {
public:
    typedef PythonSubclass modeled_type;

    static void copyConstructFromPythonInstanceConcrete(PythonSubclass* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
        std::pair<Type*, instance_ptr> typeAndPtr = extractTypeAndPtrFrom(pyRepresentation);
        Type* argType = typeAndPtr.first;
        instance_ptr argDataPtr = typeAndPtr.second;

        if (argType && Type::typesEquivalent(argType, eltType)) {
            eltType->getBaseType()->copy_constructor(tgt, argDataPtr);
            return;
        }

        if (argType && Type::typesEquivalent(argType, eltType->getBaseType())) {
            eltType->getBaseType()->copy_constructor(tgt, argDataPtr);
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
    }

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
        return specializeForType((PyObject*)this, [&](auto& subtype) {
            return subtype.tp_getattr_concrete(pyAttrName, attrName);
        }, type()->getBaseType());
    }

    int tp_setattr_concrete(PyObject* pyAttrName, PyObject* attrVal) {
        return specializeForTypeReturningInt((PyObject*)this, [&](auto& subtype) {
            return subtype.tp_setattr_concrete(pyAttrName, attrVal);
        }, type()->getBaseType());
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* eltType, PyObject* pyRepresentation, ConversionLevel level) {
        Type* argType = extractTypeFrom(pyRepresentation->ob_type);

        if (argType && Type::typesEquivalent(argType, eltType)) {
            return true;
        }

        if (argType && Type::typesEquivalent(argType, eltType->getBaseType())) {
            return true;
        }

        return false;
    }

    static void constructFromPythonArgumentsConcrete(Type* t, uint8_t* data, PyObject* args, PyObject* kwargs) {
        constructFromPythonArguments(data, (Type*)t->getBaseType(), args, kwargs);
    }

    Py_ssize_t mp_and_sq_length_concrete() {
        return specializeForTypeReturningSizeT((PyObject*)this, [&](auto& subtype) {
            return subtype.mp_and_sq_length_concrete();
        }, type()->getBaseType());
    }

    int pyInquiryConcrete(const char* op, const char* opErrRep) {
        // op == '__bool__'
        return 1;
    }
};
