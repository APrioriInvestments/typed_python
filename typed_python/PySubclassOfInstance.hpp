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

class PySubclassOfInstance : public PyInstance {
public:
    typedef SubclassOfType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(SubclassOfType* subclassOf, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
        Type* t = PyInstance::tryUnwrapPyInstanceToType(pyRepresentation);

        if (t && (t == subclassOf->getSubclassOf() || t->isSubclassOf(subclassOf->getSubclassOf()))) {
            *((Type**)tgt) = t;
            return;
        }

        if (t) {
            throw std::logic_error("Cannot construct a " + subclassOf->name() + " from the type " + t->name());
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(subclassOf, tgt, pyRepresentation, level);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return true;
    }

    static PyObject* extractPythonObjectConcrete(modeled_type* subclassOfT, instance_ptr data) {
        return incref(typePtrToPyTypeRepresentation(*((Type**)data)));
    }

    static bool compare_to_python_concrete(SubclassOfType* subclassOf, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        Type* t = PyInstance::tryUnwrapPyInstanceToType(other);

        return subclassOf->cmp(self, (instance_ptr)&t, pyComparisonOp, false);
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(SubclassOfType* subclassOfT, PyTypeObject* pyType) {
        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(pyType->tp_dict, "Type", typePtrToPyTypeRepresentation(subclassOfT->getSubclassOf()));
    }
};
