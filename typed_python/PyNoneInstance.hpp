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

class PyNoneInstance : public PyInstance {
public:
    typedef NoneType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(NoneType* none, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
        if (pyRepresentation == Py_None) {
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(none, tgt, pyRepresentation, level);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return pyRepresentation == Py_None;
    }

    static PyObject* extractPythonObjectConcrete(NoneType* valueType, instance_ptr data) {
        return incref(Py_None);
    }

    static bool compare_to_python_concrete(NoneType* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        return cmpResultToBoolForPyOrdering(
            pyComparisonOp,
            other == Py_None ? 0 : 1
            );
    }
};
