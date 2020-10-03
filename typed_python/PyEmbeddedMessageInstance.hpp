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

class PyEmbeddedMessageInstance : public PyInstance {
public:
    typedef EmbeddedMessageType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(
        EmbeddedMessageType* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level
    ) {
        if (PyBytes_Check(pyRepresentation) && level >= ConversionLevel::Implicit) {
            EmbeddedMessageType::Make()->constructor(
                tgt,
                PyBytes_GET_SIZE(pyRepresentation),
                PyBytes_AsString(pyRepresentation)
            );
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return PyBytes_Check(pyRepresentation);
    }

    static PyObject* extractPythonObjectConcrete(modeled_type* bytesType, instance_ptr data) {
        return PyBytes_FromStringAndSize(
            (const char*)EmbeddedMessageType::Make()->eltPtr(data, 0),
            EmbeddedMessageType::Make()->count(data)
        );
    }

    static bool compare_to_python_concrete(modeled_type* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        if (!PyBytes_Check(other)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }

        if (PyBytes_GET_SIZE(other) < ((EmbeddedMessageType*)t)->count(self)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }

        if (PyBytes_GET_SIZE(other) > ((EmbeddedMessageType*)t)->count(self)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }

        return cmpResultToBoolForPyOrdering(
            pyComparisonOp,
            memcmp(
                PyBytes_AsString(other),
                ((EmbeddedMessageType*)t)->eltPtr(self, 0),
                PyBytes_GET_SIZE(other)
                )
            );
    }
};
