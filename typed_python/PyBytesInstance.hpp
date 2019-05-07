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

class PyBytesInstance : public PyInstance {
public:
    typedef BytesType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(BytesType* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        if (PyBytes_Check(pyRepresentation)) {
            BytesType().constructor(
                tgt,
                PyBytes_GET_SIZE(pyRepresentation),
                PyBytes_AsString(pyRepresentation)
                );
            return;
        }
        throw std::logic_error("Can't initialize a Bytes object from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return PyBytes_Check(pyRepresentation);
    }

    static PyObject* extractPythonObjectConcrete(BytesType* bytesType, instance_ptr data) {
        return PyBytes_FromStringAndSize(
            (const char*)BytesType().eltPtr(data, 0),
            BytesType().count(data)
            );
    }

    static bool compare_to_python_concrete(BytesType* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        if (!PyBytes_Check(other)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }

        if (PyBytes_GET_SIZE(other) < ((BytesType*)t)->count(self)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }

        if (PyBytes_GET_SIZE(other) > ((BytesType*)t)->count(self)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }

        return cmpResultToBoolForPyOrdering(
            pyComparisonOp,
            memcmp(
                PyBytes_AsString(other),
                ((BytesType*)t)->eltPtr(self, 0),
                PyBytes_GET_SIZE(other)
                )
            );
    }
};

