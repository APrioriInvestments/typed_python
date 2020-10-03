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

class PyBytesInstance : public PyInstance {
public:
    typedef BytesType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(BytesType* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
        if (level >= ConversionLevel::New) {
            if (!PyBytes_Check(pyRepresentation)) {
                std::pair<Type*, instance_ptr> typeAndPtrOfArg = extractTypeAndPtrFrom(pyRepresentation);

                // fastpath for ListOf(UInt8) and TupleOf(UInt8)
                if (typeAndPtrOfArg.first && typeAndPtrOfArg.first->isTupleOrListOf() &&
                        ((TupleOrListOfType*)typeAndPtrOfArg.first)->getEltType()->getTypeCategory() == Type::TypeCategory::catUInt8) {
                    TupleOrListOfType* lstType = (TupleOrListOfType*)typeAndPtrOfArg.first;

                    BytesType::Make()->constructor(
                        tgt,
                        lstType->count(typeAndPtrOfArg.second),
                        (const char*)lstType->eltPtr(typeAndPtrOfArg.second, 0)
                    );
                    return;
                }

                PyObjectStealer argTup(PyTuple_New(1));
                PyObjectStealer emptyDict(PyDict_New());
                PyTuple_SetItem(argTup, 0, incref(pyRepresentation));

                PyObjectStealer newBytesObj(
                    PyBytes_Type.tp_new(&PyBytes_Type, argTup, emptyDict)
                );

                if (!newBytesObj) {
                    // take the interpreter's error
                    throw PythonExceptionSet();
                }

                BytesType::Make()->constructor(
                    tgt,
                    PyBytes_GET_SIZE((PyObject*)newBytesObj),
                    PyBytes_AsString((PyObject*)newBytesObj)
                );
                return;
            }
        }

        if (PyBytes_Check(pyRepresentation)) {
            BytesType::Make()->constructor(
                tgt,
                PyBytes_GET_SIZE(pyRepresentation),
                PyBytes_AsString(pyRepresentation)
            );
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        // anything iterable can be a 'bytes'
        if (level >= ConversionLevel::New) {
            return true;
        }

        return PyBytes_Check(pyRepresentation);
    }

    static PyObject* extractPythonObjectConcrete(BytesType* bytesType, instance_ptr data) {
        return PyBytes_FromStringAndSize(
            (const char*)BytesType::Make()->eltPtr(data, 0),
            BytesType::Make()->count(data)
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
