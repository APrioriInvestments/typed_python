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

class PyStringInstance : public PyInstance {
public:
    typedef StringType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(StringType* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
        if (level >= ConversionLevel::New) {
            PyObjectStealer newStr(PyObject_Str(pyRepresentation));
            if (!newStr) {
                // defer to python execption
                throw PythonExceptionSet();
            }

            copyConstructFromPythonInstanceConcrete(eltType, tgt, newStr, ConversionLevel::Signature);
            return;
        }

        if (PyUnicode_Check(pyRepresentation)) {
            if (PyUnicode_READY(pyRepresentation) == -1) throw PythonExceptionSet();

            auto kind = PyUnicode_KIND(pyRepresentation);
            assert(
                kind == PyUnicode_1BYTE_KIND ||
                kind == PyUnicode_2BYTE_KIND ||
                kind == PyUnicode_4BYTE_KIND
                );
            StringType::Make()->constructor(
                tgt,
                kind == PyUnicode_1BYTE_KIND ? 1 :
                kind == PyUnicode_2BYTE_KIND ? 2 :
                                                4,
                PyUnicode_GET_LENGTH(pyRepresentation),
                kind == PyUnicode_1BYTE_KIND ? (const char*)PyUnicode_1BYTE_DATA(pyRepresentation) :
                kind == PyUnicode_2BYTE_KIND ? (const char*)PyUnicode_2BYTE_DATA(pyRepresentation) :
                                               (const char*)PyUnicode_4BYTE_DATA(pyRepresentation)
                );
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, level);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        if (level >= ConversionLevel::New) {
            return true;
        }

        return PyUnicode_Check(pyRepresentation);
    }

    static PyObject* extractPythonObjectConcrete(StringType* t, instance_ptr data) {
        int bytes_per_codepoint = StringType::Make()->bytes_per_codepoint(data);

        return PyUnicode_FromKindAndData(
            bytes_per_codepoint == 1 ? PyUnicode_1BYTE_KIND :
            bytes_per_codepoint == 2 ? PyUnicode_2BYTE_KIND :
                                       PyUnicode_4BYTE_KIND,
            StringType::Make()->eltPtr(data, 0),
            StringType::Make()->count(data)
            );
    }

    static bool compare_to_python_concrete(StringType* t, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        if (!PyUnicode_Check(other)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }

        auto kind = PyUnicode_KIND(other);

        int bytesPer = kind == PyUnicode_1BYTE_KIND ? 1 :
            kind == PyUnicode_2BYTE_KIND ? 2 : 4;

        int bytesPerCodepoint = t->bytes_per_codepoint(self);

        if (bytesPer != bytesPerCodepoint) {
            uint8_t* data =
                kind == PyUnicode_1BYTE_KIND ? (uint8_t*)PyUnicode_1BYTE_DATA(other) :
                kind == PyUnicode_2BYTE_KIND ? (uint8_t*)PyUnicode_2BYTE_DATA(other) :
                                               (uint8_t*)PyUnicode_4BYTE_DATA(other);
            int res = StringType::cmpStatic(*(StringType::layout**)self, data, PyUnicode_GET_LENGTH(other), bytesPer);
            return cmpResultToBoolForPyOrdering(pyComparisonOp, res);
        }

        if (PyUnicode_GET_LENGTH(other) < t->count(self)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, -1);
        }

        if (PyUnicode_GET_LENGTH(other) > t->count(self)) {
            return cmpResultToBoolForPyOrdering(pyComparisonOp, 1);
        }

        return cmpResultToBoolForPyOrdering(
            pyComparisonOp,
            memcmp(
                kind == PyUnicode_1BYTE_KIND ? (const char*)PyUnicode_1BYTE_DATA(other) :
                kind == PyUnicode_2BYTE_KIND ? (const char*)PyUnicode_2BYTE_DATA(other) :
                                               (const char*)PyUnicode_4BYTE_DATA(other),
                ((StringType*)t)->eltPtr(self, 0),
                PyUnicode_GET_LENGTH(other) * bytesPer
                )
            );
    }

};
