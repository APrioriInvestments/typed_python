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

class PyPythonObjectOfTypeInstance : public PyInstance {
public:
    typedef PythonObjectOfType modeled_type;

    static void copyConstructFromPythonInstanceConcrete(
        PythonObjectOfType* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level
    ) {
        int isinst = PyObject_IsInstance(pyRepresentation, (PyObject*)eltType->pyType());
        if (isinst == -1) {
            isinst = 0;
            PyErr_Clear();
        }

        if (!isinst) {
            throw std::logic_error("Object of type " + std::string(pyRepresentation->ob_type->tp_name) +
                    " is not an instance of " + ((PythonObjectOfType*)eltType)->pyType()->tp_name);
        }

        eltType->initializeFromPyObject(tgt, pyRepresentation);
        return;
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        int isinst = PyObject_IsInstance(pyRepresentation, (PyObject*)type->pyType());

        if (isinst == -1) {
            isinst = 0;
            PyErr_Clear();
        }

        return isinst > 0;
    }

    static PyObject* extractPythonObjectConcrete(PythonObjectOfType* valueType, instance_ptr data) {
        return incref(valueType->getPyObj(data));
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(PythonObjectOfType* inType, PyTypeObject* pyType) {
        //expose 'ElementType' as a member of the type object
        PyDict_SetItemString(
            pyType->tp_dict,
            "PyType",
            (PyObject*)inType->pyType()
        );
    }

    static bool compare_to_python_concrete(PythonObjectOfType* oType, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        return PyObject_RichCompareBool(oType->getPyObj(self), other, pyComparisonOp);
    }
};
