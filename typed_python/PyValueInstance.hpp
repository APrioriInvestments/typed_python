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

class PyValueInstance : public PyInstance {
public:
    typedef Value modeled_type;

    static bool pyObjectsEquivalent(PyObject* lhs, PyObject* rhs) {
        if (lhs == rhs) {
            return true;
        }

        if (PyType_Check(lhs) && PyType_Check(rhs)) {
            Type* lhsType = PyInstance::extractTypeFrom((PyTypeObject*)lhs);
            Type* rhsType = PyInstance::extractTypeFrom((PyTypeObject*)rhs);

            if (lhsType && rhsType && Type::typesEquivalent(lhsType, rhsType)) {
                return true;
            }
        }

        return false;
    }

    static void copyConstructFromPythonInstanceConcrete(
        Value* v, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level
    ) {
        const Instance& elt = v->value();

        if (elt.type()->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType &&
            pyObjectsEquivalent(((PythonObjectOfType*)elt.type())->getPyObj(elt.data()), pyRepresentation)) {
            return;
        }

        else if (compare_to_python(elt.type(), elt.data(), pyRepresentation, level >= ConversionLevel::New ? false : true, Py_EQ)) {
            //it's the value we want
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(v, tgt, pyRepresentation, level);
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* valType, PyObject* pyRepresentation, ConversionLevel level) {
        if (valType->value().type()->getTypeCategory() == Type::TypeCategory::catPythonObjectOfType) {
            PyObject* ourObj = ((PythonObjectOfType*)valType->value().type())->getPyObj(valType->value().data());

            return pyObjectsEquivalent(ourObj, pyRepresentation);
        }

        return compare_to_python(
            valType->value().type(),
            valType->value().data(),
            pyRepresentation, level >= ConversionLevel::New ? false : true,
            Py_EQ
        );
    }

    static PyObject* extractPythonObjectConcrete(Value* valueType, instance_ptr data) {
        return extractPythonObject(valueType->value().data(), valueType->value().type());
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(Value* v, PyTypeObject* pyType) {
        //expose the actual Instance we represent as a member of the type object
        PyObject* pyObj = PyInstance::extractPythonObject(v->value());

        if (!pyObj) {
            throw PythonExceptionSet();
        }

        PyDict_SetItemString(
            pyType->tp_dict,
            "Value",
            pyObj
        );
    }
};
