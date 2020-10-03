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

class PyAlternativeInstance : public PyInstance {
public:
    typedef Alternative modeled_type;

    Alternative* type();

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    static void mirrorTypeInformationIntoPyTypeConcrete(Alternative* alt, PyTypeObject* pyType);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        Type* argType = extractTypeFrom(pyRepresentation->ob_type);

        if (argType && argType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative &&
                ((ConcreteAlternative*)argType)->getAlternative() == type) {
            return true;
        }

        return false;
    }

    static void copyConstructFromPythonInstanceConcrete(Alternative* altType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);
};

class PyConcreteAlternativeInstance : public PyInstance {
public:
    typedef ConcreteAlternative modeled_type;

    ConcreteAlternative* type();

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* ternary, const char* op, const char* opErr);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);
    PyObject* pyOperatorConcreteReverse(PyObject* rhs, const char* op, const char* opErr);

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr);

    std::pair<bool, PyObject*> callMethod(const char* name, PyObject* arg0=nullptr, PyObject* arg1=nullptr, PyObject* arg2=nullptr);

    int64_t tryCallHashMethod();

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    int sq_contains_concrete(PyObject* item);

    Py_ssize_t mp_and_sq_length_concrete();

    PyObject* mp_subscript_concrete(PyObject* item);

    int mp_ass_subscript_concrete(PyObject* item, PyObject* v);

    PyObject* tp_iter_concrete();

    PyObject* tp_iternext_concrete();

    static bool compare_to_python_concrete(ConcreteAlternative* altT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return extractTypeFrom(pyRepresentation->ob_type) == type;
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(ConcreteAlternative* alt, PyTypeObject* pyType);

    static void constructFromPythonArgumentsConcrete(ConcreteAlternative* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    static PyMethodDef* typeMethodsConcrete(Type* t);
private:
    static PyObject* altFormat(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altBytes(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altDir(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altReversed(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altComplex(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altRound(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altTrunc(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altFloor(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altCeil(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altEnter(PyObject* o, PyObject* args, PyObject* kwargs);
    static PyObject* altExit(PyObject* o, PyObject* args, PyObject* kwargs);
};
