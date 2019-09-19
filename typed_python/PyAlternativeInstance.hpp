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

class PyAlternativeInstance : public PyInstance {
public:
    typedef Alternative modeled_type;

    Alternative* type();

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* ternary, const char* op, const char* opErr);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr);

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    static void mirrorTypeInformationIntoPyTypeConcrete(Alternative* alt, PyTypeObject* pyType);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, bool isExplicit) {
      Type* argType = extractTypeFrom(pyRepresentation->ob_type);

      if (argType && argType->getTypeCategory() == Type::TypeCategory::catConcreteAlternative &&
              ((ConcreteAlternative*)argType)->getAlternative() == type) {
        return true;
      }

      return false;
    }

    static void copyConstructFromPythonInstanceConcrete(Alternative* altType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit);
};

class PyConcreteAlternativeInstance : public PyInstance {
public:
    typedef ConcreteAlternative modeled_type;

    ConcreteAlternative* type();

    PyObject* pyTernaryOperatorConcrete(PyObject* rhs, PyObject* ternary, const char* op, const char* opErr);

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr);

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr);

    std::pair<bool, PyObject*> callMethod(const char* name, PyObject* arg0=nullptr, PyObject* arg1=nullptr);

    int pyInquiryConcrete(const char* op, const char* opErrRep);

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs);

    int sq_contains_concrete(PyObject* item);

    Py_ssize_t mp_and_sq_length_concrete();

    static bool compare_to_python_concrete(ConcreteAlternative* altT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, bool isExplicit) {
        return extractTypeFrom(pyRepresentation->ob_type) == type;
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(ConcreteAlternative* alt, PyTypeObject* pyType);

    static void constructFromPythonArgumentsConcrete(ConcreteAlternative* t, uint8_t* data, PyObject* args, PyObject* kwargs);

};
