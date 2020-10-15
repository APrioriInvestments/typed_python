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

class PyCompositeTypeInstance : public PyInstance {
public:
    typedef CompositeType modeled_type;

    // return the CompositeType (not the subclass) that we're representing here.
    CompositeType* type();

    // return the actual type (possibly a subclass) of this instance.
    Type* actualType();

    PyObject* sq_item_concrete(Py_ssize_t ix);

    Py_ssize_t mp_and_sq_length_concrete();

    static void copyConstructFromPythonInstanceConcrete(CompositeType* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);

    static bool compare_to_python_concrete(CompositeType* tupT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp);

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return true;
    }

    int pyInquiryConcrete(const char* op, const char* opErrRep);
};

class PyTupleInstance : public PyCompositeTypeInstance {
public:
    typedef Tuple modeled_type;

    Tuple* type();

    static void mirrorTypeInformationIntoPyTypeConcrete(Tuple* tupleT, PyTypeObject* pyType);
};

class PyNamedTupleInstance : public PyCompositeTypeInstance {
public:
    typedef NamedTuple modeled_type;

    NamedTuple* type();

    static void copyConstructFromPythonInstanceConcrete(NamedTuple* namedTupleT, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level);

    static void constructFromPythonArgumentsConcrete(
        NamedTuple* t,
        uint8_t* data,
        PyObject* args,
        PyObject* kwargs
    );

    static void constructFromPythonArgumentsConcreteWithLevel(
        NamedTuple* t,
        uint8_t* data,
        PyObject* args,
        PyObject* kwargs,
        ConversionLevel level
    );

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    static void mirrorTypeInformationIntoPyTypeConcrete(NamedTuple* tupleT, PyTypeObject* pyType);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static PyObject* replacing(PyObject* o, PyObject* args, PyObject* kwargs);

private:

    static int findElementIndex(const std::vector<std::string>& container, const std::string &element);
};
