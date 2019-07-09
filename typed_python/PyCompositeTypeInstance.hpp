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

class PyCompositeTypeInstance : public PyInstance {
public:
    typedef CompositeType modeled_type;

    CompositeType* type();

    PyObject* sq_item_concrete(Py_ssize_t ix);

    Py_ssize_t mp_and_sq_length_concrete();

    static void copyConstructFromPythonInstanceConcrete(CompositeType* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        if (PyTuple_Check(pyRepresentation)) {
            if (eltType->getTypes().size() != PyTuple_Size(pyRepresentation)) {
                throw std::runtime_error("Wrong number of arguments to construct " + eltType->name());
            }

            eltType->constructor(tgt,
                [&](uint8_t* eltPtr, int64_t k) {
                    PyObjectHolder arg(PyTuple_GetItem(pyRepresentation, k));
                    copyConstructFromPythonInstance(eltType->getTypes()[k], eltPtr, arg);
                    }
                );
            return;
        }
        if (PyList_Check(pyRepresentation)) {
            if (eltType->getTypes().size() != PyList_Size(pyRepresentation)) {
                throw std::runtime_error("Wrong number of arguments to construct " + eltType->name());
            }

            eltType->constructor(tgt,
                [&](uint8_t* eltPtr, int64_t k) {
                    PyObjectHolder listItem(PyList_GetItem(pyRepresentation,k));
                    copyConstructFromPythonInstance(eltType->getTypes()[k], eltPtr, listItem);
                    }
                );
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, isExplicit);
    }

    static bool compare_to_python_concrete(CompositeType* tupT, instance_ptr self, PyObject* other, bool exact, int pyComparisonOp) {
        auto convert = [&](char cmpValue) { return cmpResultToBoolForPyOrdering(pyComparisonOp, cmpValue); };

        if (!PyTuple_Check(other)) {
            return convert(-1);
        }

        int lenO = PyTuple_Size(other);
        int lenS = tupT->getTypes().size();

        for (long k = 0; k < lenO && k < lenS; k++) {
            PyObjectHolder arg(PyTuple_GetItem(other, k));

            if (!compare_to_python(tupT->getTypes()[k], tupT->eltPtr(self, k), arg, exact, Py_EQ)) {
                if (pyComparisonOp == Py_EQ || pyComparisonOp == Py_NE)
                    return convert(1);  // for EQ, NE, don't compare further
                if (compare_to_python(tupT->getTypes()[k], tupT->eltPtr(self, k), arg, exact, Py_LT)) {
                    return convert(-1);
                }
                return convert(1);
            }
        }

        if (lenS < lenO) { return convert(-1); }
        if (lenS > lenO) { return convert(1); }

        return convert(0);
    }


    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, bool isExplicit) {
        return PyTuple_Check(pyRepresentation) || PyList_Check(pyRepresentation) || PyDict_Check(pyRepresentation);
    }
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

    static void copyConstructFromPythonInstanceConcrete(NamedTuple* namedTupleT, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit);

    static void constructFromPythonArgumentsConcrete(NamedTuple* t, uint8_t* data, PyObject* args, PyObject* kwargs);

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName);

    static void mirrorTypeInformationIntoPyTypeConcrete(NamedTuple* tupleT, PyTypeObject* pyType);

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal);

    static PyMethodDef* typeMethodsConcrete(Type* t);

    static PyObject* replacing(PyObject* o, PyObject* args, PyObject* kwargs);

private:

    static int findElementIndex(const std::vector<std::string>& container, const std::string &element);
};
