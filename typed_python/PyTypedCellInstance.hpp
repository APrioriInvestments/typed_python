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
#include "TypedCellType.hpp"

class PyTypedCellInstance : public PyInstance {
public:
    typedef TypedCellType modeled_type;

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return false;
    }

    static PyMethodDef* typeMethodsConcrete(Type* t) {
        return new PyMethodDef [5] {
            {"set", (PyCFunction)PyTypedCellInstance::set, METH_VARARGS | METH_KEYWORDS, NULL},
            {"clear", (PyCFunction)PyTypedCellInstance::clear, METH_VARARGS | METH_KEYWORDS, NULL},
            {"isSet", (PyCFunction)PyTypedCellInstance::isSet, METH_VARARGS | METH_KEYWORDS, NULL},
            {"get", (PyCFunction)PyTypedCellInstance::get, METH_VARARGS | METH_KEYWORDS, NULL},
            {NULL, NULL}
        };
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(TypedCellType* t, PyTypeObject* pyType) {
        PyDict_SetItemString(pyType->tp_dict, "HeldType", typePtrToPyTypeRepresentation(t->getHeldType()));
    }

    static PyObject* set(PyObject* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {"value", NULL};

        PyObject* value;

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &value)) {
            return NULL;
        }

        PyTypedCellInstance* self_w = (PyTypedCellInstance*)self;
        TypedCellType* self_t = (TypedCellType*)self_w->type();

        return translateExceptionToPyObject([&]() {
            self_t->set(self_w->dataPtr(), [&](instance_ptr data) {
                PyInstance::copyConstructFromPythonInstance(
                    self_t->getHeldType(),
                    data,
                    value,
                    ConversionLevel::Implicit
                );
            });

            return incref(Py_None);
        });
    }

    static PyObject* clear(PyObject* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        PyTypedCellInstance* self_w = (PyTypedCellInstance*)self;
        TypedCellType* self_t = (TypedCellType*)self_w->type();

        self_t->clear(self_w->dataPtr());

        return incref(Py_None);
    }

    static PyObject* isSet(PyObject* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        PyTypedCellInstance* self_w = (PyTypedCellInstance*)self;
        TypedCellType* self_t = (TypedCellType*)self_w->type();

        return incref(self_t->isSet(self_w->dataPtr()) ? Py_True : Py_False);
    }

    static PyObject* get(PyObject* self, PyObject* args, PyObject* kwargs) {
        static const char *kwlist[] = {NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
            return NULL;
        }

        PyTypedCellInstance* self_w = (PyTypedCellInstance*)self;
        TypedCellType* self_t = (TypedCellType*)self_w->type();

        if (!self_t->isSet(self_w->dataPtr())) {
            PyErr_SetString(PyExc_RuntimeError, "Cell is empty");
            return NULL;
        }

        return PyInstance::extractPythonObject(
            self_t->get(self_w->dataPtr()),
            self_t->getHeldType()
        );
    }
};
