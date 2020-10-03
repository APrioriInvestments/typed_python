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

class PyForwardInstance : public PyInstance {
public:
    typedef Forward modeled_type;

    static bool pyValCouldBeOfTypeConcrete(Type* t, PyObject* pyRepresentation, ConversionLevel level) {
        return false;
    }

    static void copyConstructFromPythonInstanceConcrete(Forward* eltType, instance_ptr tgt, PyObject* pyRepresentation, ConversionLevel level) {
        throw std::logic_error("Can't construct instances of a Forward.");
    }

    static void constructFromPythonArgumentsConcrete(Forward* t, uint8_t* data, PyObject* args, PyObject* kwargs) {
        throw std::logic_error("Can't construct instances of a Forward.");
    }


    static PyObject* forwardDefine(PyObject *o, PyObject* args) {
        if (PyTuple_Size(args) != 1) {
            PyErr_SetString(PyExc_TypeError, "Forward.define takes one argument");
            return NULL;
        }

        PyObjectHolder target(PyTuple_GetItem(args,0));

        Forward* self_type = (Forward*)PyInstance::unwrapTypeArgToTypePtr(o);
        if (!self_type) {
            PyErr_SetString(PyExc_TypeError, "Forward.define unexpected error");
            return NULL;
        }
        Type* target_type = PyInstance::unwrapTypeArgToTypePtr(target);
        if (!target_type) {
            PyErr_SetString(PyExc_TypeError, "Forward.define requires a type argument");
            return NULL;
        }

        return translateExceptionToPyObject([&]() {
            Type* result = self_type->define(target_type);
            return incref((PyObject*)PyInstance::typeObj(result));
        });
    }

    static PyObject* forwardGet(PyObject* o, PyObject* args) {
        if (PyTuple_Size(args) != 0) {
            PyErr_SetString(PyExc_TypeError, "Forward.get takes zero arguments");
            return NULL;
        }

        Forward* self_type = (Forward*)PyInstance::unwrapTypeArgToTypePtr(o);
        if (!self_type) {
            PyErr_SetString(PyExc_TypeError, "Forward.get unexpected error");
            return NULL;
        }

        Type* result = self_type->getTarget();

        if (!result) {
            return incref(Py_None);
        }

        return incref(PyInstance::typePtrToPyTypeRepresentation(result));
    }


    static PyObject* forwardGetReferencing(PyObject* o, PyObject* args) {
        if (PyTuple_Size(args) != 0) {
            PyErr_SetString(PyExc_TypeError, "Forward.getReferencing takes zero arguments");
            return NULL;
        }

        Forward* self_type = (Forward*)PyInstance::unwrapTypeArgToTypePtr(o);
        if (!self_type) {
            PyErr_SetString(PyExc_TypeError, "Forward.getReferencing unexpected error");
            return NULL;
        }

        PyObject* res = PyList_New(0);

        for (auto t: self_type->getReferencing()) {
            PyList_Append(res, PyInstance::typePtrToPyTypeRepresentation(t));
        }

        return res;
    }

    static PyMethodDef* typeMethodsConcrete(Type* t) {
        return new PyMethodDef [4] {
            {"define", (PyCFunction)PyForwardInstance::forwardDefine, METH_VARARGS | METH_CLASS, NULL},
            {"get", (PyCFunction)PyForwardInstance::forwardGet, METH_VARARGS | METH_CLASS, NULL},
            {"getReferencing", (PyCFunction)PyForwardInstance::forwardGetReferencing, METH_VARARGS | METH_CLASS, NULL},
            {NULL, NULL}
        };
    }

};
