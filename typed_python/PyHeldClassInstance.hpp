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
#include "PyClassInstance.hpp"

class PyHeldClassInstance : public PyInstance {
public:
    typedef HeldClass modeled_type;

    static PyObject* extractPythonObjectConcrete(Type* eltType, instance_ptr data) {
        // return PyInstance::initialize(eltType, [&](instance_ptr selfData) {
        //     eltType->copy_initialize(selfData, data);
        // });
        return NULL;
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation, ConversionLevel level) {
        return true;
    }

    PyObject* tp_call_concrete(PyObject* args, PyObject* kwargs) {
        return PyClassInstance::tpCallConcreteGeneric((PyObject*)this, type(), dataPtr(), args, kwargs);
    }

    PyObject* mp_subscript_concrete(PyObject* item) {
        return PyClassInstance::mpSubscriptGeneric((PyObject*)this, type(), dataPtr(), item);
    }

    int mp_ass_subscript_concrete(PyObject* item, PyObject* v) {
        return PyClassInstance::mpAssignSubscriptGeneric(
            (PyObject*)this, type(), dataPtr(), item, v
        );
    }

    int sq_contains_concrete(PyObject* item) {
        return PyClassInstance::sqContainsGeneric((PyObject*)this, type(), dataPtr(), item);
    }

    Py_ssize_t mp_and_sq_length_concrete() {
        return PyClassInstance::mpAndSqLengthGeneric((PyObject*)this, type(), dataPtr());
    }

    PyObject* tp_iter_concrete() {
        return PyClassInstance::tpIterGeneric((PyObject*)this, type(), dataPtr());
    }

    PyObject* tp_iternext_concrete() {
        return PyClassInstance::tpIternextGeneric((PyObject*)this, type(), dataPtr());
    }

    PyObject* pyUnaryOperatorConcrete(const char* op, const char* opErr) {
        return PyClassInstance::pyUnaryOperatorConcreteGeneric(
            (PyObject*)this, type(), dataPtr(), op, opErr
        );
    }

    PyObject* pyOperatorConcrete(PyObject* rhs, const char* op, const char* opErr) {
        return PyClassInstance::pyOperatorConcreteGeneric(
            (PyObject*)this, type(), dataPtr(), rhs, op, opErr
        );
    }

    PyObject* pyOperatorConcreteReverse(PyObject* lhs, const char* op, const char* opErr) {
        return PyClassInstance::pyOperatorConcreteReverseGeneric(
            (PyObject*)this, type(), dataPtr(), lhs, op, opErr
        );
    }

    PyObject* pyTernaryOperatorConcrete(
        PyObject* rhs, PyObject* ternaryArg, const char* op, const char* opErr
    ) {
        return PyClassInstance::pyTernaryOperatorConcreteGeneric(
            (PyObject*)this, type(), dataPtr(), rhs, ternaryArg, op, opErr
        );
    }

    int pyInquiryConcrete(const char* op, const char* opErrRep) {
        return PyClassInstance::pyInquiryGeneric(
            (PyObject*)this, type(), dataPtr(), op, opErrRep
        );
    }

    static PyObject* clsFormat(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsFormatGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsBytes(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsBytesGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsDir(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsDirGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsReversed(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsReversedGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsComplex(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsComplexGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsRound(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsRoundGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsTrunc(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsTruncGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsFloor(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsFloorGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsCeil(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsCeilGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsEnter(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsEnterGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }
    static PyObject* clsExit(PyObject* o, PyObject* args, PyObject* kwargs) {
        return PyClassInstance::clsExitGeneric(
            o, ((PyInstance*)o)->type(), ((PyInstance*)o)->dataPtr(), args, kwargs
        );
    }

    static void initializeClassWithDefaultArguments(
        HeldClass* cls,
        instance_ptr data,
        PyObject* args,
        PyObject* kwargs
    ) {
        if (PyTuple_Size(args)) {
            PyErr_Format(PyExc_TypeError,
                "default __init__ for instances of '%s' doesn't accept positional arguments.",
                cls->name().c_str()
                );
            throw PythonExceptionSet();
        }

        if (!kwargs) {
            return;
        }

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            int res = PyClassInstance::tpSetattrGeneric(nullptr, cls, data, key, value);

            if (res != 0) {
                throw PythonExceptionSet();
            }
        }
    }

    static void constructFromPythonArgumentsConcrete(
        HeldClass* classT,
        instance_ptr data,
        PyObject* args,
        PyObject* kwargs
    ) {
        classT->constructor(data, true /* allowEmpty */);

        auto it = classT->getMemberFunctions().find("__init__");
        if (it == classT->getMemberFunctions().end()) {
            //run the default constructor
            PyHeldClassInstance::initializeClassWithDefaultArguments(classT, data, args, kwargs);
            return;
        }

        Function* initMethod = it->second;

        PyObjectStealer refAsObject(
            PyInstance::initialize(classT->getRefToType(), [&](instance_ptr selfData) {
                ((instance_ptr*)selfData)[0] = data;
            })
        );

        auto res = PyFunctionInstance::tryToCallAnyOverload(initMethod, nullptr, refAsObject, args, kwargs);

        if (!res.first) {
            throw std::runtime_error("Cannot find a valid overload of __init__ with these arguments.");
        }

        if (res.second) {
            decref(res.second);
        } else {
            throw PythonExceptionSet();
        }
    }


    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
        return PyClassInstance::tpGetattrGeneric((PyObject*)this, type(), dataPtr(), pyAttrName, attrName);
    }

    int tp_setattr_concrete(PyObject* attrName, PyObject* attrVal) {
        return PyClassInstance::tpSetattrGeneric((PyObject*)this, type(), dataPtr(), attrName, attrVal);
    }

    static void mirrorTypeInformationIntoPyTypeConcrete(HeldClass* classT, PyTypeObject* pyType) {
        PyClassInstance::mirrorTypeInformationIntoPyTypeConcrete(
            classT->getClassType(),
            pyType,
            true /* asHeldClass */
        );
    }

    static PyMethodDef* typeMethodsConcrete(Type* t) {
        // List of magic methods that are not attached to direct function pointers in PyTypeObject.
        //   These need to be defined by adding entries to PyTypeObject.tp_methods
        //   and we need to avoid adding them to PyTypeObject.tp_dict ourselves.
        //   Also, we only want to add the entry to tp_methods if they are explicitly defined.
        const std::map<const char*, PyCFunction> special_magic_methods = {
                {"__format__", (PyCFunction)clsFormat},
                {"__bytes__", (PyCFunction)clsBytes},
                {"__dir__", (PyCFunction)clsDir},
                {"__reversed__", (PyCFunction)clsReversed},
                {"__complex__", (PyCFunction)clsComplex},
                {"__round__", (PyCFunction)clsRound},
                {"__trunc__", (PyCFunction)clsTrunc},
                {"__floor__", (PyCFunction)clsFloor},
                {"__ceil__", (PyCFunction)clsCeil},
                {"__enter__", (PyCFunction)clsEnter},
                {"__exit__", (PyCFunction)clsExit}
            };

        int cur = 0;
        auto clsMethods = ((HeldClass*)t)->getMemberFunctions();
        PyMethodDef* ret = new PyMethodDef[special_magic_methods.size() + 1];
        for (auto m: special_magic_methods) {
            if (clsMethods.find(m.first) != clsMethods.end()) {
                ret[cur++] =  {m.first, m.second, METH_VARARGS | METH_KEYWORDS, NULL};
            }
        }
        ret[cur] = {NULL, NULL};
        return ret;
    }
};
