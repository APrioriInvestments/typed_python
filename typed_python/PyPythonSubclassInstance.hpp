#pragma once

#include "PyInstance.hpp"

class PyPythonSubclassInstance : public PyInstance {
public:
    typedef PythonSubclass modeled_type;

    static void copyConstructFromPythonInstanceConcrete(PythonSubclass* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        Type* argType = extractTypeFrom(pyRepresentation->ob_type);

        if (argType && argType == eltType) {
            eltType->getBaseType()->copy_constructor(tgt, ((PyInstance*)pyRepresentation)->dataPtr());
            return;
        }

        if (argType && argType == eltType->getBaseType()) {
            eltType->getBaseType()->copy_constructor(tgt, ((PyInstance*)pyRepresentation)->dataPtr());
            return;
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation, isExplicit);
    }

    PyObject* tp_getattr_concrete(PyObject* pyAttrName, const char* attrName) {
        return specializeForType((PyObject*)this, [&](auto& subtype) {
            return subtype.tp_getattr_concrete(pyAttrName, attrName);
        }, type()->getBaseType());
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }

    static void constructFromPythonArgumentsConcrete(Type* t, uint8_t* data, PyObject* args, PyObject* kwargs) {
        constructFromPythonArguments(data, (Type*)t->getBaseType(), args, kwargs);
    }

    Py_ssize_t mp_and_sq_length_concrete() {
        return specializeForTypeReturningSizeT((PyObject*)this, [&](auto& subtype) {
            return subtype.mp_and_sq_length_concrete();
        }, type()->getBaseType());

    }

};

