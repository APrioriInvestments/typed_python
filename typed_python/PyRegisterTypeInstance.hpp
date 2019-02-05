#pragma once

#include "PyInstance.hpp"

template<class T>
class PyRegisterTypeInstance : public PyInstance {
public:
    typedef RegisterType<T> modeled_type;

    static void copyConstructFromPythonInstanceConcrete(RegisterType<T>* eltType, instance_ptr tgt, PyObject* pyRepresentation) {
        Type::TypeCategory cat = eltType->getTypeCategory();

        if (cat == Type::TypeCategory::catInt64 ||
            cat == Type::TypeCategory::catInt32 ||
            cat == Type::TypeCategory::catInt16 ||
            cat == Type::TypeCategory::catInt8 ||
            cat == Type::TypeCategory::catUInt64 ||
            cat == Type::TypeCategory::catUInt32 ||
            cat == Type::TypeCategory::catUInt16 ||
            cat == Type::TypeCategory::catUInt8 ||
            cat == Type::TypeCategory::catBool
            ) {
            if (PyLong_Check(pyRepresentation)) {
                ((T*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize an " + eltType->name() + " from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        if (cat == Type::TypeCategory::catFloat64 ||
            cat == Type::TypeCategory::catFloat32) {
            if (PyLong_Check(pyRepresentation)) {
                ((T*)tgt)[0] = PyLong_AsLong(pyRepresentation);
                return;
            }
            if (PyFloat_Check(pyRepresentation)) {
                ((T*)tgt)[0] = PyFloat_AsDouble(pyRepresentation);
                return;
            }
            throw std::logic_error("Can't initialize a " + eltType->name() + " from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        }

        PyInstance::copyConstructFromPythonInstanceConcrete(eltType, tgt, pyRepresentation);
    }
};

