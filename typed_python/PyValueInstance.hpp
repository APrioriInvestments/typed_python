#pragma once

#include "PyInstance.hpp"

class PyValueInstance : public PyInstance {
public:
    typedef Value modeled_type;

    static void copyConstructFromPythonInstanceConcrete(Value* v, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        const Instance& elt = v->value();

        if (compare_to_python(elt.type(), elt.data(), pyRepresentation, isExplicit ? false : true, Py_NE)) {
            throw std::logic_error("Can't initialize a " + v->name() + " from an instance of " +
                std::string(pyRepresentation->ob_type->tp_name));
        } else {
            //it's the value we want
            return;
        }
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* valType, PyObject* pyRepresentation) {
        return compare_to_python(valType->value().type(), valType->value().data(), pyRepresentation, true, Py_EQ);
    }

    static PyObject* extractPythonObjectConcrete(Value* valueType, instance_ptr data) {
        return extractPythonObject(valueType->value().data(), valueType->value().type());
    }
};

