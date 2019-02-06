#pragma once

#include "PyInstance.hpp"

class PyStringInstance : public PyInstance {
public:
    typedef String modeled_type;

    static void copyConstructFromPythonInstanceConcrete(String* eltType, instance_ptr tgt, PyObject* pyRepresentation, bool isExplicit) {
        if (PyUnicode_Check(pyRepresentation)) {
            auto kind = PyUnicode_KIND(pyRepresentation);
            assert(
                kind == PyUnicode_1BYTE_KIND ||
                kind == PyUnicode_2BYTE_KIND ||
                kind == PyUnicode_4BYTE_KIND
                );
            String().constructor(
                tgt,
                kind == PyUnicode_1BYTE_KIND ? 1 :
                kind == PyUnicode_2BYTE_KIND ? 2 :
                                                4,
                PyUnicode_GET_LENGTH(pyRepresentation),
                kind == PyUnicode_1BYTE_KIND ? (const char*)PyUnicode_1BYTE_DATA(pyRepresentation) :
                kind == PyUnicode_2BYTE_KIND ? (const char*)PyUnicode_2BYTE_DATA(pyRepresentation) :
                                               (const char*)PyUnicode_4BYTE_DATA(pyRepresentation)
                );
            return;
        }
        throw std::logic_error("Can't initialize a String from an instance of " +
            std::string(pyRepresentation->ob_type->tp_name));
    }

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return PyUnicode_Check(pyRepresentation);
    }

    static PyObject* extractPythonObjectConcrete(String* t, instance_ptr data) {
        int bytes_per_codepoint = String().bytes_per_codepoint(data);

        return PyUnicode_FromKindAndData(
            bytes_per_codepoint == 1 ? PyUnicode_1BYTE_KIND :
            bytes_per_codepoint == 2 ? PyUnicode_2BYTE_KIND :
                                       PyUnicode_4BYTE_KIND,
            String().eltPtr(data, 0),
            String().count(data)
            );
    }

};

