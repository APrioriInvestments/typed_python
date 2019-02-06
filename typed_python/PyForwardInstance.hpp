#pragma once

#include "PyInstance.hpp"

class PyForwardInstance : public PyInstance {
public:
    typedef Forward modeled_type;

    static bool pyValCouldBeOfTypeConcrete(Type* t, PyObject* pyRepresentation) {
        return false;
    }
};

