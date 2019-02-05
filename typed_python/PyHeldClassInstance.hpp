#pragma once

#include "PyInstance.hpp"

class PyHeldClassInstance : public PyInstance {
public:
    typedef HeldClass modeled_type;

    static bool pyValCouldBeOfTypeConcrete(modeled_type* type, PyObject* pyRepresentation) {
        return true;
    }
};
