#pragma once

#include "PyInstance.hpp"

class PyBoundMethodInstance : public PyInstance {
public:
    typedef BoundMethod modeled_type;

    BoundMethod* type();
};
