#pragma once

#include <unordered_map>
#include "Slab.hpp"

class DeepcopyContext {
public:
    DeepcopyContext(Slab* inSlab) : slab(inSlab) {
    }

    std::unordered_map<instance_ptr, instance_ptr> alreadyAllocated;

    Slab* slab;

    std::unordered_map<Type*, PyObject*> tpTypeMap;

    std::unordered_map<PyObject*, PyObject*> pyTypeMap;
};
