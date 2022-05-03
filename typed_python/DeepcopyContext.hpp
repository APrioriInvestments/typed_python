#pragma once

#include <unordered_map>
#include "Slab.hpp"

class DeepcopyContext {
public:
    DeepcopyContext(Slab* inSlab) : slab(inSlab) {
    }

    void memoize(PyObject* source, PyObject* dest) {
        pyObjectsToKeepAlive.push_back(PyObjectHolder(source));
        pyObjectsToKeepAlive.push_back(PyObjectHolder(dest));
        alreadyAllocated[(instance_ptr)source] = (instance_ptr)dest;
    }

    std::unordered_map<instance_ptr, instance_ptr> alreadyAllocated;

    std::vector<PyObjectHolder> pyObjectsToKeepAlive;

    Slab* slab;

    std::unordered_map<Type*, PyObject*> tpTypeMap;

    std::unordered_map<PyObject*, PyObject*> pyTypeMap;
};
