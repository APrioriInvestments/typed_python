#include "AllTypes.hpp"

bool PythonSubclass::isBinaryCompatibleWithConcrete(Type* other) {
    Type* nonPyBase = m_base;
    while (nonPyBase->getTypeCategory() == TypeCategory::catPythonSubclass) {
        nonPyBase = nonPyBase->getBaseType();
    }

    Type* otherNonPyBase = other;
    while (otherNonPyBase->getTypeCategory() == TypeCategory::catPythonSubclass) {
        otherNonPyBase = otherNonPyBase->getBaseType();
    }

    return nonPyBase->isBinaryCompatibleWith(otherNonPyBase);
}

// static
PythonSubclass* PythonSubclass::Make(Type* base, PyTypeObject* pyType) {
    static std::mutex guard;

    std::lock_guard<std::mutex> lock(guard);

    static std::map<PyTypeObject*, PythonSubclass*> m;

    auto it = m.find(pyType);

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(pyType, new PythonSubclass(base, pyType))
            ).first;
    }

    if (it->second->getBaseType() != base) {
        throw std::runtime_error(
            "Expected to find the same base type. Got "
                + it->second->getBaseType()->name() + " != " + base->name()
            );
    }

    return it->second;
}

