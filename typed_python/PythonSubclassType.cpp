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

    typedef std::pair<Type*, PyTypeObject*> keytype;

    static std::map<keytype, PythonSubclass*> m;

    auto it = m.find(keytype(base, pyType));

    if (it == m.end()) {
        it = m.insert(
            std::make_pair(keytype(base,pyType), new PythonSubclass(base, pyType))
            ).first;
    }

    return it->second;
}

