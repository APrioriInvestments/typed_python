#include "FunctionGlobal.hpp"
#include "CompilerVisibleObjectVisitor.hpp"


/* static */
std::pair<std::string, FunctionGlobal> FunctionGlobal::DottedGlobalsLookup(
    PyObject* funcGlobals,
    std::string dotSequence
) {
    if (!PyDict_Check(funcGlobals)) {
        throw std::runtime_error("Can't handle a non-dict function globals");
    }

    std::string shortGlobalName = dotSequence;

    size_t indexOfDot = shortGlobalName.find('.');
    if (indexOfDot != std::string::npos) {
        shortGlobalName = shortGlobalName.substr(0, indexOfDot);
    }

    PyObject* globalVal = PyDict_GetItemString(funcGlobals, shortGlobalName.c_str());

    if (!globalVal) {
        PyObject* builtins = PyDict_GetItemString(funcGlobals, "__builtins__");

        if (builtins && PyDict_Check(builtins) && PyDict_GetItemString(builtins, shortGlobalName.c_str())) {
            return DottedGlobalsLookup(builtins, dotSequence);
        }
    }

    std::string moduleName = CompilerVisibleObjectVisitor::isPyObjectGloballyIdentifiableModuleDict(funcGlobals);
    if (moduleName.size()) {
        return std::make_pair(
            shortGlobalName,
            FunctionGlobal::NamedModuleMember(
                funcGlobals,
                moduleName,
                shortGlobalName
            )
        );
    }

    // TODO: use the remainder of the dot sequence usefully
    return std::make_pair(
        shortGlobalName,
        FunctionGlobal::GlobalInDict(funcGlobals, shortGlobalName)
    );
}
