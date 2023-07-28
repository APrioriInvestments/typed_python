#include "PyObjSnapshot.hpp"
#include "PyObjGraphSnapshot.hpp"


/* static */
PyObjSnapshot* PyObjSnapshot::internalizePyObj(
    PyObject* val,
    std::unordered_map<PyObject*, PyObjSnapshot*>& constantMapCache,
    const std::map<::Type*, ::Type*>& groupMap,
    bool linkBackToOriginalObject,
    PyObjGraphSnapshot* graph
) {
    auto it = constantMapCache.find(val);

    if (it != constantMapCache.end()) {
        return it->second;
    }

    constantMapCache[val] = new PyObjSnapshot(graph);

    if (graph) {
        graph->registerSnapshot(constantMapCache[val]);
    }

    constantMapCache[val]->becomeInternalizedOf(
        val, constantMapCache, groupMap, linkBackToOriginalObject, graph
    );

    return constantMapCache[val];
}
