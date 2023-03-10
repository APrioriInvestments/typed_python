"""
try to trigger markModuleHashInvalid.
"""
import networkx as nx
import os

# this has to be set prior to be TP import
os.environ["TP_COMPILER_CACHE"] = "compiler_cache"

from typed_python import (
    Entrypoint,
    Runtime,
    Class,
    Member,
    ListOf,
    SerializationContext,
)
# from typed_python.compiler.compiler_cache import CacheDependencyGraph
import invalid_module_2  # , invalid_module_3

# from invalid_module_3 import g

@Entrypoint
def test_f(x):
    return invalid_module_2.g1(x)


from glob import glob

def delete_function(func_name):
    for folder in glob('compiler_cache/*'):
        with open(os.path.join(folder, "type_manifest.dat"), "rb") as f:
            callTargets = SerializationContext().deserialize(f.read())

        with open(os.path.join(folder, "native_type_manifest.dat"), "rb") as f:
            functionNameToNativeType = SerializationContext().deserialize(f.read())

        deletions = []
        for key in callTargets:
            if func_name in key:
                deletions.append(key)

        for key in deletions:
            del callTargets[key]

        deletions = []
        for key in functionNameToNativeType:
            if func_name in key:
                deletions.append(key)

        for key in deletions:
            del functionNameToNativeType[key]

        with open(os.path.join(folder, "type_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(callTargets))

        # write the nativetype manifest
        with open(os.path.join(folder, "native_type_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(functionNameToNativeType))


if __name__ == "__main__":

    # delete_function('test_f')
    invalid_module_2.g1(1)
    # test_f(2)
    invalid_module_2.f(1)
    invalid_module_2.g2(1)

    # inspect
    runtime = Runtime.singleton()
    converter = runtime.converter
    compiler_cache = runtime.compilerCache

    # for hash, module in compiler_cache.loadedBinarySharedObjects.items():
    #     print(module, hash)
    #     for name, hash2 in compiler_cache.nameToModuleHash.items():
    #         if hash == hash2:
    #             print('\t', name)
    #     print()

    # let us attempt to load a module.

    # what does the link do?

    # for gvd in module.globalVariableDefinitions.values():
    #     pass

    # meta = gvd.metadata

    # if meta.matches.PointerToTypedPythonObjectAsMemberOfDict:
    #     print('hit')
    # else:
    #     print('MISS')

    # with open("compiler_cache/2ea68c10b63a29f5469016e0ad05f5169076be80/globals_manifest.dat", "rb") as f:
    #     globalVarDefs = SerializationContext().deserialize(f.read())

    # val = list(globalVarDefs.values())[0]
    # print(val.metadata.sourceDict.keys())
    # print(val.metadata.name)

    # graph_deps = converter._globals_dependencies


    # cache_graph = CacheDependencyGraph(compiler_cache=compiler_cache)

    # G = cache_graph.directed_graph

    # colors = list(nx.get_node_attributes(G, 'is_global').values())
    # # # print(colors)
    # import matplotlib.pyplot as plt
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_color=colors)
    # nx.draw_networkx_labels(G, pos, font_size=16)
    # nx.draw_networkx_edges(G, pos)
    # plt.show()
