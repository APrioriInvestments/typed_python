from typing import Dict, List
from typed_python.compiler.module_definition import ModuleDefinition
from typed_python import PointerTo, ListOf, Class, SerializationContext
from typed_python import _types


class LoadedModule:
    """Represents a bundle of compiled functions that are now loaded in memory.
    Members:
        functionPointers - a map from name to NativeFunctionPointer giving the
            public interface of the module
        serializedGlobalVariableDefinitions - a map from LLVM-assigned global name to serialized GlobalVariableDefinition
            giving the loadable strings
    """
    GET_GLOBAL_VARIABLES_NAME = ModuleDefinition.GET_GLOBAL_VARIABLES_NAME

    def __init__(self, functionPointers, serializedGlobalVariableDefinitions):
        self.functionPointers = functionPointers
        assert ModuleDefinition.GET_GLOBAL_VARIABLES_NAME in self.functionPointers

        self.serializedGlobalVariableDefinitions = serializedGlobalVariableDefinitions
        self.orderedDefs = [
            self.serializedGlobalVariableDefinitions[name] for name in sorted(self.serializedGlobalVariableDefinitions)
        ]
        self.orderedDefNames = sorted(list(self.serializedGlobalVariableDefinitions.keys()))
        self.pointers = ListOf(PointerTo(int))()
        self.pointers.resize(len(self.orderedDefs))

        self.functionPointers[ModuleDefinition.GET_GLOBAL_VARIABLES_NAME](self.pointers.pointerUnsafe(0))

        self.installedGlobalVariableDefinitions = {}

        for i, o in enumerate(self.orderedDefNames):
            if 'type_pointer_' in o and 'ProxyRespondWithContents' in o:
                print("Module ", self, " has ", o, " at ", int(self.pointers[i]))

    @staticmethod
    def validateGlobalVariables(serializedGlobalVariableDefinitions: Dict[str, bytes]) -> bool:
        """Check that each global variable definition is sensible.
        Sometimes we may successfully deserialize a global variable from a cached
        module, but then some dictionary member is not valid because it was removed
        or has the wrong type. In this case, we need to evict this module from
        the cache because it's no longer valid.

        Args:
            serializedGlobalVariableDefinitions: a dict from string to a serialized GlobalVariableMetadata
        """
        for gvd in serializedGlobalVariableDefinitions.values():
            meta = SerializationContext().deserialize(gvd).metadata
            if meta.matches.PointerToTypedPythonObjectAsMemberOfDict:
                if not isinstance(meta.sourceDict, dict):
                    return False

                if meta.name not in meta.sourceDict:
                    return False

                if not isinstance(meta.sourceDict[meta.name], meta.type):
                    return False

            if meta.matches.PointerToTypedPythonObject:
                if not isinstance(meta.value, meta.type):
                    return False

            if meta.matches.ClassVtable:
                if not issubclass(meta.value, Class):
                    return False

        return True

    def linkGlobalVariables(self, variable_names: List[str] = None) -> None:
        """Walk over all global variables in `variable_names` and make sure they are populated.
        Each module has a bunch of global variables that contain references to things
        like type objects, string objects, python module members, etc.
        The metadata about these is stored in 'self.serializedGlobalVariableDefinitions' whose keys
        are names and whose values are GlobalVariableMetadata instances.
        Every module we compile exposes a member named ModuleDefinition.GET_GLOBAL_VARIABLES_NAME
        which takes a pointer to a list of pointers and fills it out with the global variables.
        When the module is loaded, all the variables are initialized to zero. This function
        walks over them and populates them, effectively linking them into the current binary.
        """

        if variable_names is None:
            i_vals = range(len(self.orderedDefs))
        else:
            i_vals = [self.orderedDefNames.index(x) for x in variable_names]

        for i in i_vals:
            assert self.pointers[i], f"Failed to get a pointer to {self.orderedDefs[i].name}"

            meta = SerializationContext().deserialize(self.orderedDefs[i]).metadata

            self.installedGlobalVariableDefinitions[i] = meta

            if meta.matches.StringConstant:
                self.pointers[i].cast(str).initialize(meta.value)

            elif meta.matches.IntegerConstant:
                self.pointers[i].cast(int).initialize(meta.value)

            elif meta.matches.BytesConstant:
                self.pointers[i].cast(bytes).initialize(meta.value)

            elif meta.matches.PointerToPyObject:
                self.pointers[i].cast(object).initialize(meta.value)

            elif meta.matches.PointerToTypedPythonObject:
                self.pointers[i].cast(meta.type).initialize(meta.value)

            elif meta.matches.PointerToTypedPythonObjectAsMemberOfDict:
                self.pointers[i].cast(meta.type).initialize(meta.sourceDict[meta.name])

            elif meta.matches.ClassMethodDispatchSlot:
                slotIx = _types.allocateClassMethodDispatch(
                    meta.clsType,
                    meta.methodName,
                    meta.retType,
                    meta.argTupleType,
                    meta.kwargTupleType
                )
                self.pointers[i].cast(int).initialize(slotIx)

            elif meta.matches.IdOfPyObject:
                self.pointers[i].cast(int).initialize(id(meta.value))

            elif meta.matches.ClassVtable:
                self.pointers[i].cast(int).initialize(
                    _types._vtablePointer(meta.value)
                )

            elif meta.matches.RawTypePointer:
                if 'ProxyRespondWithContents' in str(meta.value):
                    print(f"module {self} initializing {meta.value} at {int(self.pointers[i])}")

                self.pointers[i].cast(int).initialize(
                    _types.getTypePointer(meta.value)
                )
