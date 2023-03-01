from typed_python.compiler.module_definition import ModuleDefinition
from typed_python import PointerTo, ListOf, Class
from typed_python import _types


class LoadedModule:
    """Represents a bundle of compiled functions that are now loaded in memory.

    Members:
        functionPointers - a map from name to NativeFunctionPointer giving the
            public interface of the module
        globalVariableDefinitions - a map from name to GlobalVariableDefinition
            giving the loadable strings
    """
    GET_GLOBAL_VARIABLES_NAME = ModuleDefinition.GET_GLOBAL_VARIABLES_NAME

    def __init__(self, functionPointers, globalVariableDefinitions):
        self.functionPointers = functionPointers

        self.globalVariableDefinitions = globalVariableDefinitions

    @staticmethod
    def validateGlobalVariables(globalVariableDefinitions):
        """Check that each global variable definition is sensible.

        Sometimes we may successfully deserialize a global variable from a cached
        module, but then some dictionary member is not valid because it was removed
        or has the wrong type. In this case, we need to evict this module from
        the cache because it's no longer valid.

        Args:
            globalVariableDefinitions - a dict from string to GlobalVariableMetadata
        """
        for gvd in globalVariableDefinitions.values():
            meta = gvd.metadata

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

    def linkGlobalVariables(self):
        """Walk over all global variables in the module and make sure they are populated.

        Each module has a bunch of global variables that contain references to things
        like type objects, string objects, python module members, etc.

        The metadata about these is stored in 'self.globalVariableDefinitions' whose keys
        are names and whose values are GlobalVariableMetadata instances.

        Every module we compile exposes a member named ModuleDefinition.GET_GLOBAL_VARIABLES_NAME
        which takes a pointer to a list of pointers and fills it out with the global variables.

        When the module is loaded, all the variables are initialized to zero. This function
        walks over them and populates them, effectively linking them into the current binary.
        """
        assert ModuleDefinition.GET_GLOBAL_VARIABLES_NAME in self.functionPointers

        orderedDefs = [
            self.globalVariableDefinitions[name] for name in sorted(self.globalVariableDefinitions)
        ]

        pointers = ListOf(PointerTo(int))()
        pointers.resize(len(orderedDefs))

        self.functionPointers[ModuleDefinition.GET_GLOBAL_VARIABLES_NAME](pointers.pointerUnsafe(0))

        for i in range(len(orderedDefs)):
            assert pointers[i], f"Failed to get a pointer to {orderedDefs[i].name}"

            meta = orderedDefs[i].metadata

            if meta.matches.StringConstant:
                pointers[i].cast(str).initialize(meta.value)

            if meta.matches.IntegerConstant:
                pointers[i].cast(int).initialize(meta.value)

            elif meta.matches.BytesConstant:
                pointers[i].cast(bytes).initialize(meta.value)

            elif meta.matches.PointerToPyObject:
                pointers[i].cast(object).initialize(meta.value)

            elif meta.matches.PointerToTypedPythonObject:
                pointers[i].cast(meta.type).initialize(meta.value)

            elif meta.matches.PointerToTypedPythonObjectAsMemberOfDict:
                pointers[i].cast(meta.type).initialize(meta.sourceDict[meta.name])

            elif meta.matches.ClassMethodDispatchSlot:
                slotIx = _types.allocateClassMethodDispatch(
                    meta.clsType,
                    meta.methodName,
                    meta.retType,
                    meta.argTupleType,
                    meta.kwargTupleType
                )
                pointers[i].cast(int).initialize(slotIx)

            elif meta.matches.IdOfPyObject:
                pointers[i].cast(int).initialize(id(meta.value))

            elif meta.matches.ClassVtable:
                pointers[i].cast(int).initialize(
                    _types._vtablePointer(meta.value)
                )

            elif meta.matches.RawTypePointer:
                pointers[i].cast(int).initialize(
                    _types.getTypePointer(meta.value)
                )
