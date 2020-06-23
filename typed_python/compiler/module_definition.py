class ModuleDefinition:
    """A single module of compiled llvm code.

    Members:
        moduleText - a string containing the llvm IR for the module
        functionList - a list of the names of exported functions
        globalDefinitions - a dict from name to a GlobalDefinition
    """
    GET_GLOBAL_VARIABLES_NAME = ".get_global_variables"

    def __init__(self, moduleText, functionNameToType, globalVariableDefinitions):
        self.moduleText = moduleText
        self.functionNameToType = functionNameToType
        self.globalVariableDefinitions = globalVariableDefinitions
