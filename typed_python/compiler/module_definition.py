#   Copyright 2017-2020 typed_python Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typed_python import sha_hash


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
        self.hash = sha_hash(moduleText)
