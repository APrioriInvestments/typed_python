#   Copyright 2017-2023 typed_python Authors
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

from typed_python.compiler.native_compiler.module_definition import ModuleDefinition
from typed_python.compiler.native_compiler.typed_llvm_value import TypedLLVMValue
from typed_python.compiler.native_compiler.native_ast_to_llvm_function_converter import (
    type_to_llvm_type,
    FunctionConverter,
    llvmI64,
    populate_needed_externals
)

import typed_python.compiler.native_compiler.native_ast as native_ast
import llvmlite.ir
import os


class NativeAstToLlvmConverter:
    def __init__(self):
        object.__init__(self)
        self._modules = {}
        self._functions_by_name = {}
        self._function_definitions = {}

        # a map from function name to function type for functions that
        # are defined in external shared objects and linked in to this one.
        self._externallyDefinedFunctionTypes = {}

        # total number of instructions in each function, by name
        self._function_complexity = {}

        self._inlineRequests = []

        self._printAllNativeCalls = os.getenv("TP_COMPILER_LOG_NATIVE_CALLS")
        self.verbose = False

    def markExternal(self, functionNameToType):
        """Provide type signatures for a set of external functions."""
        self._externallyDefinedFunctionTypes.update(functionNameToType)

    def canBeInlined(self, name):
        return name not in self._externallyDefinedFunctionTypes

    def totalFunctionComplexity(self, name):
        """Return the total number of instructions contained in a function.

        The function must already have been defined in a prior parss. We use this
        information to decide which functions to repeat in new module definitions.
        """
        if name in self._function_complexity:
            return self._function_complexity[name]

        res = 0
        for block in self._functions_by_name[name].basic_blocks:
            res += len(block.instructions)

        self._function_complexity[name] = res

        return res

    def repeatFunctionInModule(self, name, module):
        """Request that the function given by 'name' be inlined into 'module'.

        It must already have been defined in another module.

        Returns:
            a fresh unpopulated function definition for the given function.
        """
        assert name in self._functions_by_name
        assert self._functions_by_name[name].module != module

        existingFunctionDef = self._functions_by_name[name]

        funcType = existingFunctionDef.type
        if funcType.is_pointer:
            funcType = funcType.pointee

        assert isinstance(funcType, llvmlite.ir.FunctionType)

        self._functions_by_name[name] = llvmlite.ir.Function(module, funcType, name)

        self._inlineRequests.append(name)

        return self._functions_by_name[name]

    def add_functions(self, names_to_definitions):
        names_to_definitions = dict(names_to_definitions)

        for name in names_to_definitions:
            assert name not in self._functions_by_name, "can't define %s twice" % name

        module_name = "module_%s" % len(self._modules)

        module = llvmlite.ir.Module(name=module_name)

        self._modules[module_name] = module

        external_function_references = {}
        populate_needed_externals(external_function_references, module)

        functionTypes = {}

        for name, function in names_to_definitions.items():
            functionTypes[name] = native_ast.Type.Function(
                output=function.output_type,
                args=[x[1] for x in function.args],
                varargs=False,
                can_throw=True
            )
            func_type = llvmlite.ir.FunctionType(
                type_to_llvm_type(function.output_type),
                [type_to_llvm_type(x[1]) for x in function.args]
            )
            self._functions_by_name[name] = llvmlite.ir.Function(module, func_type, name)

            self._functions_by_name[name].linkage = 'external'
            self._function_definitions[name] = function

        if self.verbose:
            for name in names_to_definitions:
                definition = names_to_definitions[name]
                func = self._functions_by_name[name]

                print()
                print("*************")
                print(
                    "def %s(%s): #->%s" % (
                        name,
                        ",".join(["%s=%s" % (k, str(t)) for k, t in definition.args]),
                        str(definition.output_type)
                    )
                )
                print(native_ast.indent(str(definition.body.body)))
                print("*************")
                print()

        globalDefinitions = {}
        globalDefinitionsLlvmValues = {}
        usedExternalFunctions = set()

        while names_to_definitions:
            for name in sorted(names_to_definitions):
                definition = names_to_definitions.pop(name)
                func = self._functions_by_name[name]
                func.attributes.personality = external_function_references["tp_gxx_personality_v0"]

                # for a in func.args:
                #     if a.type.is_pointer:
                #         a.add_attribute("noalias")

                arg_assignments = {}
                for i in range(len(func.args)):
                    arg_assignments[definition.args[i][0]] = \
                        TypedLLVMValue(func.args[i], definition.args[i][1])

                block = func.append_basic_block('entry')
                builder = llvmlite.ir.IRBuilder(block)

                try:
                    func_converter = FunctionConverter(
                        module,
                        globalDefinitions,
                        globalDefinitionsLlvmValues,
                        func,
                        self,
                        builder,
                        arg_assignments,
                        definition.output_type,
                        external_function_references,
                        usedExternalFunctions
                    )

                    func_converter.setup()

                    res = func_converter.convert(definition.body.body)

                    func_converter.finalize()

                    if res is not None:
                        if definition.output_type != native_ast.Void:
                            assert not builder.block.is_terminated
                            assert definition.output_type == res.native_type, (
                                definition.output_type, res.native_type
                            )

                            builder.ret(res.llvm_value)
                        else:
                            builder.ret_void()
                    else:
                        if not builder.block.is_terminated:
                            builder.unreachable()

                except Exception:
                    print("function failing = " + name)
                    import traceback
                    traceback.print_exc()
                    raise

            # each function listed here was deemed 'inlinable', which means that we
            # want to repeat its definition in this particular module.
            for name in self._inlineRequests:
                names_to_definitions[name] = self._function_definitions[name]
            self._inlineRequests.clear()

        # define a function that accepts a pointer and fills it out with a table of pointer values
        # so that we can link in any type objects that are defined within the source code.
        self.defineGlobalMetadataAccessor(module, globalDefinitions, globalDefinitionsLlvmValues)

        functionTypes[ModuleDefinition.GET_GLOBAL_VARIABLES_NAME] = native_ast.Type.Function(
            output=native_ast.Void,
            args=[native_ast.Void.pointer().pointer()]
        )

        return ModuleDefinition(
            str(module),
            functionTypes,
            globalDefinitions,
            usedExternalFunctions
        )

    def defineGlobalMetadataAccessor(self, module, globalDefinitions, globalDefinitionsLlvmValues):
        """Given a list of global variables, make a function to access them.

        The function will be named '.get_global_variables' and will accept
        a single argument that takes a PointerTo(PointerTo(None)) and fills
        it out with the values of the globalDefinitions in their lexical
        ordering.
        """
        accessorFunction = llvmlite.ir.Function(
            module,
            type_to_llvm_type(
                native_ast.Type.Function(
                    output=native_ast.Void,
                    args=[native_ast.Void.pointer().pointer()],
                    varargs=False,
                    can_throw=False
                )
            ),
            ModuleDefinition.GET_GLOBAL_VARIABLES_NAME
        )

        accessorFunction.linkage = "external"

        outPtr = accessorFunction.args[0]

        block = accessorFunction.append_basic_block('entry')
        builder = llvmlite.ir.IRBuilder(block)
        voidPtr = type_to_llvm_type(native_ast.Void.pointer())

        index = 0
        for name in sorted(globalDefinitions):
            builder.store(
                builder.bitcast(
                    globalDefinitionsLlvmValues[name].llvm_value,
                    voidPtr
                ),
                builder.gep(outPtr, [llvmI64(index)])
            )
            index += 1

        builder.ret_void()
